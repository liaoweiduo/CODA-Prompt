from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import os
import sys
import argparse
import torch
import numpy as np
import yaml
import json
import random
from trainer_ft import Trainer

def create_args():
    
    # This function prepares the variables shared across demo.py
    parser = argparse.ArgumentParser()

    # Standard Args
    parser.add_argument('--gpuid', nargs="+", type=int, default=[0],
                         help="The list of gpuid, ex:--gpuid 3 1. Negative value means cpu-only")
    parser.add_argument('--log_dir', type=str, default="outputs/out",
                         help="Save experiments results in dir for future plotting!")
    parser.add_argument('--dataroot', type=str, default="data")
    parser.add_argument('--learner_type', type=str, default='default', help="The type (filename) of learner")
    parser.add_argument('--learner_name', type=str, default='NormalNN', help="The class name of learner")
    parser.add_argument('--debug_mode', type=int, default=0, metavar='N',
                        help="activate learner specific settings for debug_mode")
    parser.add_argument('--repeat', type=int, default=1, help="Repeat the experiment N times")
    parser.add_argument('--overwrite', type=int, default=0, metavar='N', help='Train regardless of whether saved model exists')
    parser.add_argument('--lr', nargs="+", type=float, default=[0.001, 0.001], help="lr")
    parser.add_argument('--batch_size', type=int, default=32, help="temperature for distillation")

    # CL Args          
    parser.add_argument('--oracle_flag', default=False, action='store_true', help='Upper bound for oracle')
    parser.add_argument('--upper_bound_flag', default=False, action='store_true', help='Upper bound')
    parser.add_argument('--max_task', type=int, default=-1, help="number of learned task")
    parser.add_argument('--memory', type=int, default=0, help="size of memory for replay")
    parser.add_argument('--temp', type=float, default=2., dest='temp', help="temperature for distillation")
    parser.add_argument('--DW', default=False, action='store_true', help='dataset balancing')
    parser.add_argument('--prompt_param', nargs="+", type=float, default=[1, 1, 1],
                         help="e prompt pool size, e prompt length, g prompt length")

    # Slot Args
    parser.add_argument('--only_learn_slot', default=False, action='store_true', help='only learn slots')
    parser.add_argument('--slot_pre_learn_model', type=str, default='none',
                        help="The model name to the pre-learned slot attn model.")
    parser.add_argument('--t0_model_from', type=str, default='none',
                        help="The model name to warm-start from the 2nd task.")
    parser.add_argument('--slot_lr', type=float, default=0.0003, help="slot lr")
    parser.add_argument('--slot_schedule_type', type=str, default='cosine')

    # MO Args
    parser.add_argument('--concept_weight', default=False, action='store_true',
                        help='True to use concept weighting on data.')
    parser.add_argument('--target_concept_id', type=int, default=-1, help="specify specific concept to weight")

    # CFST Args
    parser.add_argument('--mode', type=str, default='continual',
                        help="choices: [continual, sys, pro, sub, non, noc]")
    parser.add_argument('--test_model', type=int, default=-1, help="-1 for last model, starting from 1. ")
    parser.add_argument('--eval_every_epoch', default=False, action='store_true')
    parser.add_argument('--use_feature_statistics', action='store_true',
                        help='to do classification')

    # Config Arg
    parser.add_argument('--config', type=str, default="configs/config.yaml",
                         help="yaml experiment config input")

    return parser

def get_args(argv):
    parser=create_args()
    args = parser.parse_args(argv)
    config = vars(args)
    config_yaml = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    config.update(config_yaml)      # make yaml overwrite args

    # config['batch_size'] = 100    # trainer_ft -> batch_size is set to 100
    return argparse.Namespace(**config)

# want to save everything printed to outfile
class Logger(object):
    def __init__(self, name):
        self.terminal = sys.stdout
        self.log = open(name, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        self.log.flush()


def start(args):
    print(f'********start mode {args.mode}********')
    metric_keys = ['acc', 'time', ]
    save_keys = ['global']
    global_only = ['time']
    avg_metrics = {}
    for mkey in metric_keys:
        avg_metrics[mkey] = {}
        for skey in save_keys: avg_metrics[mkey][skey] = []

    # make sure continual training has finished
    if args.learner_name == 'Prompt':
        print('WARNING: test on pretrained-vit')
    else:
        save_file = args.log_dir + '/results-acc/global.yaml'
        with open(save_file, 'r') as yaml_file:
            yaml_result = yaml.safe_load(yaml_file)
            his = np.asarray(yaml_result['history'])

        # num of repeats for few-shot testing
        finished_repeats = his.shape[-1]
        if args.test_model == -1:
            # change to the last model
            args.test_model = his.shape[0]  # test model id starting from 1

        assert (finished_repeats >= args.repeat
                ), f"haven't finish continual training: finished:target={finished_repeats}:{args.repeat}."

    # load results
    if args.overwrite:
        start_r = 0
    else:
        try:
            for mkey in metric_keys:
                for skey in save_keys:
                    if (not (mkey in global_only)) or (skey == 'global'):
                        save_file = args.log_dir + '/results-' + mkey + '/' + skey + f'-{args.mode}.yaml'
                        if os.path.exists(save_file):
                            with open(save_file, 'r') as yaml_file:
                                yaml_result = yaml.safe_load(yaml_file)
                                avg_metrics[mkey][skey] = np.asarray(yaml_result['history'])

            # next repeat needed
            start_r = avg_metrics[metric_keys[0]][save_keys[0]].shape[-1]

            # extend if more repeats left
            if start_r < args.repeat:
                max_task = avg_metrics['acc']['global'].shape[0]
                for mkey in metric_keys:
                    avg_metrics[mkey]['global'] = np.append(avg_metrics[mkey]['global'],
                                                            np.zeros((max_task, args.repeat - start_r)), axis=-1)
                    if (not (mkey in global_only)):
                        avg_metrics[mkey]['pt'] = np.append(avg_metrics[mkey]['pt'],
                                                            np.zeros((max_task, max_task, args.repeat - start_r)),
                                                            axis=-1)
                        avg_metrics[mkey]['pt-local'] = np.append(avg_metrics[mkey]['pt-local'],
                                                                  np.zeros((max_task, max_task, args.repeat - start_r)),
                                                                  axis=-1)

        except:
            start_r = 0

    # start_r = 0
    for r in range(start_r, args.repeat):

        print('************************************')
        print(f'* STARTING TESTING MODEL: repeat-{r+1}/task-{args.test_model}/class.pth.')
        print('************************************')

        # set random seeds
        seed = r
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        # set up a trainer
        trainer = Trainer(args, seed, metric_keys, save_keys)

        # init total run metrics storage
        max_task = trainer.max_task
        if r == 0:
            for mkey in metric_keys:
                avg_metrics[mkey]['global'] = np.zeros((max_task,args.repeat))
                if (not (mkey in global_only)):
                    avg_metrics[mkey]['pt'] = np.zeros((max_task,max_task,args.repeat))
                    avg_metrics[mkey]['pt-local'] = np.zeros((max_task,max_task,args.repeat))

        # train model
        avg_metrics = trainer.train(avg_metrics)

        # # evaluate model
        # avg_metrics = trainer.evaluate(avg_metrics)

        # save results
        for mkey in metric_keys:
            m_dir = args.log_dir+'/results-'+mkey+'/'
            if not os.path.exists(m_dir): os.makedirs(m_dir)
            for skey in save_keys:
                if (not (mkey in global_only)) or (skey == 'global'):
                    save_file = m_dir+skey+f'-{args.mode}.yaml'
                    result=avg_metrics[mkey][skey]
                    yaml_results = {}
                    if len(result.shape) > 2:
                        yaml_results['mean'] = result[:,:,:r+1].mean(axis=2).tolist()
                        if r>1: yaml_results['std'] = result[:,:,:r+1].std(axis=2).tolist()
                        yaml_results['history'] = result[:,:,:r+1].tolist()
                    else:
                        yaml_results['mean'] = result[:,:r+1].mean(axis=1).tolist()
                        if r>1: yaml_results['std'] = result[:,:r+1].std(axis=1).tolist()
                        yaml_results['history'] = result[:,:r+1].tolist()
                    with open(save_file, 'w') as yaml_file:
                        yaml.dump(yaml_results, yaml_file, default_flow_style=False)

        # Print the summary so far
        print('===Summary of experiment repeats:',r+1,'/',args.repeat,'===')
        for mkey in metric_keys:
            print(mkey, ' | mean:', avg_metrics[mkey]['global'][-1,:r+1].mean(), 'std:', avg_metrics[mkey]['global'][-1,:r+1].std())

        '''nvidia-smi'''
        print(os.system('nvidia-smi'))

if __name__ == '__main__':
    args = get_args(sys.argv[1:])

    # determinstic backend
    torch.backends.cudnn.deterministic=True

    # duplicate output stream to output file
    if not os.path.exists(args.log_dir): os.makedirs(args.log_dir)
    log_out = args.log_dir + '/output.log'
    sys.stdout = Logger(log_out)
    log_err = args.log_dir + '/err.log'
    sys.stderr = Logger(log_err)

    print(vars(args))

    start(args)
