from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import copy
import os
import sys
from datetime import datetime
import argparse
import torch
import numpy as np
import yaml
import json
import random
from trainer import Trainer

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
    parser.add_argument('--batch_size', type=int, default=32, help="batch size")
    parser.add_argument('--lr_decreace_ratio', type=float, default=1.0,
                        help="lr on prompt = ratio * lr")

    # CL Args          
    parser.add_argument('--oracle_flag', default=False, action='store_true',
                        help='Upper bound for oracle: MTL for task[:i]')
    parser.add_argument('--upper_bound_flag', default=False, action='store_true', help='Upper bound')
    parser.add_argument('--max_task', type=int, default=-1, help="number of learned task")
    parser.add_argument('--memory', type=int, default=0, help="size of memory for replay")
    parser.add_argument('--temp', type=float, default=2., dest='temp', help="temperature for distillation")
    parser.add_argument('--DW', default=False, action='store_true', help='dataset balancing')
    parser.add_argument('--prompt_param', nargs="+", type=float, default=[1, 1, 1],
                         help="e prompt pool size, e prompt length, g prompt length")
    parser.add_argument('--larger_prompt_lr', action='store_true',
                        help='if using larger prompt lr, prompt lr = 10 * head lr')
    parser.add_argument('--eval_class_wise', default=False, action='store_true')

    # Slot Args
    parser.add_argument('--n_slots', type=int, default=10, help="number of slots for one extraction")
    parser.add_argument('--n_iters', type=int, default=5, help="num of iter to extract slots")
    parser.add_argument('--slot_temp', type=float, default=1.0,
                        help="temperature to control how sharp are slot attns")
    parser.add_argument('--s2p_temp', type=float, default=1,
                        help="temperature to control how sharp are the selection of slots")
    parser.add_argument('--s2p_mode', type=str, default='attn+sig',
                        help="some options: [attn{mlp,gate}+{FPS}+sig{soft,hard,cos,avg}]")
    parser.add_argument('--slot_cross_attn_temp', type=float, default=10.,
                        help="temperature to measure how popular of the slots across the batch imgs.")

    parser.add_argument('--only_learn_slot', default=False, action='store_true', help='only learn slots')
    parser.add_argument('--slot_pre_learn_model', type=str, default='none',
                        help="The model name to the pre-learned slot attn model.")
    parser.add_argument('--t0_model_from', type=str, default='none',
                        help="The model name to warm-start from the 2nd task.")
    parser.add_argument('--slot_lr', nargs="+", type=float, default=[0.0001, 0.00001], help="slot lr")
    parser.add_argument('--slot_schedule_type', type=str, default='cosine')
    parser.add_argument('--logit_task_mask_top_k', type=int, default=10, help="no use")

    parser.add_argument('--use_intra_consistency_reg', action='store_true')
    parser.add_argument('--intra_consistency_reg_coeff', type=float, default=1,
                        help="coeff of reg on maintaining intra-consistency of slots")
    parser.add_argument('--intra_consistency_reg_mode', type=str, default='map+cos+kl',
                        help="learn(cross)+cos(dot)+l1(l2, kl)")
    parser.add_argument('--intra_consistency_reg_temp', type=float, default=1,
                        help="temp on primitive loss.")

    parser.add_argument('--use_slot_ortho_reg', action='store_true')
    parser.add_argument('--slot_ortho_reg_mode', type=str, default='cos+ce',
                        help="dot{cos}+l2{ce}")
    parser.add_argument('--slot_ortho_reg_temp', type=float, default=1,
                        help="temp on slot ortho reg for each img.")
    parser.add_argument('--slot_ortho_reg_coeff', type=float, default=0.5,
                        help="coeff of reg on slot ortho.")

    # Prompt learn Args
    parser.add_argument('--prompt_pre_learn_model', type=str, default='none',
                        help="The model name to the pre-learned prompt model.")
    parser.add_argument('--use_feature_statistics', action='store_true')
    parser.add_argument('--use_slot_statistics', action='store_true')
    parser.add_argument('--do_not_eval_during_training', action='store_true')

    parser.add_argument('--use_weight_reg', action='store_true')
    parser.add_argument('--weight_reg_coeff', type=float, default=0.0,
                        help="coeff of reg on s2p weights changes (or response)")
    parser.add_argument('--weight_reg_mode', type=str, default='weights',
                        help="weights, response")

    parser.add_argument('--use_selection_onehot_reg', action='store_true')
    parser.add_argument('--selection_onehot_reg_coeff', type=float, default=0.0,
                        help="coeff of reg on prompt selection to make it sparse")
    parser.add_argument('--selection_onehot_reg_mode', type=str, default='l1',
                        help="l1, l2")

    parser.add_argument('--use_selection_slot_similar_reg', action='store_true')
    parser.add_argument('--selection_slot_similar_reg_mode', type=str, default='l1',
                        help="l1, l2, ce")
    parser.add_argument('--selection_slot_similar_reg_coeff', type=float, default=0.0,
                        help="coeff of reg on prompt selection sim to match with the distribution of slot sim")

    parser.add_argument('--use_prompt_concept_alignment_reg', action='store_true')
    parser.add_argument('--prompt_concept_alignment_reg_coeff', type=float, default=0.0,
                        help="coeff of reg on feature similarity of masked img with concept and use prompt.")

    parser.add_argument('--use_old_samples_for_reg', action='store_true')
    parser.add_argument('--use_old_samples_for_reg_no_grad', action='store_true')
    parser.add_argument('--concept_weight', default=False, action='store_true',
                        help='True to use concept weighting on data.')
    parser.add_argument('--target_concept_id', type=int, default=-1, help="specify specific concept to weight")
    parser.add_argument('--concept_similar_reg_coeff', type=float, default=0,
                        help="coeff for concept similar reg.")
    parser.add_argument('--concept_similar_reg_coeff_sensitivity', type=float, default=0.,
                        help="sensitivity for reg on n_cls.")
    parser.add_argument('--concept_similar_reg_temp', type=float, default=0.01,
                        help="temp on logit similarity.")
    parser.add_argument('--concept_similar_reg_mode', type=str, default='dot+kl')
    parser.add_argument('--dynamic_concept_similar_reg_coeff', default=False, action='store_true',
                        help='coeff from 0 for the first epoch.')

    parser.add_argument('--use_slot_logit_similar_reg', action='store_true')
    parser.add_argument('--slot_logit_similar_reg_coeff', type=float, default=0.,
                        help="coeff for concept similar reg.")
    parser.add_argument('--slot_logit_similar_reg_coeff_sensitivity', type=float, default=0.,
                        help="sensitivity for reg on n_cls.")
    parser.add_argument('--slot_logit_similar_reg_mode', type=str, default='map+cos+kl')
    parser.add_argument('--slot_logit_similar_reg_temp', type=float, default=0.001,
                        help="temp on logit similarity.")
    parser.add_argument('--slot_logit_similar_reg_slot_temp', type=float, default=1,
                        help="temp on logit similarity.")

    parser.add_argument('--use_knowledge_distillation', action='store_true',
                        help="learn extra coda ")

    # CFST Args
    parser.add_argument('--compositional_testing', action='store_true')
    parser.add_argument('--mode', type=str, default='continual',
                        help="choices: [continual, sys, pro, sub, non, noc]")
    parser.add_argument('--test_model', type=int, default=-1, help="-1 for last model, starting from 1. ")

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
    # if config['debug_mode'] == 1:
    #     config['batch_size'] = 16
    return argparse.Namespace(**config)

# want to save everything printed to outfile
class Logger(object):
    def __init__(self, name, mode='a'):
        self.terminal = sys.stdout
        self.log = open(name, mode)

    def write(self, message):
        self.terminal.write(f"{message}")
        self.log.write(f"{message}")
        # self.log.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]: {message}")

    def flush(self):
        self.log.flush()

def comp_test(args):
    from run_ft import start
    # check args differences
    if args.dataset == 'CGQA':
        modes = ['sys', 'pro', 'sub', 'non', 'noc']
    elif args.dataset == 'COBJ':
        modes = ['sys', 'pro', 'non', 'noc']
    else:
        print(f'{args.dataset} does not implement CFST testing.')
        return

    for mode in modes:
        args_ft = copy.deepcopy(args)
        args_ft.lr = 0.001
        args_ft.use_feature_statistics = True
        args_ft.mode = mode
        start(args_ft)


if __name__ == '__main__':
    args = get_args(sys.argv[1:])

    # determinstic backend
    torch.backends.cudnn.deterministic=True

    # duplicate output stream to output file
    if not os.path.exists(args.log_dir): os.makedirs(args.log_dir)
    log_out = args.log_dir + '/output.log'
    sys.stdout = Logger(log_out)
    log_err = args.log_dir + '/err.log'
    sys.stderr = Logger(log_err, 'w')

    # save args
    with open(args.log_dir + '/args.yaml', 'w') as yaml_file:
        yaml.dump(vars(args), yaml_file, default_flow_style=False)
    print(vars(args))
    
    metric_keys = ['acc','time',]
    save_keys = ['global', 'pt', 'pt-local']
    global_only = ['time']
    avg_metrics = {}
    for mkey in metric_keys: 
        avg_metrics[mkey] = {}
        for skey in save_keys: avg_metrics[mkey][skey] = []

    # load results
    if args.overwrite:
        start_r = 0
    else:
        try:
            for mkey in metric_keys: 
                for skey in save_keys:
                    if (not (mkey in global_only)) or (skey == 'global'):
                        save_file = args.log_dir+'/results-'+mkey+'/'+skey+'.yaml'
                        if os.path.exists(save_file):
                            with open(save_file, 'r') as yaml_file:
                                yaml_result = yaml.safe_load(yaml_file)
                                avg_metrics[mkey][skey] = np.asarray(yaml_result['history'])

            # next repeat needed
            start_r = avg_metrics[metric_keys[0]][save_keys[0]].shape[-1]

            # extend if more repeats left
            max_task = avg_metrics['acc']['global'].shape[0]
            if start_r < args.repeat:
                for mkey in metric_keys: 
                    avg_metrics[mkey]['global'] = np.append(avg_metrics[mkey]['global'], np.zeros((max_task,args.repeat-start_r)), axis=-1)
                    if (not (mkey in global_only)):
                        avg_metrics[mkey]['pt'] = np.append(avg_metrics[mkey]['pt'], np.zeros((max_task,max_task,args.repeat-start_r)), axis=-1)
                        avg_metrics[mkey]['pt-local'] = np.append(avg_metrics[mkey]['pt-local'], np.zeros((max_task,max_task,args.repeat-start_r)), axis=-1)

        except:
            start_r = 0
    # start_r = 0
    for r in range(start_r, args.repeat):

        print('************************************')
        print('* STARTING TRIAL ' + str(r+1))
        print('************************************')

        # set random seeds
        seed = r
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        # torch.cuda.manual_seed_all(seed)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False

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

        # evaluate model
        avg_metrics = trainer.evaluate(avg_metrics)    

        # save results
        for mkey in metric_keys: 
            m_dir = args.log_dir+'/results-'+mkey+'/'
            if not os.path.exists(m_dir): os.makedirs(m_dir)
            for skey in save_keys:
                if (not (mkey in global_only)) or (skey == 'global'):
                    save_file = m_dir+skey+'.yaml'
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
        os.system('nvidia-smi')     # vir sys.out and not write to log file

    args.gpuid = [0]        # prevent using multi-gpu

    # do compositional testing on all available mode
    if args.compositional_testing:
        comp_test(args)

    from debug import Debugger

    args.debug_mode = 1
    args.max_task = max_task
    debugger = Debugger(level='INFO', args=vars(args))
    res = debugger.collect_results(max_task=args.max_task, draw=True, use_dataset=True)
    df_res = debugger.generate_df(save=True)
