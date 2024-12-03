import os
import sys
import argparse
import torch
import numpy as np
import random
from random import shuffle
from collections import OrderedDict
import dataloaders
from dataloaders.utils import *
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import learners
from debug import Debugger

class Trainer:

    def __init__(self, args, seed, metric_keys, save_keys):

        # process inputs
        self.args = args
        self.seed = seed
        self.metric_keys = metric_keys
        self.save_keys = save_keys
        self.log_dir = args.log_dir
        self.batch_size = args.batch_size
        self.workers = args.workers
        
        # model load directory
        self.model_top_dir = args.log_dir

        # select dataset
        self.grayscale_vis = False
        self.top_k = 1
        if args.dataset == 'CIFAR10':
            Dataset = dataloaders.iCIFAR10
            num_classes = 10
            self.dataset_size = [32,32,3]
        elif args.dataset == 'CIFAR100':
            Dataset = dataloaders.iCIFAR100
            num_classes = 100
            self.dataset_size = [32,32,3]
        elif args.dataset == 'ImageNet_R':
            Dataset = dataloaders.iIMAGENET_R
            num_classes = 200
            self.dataset_size = [224,224,3]
            self.top_k = 1
        elif args.dataset == 'DomainNet':
            Dataset = dataloaders.iDOMAIN_NET
            num_classes = 345
            self.dataset_size = [224,224,3]
            self.top_k = 1
        elif args.dataset == 'CGQA':
            num_classes = 100
            Dataset = dataloaders.CGQA
        elif args.dataset == 'COBJ':
            num_classes = 30
            Dataset = dataloaders.COBJ
        else:
            raise ValueError('Dataset not implemented!')

        # upper bound flag
        self.upper_bound_flag = args.upper_bound_flag
        if args.upper_bound_flag:
            args.other_split_size = num_classes
            args.first_split_size = num_classes

        # load tasks
        class_order = np.arange(num_classes).tolist()
        class_order_logits = np.arange(num_classes).tolist()
        if self.seed > 0 and args.rand_split:
            print('=============================================')
            print('Shuffling....')
            print('pre-shuffle:' + str(class_order))
            random.seed(self.seed)
            random.shuffle(class_order)
            print('post-shuffle:' + str(class_order))
            print('=============================================')
        self.tasks = []
        self.tasks_logits = []
        p = 0
        while p < num_classes and (args.max_task == -1 or len(self.tasks) < args.max_task):
            inc = args.other_split_size if p > 0 else args.first_split_size
            self.tasks.append(class_order[p:p+inc])
            self.tasks_logits.append(class_order_logits[p:p+inc])
            p += inc
        self.num_tasks = len(self.tasks)
        self.task_names = [str(i+1) for i in range(self.num_tasks)]

        # number of tasks to perform
        if args.max_task > 0:
            self.max_task = min(args.max_task, len(self.task_names))
        else:
            self.max_task = len(self.task_names)

        # datasets and dataloaders
        k = 1 # number of transforms per image
        if args.model_name.startswith('vit'):
            resize_imnet = True
        else:
            resize_imnet = False

        return_concepts = True      # not for training
        train_transform = dataloaders.utils.get_transform(dataset=args.dataset, phase='train', aug=args.train_aug, resize_imnet=resize_imnet)
        test_transform  = dataloaders.utils.get_transform(dataset=args.dataset, phase='test', aug=args.train_aug, resize_imnet=resize_imnet)
        self.train_dataset = Dataset(args.dataroot, train=True, lab = True, tasks=self.tasks,
                                     download_flag=True if (args.debug_mode == 0) else False, transform=train_transform,
                                     seed=self.seed, rand_split=args.rand_split, validation=args.validation,
                                     first_split_size=args.first_split_size // 10,
                                     other_split_size=args.other_split_size // 10,
                                     return_concepts=return_concepts,
                                     mode=args.mode,
                                     )
        # if args.debug_mode == 1:
        #     self.train_dataset.debug_mode()     # use val datasets to avoid large train set loading
        self.test_dataset  = Dataset(args.dataroot, train=False, lab = True, tasks=self.tasks,
                                     download_flag=True if (args.debug_mode == 0) else False, transform=test_transform,
                                     seed=self.seed, rand_split=args.rand_split, validation=args.validation,
                                     first_split_size=args.first_split_size // 10,
                                     other_split_size=args.other_split_size // 10,
                                     return_concepts=return_concepts,
                                     mode=args.mode,
                                     )

        # for oracle
        self.oracle_flag = args.oracle_flag
        self.add_dim = 0

        # Prepare the self.learner (model)
        self.learner_config = {
            'dataset': args.dataset,
            'num_classes': num_classes,
            'lr': args.lr,
            'debug_mode': args.debug_mode == 1,
            'momentum': args.momentum,
            'weight_decay': args.weight_decay,
            'schedule': args.schedule,
            'schedule_type': args.schedule_type,
            'model_type': args.model_type,
            'model_name': args.model_name,
            'optimizer': args.optimizer,
            'gpuid': args.gpuid,
            'memory': args.memory,
            'temp': args.temp,
            'out_dim': num_classes,
            'overwrite': args.overwrite == 1,
            'DW': args.DW,
            'batch_size': args.batch_size,
            'upper_bound_flag': args.upper_bound_flag,
            'tasks': self.tasks_logits,
            'top_k': self.top_k,
            'prompt_param':[self.num_tasks,args.prompt_param],
            'mode': args.mode,
            'seed': self.seed,
            'log_dir': args.log_dir,
            # slot training args
            'only_learn_slot': args.only_learn_slot,
            'slot_pre_learn_model': args.slot_pre_learn_model,
            't0_model_from': args.t0_model_from,
            'slot_lr': args.slot_lr,
            'logit_task_mask_top_k': args.logit_task_mask_top_k,
            'slot_schedule_type': args.slot_schedule_type,
            'target_concept_id': args.target_concept_id,
            'concept_weight': args.concept_weight if hasattr(args, 'concept_weight') else False,
        }
        # # pmo settings
        # if len(args.prompt_param) > 3:
        #     self.learner_config.update({
        #         'aux_root': args.dataroot,  # no use, use train dataset instead
        #         'num_aux_sampling': 8,
        #     })

        self.learner_type, self.learner_name = args.learner_type, args.learner_name
        self.learner = learners.__dict__[self.learner_type].__dict__[self.learner_name](self.learner_config)

    def task_eval(self, t_index, local=False, task='acc'):

        val_name = self.task_names[t_index]
        print('validation split name:', val_name)
        
        # eval
        self.test_dataset.load_dataset(t_index, train=True)
        test_loader  = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False, num_workers=self.workers)
        if self.args.only_learn_slot:
            if local:
                return self.learner.validation(test_loader, task_in=self.tasks_logits[t_index], task_metric=task,
                                               slot_recon_loss=True)
            else:

                return self.learner.validation(test_loader, task_metric=task,
                                               slot_recon_loss=True)

        if local:
            return self.learner.validation(test_loader, task_in=self.tasks_logits[t_index], task_metric=task)
        else:
            return self.learner.validation(test_loader, task_metric=task)

    def class_eval(self, c_index, local=False, task='acc'):
        # local not impl
        print('validation class index:', c_index)

        # eval
        test_dataset = self.test_dataset.get_single_class_dataset(c_index)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False,
                                 num_workers=self.workers)
        if self.args.only_learn_slot:
            # if local:
            #     return self.learner.validation(test_loader, task_in=self.tasks_logits[t_index], task_metric=task,
            #                                    slot_recon_loss=True)
            # else:
            return self.learner.validation(test_loader, task_metric=task,
                                               slot_recon_loss=True)

        # if local:
        #     return self.learner.validation(test_loader, task_in=self.tasks_logits[t_index], task_metric=task)
        # else:
        return self.learner.validation(test_loader, task_metric=task)

    def train(self, avg_metrics):
    
        # temporary results saving
        temp_table = {}
        for mkey in self.metric_keys: temp_table[mkey] = []
        temp_dir = self.log_dir + '/temp/'
        if not os.path.exists(temp_dir): os.makedirs(temp_dir)
        writer = SummaryWriter(temp_dir)
        debugger = Debugger(level='DEBUG')

        # for each task
        for i in range(self.max_task):      # for few-shot testing, if should start from an offset

            # save current task index
            self.current_t_index = i

            # print name
            train_name = self.task_names[i]
            print('======================', train_name, '=======================')

            # load dataset for task
            task = self.tasks_logits[i]
            if self.oracle_flag:
                self.train_dataset.load_dataset(i, train=False)
                self.learner = learners.__dict__[self.learner_type].__dict__[self.learner_name](self.learner_config)
                self.add_dim += len(task)
            else:
                self.train_dataset.load_dataset(i, train=True)
                self.add_dim = len(task)

            # set task id for model (needed for prompting)
            try:
                self.learner.model.module.task_id = i
            except:
                self.learner.model.task_id = i

            # add valid class to classifier
            self.learner.add_valid_output_dim(self.add_dim)

            # load dataset with memory
            if self.learner_config['memory'] > 0:
                self.train_dataset.append_coreset(only=False)

            # # debug
            # self.train_dataset.update_coreset(self.learner.memory_size, np.arange(self.learner.last_valid_out_dim))

            # load dataloader
            # if self.args.learn_class_id == -1:
            #     train_dataset = self.train_dataset
            # else:
            #     train_dataset = self.train_dataset.get_single_class_dataset(self.args.learn_class_id)
            #     print(f'Train on class {self.args.learn_class_id} with len {len(train_dataset)}.')
            train_dataset = self.train_dataset
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=int(self.workers))

            # increment task id in prompting modules
            if i > 0:
                try:
                    if self.learner.model.module.prompt is not None:
                        self.learner.model.module.prompt.process_task_count()
                except:
                    if self.learner.model.prompt is not None:
                        self.learner.model.prompt.process_task_count()

            # learn
            # test_loader does not use for training
            self.test_dataset.load_dataset(i, train=False)
            # self.test_dataset.load_dataset(0, train=True)
            test_loader  = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False, num_workers=self.workers)

            model_save_dir = self.model_top_dir + '/models/repeat-'+str(self.seed+1)+'/task-'+self.task_names[i]+'/'
            if not os.path.exists(model_save_dir): os.makedirs(model_save_dir)
            avg_train_time = self.learner.learn_batch(train_loader, self.train_dataset, model_save_dir, test_loader)

            # save model
            self.learner.save_model(model_save_dir)
            
            # evaluate acc
            acc_table = []
            acc_table_ssl = []
            self.reset_cluster_labels = True

            if self.args.eval_class_wise:
                unique_labels = self.test_dataset.get_unique_labels()
                for label_id, label in enumerate(unique_labels):
                    acc = self.class_eval(label)
                    if type(acc) is list:
                        for prompt_id in range(acc):
                            # log
                            self.learner.epoch_log['scaler']['Tag'].append(
                                f'val_acc/prompt_{prompt_id}/class_{label}')
                            self.learner.epoch_log['scaler']['Idx'].append(i)       # task id
                            self.learner.epoch_log['scaler']['Value'].append(acc[prompt_id])
                    else:
                        # log
                        self.learner.epoch_log['scaler']['Tag'].append(f'val_acc/class_{label}')
                        self.learner.epoch_log['scaler']['Idx'].append(i)       # task id
                        self.learner.epoch_log['scaler']['Value'].append(acc)

            for j in range(i+1):
                task_acc = self.task_eval(j)
                if type(task_acc) is list:      # on all prompts
                    task_acc = np.mean(task_acc)
                acc_table.append(task_acc)

                # log
                self.learner.epoch_log['scaler']['Tag'].append(f'val_acc/task_{j}')
                self.learner.epoch_log['scaler']['Idx'].append(i)
                self.learner.epoch_log['scaler']['Value'].append(task_acc)

            temp_table['acc'].append(np.asarray([*acc_table, *[0 for _ in range(i+1, self.max_task)], np.mean(np.asarray(acc_table))]))      # pt and mean

            # save temporary acc results
            for mkey in ['acc']:
                save_file = temp_dir + mkey + f'_seed{self.seed}.csv'
                np.savetxt(save_file, np.asarray(temp_table[mkey]), delimiter=",", fmt='%.2f')  

            if avg_train_time is not None: avg_metrics['time']['global'][i] = avg_train_time

            '''save epoch log'''
            if hasattr(self.learner, 'epoch_log') and not os.path.exists(temp_dir + f'log_seed{self.seed}_t{i}' + '.pkl'):
                self.learner.train_log_to_df()
                epoch_log = self.learner.epoch_log

                debugger.save_log(epoch_log, temp_dir + f'log_seed{self.seed}_t{i}' + '.pkl')

                # pop_labels = [
                #     f"p{idx}" if idx < self.learner_config['n_obj'] else f"m{idx - self.learner_config['n_obj']}"
                #     for idx in range(self.learner_config['n_mix'] + self.learner_config['n_obj'])
                # ]  # ['p0', 'p1', 'm0', 'm1']

                # write after check if has log
                # '''write sampled mo images'''
                # for task_idx, task in enumerate(torch_tasks):  # only visual the last task
                #     debugger.write_task(pmo, task, pop_labels[task_idx], i=i, writer=writer, prefix='mo-image')

                # '''write pool'''
                # debugger.write_pool(pool, i=i, writer=writer, prefix=f'pool')

                if len(epoch_log['scaler']) > 0:     # perform training
                    debugger.write_scaler(epoch_log['scaler'], key='all', i=i, writer=writer, inner=True)
                    # debugger.write_scaler(epoch_log['scaler'], key='loss/ce_loss', i=i, writer=writer, inner=True)
                    # debugger.write_scaler(epoch_log['scaler'], key='loss/hv_loss', i=i, writer=writer, inner=True)
                    # debugger.write_scaler(epoch_log['scaler'], key='loss/mo_loss', i=i, writer=writer, inner=True)
                    # debugger.write_scaler(epoch_log['scaler'], key='loss/s2p_loss', i=i, writer=writer, inner=True)
                    # debugger.write_scaler(epoch_log['scaler'], key='loss/mk_loss', i=i, writer=writer, inner=True)
                    # debugger.write_scaler(epoch_log['scaler'], key='loss/ccl_loss', i=i, writer=writer, inner=True)
                    # debugger.write_scaler(epoch_log['scaler'], key='loss/slot_recon_loss', i=i, writer=writer, inner=True)
                    # # debugger.write_scaler(epoch_log['scaler'], key='val_acc_phase1', i=i, writer=writer, inner=True)
                    # debugger.write_scaler(epoch_log['scaler'], key='val_acc', i=i, writer=writer, inner=True)
                    # debugger.write_scaler(epoch_log['scaler'], key='alpha', i=i, writer=writer, inner=True)
                    # # debugger.write_scaler(epoch_log['scaler'], key='loss/et_loss', i=i, writer=writer, inner=True)
                    # # debugger.write_scaler(epoch_log['scaler'], key='et/loss', i=i, writer=writer, inner=True)
                    # # debugger.write_scaler(epoch_log['scaler'], key='et/acc', i=i, writer=writer, inner=True)

                '''write mo'''
                if len(epoch_log['mo']) > 0:
                    # debugger.write_mo(epoch_log['mo'], None, i=i, writer=writer, target='acc')
                    debugger.write_mo(epoch_log['mo'], None, i=i, writer=writer, target='loss')
                    debugger.write_mo(epoch_log['mo'], None, i=i, writer=writer, target='norm_loss')

                    '''write hv acc/loss'''
                    # debugger.write_hv(epoch_log['mo'], i, ref=0, writer=writer, target='acc', norm=False)
                    debugger.write_hv(epoch_log['mo'], i, ref=1, writer=writer, target='loss', norm=False)
                    '''write avg_span acc/loss: E_i(max(f_i) - min(f_i))'''
                    # debugger.write_avg_span(epoch_log['mo'], i, writer=writer, target='acc', norm=False)
                    debugger.write_avg_span(epoch_log['mo'], i, writer=writer, target='loss', norm=False)
                    '''write min crowding distance'''
                    # debugger.write_min_crowding_distance(epoch_log['mo'], i, writer=writer, target='acc', norm=False)
                    debugger.write_min_crowding_distance(epoch_log['mo'], i, writer=writer, target='loss', norm=False)

            # '''nvidia-smi'''
            # print(os.system('nvidia-smi'))

        '''Close the writers'''
        writer.close()

        return avg_metrics 
    
    def summarize_acc(self, acc_dict, acc_table, acc_table_pt):

        # unpack dictionary
        avg_acc_all = acc_dict['global']
        avg_acc_pt = acc_dict['pt']
        avg_acc_pt_local = acc_dict['pt-local']

        # Calculate average performance across self.tasks
        # Customize this part for a different performance metric
        avg_acc_history = [0] * self.max_task
        for i in range(self.max_task):
            train_name = self.task_names[i]
            cls_acc_sum = 0
            for j in range(i+1):
                val_name = self.task_names[j]
                cls_acc_sum += acc_table[val_name][train_name]
                avg_acc_pt[j,i,self.seed] = acc_table[val_name][train_name]
                avg_acc_pt_local[j,i,self.seed] = acc_table_pt[val_name][train_name]
            avg_acc_history[i] = cls_acc_sum / (i + 1)

        # Gather the final avg accuracy
        avg_acc_all[:,self.seed] = avg_acc_history

        # repack dictionary and return
        return {'global': avg_acc_all,'pt': avg_acc_pt,'pt-local': avg_acc_pt_local}

    def evaluate(self, avg_metrics):

        self.learner = learners.__dict__[self.learner_type].__dict__[self.learner_name](self.learner_config)

        # store results
        metric_table = {}
        metric_table_local = {}
        for mkey in self.metric_keys:
            metric_table[mkey] = {}
            metric_table_local[mkey] = {}
            
        for i in range(self.max_task):

            # increment task id in prompting modules
            if i > 0:
                try:
                    if self.learner.model.module.prompt is not None:
                        self.learner.model.module.prompt.process_task_count()
                except:
                    if self.learner.model.prompt is not None:
                        self.learner.model.prompt.process_task_count()

            # load model
            model_save_dir = self.model_top_dir + '/models/repeat-'+str(self.seed+1)+'/task-'+self.task_names[i]+'/'
            self.learner.task_count = i 
            self.learner.add_valid_output_dim(len(self.tasks_logits[i]))
            self.learner.pre_steps()
            self.learner.load_model(model_save_dir)

            # set task id for model (needed for prompting)
            try:
                self.learner.model.module.task_id = i
            except:
                self.learner.model.task_id = i

            # evaluate acc
            metric_table['acc'][self.task_names[i]] = OrderedDict()
            metric_table_local['acc'][self.task_names[i]] = OrderedDict()
            self.reset_cluster_labels = True
            for j in range(i+1):
                val_name = self.task_names[j]
                acc = self.task_eval(j)
                if type(acc) is list:      # on all prompts
                    acc = np.mean(acc)
                metric_table['acc'][val_name][self.task_names[i]] = acc
            for j in range(i+1):
                val_name = self.task_names[j]
                acc = self.task_eval(j, local=True)
                if type(acc) is list:      # on all prompts
                    acc = np.mean(acc)
                metric_table_local['acc'][val_name][self.task_names[i]] = acc

        # summarize metrics
        avg_metrics['acc'] = self.summarize_acc(avg_metrics['acc'], metric_table['acc'],  metric_table_local['acc'])

        return avg_metrics