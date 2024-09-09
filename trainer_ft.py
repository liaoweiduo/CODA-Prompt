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
import learners

class Trainer:

    def __init__(self, args, seed, metric_keys, save_keys):

        # process inputs
        self.seed = seed
        self.metric_keys = metric_keys
        self.save_keys = save_keys
        self.log_dir = args.log_dir
        self.batch_size = 100   # args.batch_size
        self.workers = args.workers

        self.test_model = args.test_model
        
        # model load directory
        self.model_top_dir = args.log_dir

        # select dataset
        self.grayscale_vis = False
        self.top_k = 1
        if args.dataset == 'CGQA':
            num_classes = 100
            Dataset = dataloaders.CGQA
        elif args.dataset == 'COBJ':
            num_classes = 30
            Dataset = dataloaders.COBJ
        else:
            raise ValueError('Dataset not implemented!')

        # upper bound flag
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
        self.num_tasks = len(self.tasks)        # 10

        # datasets and dataloaders
        k = 1 # number of transforms per image
        if args.model_name.startswith('vit'):
            resize_imnet = True
        else:
            resize_imnet = False
        train_transform = dataloaders.utils.get_transform(dataset=args.dataset, phase='train', aug=args.train_aug, resize_imnet=resize_imnet)
        test_transform  = dataloaders.utils.get_transform(dataset=args.dataset, phase='test', aug=args.train_aug, resize_imnet=resize_imnet)
        self.train_dataset = Dataset(args.dataroot, train=True, lab = True, tasks=self.tasks,
                                     download_flag=True, transform=train_transform,
                                     seed=self.seed, rand_split=args.rand_split, validation=args.validation,
                                     mode=args.mode)
        self.test_dataset  = Dataset(args.dataroot, train=False, tasks=self.tasks,
                                     download_flag=False, transform=test_transform,
                                     seed=self.seed, rand_split=args.rand_split, validation=args.validation,
                                     mode=args.mode)

        self.max_task = self.train_dataset.benchmark.n_experiences      # 300
        self.task_names = [str(i+1) for i in range(self.max_task)]      # 300

        # for oracle
        self.oracle_flag = args.oracle_flag
        self.add_dim = self.num_tasks

        args.schedule = [20]

        # Prepare the self.learner (model)
        self.learner_config = {'num_classes': num_classes,
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
                        }
        self.learner_type, self.learner_name = args.learner_type, args.learner_name
        self.learner = learners.__dict__[self.learner_type].__dict__[self.learner_name](self.learner_config)

    def task_eval(self, t_index, local=False, task='acc'):

        val_name = self.task_names[t_index]
        print('validation split name:', val_name)
        
        # eval
        self.test_dataset.load_dataset(t_index, train=True)
        test_loader  = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False, num_workers=self.workers)
        if local:
            return self.learner.validation(test_loader, task_in = self.tasks_logits[0], task_metric=task)
        else:
            return self.learner.validation(test_loader, task_metric=task)

    def train(self, avg_metrics):
    
        # temporary results saving
        temp_table = {}
        for mkey in self.metric_keys: temp_table[mkey] = []
        temp_dir = self.log_dir + '/temp/'
        if not os.path.exists(temp_dir): os.makedirs(temp_dir)

        # for each task
        for i in range(self.max_task):      # for few-shot testing, it should start from an offset

            self.learner = learners.__dict__[self.learner_type].__dict__[self.learner_name](self.learner_config)

            # do prompt increases, since prompt may extend as num_task increasing for some methods
            for j in range(self.test_model):        # load task-args.test_model model
                # increment task id in prompting modules
                if j > 0:
                    try:
                        if self.learner.model.module.prompt is not None:
                            self.learner.model.module.prompt.process_task_count()
                    except:
                        if self.learner.model.prompt is not None:
                            self.learner.model.prompt.process_task_count()
                self.learner.task_count = j
                self.learner.pre_steps()

            self.learner.add_valid_output_dim(len(self.tasks_logits[0]))      # only for one task
            # this is important in model updating (mask out valid_out_dim)
            # load model
            model_save_dir = self.model_top_dir + '/models/repeat-'+str(self.seed+1)+f'/task-{self.test_model}/'
            self.learner.load_model(model_save_dir, drop_last=True)

            # set task id for model (needed for prompting)
            try:
                self.learner.model.module.task_id = self.num_tasks
                # use task_offset. this num_tasks is continual training's num_tasks
            except:
                self.learner.model.task_id = self.num_tasks

            # save current task index
            self.current_t_index = self.num_tasks

            # print name
            train_name = self.task_names[i]
            print('======================', train_name, '=======================')

            # load dataset for task
            task = self.tasks_logits[0]
            self.train_dataset.load_dataset(i, train=True)

            # load dataloader
            train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=False, num_workers=int(self.workers))

            # learn
            self.test_dataset.load_dataset(i, train=False)
            test_loader  = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False, num_workers=self.workers)
            # no use during training
            # model_save_dir = self.model_top_dir + '/models/repeat-'+str(self.seed+1)+'/task-'+self.task_names[i]+'/'
            # if not os.path.exists(model_save_dir): os.makedirs(model_save_dir)
            # set model_save_dir to None to enable training
            avg_train_time = self.learner.learn_batch(train_loader, self.train_dataset, None, test_loader)

            # # save model
            # self.learner.save_model(model_save_dir)

            # eval and save to avg_metrics
            acc = self.task_eval(i)
            # acc_local = self.task_eval(i, local=True)
            avg_metrics['acc']['global'][i, self.seed] = acc        # [max_task, repeat]
            if avg_train_time is not None: avg_metrics['time']['global'][i] = avg_train_time

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
                metric_table['acc'][val_name][self.task_names[i]] = self.task_eval(j)
            for j in range(i+1):
                val_name = self.task_names[j]
                metric_table_local['acc'][val_name][self.task_names[i]] = self.task_eval(j, local=True)

        # summarize metrics
        avg_metrics['acc'] = self.summarize_acc(avg_metrics['acc'], metric_table['acc'],  metric_table_local['acc'])

        return avg_metrics