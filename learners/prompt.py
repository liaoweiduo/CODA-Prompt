from __future__ import print_function
import sys
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from types import MethodType
import models
from utils.metric import accuracy, AverageMeter, Timer
import numpy as np
from torch.optim import Optimizer
import contextlib
import os
import copy
import torchvision
from torch.autograd import Variable, Function
import pandas as pd
from datetime import datetime

from .default import NormalNN, weight_reset, accumulate_acc
from utils.schedulers import CosineSchedule


class Prompt(NormalNN):

    def __init__(self, learner_config):
        self.prompt_param = learner_config['prompt_param']      # [self.num_tasks,args.prompt_param]
        super(Prompt, self).__init__(learner_config)

    def update_model(self, inputs, targets):

        # logits
        logits, prompt_loss = self.model(inputs, train=True)
        logits = logits[:,:self.valid_out_dim]

        # # debug
        # print(f'prompt_loss: {prompt_loss}')

        # ce with heuristic
        if self.memory_size == 0:       # replay-based will have old tasks which may cause inf loss
            # logits[:,:self.last_valid_out_dim] = -float('inf')
            logits[:,:self.last_valid_out_dim] = logits[:, :self.last_valid_out_dim].detach().clone()
        dw_cls = self.dw_k[-1 * torch.ones(targets.size()).long()]
        total_loss = self.criterion(logits, targets.long(), dw_cls)

        # # debug
        # print(f'classification loss: {total_loss}')

        # ce loss
        total_loss = total_loss + prompt_loss.sum()

        # step
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return total_loss.detach(), logits

    # sets model optimizers
    def init_optimizer(self, target=None, schedule=None):

        # parse optimizer args
        # Multi-GPU
        if len(self.config['gpuid']) > 1:
            last = self.model.module.last
            prompt = self.model.module.prompt
        else:
            last = self.model.last
            prompt = self.model.prompt

        if self.config['mode'] in ['sys', 'pro', 'sub', 'non', 'noc']:
            # if fewshot testing self.config['mode'], only learn classifier: model.last
            params_to_opt = list(last.parameters())
        elif target == 'last':
            params_to_opt = list(last.parameters())
        elif target == 'prompt':
            params_to_opt = list(prompt.parameters())
        elif target == 'p':
            params_to_opt = [param for key, param in prompt.named_parameters()
                             if 'e_p_' in key] + list(last.parameters())
        elif target == 'ka':
            params_to_opt = [param for key, param in prompt.named_parameters()
                             if 'e_k_' in key or 'e_a_' in key] + list(last.parameters())
        else:
            params_to_opt = list(prompt.parameters()) + list(last.parameters())

        print('*****************************************')
        optimizer_arg = {'params':params_to_opt,
                         'lr':self.config['lr'],
                         'weight_decay':self.config['weight_decay']}
        if self.config['optimizer'] in ['SGD','RMSprop']:
            optimizer_arg['momentum'] = self.config['momentum']
        elif self.config['optimizer'] in ['Rprop']:
            optimizer_arg.pop('weight_decay')
        elif self.config['optimizer'] == 'amsgrad':
            optimizer_arg['amsgrad'] = True
            self.config['optimizer'] = 'Adam'
        elif self.config['optimizer'] == 'Adam':
            optimizer_arg['betas'] = (self.config['momentum'],0.999)

        # create optimizers
        self.optimizer = torch.optim.__dict__[self.config['optimizer']](**optimizer_arg)

        if schedule is None:
            schedule = self.schedule

        # create schedules
        if self.schedule_type == 'cosine':
            self.scheduler = CosineSchedule(self.optimizer, K=schedule[-1])
        elif self.schedule_type == 'decay':
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=schedule, gamma=0.1)

    def create_model(self):
        pass

    def cuda(self):
        torch.cuda.set_device(self.config['gpuid'][0])
        self.model = self.model.cuda()
        self.criterion_fn = self.criterion_fn.cuda()

        # Multi-GPU
        if len(self.config['gpuid']) > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.config['gpuid'], output_device=self.config['gpuid'][0])
        return self


# Our method!
class CODAPrompt(Prompt):

    def __init__(self, learner_config):
        super(CODAPrompt, self).__init__(learner_config)

    def create_model(self):
        cfg = self.config
        model = models.__dict__[cfg['model_type']].__dict__[cfg['model_name']](out_dim=self.out_dim, prompt_flag = 'coda',prompt_param=self.prompt_param)
        return model


# CODA-Prompt with memory replay
class CODAPromptR(Prompt):
    """
    can overwrite learn_batch to inject some modification.
    """
    def __init__(self, learner_config):
        super(CODAPromptR, self).__init__(learner_config)

        self.task_dim_list = []    # list: [num_tsks, dim] pattern: [0,1,2,3,4,5,6,7,8,9], [10,11,12,13,...]

    def add_valid_output_dim(self, dim=0):
        """Difference:
        Maintain a task_dim_list, recording dim ranges for all tasks.
        """
        # This function is kind of ad-hoc, but it is the simplest way to support incremental class learning
        self.log('Incremental class: Old valid output dimension:', self.valid_out_dim)
        dim_from = self.valid_out_dim
        self.valid_out_dim += dim
        self.log('Incremental class: New Valid output dimension:', self.valid_out_dim)
        dim_to = self.valid_out_dim
        self.task_dim_list.append(np.arange(dim_from, dim_to))

        return self.valid_out_dim

    def create_model(self):
        cfg = self.config
        model = models.__dict__[cfg['model_type']].__dict__[cfg['model_name']](out_dim=self.out_dim, prompt_flag = 'coda_r',prompt_param=self.prompt_param)
        return model

    def learn_batch(self, train_loader, train_dataset, model_save_dir, val_loader=None):
        """Difference:
        forward tasks_id for a batch of samples to update_model()
        """

        # try to load model
        need_train = True
        if not self.overwrite:
            try:
                self.load_model(model_save_dir)
                need_train = False
            except:
                pass

        # trains
        if self.reset_optimizer:  # Reset optimizer before learning each task
            self.log('Optimizer is reset!')
            self.init_optimizer()
        if need_train:
            # data weighting
            self.data_weighting(train_dataset)
            losses = AverageMeter()
            acc = AverageMeter()
            batch_time = AverageMeter()
            batch_timer = Timer()
            for epoch in range(self.config['schedule'][-1]):
                self.epoch = epoch

                if epoch > 0: self.scheduler.step()
                for param_group in self.optimizer.param_groups:
                    self.log('LR:', param_group['lr'])
                batch_timer.tic()
                for i, (x, y, task) in enumerate(train_loader):

                    # verify in train mode
                    self.model.train()

                    # send data to gpu
                    if self.gpu:
                        x = x.cuda()
                        y = y.cuda()
                        task = task.cuda()

                    # # debug
                    # print(f'x shape: {x.shape}, y: {y}, task: {task}')

                    # model update
                    loss, output = self.update_model(x, y, task)

                    # measure elapsed time
                    batch_time.update(batch_timer.toc())
                    batch_timer.tic()

                    # measure accuracy and record loss
                    y = y.detach()
                    accumulate_acc(output, y, task, acc, topk=(self.top_k,))
                    losses.update(loss, y.size(0))
                    batch_timer.tic()

                # eval update
                self.log(
                    'Epoch:{epoch:.0f}/{total:.0f}'.format(epoch=self.epoch + 1, total=self.config['schedule'][-1]))
                self.log(' * Loss {loss.avg:.3f} | Train Acc {acc.avg:.3f} | Time {time.avg:.3f}'.format(loss=losses,
                                                                                                         acc=acc,
                                                                                                         time=batch_time))

                # reset
                losses = AverageMeter()
                acc = AverageMeter()

        self.model.eval()

        self.last_valid_out_dim = self.valid_out_dim
        self.first_task = False

        # Extend memory
        self.task_count += 1
        if self.memory_size > 0:
            train_dataset.update_coreset(self.memory_size, np.arange(self.last_valid_out_dim))

        try:
            return batch_time.avg
        except:
            return None

    def update_model(self, inputs, targets, tasks):
        """Difference: # for skipping changes
        Intro tasks_id for batch of samples to support ce with heuristic
        # Change self.model.prompt.batch_task_ids to this batch of samples
        #     to support awareness of task id for prompt's forward
        Forward batch_task_idxs to model and thus DataParallel
            will correctly locate sample with its task id.
        # Logits mask according to sample's taskid but not current taskid.
        #     dim region is based on self.task_dim_list to np array.
        """
        # record task ids
        # for prompt to select correct KAP
        # abandon: if use parallel model, inputs will be distributed to different devices
        #   but batch_task_ids will not
        # try:
        #     self.model.module.prompt.batch_task_ids = tasks
        # except:
        #     self.model.prompt.batch_task_ids = tasks

        if self.debug_mode:
            print(f'train batch: \ntargets:{targets} \ntasks:{tasks}')

        # logits
        logits, prompt_loss = self.model(inputs, task_id=tasks, train=True)     # replay-based need task_ids
        logits = logits[:,:self.valid_out_dim]

        if self.debug_mode:
            print(f'logits: {logits}')
            print(f'prompt_loss: {prompt_loss}')

        # ce with heuristic
        '''old version'''
        # logits[:,:self.last_valid_out_dim] = -float('inf')
        '''according to sample_task_id'''
        task_dim_list = self.task_dim_list     # [[0,1,2,...,9],[10,11,...,19],...]
        mask = torch.ones_like(logits, dtype=torch.bool)
        filter_indices = torch.tensor(
            [[idx, value] for idx, task in enumerate(tasks) for value in task_dim_list[task]],
            device=logits.device
        )
        mask[filter_indices[:, 0], filter_indices[:, 1]] = False    # valid region to 0, thus masked_fill all 1 to -inf
        # mask = torch.BoolTensor(mask).to(logits.device)
        logits = logits.masked_fill(mask, value=-float('inf'))

        if self.debug_mode:
            print(f'masked logits: {logits}')

        dw_cls = self.dw_k[-1 * torch.ones(targets.size()).long()]
        total_loss = self.criterion(logits, targets.long(), dw_cls)

        if self.debug_mode:
            print(f'classification loss: {total_loss}')

        # ce loss
        total_loss = total_loss + prompt_loss.sum()

        # step
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return total_loss.detach(), logits


# @article{wang2022dualprompt,
#   title={DualPrompt: Complementary Prompting for Rehearsal-free Continual Learning},
#   author={Wang, Zifeng and Zhang, Zizhao and Ebrahimi, Sayna and Sun, Ruoxi and Zhang, Han and Lee, Chen-Yu and Ren, Xiaoqi and Su, Guolong and Perot, Vincent and Dy, Jennifer and others},
#   journal={European Conference on Computer Vision},
#   year={2022}
# }
class DualPrompt(Prompt):

    def __init__(self, learner_config):
        super(DualPrompt, self).__init__(learner_config)

    def create_model(self):
        cfg = self.config
        model = models.__dict__[cfg['model_type']].__dict__[cfg['model_name']](out_dim=self.out_dim, prompt_flag = 'dual', prompt_param=self.prompt_param)
        return model

# @inproceedings{wang2022learning,
#   title={Learning to prompt for continual learning},
#   author={Wang, Zifeng and Zhang, Zizhao and Lee, Chen-Yu and Zhang, Han and Sun, Ruoxi and Ren, Xiaoqi and Su, Guolong and Perot, Vincent and Dy, Jennifer and Pfister, Tomas},
#   booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
#   pages={139--149},
#   year={2022}
# }
class L2P(Prompt):

    def __init__(self, learner_config):
        super(L2P, self).__init__(learner_config)

    def create_model(self):
        cfg = self.config
        model = models.__dict__[cfg['model_type']].__dict__[cfg['model_name']](out_dim=self.out_dim, prompt_flag = 'l2p',prompt_param=self.prompt_param)
        return model