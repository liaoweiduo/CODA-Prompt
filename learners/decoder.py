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


class Decoder(NormalNN):
    """Decoder for ViT features"""
    def __init__(self, learner_config):
        self.prompt_param = learner_config['prompt_param']      # [self.num_tasks,args.prompt_param]
        super(Decoder, self).__init__(learner_config)

    def init_train_log(self):
        self.epoch_log = dict()
        # Tag: acc/loss
        self.epoch_log['mo'] = {'Tag': [], 'Pop_id': [], 'Obj_id': [], 'Epoch_id': [], 'Inner_id': [], 'Value': []}
        # 'loss/hv_loss'
        self.epoch_log['scaler'] = {'Tag': [], 'Idx': [], 'Value': []}

    def train_log_to_df(self):
        self.epoch_log['mo'] = pd.DataFrame(self.epoch_log['mo'])
        self.epoch_log['scaler'] = pd.DataFrame(self.epoch_log['scaler'])

    def update_model(self, inputs, targets):

        # logits
        logits, cls_token = self.model(inputs, train=True)
        logits = logits[:,:self.valid_out_dim]

        # # debug
        # print(f'logits: {logits.shape}, cls_token: {cls_token.shape}')

        # ce with heuristic
        if self.memory_size == 0:       # replay-based will have old tasks which may cause inf loss
            logits[:,:self.last_valid_out_dim] = -float('inf')
            # logits[:,:self.last_valid_out_dim] = logits[:, :self.last_valid_out_dim].detach().clone()
        dw_cls = self.dw_k[-1 * torch.ones(targets.size()).long()]
        total_loss = self.criterion(logits, targets.long(), dw_cls)

        # # debug
        # print(f'classification loss: {total_loss}')

        # step
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return total_loss.detach(), logits

    def learn_batch(self, train_loader, train_dataset, model_save_dir, val_loader=None):
        # return self.learn_batch_diff_stage(train_loader, train_dataset, model_save_dir, val_loader)

        self.init_train_log()

        # self.train_dataset = train_dataset
        # self.aux.update_source(train_dataset)       # aux samples from the current task

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
                for i, sample in enumerate(train_loader):
                    self.batch_idx = i

                    concepts = None
                    if train_dataset.return_concepts:
                        x, y, concepts, task = sample
                    else:
                        x, y, task = sample

                    # verify in train mode
                    self.model.train()

                    # send data to gpu
                    if self.gpu:
                        x = x.cuda()
                        y = y.cuda()
                        # if concepts is not None:
                        #     concepts = concepts.cuda()      # [bs, 224, 224]
                        # task = task.cuda()

                    # # debug
                    # print(f'x shape: {x.shape}, y: {y}, task: {task}')

                    # model update
                    loss, output = self.update_model(x, y)      # , task

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
                self.log(
                    ' * Loss {loss.avg:.3f} | '
                    'Train Acc {acc.avg:.3f} | '
                    'Time {time.avg:.3f}*{i}'.format(
                        loss=losses, acc=acc, time=batch_time, i=len(train_loader)))

                # reset
                losses = AverageMeter()
                acc = AverageMeter()

                # validation
                if val_loader is not None:
                    val_acc = self.validation(val_loader)
                    # log
                    self.epoch_log['scaler']['Tag'].append(f'val_acc')
                    self.epoch_log['scaler']['Idx'].append(self.epoch)
                    self.epoch_log['scaler']['Value'].append(val_acc)

                if self.epoch % 10 == 0:
                    '''nvidia-smi'''
                    self.log(os.system('nvidia-smi'))

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

    # sets model optimizers
    def init_optimizer(self, target=None, schedule=None):

        # parse optimizer args
        # Multi-GPU
        if len(self.config['gpuid']) > 1:
            last = self.model.module.last
            decoder = self.model.module.decoder
        else:
            last = self.model.last
            decoder = self.model.decoder

        if self.config['mode'] in ['sys', 'pro', 'sub', 'non', 'noc']:
            # if fewshot testing self.config['mode'], only learn classifier: model.last
            params_to_opt = list(last.parameters())
        elif target == 'last':
            params_to_opt = list(last.parameters())
        elif target == 'decoder':
            params_to_opt = list(decoder.parameters())
        # elif target == 'p':
        #     params_to_opt = [param for key, param in prompt.named_parameters()
        #                      if 'e_p_' in key] + list(last.parameters())
        # elif target == 'ka':
        #     params_to_opt = [param for key, param in prompt.named_parameters()
        #                      if 'e_k_' in key or 'e_a_' in key] + list(last.parameters())
        else:
            params_to_opt = list(decoder.parameters()) + list(last.parameters())

        print('******************* init optimizer **********************')
        print(f'target: {target} len {len(params_to_opt)}')

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


class SLOTDecoder(Decoder):

    def __init__(self, learner_config):
        super(SLOTDecoder, self).__init__(learner_config)

    def create_model(self):
        cfg = self.config
        model = models.__dict__[cfg['model_type']].__dict__[cfg['model_name']](out_dim=self.out_dim, prompt_flag = 'slot',prompt_param=self.prompt_param)
        return model

