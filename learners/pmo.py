from __future__ import print_function
import sys
import math
from typing import Optional, Union
import pickle

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
from .prompt import Prompt, CODAPromptCond
from utils.schedulers import CosineSchedule
from .pmo_utils import Pool, Mixer, available_setting, task_to_device, cal_hv_weights, normalize
from models.losses import prototype_loss
from mo_optimizers.functions_evaluation import fastNonDominatedSort
import dataloaders


# Our PMO (Prompt & Multi-Objective)
class PMOPrompt(Prompt):
    def __init__(self, learner_config):
        super(PMOPrompt, self).__init__(learner_config)

        self.train_dataset = None
        self.t = 0
        self.epoch = 0
        self.epochs = 0     # total epoch in this task
        self.concept_weight = self.config['concept_weight']  # True to use concept to weight data.

        config = self.config['prompt_param'][1]
        while len(config) < 3:      # deal with old exps that have not enough config
            config.append(0)
        # mo
        # self.n_obj = int(self.config['prompt_param'][1][3])
        # self.num_aux_sampling = int(self.config['num_aux_sampling'])
        try:
            prompt = self.model.module.prompt
        except:
            prompt = self.model.prompt
        self.e_layers = prompt.e_layers
        self.n_prompt_per_task = prompt.n_prompt_per_task
        # self.n_obj_avail = prompt.n_obj
        self.FPS = prompt.FPS

        # cls statistics
        self.cls_stats = {}

    def create_model(self):
        cfg = self.config
        model = models.__dict__[cfg['model_type']].__dict__[cfg['model_name']](out_dim=self.out_dim, prompt_flag = 'pmo',prompt_param=self.prompt_param, use_vit_emb=True, use_vit_fea=False)
        return model

    def load_model(self, filename, drop_last=False, freeze=False):
        state_dict = torch.load(filename + 'class.pth')
        # complete with/without module.
        for key in list(state_dict.keys()):
            if 'module' in key:
                state_dict[key[7:]] = state_dict[key]
            else:
                state_dict[f'module.{key}'] = state_dict[key]
        if drop_last:
            del state_dict['module.last.weight']; del state_dict['module.last.bias']
            del state_dict['last.weight']; del state_dict['last.bias']
            # if 'module.last.weight' in state_dict:
            #     del state_dict['module.last.weight']; del state_dict['module.last.bias']
            # else:
            #     del state_dict['last.weight']; del state_dict['last.bias']
            # self.model.load_state_dict(state_dict, strict=False)
        self.model.load_state_dict(state_dict, strict=False)
        self.log('=> Load Done')
        # self.log(f'=> Load Done with params {list(state_dict.keys())}')

        if freeze:
            self.log('=> Freeze backbone')     # on CFST
            for k, p in self.model.named_parameters():
                if 'last' not in k:
                    p.requires_grad = False

        # freeze selection when concept_weight = True
        if self.config['concept_weight']:
            for k, p in self.model.prompt.named_parameters():
                if 'e_k_' in k or 'e_a_' in k:
                    p.requires_grad = False

        if self.gpu:
            self.model = self.model.cuda()
        self.model.eval()

    # data weighting
    def data_weighting(self, dataset, num_seen=None):
        # assign specific weight on cls with target concept.
        # if dataset.target_sample_info is not None:      # continual
        if self.concept_weight:
            concepts = dataset.get_concepts()  # [n_cls * [list of concepts: e.g., 1, 10]]
            num_concepts = dataset.num_concepts
            self.dw_k = torch.tensor(np.ones((num_concepts, self.valid_out_dim + 1), dtype=np.float32))

            for target_concept in range(num_concepts):
                # target_concept = self.target_concept_id
                for cls_id in range(self.valid_out_dim):
                    if target_concept in concepts[cls_id]:
                        self.dw_k[target_concept, cls_id] = 2
        else:
            self.dw_k = torch.tensor(np.ones(self.valid_out_dim + 1, dtype=np.float32))

        # cuda
        if self.cuda:
            self.dw_k = self.dw_k.cuda()

    def learn_batch(self, train_loader, train_dataset, model_save_dir, val_loader=None):
        # return self.learn_batch_diff_stage(train_loader, train_dataset, model_save_dir, val_loader)

        self.init_train_log()

        self.train_dataset = train_dataset
        if hasattr(train_dataset, 'num_concepts'):
            self.num_concepts = train_dataset.num_concepts

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
            target = None
            if self.concept_weight:
                target = 'p'
            self.init_optimizer(target=target)

        if need_train:
            losses = AverageMeter()
            acc = AverageMeter()
            batch_time = AverageMeter()
            batch_timer = Timer()

            # data weighting
            self.data_weighting(train_dataset)

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
                        if concepts is not None:
                            concepts = concepts.cuda()      # [bs, 224, 224]
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

                # if self.epoch % 10 == 0:
                if self.epoch == 0:
                    '''nvidia-smi'''
                    self.log(os.system('nvidia-smi'))

                # reset
                losses = AverageMeter()
                acc = AverageMeter()

                # # validation
                # if val_loader is not None:
                #     val_acc = self.validation(val_loader)
                #     # log
                #     self.epoch_log['scaler']['Tag'].append(f'val_acc')
                #     self.epoch_log['scaler']['Idx'].append(self.epoch)
                #     self.epoch_log['scaler']['Value'].append(val_acc)

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

    def learn_batch_diff_stage(self, train_loader, train_dataset, model_save_dir, val_loader=None):
        """after train cal statistics"""
        self.init_train_log()

        self.train_dataset = train_dataset
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

        # data weighting
        self.data_weighting(train_dataset)
        acc = AverageMeter()
        if need_train:
            losses = AverageMeter()
            batch_time = AverageMeter()
            batch_timer = Timer()
            n_epoch = self.config['schedule'][-1] if train_dataset.t == 0 else 5
            for epoch in range(n_epoch):
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
                    loss, output = self.update_model_pop_prompt(x, y)      # , task

                    # measure elapsed time
                    batch_time.update(batch_timer.toc())
                    batch_timer.tic()

                    # measure accuracy and record loss
                    y = y.detach()
                    # accumulate_acc(output, y, task, acc, topk=(self.top_k,))
                    losses.update(loss, y.size(0))
                    batch_timer.tic()

                # eval update
                self.log(
                    'Epoch:{epoch:.0f}/{total:.0f}'.format(epoch=self.epoch + 1, total=self.config['schedule'][-1]))
                self.log(
                    ' * Loss {loss.avg:.3f} | '
                    'Time {time.avg:.3f}*{i}'.format(
                        loss=losses, time=batch_time, i=len(train_loader)))

                if self.epoch % 10 == 0:
                    '''nvidia-smi'''
                    self.log(os.system('nvidia-smi'))

                # reset
                losses = AverageMeter()
                acc = AverageMeter()

            # collect class statistics
            self.collect_statistics(train_loader, train_dataset)

            # # validation
            # if val_loader is not None:
            #     val_acc = self.validation(val_loader)
            #     # log
            #     self.epoch_log['scaler']['Tag'].append(f'val_acc')
            #     self.epoch_log['scaler']['Idx'].append(self.epoch)
            #     self.epoch_log['scaler']['Value'].append(val_acc)

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

    def update_model(self, inputs, targets):
        """normal update model"""

        if self.concept_weight:
            total_loss = []
            for concept_id in range(self.num_concepts):
                # logits
                out = self.model(inputs, train=True, prompt_id=concept_id)      # specify prompt to use
                # out -> [bs, 100]
                if len(out) == 2:
                    logits, prompt_loss = out
                    prompt_loss = prompt_loss.sum()
                else:
                    logits = out
                    prompt_loss = 0
                logits = logits[:, :self.valid_out_dim]

                # ce with heuristic
                if self.memory_size == 0:  # replay-based will have old tasks which may cause inf loss
                    logits[:, :self.last_valid_out_dim] = -float('inf')
                    # logits[:,:self.last_valid_out_dim] = logits[:, :self.last_valid_out_dim].detach().clone()

                dw_cls = self.dw_k[concept_id, targets.long()]
                # dw_cls = self.dw_k[-1 * torch.ones(targets.size()).long()]
                ce_loss = self.criterion(logits, targets.long(), dw_cls)

                # ce loss
                total_loss.append(ce_loss + prompt_loss)

            total_loss = torch.stack(total_loss).mean()

        else:
            # logits
            out = self.model(inputs, train=True)        # use selection
            # out -> learning prompt: [bs, n_concepts, 100] or learning selection: [bs, 100]
            if len(out) == 2:
                logits, prompt_loss = out
                prompt_loss = prompt_loss.sum()
            else:
                logits = out
                prompt_loss = 0
            logits = logits[:, :self.valid_out_dim]

            # ce with heuristic
            if self.memory_size == 0:  # replay-based will have old tasks which may cause inf loss
                logits[:, :self.last_valid_out_dim] = -float('inf')
                # logits[:,:self.last_valid_out_dim] = logits[:, :self.last_valid_out_dim].detach().clone()

            dw_cls = self.dw_k[targets.long()]
            # dw_cls = self.dw_k[-1 * torch.ones(targets.size()).long()]
            total_loss = self.criterion(logits, targets.long(), dw_cls)

            # # debug
            # print(f'classification loss: {total_loss}')

            # ce loss
            total_loss = total_loss + prompt_loss

        # step
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return total_loss.detach(), logits

    def update_model_with_hv(self, inputs, targets, pre_learn=False):
        """Difference:
            Cal mo loss matrix and hv loss.
            backward one loss by one since hv loss is huge
        """

        self.optimizer.zero_grad()
        try:
            prompt = self.model.module.prompt
            last = self.model.module.last
        except:
            prompt = self.model.prompt
            last = self.model.last

        grads = {}
        params_to_opt = {key: param for key, param in prompt.named_parameters() if 'e_k_' in key or 'e_a_' in key}
        params_for_ce = {key: param for key, param in last.named_parameters()}
        params_for_ce.update({key: param for key, param in prompt.named_parameters() if 'e_p_' in key})

        for k, p in params_to_opt.items():      # use ce+hv
            grads[k] = {'shape': p.shape, 'grads': []}
        for k, p in params_for_ce.items():      # use ce
            grads[k] = {'shape': p.shape, 'grads': []}

        # logits
        logits, prompt_loss = self.model(inputs, train=True, pre_learn=pre_learn)
        logits = logits[:,:self.valid_out_dim]

        if self.debug_mode:
            # print(f'logits: {logits}')
            print(f'prompt_loss: {prompt_loss}')

        # ce with heuristic
        if self.memory_size == 0:       # replay-based will have old tasks which may cause inf loss
            logits[:,:self.last_valid_out_dim] = -float('inf')
        dw_cls = self.dw_k[-1 * torch.ones(targets.size()).long()]
        total_loss = self.criterion(logits, targets.long(), dw_cls)

        self.epoch_log['scaler']['Tag'].append('loss/ce_loss')
        self.epoch_log['scaler']['Idx'].append(self.epoch)
        self.epoch_log['scaler']['Value'].append(total_loss.item())

        if self.debug_mode:
            print(f'classification loss: {total_loss} and {total_loss.grad_fn}')

        # ce loss
        total_loss = total_loss + prompt_loss.sum()

        self.optimizer.zero_grad()
        total_loss.backward()

        for k, p in params_to_opt.items():
            grads[k]['grads'].append(p.grad)

        for k, p in params_for_ce.items():
            grads[k]['grads'].append(p.grad)

        if not pre_learn:       # for pre_learn, only ce loss is used.
            # hv loss calculation without affecting RNG state
            state = np.random.get_state()
            # seed with taskid epochid batchid
            np.random.seed(self.seed + self.train_dataset.t + self.epoch + self.batch_idx)

            '''hv loss'''
            self.optimizer.zero_grad()
            for l in self.e_layers:
                # if self.train_dataset.t > 0:        # start from the second task
                repeat = 1
                pop_size = self.num_aux_sampling
                mo_matrix = self.obtain_mo_matrix(hard_l=l, use_old_prompts=True,
                                                  mask=self.mask, mask_mode=self.mask_mode,
                                                  train=True, repeat=repeat)   # [obj10, pop20]

                if self.debug_mode:
                    print(f'mo_matrix: {mo_matrix}')

                hv_loss = 0
                maximization = True if self.mask_mode == 'maskout' else False
                if mo_matrix is not None:
                    # if maximization:
                    normed_mo_matrix = normalize(mo_matrix, noise=True)
                    # else:
                    #     with torch.no_grad():
                    #         normed_mo_matrix = normalize_to_simplex(mo_matrix, noise=True)

                    # repeat for hv_loss
                    ref = 1  # dynamic 1.5*max for minimization or 1 for reverse
                    weights = [
                        cal_hv_weights(
                            normed_mo_matrix[:, r*pop_size:(r+1)*pop_size], ref, reverse=maximization)
                        for r in range(repeat)
                    ]

                    if self.debug_mode:
                        print(f'weights: {weights}')

                    weights = torch.cat(weights, dim=1)     # [obj, pop]

                    # if maximization:    # use normed mo
                    mo_matrix = normed_mo_matrix
                    hv_loss = torch.sum(mo_matrix * weights, dim=0)     # to vector over samples
                    hv_loss = torch.sum(hv_loss)
                    # sum to balance over prompts, mean to align to 1 sample's ce loss

                    if maximization:
                        hv_loss = torch.exp(hv_loss)                  # exp() make loss \in [0, 1]
                    else:
                        hv_loss = hv_loss
                    # hv_loss = hv_loss * self.hv_coeff

                    hv_loss.backward()

                    hv_loss = hv_loss.item()

                    if self.debug_mode:
                        print(f'hv loss in layer{l}: {hv_loss}')

                # else:
                #     print(f'ERROR: mo_matrix is None, skip layer{l}')

                self.epoch_log['scaler']['Tag'].append('loss/hv_loss')
                self.epoch_log['scaler']['Idx'].append(self.epoch)
                self.epoch_log['scaler']['Value'].append(hv_loss)

            np.random.set_state(state)

            for k, p in params_to_opt.items():
                grads[k]['grads'].append(p.grad)

            # cal grad for 2 opt processes
            # params_to_opt
            l1 = torch.cat([grads[k]['grads'][0].flatten() for k, p in params_to_opt.items()])
            l2 = torch.cat([grads[k]['grads'][1].flatten() for k, p in params_to_opt.items()])
            alpha = torch.nn.functional.relu(torch.sum(l2 * (l1 - l2)) / torch.sum((l1 - l2) * (l1 - l2)))

            if self.debug_mode:
                print(f'hv alpha: {alpha}')
            self.epoch_log['scaler']['Tag'].append('alpha')
            self.epoch_log['scaler']['Idx'].append(self.epoch)
            self.epoch_log['scaler']['Value'].append(alpha.item())

            self.optimizer.zero_grad()
            for k, p in params_to_opt.items():
                p.grad = alpha * grads[k]['grads'][0] + (1 - alpha) * grads[k]['grads'][1]
            for k, p in params_for_ce.items():
                p.grad = grads[k]['grads'][0]

        # step
        self.optimizer.step()

        return total_loss.detach(), logits

    def update_model_pop_prompt(self, inputs, targets, pre_learn=False):
        """Difference:
        Forward all prompts and use min{ce loss} to backward
        """
        self.optimizer.zero_grad()
        try:
            prompt = self.model.module.prompt
            last = self.model.module.last
        except:
            prompt = self.model.prompt
            last = self.model.last

        # forward all prompts
        # for l in self.e_layers:
        mo_matrix, features = self.obtain_mo_matrix_pop_prompt(
            None, use_old_prompts=False if prompt.FPS else True,
            mask=self.mask, mask_mode=self.mask_mode,
            train=True,
            samples=inputs,
            labels=targets,
            group_by_labels=False,
            return_features=True,
        )  # [bs, 21]
        # features: [bs, 21, 768]

        if self.debug_mode:
            print(f'mo_matrix {mo_matrix.shape}: {mo_matrix}')

        '''log'''
        for obj_idx in range(mo_matrix.shape[0]):
            for pop_idx in range(mo_matrix.shape[1]):
                self.epoch_log['mo']['Tag'].append('loss')
                self.epoch_log['mo']['Pop_id'].append(pop_idx)
                self.epoch_log['mo']['Obj_id'].append(obj_idx)
                self.epoch_log['mo']['Epoch_id'].append(self.epoch)
                self.epoch_log['mo']['Inner_id'].append(0)
                self.epoch_log['mo']['Value'].append(mo_matrix[obj_idx, pop_idx].item())

        # # min {ce loss}
        # loss = torch.min(mo_matrix, dim=1)[0]      # [bs]
        # min-2 {ce loss}
        loss = torch.mean(torch.sort(mo_matrix, dim=1)[0][:, :2], dim=1)        # [bs]
        loss = torch.mean(loss)

        loss.backward()

        if self.debug_mode:
            print(f'loss: {loss.item()}')

        self.epoch_log['scaler']['Tag'].append('loss/mo_loss')
        self.epoch_log['scaler']['Idx'].append(self.epoch)
        self.epoch_log['scaler']['Value'].append(loss.item())

        # # predict according to logits on all objs
        # # logits: [bs, 21, 100] -> [bs, 100] => mean, max
        # logits = torch.max(logits, dim=1)[0]      # [bs, 100]
        # fake logits
        logits = torch.max(features[:, :, :100], dim=1)[0]

        # step
        self.optimizer.step()

        return loss.detach(), logits

    def update_model_pop_prompt_old(self, inputs, targets, pre_learn=False):
        """Difference:
            Cal mo loss matrix and hv loss.
            backward one loss by one since hv loss is huge
        """
        self.optimizer.zero_grad()
        try:
            prompt = self.model.module.prompt
            last = self.model.module.last
        except:
            prompt = self.model.prompt
            last = self.model.last

        grads = {}
        params_to_opt = {key: param for key, param in prompt.named_parameters() if 'e_p_' in key}
        # params_to_opt = {key: param for key, param in prompt.named_parameters()}
        params_for_ce = {key: param for key, param in last.named_parameters()}
        params_for_ce.update({key: param for key, param in prompt.named_parameters() if 'e_k_' in key or 'e_a_' in key})

        for k, p in params_to_opt.items():  # use ce+hv
            grads[k] = {'shape': p.shape, 'grads': []}
        for k, p in params_for_ce.items():  # use ce
            grads[k] = {'shape': p.shape, 'grads': []}

        # logits
        logits, prompt_loss = self.model(inputs, train=True, pre_learn=pre_learn)
        logits = logits[:, :self.valid_out_dim]

        if self.debug_mode:
            # print(f'logits: {logits}')
            print(f'prompt_loss: {prompt_loss}')

        # ce with heuristic
        if self.memory_size == 0:  # replay-based will have old tasks which may cause inf loss
            logits[:, :self.last_valid_out_dim] = -float('inf')
        dw_cls = self.dw_k[-1 * torch.ones(targets.size()).long()]
        total_loss = self.criterion(logits, targets.long(), dw_cls)

        self.epoch_log['scaler']['Tag'].append('loss/ce_loss')
        self.epoch_log['scaler']['Idx'].append(self.epoch)
        self.epoch_log['scaler']['Value'].append(total_loss.item())

        if self.debug_mode:
            print(f'classification loss: {total_loss} and {total_loss.grad_fn}')

        # ce loss
        total_loss = total_loss + prompt_loss.sum()

        self.optimizer.zero_grad()
        total_loss.backward()

        for k, p in params_to_opt.items():
            grads[k]['grads'].append(p.grad.clone())        # ce grad

        for k, p in params_for_ce.items():
            grads[k]['grads'].append(p.grad.clone())

        if not pre_learn:  # for pre_learn, only ce loss is used.
            # hv loss calculation without affecting RNG state
            state = np.random.get_state()
            # seed with taskid epochid batchid
            np.random.seed(self.seed + self.train_dataset.t + self.epoch + self.batch_idx)

            '''hv loss'''
            # group samples by label
            labels = torch.unique(targets)
            # select n_obj groups
            selected_labels = np.sort(
                np.random.choice(labels.cpu().numpy(), self.n_obj, replace=False)).astype(int)
            # select mostly 3 imgs for each target
            selected_inputs = torch.cat([inputs[targets == label][:3] for label in selected_labels])
            selected_targets = torch.cat([targets[targets == label][:3] for label in selected_labels])

            self.optimizer.zero_grad()
            # for l in self.e_layers:
            mo_matrix = self.obtain_mo_matrix_pop_prompt(
                None, use_old_prompts=False if prompt.FPS else True,
                mask=self.mask, mask_mode=self.mask_mode,
                train=True,
                samples=selected_inputs,
                labels=selected_targets,
            )  # [obj2, pop120+1]

            if self.debug_mode:
                print(f'mo_matrix: {mo_matrix}')

            maximization = True if self.mask_mode == 'maskout' else False

            if mo_matrix is not None:       # None when n_obj <= 0 or any target only has 1 image

                '''log'''
                for obj_idx in range(mo_matrix.shape[0]):
                    for pop_idx in range(mo_matrix.shape[1]):
                        self.epoch_log['mo']['Tag'].append('loss')
                        self.epoch_log['mo']['Pop_id'].append(pop_idx)
                        self.epoch_log['mo']['Obj_id'].append(obj_idx)
                        self.epoch_log['mo']['Epoch_id'].append(self.epoch)
                        self.epoch_log['mo']['Inner_id'].append(0)
                        self.epoch_log['mo']['Value'].append(mo_matrix[obj_idx, pop_idx].item())

                mo_matrix = normalize(mo_matrix)

                '''log after norm: if delete, also remove log writer in trainer.py'''
                for obj_idx in range(mo_matrix.shape[0]):
                    for pop_idx in range(mo_matrix.shape[1]):
                        self.epoch_log['mo']['Tag'].append('norm_loss')
                        self.epoch_log['mo']['Pop_id'].append(pop_idx)
                        self.epoch_log['mo']['Obj_id'].append(obj_idx)
                        self.epoch_log['mo']['Epoch_id'].append(self.epoch)
                        self.epoch_log['mo']['Inner_id'].append(0)
                        self.epoch_log['mo']['Value'].append(mo_matrix[obj_idx, pop_idx].item())

                ref = 1  # dynamic 1.5*max for minimization or 1 for reverse
                weights = cal_hv_weights(mo_matrix, ref, reverse=maximization)

                if self.debug_mode:
                    print(f'weights: {weights}')

                hv_loss = torch.sum(mo_matrix * weights, dim=0)  # to vector over samples
                hv_loss = torch.sum(hv_loss)
                # sum to balance over prompts, mean to align to 1 sample's ce loss

                if maximization:
                    hv_loss = torch.exp(hv_loss)  # exp() make loss \in [0, 1]
                else:
                    hv_loss = hv_loss
                # hv_loss = hv_loss * self.hv_coeff

                hv_loss.backward()

                if self.debug_mode:
                    print(f'hv loss: {hv_loss.item()}')

                self.epoch_log['scaler']['Tag'].append('loss/hv_loss')
                self.epoch_log['scaler']['Idx'].append(self.epoch)
                self.epoch_log['scaler']['Value'].append(hv_loss.item())

                for k, p in params_to_opt.items():
                    grads[k]['grads'].append(p.grad.clone())        # hv grad

                # cal grad for 2 opt processes
                # params_to_opt
                alphas = []
                for k, p in params_to_opt.items():
                    l1 = grads[k]['grads'][0].flatten()       # although many parts are 0, because only use 10 prompts
                    l2 = grads[k]['grads'][1].flatten()
                    # norm each grad ? will cause alpha to be 0.5
                    # l1 = l1 / torch.norm(l1, p=2)
                    # l2 = l2 / torch.norm(l2, p=2)

                    alpha = -torch.sum(l2 * (l1 - l2)) / torch.sum((l1 - l2) * (l1 - l2))
                    alpha = torch.clip(alpha, 0, 1)
                    # alpha = torch.clip(alpha - 0.5, 0, 0.5) + 0.5     # [0.5 - 1]
                    grads[k]['alpha'] = alpha
                    alphas.append(alpha)

                mean_alpha = torch.mean(torch.stack(alphas)).item()
                if self.debug_mode:
                    print(f'hv mean alpha: {mean_alpha}')
                self.epoch_log['scaler']['Tag'].append('alpha')
                self.epoch_log['scaler']['Idx'].append(self.epoch)
                self.epoch_log['scaler']['Value'].append(mean_alpha)

                self.optimizer.zero_grad()
                for k, p in params_to_opt.items():
                    alpha = grads[k]['alpha']
                    p.grad = alpha * grads[k]['grads'][0] + (1 - alpha) * grads[k]['grads'][1]
                for k, p in params_for_ce.items():
                    p.grad = grads[k]['grads'][0]

            else:   # put grad back to param
                self.optimizer.zero_grad()
                for k, p in params_to_opt.items():
                    p.grad = grads[k]['grads'][0]
                for k, p in params_for_ce.items():
                    p.grad = grads[k]['grads'][0]

            np.random.set_state(state)

        # step
        self.optimizer.step()

        return total_loss.detach(), logits

    def obtain_mo_matrix(self, hard_l, pop_size=None, use_old_prompts=False,
                         add_noise=False, mask: Optional[Union[float, str]] = 0., mask_mode='maskout',
                         train=True, return_labels=False, repeat=1):
        """Return mo_matrix: Torch tensor [obj, pop]
        proj: whether to project mo matrix to simplex.
        mask:   int to be constant prompts,
                None to be only selecting specific prompt,
                randn/uniform/ortho to be random prompts
        """
        if self.n_obj <= 0:
            return None

        ncc_losses_mo = []  # [obj_idx]: obj_idx * tensor[pop_idx]
        # how many prompt for 1 obj according to n_obj
        '''sampling by aux'''
        if pop_size is None:
            pop_size = self.num_aux_sampling
        samples, labels = self.aux.sampling(pop_size * repeat, sort=False)     # [1000, 3, 224, 224]
        if self.gpu:
            samples = samples.cuda()
            labels = labels.cuda()
        n_samples = samples.shape[0]

        '''forward to get objectives'''
        if hard_l in self.e_layers:
            selected_obj_idxs = []
            old_objs = list(range(self.n_obj_avail * self.task_count))
            new_objs = list(range(self.n_obj_avail * self.task_count, self.n_obj_avail * (self.task_count + 1)))
            n_obj = self.n_obj
            # select from old prompts
            if self.task_count == 0:
                use_old_prompts = False           # first task use 3 new prompts
            if use_old_prompts is True:     # use all old prompt as 1 obj
                n_obj = self.n_obj - 1      # the first obj is to use all old prompt
                selected_obj_idxs.append(np.asarray([-1]).astype(int))
            elif type(use_old_prompts) is int:      # select some old objs
                n_obj = self.n_obj - use_old_prompts
                selected_obj_idxs.append(np.sort(
                    np.random.choice(old_objs, use_old_prompts, replace=False)).astype(int))

            # random select self.n_obj obj_idx from self.n_prompt_per_task
            selected_obj_idxs.append(np.sort(
                np.random.choice(new_objs, n_obj, replace=False)).astype(int))

            selected_obj_idxs = np.concatenate(selected_obj_idxs)

            if self.debug_mode:
                print(f'selected_obj_idxs: {selected_obj_idxs}')

            # # detach classifier
            # try:
            #     last = copy.deepcopy(self.model.last)
            # except:
            #     last = copy.deepcopy(self.model.module.last)
            # for p in last.parameters():
            #     p.requires_grad = False
            for re_idx, obj_idx in enumerate(selected_obj_idxs):
                # pen: penultimate features; train: same forward as batch training.
                out = self.model(samples, pen=False, train=train,
                                 hard_obj_idx=obj_idx, hard_l=hard_l,
                                 mask=mask, mask_mode=mask_mode,
                                 # register_blk=hard_l,
                                 debug_mode=self.debug_mode)
                logits = out[0] if train else out
                # logits = last(logits)     # detached last

                # [100, 768]
                # objs = torch.var(logits, dim=1)  # torch[100]
                logits = logits[:, :self.valid_out_dim]
                # ce with heuristic
                logits[:, :self.last_valid_out_dim] = -float('inf')
                # logits[:, :self.last_valid_out_dim] = logits[:, :self.last_valid_out_dim].detach().clone()
                dw_cls = self.dw_k[-1 * torch.ones(labels.size()).long()]
                objs = self.criterion_fn(logits, labels.long()) * dw_cls        # [100]

                # add noise on objs
                if add_noise:
                    noise = 1e-9
                    noise = torch.from_numpy(np.random.rand(*objs.shape)).float().to(objs.device) * noise * 2 - noise
                    # noise = torch.randn_like(objs) * noise * 2 - noise
                    objs = objs + noise

                # collect objs
                ncc_losses_mo.append(objs)

                self.epoch_log['mo']['Tag'].extend(['loss' for _ in range(n_samples)])
                self.epoch_log['mo']['Pop_id'].extend([s_i for s_i in range(n_samples)])
                self.epoch_log['mo']['Obj_id'].extend([re_idx for _ in range(n_samples)])
                self.epoch_log['mo']['Epoch_id'].extend([self.epoch for _ in range(n_samples)])
                self.epoch_log['mo']['Inner_id'].extend([hard_l for _ in range(n_samples)])
                self.epoch_log['mo']['Value'].extend(list(objs.detach().cpu().numpy()))

            # ncc_losses = torch.mean(
            #     torch.stack([torch.stack(ncc_losses_mo[f'l{hard_l}']) for hard_l in self.e_layers]), dim=0)
            # # [5, 2, 100] -> [2, 100]
            ncc_losses = torch.stack(ncc_losses_mo)
        else:
            ncc_losses = None
        if return_labels:
            return ncc_losses, samples, labels
        else:
            return ncc_losses

    def return_front(self, mo_matrix, idx=-1):
        """Return the specific front from mo_matrix: [obj, pop]"""
        n_obj, n_pop = mo_matrix.shape
        mo_obj = mo_matrix.detach().cpu().numpy()

        # non-dom sorting to create multiple fronts
        subfront_indices = fastNonDominatedSort(mo_obj)
        number_of_fronts = np.max(subfront_indices) + 1  # +1 because of 0 indexing

        if idx >= number_of_fronts or idx == -1:
            idx = number_of_fronts - 1      # return the last front (dominated ones)

        return mo_matrix[:, subfront_indices == idx]

    def obtain_mo_matrix_pop_prompt(self, hard_l=None, use_old_prompts=False,
                                    mask: Optional[Union[float, str]] = None, mask_mode='use',
                                    dis='cos', check=False,
                                    train=True,
                                    samples=None, labels=None, addition_concepts=None,
                                    return_nui_labels=False,
                                    group_by_labels=True,
                                    return_features=False
                                    ):
        """Return mo_matrix: Torch tensor [obj, pop]
        Obj: samples; Pop: prompts
        If use old obj, then add 1 individual to use all old prompts
        If addition_concepts is not None: [num, n_prompts], append and return [obj, pop+num]
        """
        if self.n_obj <= 0:
            return None

        def sampling(n_obj, min_samples=10):
            """Sample and check whether number of list of samples contains at least min_samples
            is larger than n_obj, and return cat-ed samples and labels.
            :param n_obj: number of column that need to satisfied.
            :param min_samples:

            :return cat-ed samples and labels.
            """
            _dict = {}
            while len([_dict[_l] for _l in _dict.keys() if _dict[_l].shape[0] >= min_samples]) < n_obj:
                ss, ls, _ = self.aux.sampling(self.num_aux_sampling, sort=True)  # [100, 3, 224, 224]
                for label in sorted(set(ls)):
                    if label in _dict:
                        _dict[label] = torch.cat((_dict[label], ss[ls == label]))
                    else:
                        _dict[label] = ss[ls == label]

            _samples, _labels = [], []
            for label, ss in _dict.items():
                if ss.shape[0] >= min_samples:
                    _samples.append(ss[:min_samples])
                    _labels.append(torch.fill_(torch.empty(min_samples), label))

            _samples = torch.stack(_samples)        # [10, 10, 3, 224, 224] [n_cls, min_samples, *img_size]
            _labels = torch.stack(_labels)          # [10, 10]  [n_cls, min_samples]
            index = np.random.choice(len(_samples), n_obj, replace=False)

            return _samples[index].view(n_obj*min_samples, *_samples.shape[-3:]), _labels[index].view(n_obj*min_samples)

        if samples is None:
            '''sampling by aux'''
            num_sample_per_obj = 3
            samples, labels = sampling(self.n_obj, min_samples=num_sample_per_obj)
            # dead while when n_obj > n_class_per_task

            if self.gpu:
                samples = samples.cuda()
                labels = labels.cuda()

        n_samples = samples.shape[0]
        nui_labels = torch.unique(labels)
        # check if any label has only 1 sample
        if check:
            check = True
            flag = []
            for label in nui_labels:
                if len(labels[labels == label]) < 2:
                    check = False
                    flag.append(label.item())
            if not check:
                print(f'mo: label ({flag}) has/have only 1 sample')
                return None

        '''forward all prompts get objectives [n_samples, pop], pop may with old prompt'''
        if self.FPS:
            old_objs = []
            new_objs = list(range(self.n_obj_avail))
        else:
            old_objs = list(range(self.n_obj_avail * self.task_count))
            new_objs = list(range(self.n_obj_avail * self.task_count, self.n_obj_avail * (self.task_count + 1)))
        target_objs = old_objs + new_objs if use_old_prompts else new_objs

        # n_obj = self.n_obj
        ncc_losses = None
        out = None
        features = None
        if hard_l is None or hard_l in self.e_layers:       # None for all layer to use specific prompt
            # rearrange samples labels and hard_obj_idx
            stack_samples = torch.stack([samples for _ in range(len(target_objs))],
                                        dim=1).reshape(-1, *samples.shape[1:])
            # [bs * n_obj, 3, 224, 224]
            prompt_idxs = torch.as_tensor([obj for _ in range(n_samples) for obj in target_objs]).cuda()
            # int64 tensor [bs * n_obj] -> thus can be separated to diff devices

            if self.debug_mode:
                print('mo samples', stack_samples.shape)

            # pen: penultimate features; train: same forward as batch training.
            out, features = self.model(
                stack_samples, pen=True, train=train,
                hard_obj_idx=prompt_idxs, hard_l=hard_l,
                mask=mask, mask_mode=mask_mode,
                # register_blk=hard_l,
                debug_mode=self.debug_mode)
            # features: [bs*n_prompt, 768]
            # out is logits: [bs*n_prompt, 100]

            ## pair-wise sim loss
            out = out.reshape(n_samples, len(target_objs), out.shape[-1])   # [bs, n_prompt, 100]
            features = features.reshape(n_samples, len(target_objs), features.shape[-1])   # [bs, n_prompt, 768]

            if addition_concepts is not None:       # [n_concepts, 21]
                n_concepts = addition_concepts.size(0)
                out = [out]
                features = [features]
                stack_samples = torch.stack([samples for _ in range(n_concepts)],
                                            dim=1).reshape(-1, *samples.shape[1:])
                # [bs*n_concepts, 100]
                addition_concepts = torch.stack([addition_concepts for _ in range(n_samples)],
                                                dim=0).reshape(-1, *addition_concepts.shape[1:])
                # [bs*n_concepts, 21]

                if self.debug_mode:
                    print('addition concepts mo samples', stack_samples.shape)

                out_, features_ = self.model(
                    stack_samples, pen=True, train=train,
                    hard_l=hard_l,
                    mask=mask, mask_mode=mask_mode,
                    concepts=addition_concepts,
                    # register_blk=hard_l,
                    debug_mode=self.debug_mode)

                ## pair-wise sim loss
                out_ = out_.reshape(n_samples, n_concepts, out_.shape[-1])
                features_ = features_.reshape(n_samples, n_concepts, features_.shape[-1])
                # [bs, n_concepts, 768]
                out.append(out_)
                features.append(features_)
                out = torch.cat(out, dim=1)       # [bs, n_prompt + n_concepts, 100]
                features = torch.cat(features, dim=1)       # [bs, n_prompt + n_concepts, 768]

            ncc_losses = self.obtain_loss(out, labels, mode='last', group_by_labels=group_by_labels)

        if return_nui_labels:
            return ncc_losses, nui_labels
        if return_features:
            return ncc_losses, features

        return ncc_losses

    def obtain_loss(self, out, labels, mode='last', group_by_labels=True):
        nui_labels = torch.unique(labels)
        if mode == 'last':
            # use ce loss
            # out: [bs, n_prompt, 100]
            logits = out
            # broadcast labels
            bs, pop, _ = logits.shape
            logits = logits.view(-1, logits.size(-1))       # [bs*n_prompt, 100]
            cat_labels = torch.stack([labels for _ in range(pop)], dim=1).view(-1)  # [bs*n_prompt]

            logits = logits[:, :self.valid_out_dim]
            # ce with heuristic
            logits[:, :self.last_valid_out_dim] = -float('inf')
            # logits[:, :self.last_valid_out_dim] = logits[:, :self.last_valid_out_dim].detach().clone()
            # dw_cls = self.dw_k[-1 * torch.ones(cat_labels.size()).long()]
            # objs = (self.criterion_fn(logits, cat_labels.long()) * dw_cls).view(bs, pop)  # [bs, n_prompt]
            objs = (self.criterion_fn(logits, cat_labels.long())).view(bs, pop)  # [bs, n_prompt]
            if group_by_labels:
                '''group objectives [n_samples, n_prompt] -> [n_label, n_prompt]'''
                ncc_losses = torch.stack([torch.mean(objs[labels == label], dim=0) for label in nui_labels])
            else:
                ncc_losses = objs
        elif mode in ['cos', 'dot', 'l2']:
            # use prototype loss
            # out: [bs, 768]
            features = out
            # group according to labels
            ncc_losses = []
            grouped_features = []     # each group of samples can have different numbers, can not stack
            anchors = []
            for label in nui_labels:
                label_features = features[labels == label]          # [n_img, n_prompt+, 768]
                grouped_features.append(label_features)

                # obtain anchor
                anchor = torch.mean(label_features, dim=0)       # [n_prompt+, 768]
                anchors.append(anchor)

            anchors = torch.stack(anchors, dim=0)       # [n_labels, n_prompt+, 768]
            anchors = anchors.unsqueeze(0)              # [1, n_label, n_prompt+, 768]

            # loss
            for idx, label in enumerate(nui_labels):
                label_features = grouped_features[idx]          # [n_img, n_prompt+, 768]
                label_features = label_features.unsqueeze(1)    # [n_img, 1, n_prompt+, 768]

                if mode == 'cos':
                    cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
                    sim = cos(anchors, label_features)  # [n_img, n_label, n_prompt+]
                    # [-1+1]
                    # sim[idx] - sum(sim[others/{idx}])
                    posi_sim = 1-sim[:, idx]
                    nega_sim = torch.cat([sim[:, :idx], sim[:, idx+1:]], dim=1)
                    nega_sim[nega_sim < 0] = 0      # max{nega_sim, 0}
                    sim = posi_sim + torch.sum(nega_sim, dim=1)     # [n_img, n_prompt+]
                    # option: or use ce loss on sim -> sim as logits
                    ncc_losses.append(torch.mean(sim, dim=0))     # [n_prompt+]
                # elif dis == 'dot':
                #     dist = - (anchors * label_features).sum(-1)
                #     ncc_losses.append(torch.mean(sim, dim=0))     # [n_prompt+]
                # elif dis == 'l2':
                #     d = nn.PairwiseDistance(p=2)
                #     dist = d(anchors, label_features)     # [0, +inf]
                #     ncc_losses.append(torch.mean(dist, dim=0))     # [n_prompt+]
                else:
                    raise Exception('Unknown distance {}'.format(mode))

                ncc_losses = torch.stack(ncc_losses)        # [n_label(obj), n_prompt (new, old, add_concepts)]

        else:
            raise Exception(f'not implemented mode: {mode}')

        return ncc_losses

    def validation(self, dataloader, model=None, task_in=None, task_metric='acc', verbal=True, task_global=False,
                   **kwargs):
        if not self.concept_weight:
            return super().validation(dataloader, model, task_in, task_metric, verbal, task_global)
        else:
            accs = []
            for concept_id in range(self.num_concepts):
                acc = super().validation(dataloader, model, task_in, task_metric, verbal, task_global,
                                         prompt_id=concept_id)
                accs.append(acc)
            return accs

    def validation_forward_mo(self, dataloader, model=None, task_in=None, task_metric='acc', verbal=True, task_global=False):
        """Different: forward mo and use cls statistics for logits"""
        # pass task to forward if task-awareness
        if model is None:
            model = self.model

        # This function doesn't distinguish tasks.
        batch_timer = Timer()
        acc = AverageMeter()
        batch_timer.tic()

        orig_mode = model.training
        model.eval()

        # load statistics for evaluating
        self.load_statistics()

        for i, sample in enumerate(dataloader):
            concepts = None
            if len(sample) == 3:
                (input, target, task) = sample
            else:   # contain concepts
                (input, target, concepts, task) = sample

            if self.debug_mode:
                print(
                    f'batch{i}: \nlen: {len(target)} target:{(target.min(), target.max())} task:{(task.min(), task.max())}')
            if self.gpu:
                with torch.no_grad():
                    input = input.cuda()
                    target = target.cuda()
            #         if concepts is not None:
            #             concepts = concepts.cuda()  # [bs, 1, 21]
            # if concepts is not None:
            #     concepts = self.process_concepts(concepts, self.num_prompts)        # [bs, 21]
            with torch.no_grad():
                if task_in is None:
                    # output = model.forward(input, task_id=task[0].item())[:, :self.valid_out_dim]
                    # output = model.forward(input,
                    #                        concepts=concepts if self.use_concept_labels_as_aqk else None
                    #                        )[:, :self.valid_out_dim]

                    # forward all prompts
                    _, features = self.obtain_mo_matrix_pop_prompt(
                        None,
                        mask=self.mask, mask_mode=self.mask_mode,
                        train=False,
                        samples=input,
                        labels=target,
                        group_by_labels=False,
                        return_features=True,
                    )  # [bs, 21]

                    # predict
                    # [bs, 21, 768] -> [bs, 100]
                    output = self.predict_mo(features)[:, :self.valid_out_dim]        # [bs, n_cls]

                    # if self.debug_mode:
                    #     print(f'batch{i}: \noutput:{output}')

                    acc = accumulate_acc(output, target, task, acc, topk=(self.top_k,))
                else:
                    mask = target >= task_in[0]
                    mask_ind = mask.nonzero().view(-1)
                    input, target = input[mask_ind], target[mask_ind]

                    mask = target < task_in[-1]
                    mask_ind = mask.nonzero().view(-1)
                    input, target = input[mask_ind], target[mask_ind]

                    if len(target) > 1:
                        # forward all prompts
                        _, features = self.obtain_mo_matrix_pop_prompt(
                            None,
                            mask=self.mask, mask_mode=self.mask_mode,
                            train=False,
                            samples=input,
                            labels=target,
                            group_by_labels=False,
                            return_features=True,
                        )  # [bs, 21]

                        # predict
                        # [bs, 21, 768] -> [bs, 100]
                        output = self.predict_mo(features)        # [bs, n_cls]

                        if task_global:
                            # output = model.forward(input, task_id=task[0].item())[:, :self.valid_out_dim]
                            output = output[:, :self.valid_out_dim]
                            acc = accumulate_acc(output, target, task, acc, topk=(self.top_k,))
                        else:
                            # output = model.forward(input, task_id=task[0].item())[:, task_in]
                            output = output[:, task_in]
                            acc = accumulate_acc(output, target - task_in[0], task, acc, topk=(self.top_k,))

        model.train(orig_mode)

        if verbal:
            self.log(' * Val Acc {acc.avg:.3f}, Total time {time:.2f}'
                     .format(acc=acc, time=batch_timer.toc()))
        return acc.avg

    def collect_statistics(self, train_loader, train_dataset, model=None, verbal=True):
        if model is None:
            model = self.model

        try:
            prompt = model.module.prompt
        except:
            prompt = model.prompt

        # This function doesn't distinguish tasks.
        batch_timer = Timer()
        batch_timer.tic()

        orig_mode = model.training
        model.eval()
        batch_timer.tic()
        for i, sample in enumerate(train_loader):
            concepts = None
            if train_dataset.return_concepts:
                x, y, concepts, task = sample
            else:
                x, y, task = sample

            # send data to gpu
            if self.gpu:
                x = x.cuda()
                y = y.cuda()
                # if concepts is not None:
                #     concepts = concepts.cuda()      # [bs, 224, 224]
                # task = task.cuda()

            # # debug
            # print(f'x shape: {x.shape}, y: {y}, task: {task}')
            with torch.no_grad():
                _, features = self.obtain_mo_matrix_pop_prompt(
                    None, use_old_prompts=False if prompt.FPS else True,
                    mask=self.mask, mask_mode=self.mask_mode,
                    samples=x,
                    labels=y,
                    group_by_labels=False,
                    return_features=True,
                )
                # features: [bs, 21, 768]

            # vars = torch.softmax(torch.var(features, dim=-1), dim=-1)   # [bs, 21]

            # accumulate aligning label   [21, 768]
            uni_y = torch.unique(y)
            for label in uni_y:
                label_features = features[y == label]
                l = label_features.shape[0]
                label = label.item()
                if label in self.cls_stats.keys():
                    self.cls_stats[label] = (self.cls_stats[label]*self.cls_stats[f'n_{label}'] +
                                             torch.sum(label_features, dim=0)
                                             ) / (self.cls_stats[f'n_{label}'] + l)
                    self.cls_stats[f'n_{label}'] += l
                else:
                    self.cls_stats[label] = torch.sum(label_features, dim=0)
                    self.cls_stats[f'n_{label}'] = l

        model.train(orig_mode)

        if verbal:
            self.log(' * Collect statistics: Total time {time:.2f}'
                     .format(time=batch_timer.toc()))

        # save statistics
        stats_path = os.path.join(self.config['log_dir'], 'temp', f'cls_stats.pkl')
        print('=> Saving statistics to:', stats_path)
        with open(stats_path, 'wb') as f:
            pickle.dump(self.cls_stats, f)
        print('=> Save Done')

    def load_statistics(self, verbose=False):
        stats_path = os.path.join(self.config['log_dir'], 'temp', f'cls_stats.pkl')
        if verbose:
            print('=> Load statistics from:', stats_path)
        with open(stats_path, 'rb') as f:
            self.cls_stats = pickle.load(f)

    def predict_mo(self, mo_features):
        """mo_features: [bs, 21, 768] -> [bs, 21]. neg-kl"""

        # # calculate softmax-ed variance
        # mo_logits = torch.softmax(torch.var(mo_features, dim=-1), dim=-1)  # [bs, 21]
        #
        # n_cls = len(self.cls_stats) // 2        # pattern and n_img
        # target = torch.stack([torch.softmax(torch.var(self.cls_stats[idx], dim=-1), dim=-1)
        #                       for idx in range(n_cls)], dim=0)      # [n_cls, 21]
        #
        # logits = -F.kl_div(mo_logits.unsqueeze(1).log(),
        #                    target.unsqueeze(0), reduction='none').mean(-1)    # [bs, n_cls]
        #
        # logits = torch.softmax(logits, dim=-1)      # [bs, n_cls]

        # calculate softmax-ed cos-sim
        norm_features = nn.functional.normalize(mo_features, dim=-1)

        n_cls = len(self.cls_stats) // 2  # pattern and n_img
        target = torch.stack([self.cls_stats[idx]
                              for idx in range(n_cls)], dim=0)  # [n_cls, 21, 768]
        norm_target = nn.functional.normalize(target, dim=-1)

        # cosine sim
        logits = torch.einsum('bpd,cpd->bc', norm_features, norm_target)  # [bs, n_cls]

        return logits


class Auxiliary:
    """Provide auxiliary samples for supporting evaluating prompts"""
    def __init__(self, source=None):
        """source can be a val_dataset"""
        self.source = source

    def update_source(self, source):
        self.source = source

    def sampling(self, num_samples=100, sort=True):
        """
        :param num_samples:
        :param sort: return sorted samples according to targets. Inactivated if no target provided.
        """
        assert self.source is not None
        indexs = np.arange(len(self.source))
        selected = np.random.choice(indexs, num_samples, replace=True if num_samples > len(indexs) else False)

        imgs = []
        targets = []
        concepts = []
        for idx in selected:
            data = self.source[idx]
            imgs.append(data[0])
            targets.append(data[1])
            if len(data) > 2:
                concepts.append(data[2])
        imgs = torch.stack(imgs)
        targets = np.stack(targets)
        if len(concepts) > 1:
            concepts = torch.stack(concepts)

        if sort:
            sorted_indexs = np.argsort(targets)
            imgs = imgs[sorted_indexs]
            targets = targets[sorted_indexs]
            if len(concepts) > 1:
                concepts = concepts[sorted_indexs]

        if len(concepts) > 1:
            return imgs, torch.from_numpy(targets), concepts
        else:
            return imgs, torch.from_numpy(targets)


if __name__ == '__main__':
    import dataloaders
    dataset = dataloaders.CGQA('/mnt/d/OneDrive - City University of Hong Kong - Student/datasets',
                               train=False, validation=True, download_flag=False, seed=0)
    dataset.load_dataset(9, train=False)
    aux = Auxiliary(dataset)


