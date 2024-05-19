from __future__ import print_function
import sys
import math
from typing import Optional, Union

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
from .prompt import Prompt
from utils.schedulers import CosineSchedule
from .pmo_utils import Pool, Mixer, available_setting, task_to_device, cal_hv_weights, normalize_to_simplex
from models.losses import prototype_loss
from mo_optimizers.functions_evaluation import fastNonDominatedSort
import dataloaders


# Our PMO (Pool & Multi-Objective)
class PMOPrompt(Prompt):
    def __init__(self, learner_config):
        super(PMOPrompt, self).__init__(learner_config)
        # self.pool_size = self.prompt_param[1][3]        # =0 if do not enable pool hv loss
        # self.pool = Pool(self.pool_size, self.seed)

        self.train_dataset = None
        self.epoch = 0

        # load aux
        # aux_dataset = dataloaders.CGQA(
        #     self.config['aux_root'],
        #     train=False, validation=True, download_flag=False, seed=self.config['seed'])
        # aux_dataset.load_dataset(9, train=False)   # consider all samples: 100 classes with 5000 samples.
        # self.aux = Auxiliary(aux_dataset)
        self.aux = Auxiliary()

        # mo
        self.n_obj = int(self.config['n_obj'])
        self.num_aux_sampling = int(self.config['num_aux_sampling'])
        self.mask = self.config['prompt_param'][1][4]               # constant float or randn or uniform or ortho
        self.mask_mode = int(self.config['prompt_param'][1][5])     # maskout or use
        if int(self.mask) == -10000:
            self.mask = 'randn'
        elif int(self.mask) == -10001:
            self.mask = 'uniform'
        elif int(self.mask) == -10002:
            self.mask = 'ortho'
        elif int(self.mask) == -10003:
            self.mask = None
        if self.mask_mode == 0:
            self.mask_mode = 'maskout'
        elif self.mask_mode == 1:
            self.mask_mode = 'use'
        else:
            raise Exception(f'Unknown mask mode {self.mask_mode}')
        print(f'Mask info: {self.mask_mode}->{self.mask}')
        self.hv_coeff = self.config['hv_coeff']     # -1 if use LCQP

        try:
            prompt = self.model.module.prompt
        except:
            prompt = self.model.prompt
        self.e_layers = prompt.e_layers
        self.n_prompt_per_task = prompt.n_prompt_per_task
        self.n_obj_avail = prompt.n_obj
        self.FPS = prompt.FPS

        # log
        self.epoch_log = dict()
        self.init_train_log()

    def init_train_log(self):
        self.epoch_log = dict()
        # Tag: acc/loss
        self.epoch_log['mo'] = {'Tag': [], 'Pop_id': [], 'Obj_id': [], 'Epoch_id': [], 'Inner_id': [], 'Value': []}
        # 'loss/hv_loss'
        self.epoch_log['scaler'] = {'Tag': [], 'Idx': [], 'Value': []}

    def train_log_to_df(self):
        self.epoch_log['mo'] = pd.DataFrame(self.epoch_log['mo'])
        self.epoch_log['scaler'] = pd.DataFrame(self.epoch_log['scaler'])

    def create_model(self, use_vit_emb=False):
        cfg = self.config
        model = models.__dict__[cfg['model_type']].__dict__[cfg['model_name']](out_dim=self.out_dim, prompt_flag = 'pmo',prompt_param=self.prompt_param, use_vit_emb=use_vit_emb)
        return model

    def learn_batch(self, train_loader, train_dataset, model_save_dir, val_loader=None):
        """Difference:
        Init training log for mo.
        Change aux dataset.
        Save batch_idx
        See nvidia-smi.
        """
        # return self.learn_batch_diff_stage(train_loader, train_dataset, model_save_dir, val_loader)

        self.init_train_log()

        self.train_dataset = train_dataset
        self.aux.update_source(train_dataset)       # aux samples from the current task

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
                    self.batch_idx = i

                    # verify in train mode
                    self.model.train()

                    # send data to gpu
                    if self.gpu:
                        x = x.cuda()
                        y = y.cuda()
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

                if self.epoch == 0:
                    '''nvidia-smi'''
                    self.log(os.system('nvidia-smi'))

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
        """iteratively learn"""
        self.init_train_log()

        self.train_dataset = train_dataset
        self.aux.update_source(train_dataset)       # aux samples from the current task

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

            # pre_learn_epochs = 5
            iter_epochs = 5
            phase = 'p'        # phase start from 'ka' and p-ka loop
            for epoch in range(self.config['schedule'][-1]):
                self.epoch = epoch

                if epoch % iter_epochs == 0:
                    # update lr
                    self.config['lr'] = self.optimizer.param_groups[0]['lr']
                    phase = 'p' if phase == 'ka' else 'ka'
                    self.log(f'epoch: {epoch}: Optimizer is reset for {phase}')
                    self.init_optimizer(target=phase, schedule=[s-epoch for s in self.schedule])

                for param_group in self.optimizer.param_groups:
                    self.log('LR:', param_group['lr'])
                batch_timer.tic()
                for i, (x, y, task) in enumerate(train_loader):
                    self.batch_idx = i

                    # verify in train mode
                    self.model.train()

                    # send data to gpu
                    if self.gpu:
                        x = x.cuda()
                        y = y.cuda()
                        # task = task.cuda()

                    # # debug
                    # print(f'x shape: {x.shape}, y: {y}, task: {task}')

                    # model update
                    # loss, output = self.update_model(x, y, pre_learn=True if epoch < pre_learn_epochs else False)
                    loss, output = self.update_model(x, y)

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

                if self.epoch == 0:
                    '''nvidia-smi'''
                    print(os.system('nvidia-smi'))

                self.scheduler.step()

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

    def update_model(self, inputs, targets, pre_learn=False):
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
                    normed_mo_matrix = normalize_to_simplex(mo_matrix, noise=True)
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
        # params_to_opt = {key: param for key, param in prompt.named_parameters() if 'e_k_' in key or 'e_a_' in key}
        params_to_opt = {key: param for key, param in prompt.named_parameters()}
        params_for_ce = {key: param for key, param in last.named_parameters()}
        # params_for_ce.update({key: param for key, param in prompt.named_parameters() if 'e_p_' in key})

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
            selected_inputs = torch.cat([inputs[targets == label] for label in selected_labels])
            selected_targets = torch.cat([targets[targets == label] for label in selected_labels])

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
                mo_matrix = normalize_to_simplex(mo_matrix)
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
                    # objs_max = torch.max(objs).detach()
                    # objs_min = torch.min(objs).detach()
                    # noise = (objs_max - objs_min) / len(objs)*2    # scope of noise
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
                                    mask: Optional[Union[float, str]] = 0., mask_mode='maskout',
                                    train=True,
                                    samples=None, labels=None):
        """Return mo_matrix: Torch tensor [obj, pop]
        Obj: samples; Pop: prompts
        If use old obj, then add 1 individual to use all old prompts
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
                ss, ls = self.aux.sampling(self.num_aux_sampling, sort=True)  # [100, 3, 224, 224]
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
            num_sample_per_obj = 10
            samples, labels = sampling(self.n_obj, min_samples=num_sample_per_obj)
            # dead while when n_obj > n_class_per_task

            if self.gpu:
                samples = samples.cuda()
                labels = labels.cuda()

        n_samples = samples.shape[0]
        nui_labels = torch.unique(labels)
        # check if any label has only 1 sample
        check = True
        for label in nui_labels:
            if len(labels[labels == label]) < 2:
                check = False
        if not check:
            return None

        '''forward all prompts get objectives [n_samples, pop], pop may with old prompt'''
        ncc_losses = []

        if self.FPS:
            old_objs = []
            new_objs = list(range(self.n_obj_avail))
        else:
            old_objs = list(range(self.n_obj_avail * self.task_count))
            new_objs = list(range(self.n_obj_avail * self.task_count, self.n_obj_avail * (self.task_count + 1)))
        n_obj = self.n_obj

        if hard_l is None or hard_l in self.e_layers:       # None for all layer to use specific prompt
            for prompt_idx in new_objs:
                # pen: penultimate features; train: same forward as batch training.
                out = self.model(samples, pen=True, train=train,
                                 hard_obj_idx=prompt_idx, hard_l=hard_l,
                                 mask=mask, mask_mode=mask_mode,
                                 # register_blk=hard_l,
                                 debug_mode=self.debug_mode)
                logits = out[0] if train else out       # pen=True, logits is features: [bs, 768]

                ## ce loss
                # # [bs, 100]
                # logits = logits[:, :self.valid_out_dim]
                # # ce with heuristic
                # logits[:, :self.last_valid_out_dim] = -float('inf')
                # dw_cls = self.dw_k[-1 * torch.ones(labels.size()).long()]
                # objs = self.criterion_fn(logits, labels.long()) * dw_cls        # [bs]
                # ncc_losses.append(objs)

                ## pair-wise sim loss
                # logits = logits / torch.norm(logits, dim=1, keepdim=True)     # [bs, 768]
                cos = nn.CosineSimilarity(dim=1, eps=1e-6)
                # group according to labels
                objs = []
                for label in nui_labels:
                    label_logits = logits[labels == label]
                    # for each group, cal cos sim = avg cos sim (avg(label_logits), label_logits)
                    label_logits_anchor = torch.mean(label_logits, dim=0)
                    cos_sim = cos(label_logits_anchor, label_logits) + 1     # [bs]
                    # +1 to scope [-1, 1] -> [0, 2]
                    objs.append(torch.mean(cos_sim))
                objs = torch.stack(objs)
                ncc_losses.append(objs)

            if use_old_prompts:
                # out = self.model(samples, train=train, pre_learn=True,
                #                  debug_mode=self.debug_mode)
                out = self.model(samples, pen=False, train=train,
                                 hard_obj_idx=-1, hard_l=hard_l,
                                 mask=mask, mask_mode=mask_mode,
                                 # register_blk=hard_l,
                                 debug_mode=self.debug_mode)
                logits = out[0] if train else out

                ## ce loss
                # # [bs, 100]
                # logits = logits[:, :self.valid_out_dim]
                # # ce with heuristic
                # logits[:, :self.last_valid_out_dim] = -float('inf')
                # dw_cls = self.dw_k[-1 * torch.ones(labels.size()).long()]
                # objs = self.criterion_fn(logits, labels.long()) * dw_cls        # [bs]
                # ncc_losses.append(objs)

                ## pair-wise sim loss
                # logits = logits / torch.norm(logits, dim=1, keepdim=True)     # [bs, 768]
                cos = nn.CosineSimilarity(dim=1, eps=1e-6)
                # group according to labels
                objs = []
                for label in nui_labels:
                    label_logits = logits[labels == label]
                    # for each group, cal cos sim = avg cos sim (avg(label_logits), label_logits)
                    label_logits_anchor = torch.mean(label_logits, dim=0)
                    cos_sim = cos(label_logits_anchor, label_logits) + 1     # [bs]
                    # +1 to scope [-1, 1] -> [0, 2]
                    objs.append(torch.mean(cos_sim))
                objs = torch.stack(objs)
                ncc_losses.append(objs)

        ncc_losses = torch.stack(ncc_losses, dim=1)

        # '''group objectives [n_samples, pop] -> [n_obj, pop]'''
        # ncc_losses = torch.stack([torch.mean(ncc_losses[labels == label], dim=0) for label in nui_labels])

        '''log'''
        for obj_idx in range(ncc_losses.shape[0]):
            for pop_idx in range(ncc_losses.shape[1]):
                self.epoch_log['mo']['Tag'].append('loss')
                self.epoch_log['mo']['Pop_id'].append(pop_idx)
                self.epoch_log['mo']['Obj_id'].append(obj_idx)
                self.epoch_log['mo']['Epoch_id'].append(self.epoch)
                self.epoch_log['mo']['Inner_id'].append(0)
                self.epoch_log['mo']['Value'].append(ncc_losses[obj_idx, pop_idx].item())

        return ncc_losses


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
        for idx in selected:
            data = self.source[idx]
            imgs.append(data[0])
            targets.append(data[1])
        imgs = torch.stack(imgs)
        targets = np.stack(targets)

        if sort:
            sorted_indexs = np.argsort(targets)
            imgs = imgs[sorted_indexs]
            targets = targets[sorted_indexs]

        return imgs, torch.from_numpy(targets)


if __name__ == '__main__':
    import dataloaders
    dataset = dataloaders.CGQA('/mnt/d/OneDrive - City University of Hong Kong - Student/datasets',
                               train=False, validation=True, download_flag=False, seed=0)
    dataset.load_dataset(9, train=False)
    aux = Auxiliary(dataset)


