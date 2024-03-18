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
from .prompt import Prompt
from utils.schedulers import CosineSchedule
from .pmo_utils import Pool, Mixer, available_setting, task_to_device, cal_hv_loss
from models.losses import prototype_loss
import dataloaders


# Our PMO (Pool & Multi-Objective)
class PMOPrompt(Prompt):
    def __init__(self, learner_config):
        super(PMOPrompt, self).__init__(learner_config)
        # self.pool_size = self.prompt_param[1][3]        # =0 if do not enable pool hv loss
        # self.pool = Pool(self.pool_size, self.seed)

        self.train_dataset = None

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

        try:
            prompt = self.model.module.prompt
        except:
            prompt = self.model.prompt
        self.e_layers = prompt.e_layers
        self.n_prompt_per_task = prompt.n_prompt_per_task

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

    def create_model(self):
        cfg = self.config
        model = models.__dict__[cfg['model_type']].__dict__[cfg['model_name']](out_dim=self.out_dim, prompt_flag = 'pmo',prompt_param=self.prompt_param)
        return model

    def learn_batch(self, train_loader, train_dataset, model_save_dir, val_loader=None):
        """Difference:
        Init training log for mo.
        Change aux dataset.
        Save batch_idx
        See nvidia-smi.
        """
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

                if self.epoch == 0:
                    '''nvidia-smi'''
                    print(os.system('nvidia-smi'))

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

    def update_model(self, inputs, targets):
        """Difference:
            Cal mo loss matrix and hv loss.
            backward one loss by one since hv loss is huge
        """

        self.optimizer.zero_grad()

        # logits
        logits, prompt_loss = self.model(inputs, train=True)
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

        total_loss.backward()

        # hv loss calculation without affecting RNG state
        state = np.random.get_state()
        # seed with taskid epochid batchid
        np.random.seed(self.seed + self.train_dataset.t + self.epoch + self.batch_idx)

        '''hv loss'''
        for l in self.e_layers:
            # if self.train_dataset.t > 0:        # start from the second task
            mo_matrix = self.obtain_mo_matrix(hard_l=l, add_noise=True, mask=True)   # [10, 20]

            if self.debug_mode:
                print(f'mo_matrix: {mo_matrix}')

            hv_loss = 0
            if mo_matrix is not None:
                # norm mo matrix?
                ref = None  # dynamic 1.5*max or 0
                hv_loss = cal_hv_loss(mo_matrix, ref, reverse=True)       # not normalized mo matrix

                # total_loss = total_loss + hv_loss
                hv_loss = torch.mean(hv_loss)                       # align to 1 ce loss
                coeff_hv_loss = torch.exp(hv_loss)*10               # exp() make loss \in [0, 1]
                coeff_hv_loss.backward()
                hv_loss = hv_loss.item()

                if self.debug_mode:
                    print(f'hv loss in layer{l}: {hv_loss}')

            # else:
            #     print(f'ERROR: mo_matrix is None, skip layer{l}')

            self.epoch_log['scaler']['Tag'].append('loss/hv_loss')
            self.epoch_log['scaler']['Idx'].append(self.epoch)
            self.epoch_log['scaler']['Value'].append(hv_loss)

        np.random.set_state(state)

        # step
        self.optimizer.step()

        return total_loss.detach(), logits

    def obtain_mo_matrix(self, hard_l, add_noise=True, mask=True):
        """Return mo_matrix: Torch tensor [obj, pop]"""
        if self.n_obj <= 0:
            return None

        ncc_losses_mo = []  # [obj_idx]: obj_idx * tensor[pop_idx]
        # how many prompt for 1 obj according to n_obj
        '''sampling by aux'''
        samples, labels = self.aux.sampling(self.num_aux_sampling, sort=True)     # [1000, 3, 224, 224]
        if self.gpu:
            samples = samples.cuda()
            labels = labels.cuda()
        n_samples = samples.shape[0]

        '''forward to get objectives'''
        if hard_l in self.e_layers:
            # random select self.n_obj obj_idx from self.n_prompt_per_task
            selected_obj_idxs = np.sort(np.random.choice(self.n_prompt_per_task, self.n_obj, replace=False))
            for re_idx, obj_idx in enumerate(selected_obj_idxs):
                # pen: penultimate features; train: same forward as batch training.
                logits, _ = self.model(samples, pen=False, train=True,
                                       hard_obj_idx=obj_idx, hard_l=hard_l, mask=mask,
                                       debug_mode=self.debug_mode)
                # [100, 768]
                # objs = torch.var(logits, dim=1)  # torch[100]
                logits = logits[:, :self.valid_out_dim]
                # ce with heuristic
                logits[:, :self.last_valid_out_dim] = -float('inf')
                dw_cls = self.dw_k[-1 * torch.ones(labels.size()).long()]
                objs = self.criterion_fn(logits, labels.long()) * dw_cls        # [100]

                # add noise on objs
                if add_noise:
                    objs_max = torch.max(objs).detach()
                    objs_min = torch.min(objs).detach()
                    noise = (objs_max - objs_min) / len(objs)*2    # scope of noise
                    noise = torch.from_numpy(np.random.randn(*objs.shape)).to(objs.device) * noise
                    # noise = torch.randn_like(objs) * noise
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

        return ncc_losses

    def obtain_mo_matrix_pop_prompt(self, hard_l, add_noise=True, mask=True):
        """Return mo_matrix: Torch tensor [obj, pop]
        Obj: samples; Pop: prompts
        """
        if self.n_obj <= 0:
            return None

        '''sampling by aux'''
        samples, labels = [], []
        while len(samples) < self.n_obj:
            s, l = self.aux.sampling(100, sort=True)     # [1000, 3, 224, 224]




        if self.gpu:
            samples = samples.cuda()
            labels = labels.cuda()
        n_samples = samples.shape[0]

        ncc_losses_mo = torch.zeros((self.n_obj, self.n_prompt_per_task), device=samples.device)

        '''forward to get objectives'''
        if hard_l in self.e_layers:
            # random select self.n_obj obj_idx from self.n_prompt_per_task
            selected_obj_idxs = np.sort(np.random.choice(self.n_prompt_per_task, self.n_obj, replace=False))
            for re_idx, obj_idx in enumerate(selected_obj_idxs):
                # pen: penultimate features; train: same forward as batch training.
                logits, _ = self.model(samples, pen=False, train=True,
                                       hard_obj_idx=obj_idx, hard_l=hard_l, mask=mask,
                                       debug_mode=self.debug_mode)
                # [100, 768]
                # objs = torch.var(logits, dim=1)  # torch[100]
                logits = logits[:, :self.valid_out_dim]
                # ce with heuristic
                logits[:, :self.last_valid_out_dim] = -float('inf')
                dw_cls = self.dw_k[-1 * torch.ones(labels.size()).long()]
                objs = self.criterion_fn(logits, labels.long()) * dw_cls        # [100]

                # add noise on objs
                if add_noise:
                    objs_max = torch.max(objs).detach()
                    objs_min = torch.min(objs).detach()
                    noise = (objs_max - objs_min) / len(objs)*10    # scope of noise
                    noise = torch.from_numpy(np.random.randn(*objs.shape)).to(objs.device) * noise
                    # noise = torch.randn_like(objs) * noise
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


