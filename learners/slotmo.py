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
from .prompt import Prompt
from utils.schedulers import CosineSchedule
from .pmo_utils import Pool, Mixer, available_setting, task_to_device, cal_hv_weights, normalize
from models.losses import prototype_loss
from mo_optimizers.functions_evaluation import fastNonDominatedSort
import dataloaders


# Our PMO (Pool & Multi-Objective)
class SLOTPrompt(Prompt):
    def __init__(self, learner_config):
        super(SLOTPrompt, self).__init__(learner_config)
        # self.pool_size = self.prompt_param[1][3]        # =0 if do not enable pool hv loss
        # self.pool = Pool(self.pool_size, self.seed)

        self.train_dataset = None
        self.t = 0
        self.epoch = 0

        # load aux
        # aux_dataset = dataloaders.CGQA(
        #     self.config['aux_root'],
        #     train=False, validation=True, download_flag=False, seed=self.config['seed'])
        # aux_dataset.load_dataset(9, train=False)   # consider all samples: 100 classes with 5000 samples.
        # self.aux = Auxiliary(aux_dataset)
        self.aux = Auxiliary()

        # mo
        self.n_opt_slots = int(self.config['prompt_param'][1][3])               # num of slots considered to be opted 5

        try:
            prompt = self.model.module.prompt
        except:
            prompt = self.model.prompt
        self.e_layers = prompt.e_layers
        self.FPS = prompt.FPS
        self.e_pool_size = prompt.e_pool_size       # 100
        self.key_d = prompt.key_d                   # 64

        # cls statistics
        self.cls_stats = {}

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
        model = models.__dict__[cfg['model_type']].__dict__[cfg['model_name']](out_dim=self.out_dim, prompt_flag = 'slot',prompt_param=self.prompt_param, use_vit_emb=False)
        return model

    def load_model(self, filename, drop_last=False):
        # # random init pool
        # if self.pool is None:
        #     self.register_buffer('pool', torch.randn(self.e_pool_size, self.key_d).float())

        super().load_model(filename, drop_last=drop_last)

    def learn_batch(self, train_loader, train_dataset, model_save_dir, val_loader=None):
        self.init_train_log()

        self.train_dataset = train_dataset
        self.t = train_dataset.t
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

        # data weighting
        self.data_weighting(train_dataset)
        if need_train:
            losses = AverageMeter()
            acc = AverageMeter()
            batch_time = AverageMeter()
            batch_timer = Timer()

            if self.t == 0:     # first task do phase I + II + III
                '''phase I: train slots attn'''
                self.log('Phase I: train slot attn!')
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
                    # 'Train Acc {acc.avg:.3f} | '      , acc=acc

                    if self.epoch % 10 == 0:
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

                '''phase II: maintain pool'''
                self.log('Phase II: maintain pool!')
                batch_timer.tic()
                for i, sample in enumerate(train_loader):
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

                    try:
                        model = self.model.module
                    except:
                        model = self.model

                    model.obtain_q(x, maintain_pool=True)

                # measure elapsed time
                t = batch_timer.toc()
                batch_timer.tic()
                self.log(
                    ' * maintain pool | '
                    'Time {t:.3f}'.format(t=t))

            '''phase III: use the closest slots in the pool and train slot2prompt mapping and classifier'''
            if self.reset_optimizer:  # Reset optimizer before learning each task
                self.log('Optimizer is reset!')
                self.init_optimizer()

            losses = AverageMeter()
            acc = AverageMeter()
            batch_time = AverageMeter()
            batch_timer = Timer()

            # total = self.config['schedule'][-1]
            total = 5
            for epoch in range(total):       # self.config['schedule'][-1]
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
                    loss, output = self.update_model(x, y, match_pool=True)  # , task

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
                    'Epoch:{epoch:.0f}/{total:.0f}'.format(epoch=self.epoch + 1, total=total))
                self.log(
                    ' * Loss {loss.avg:.3f} | '
                    'Time {time.avg:.3f}*{i}'.format(
                        loss=losses, time=batch_time, i=len(train_loader)))
                # 'Train Acc {acc.avg:.3f} | '      , acc=acc

                if self.epoch % 10 == 0:
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

    def update_model(self, inputs, targets, match_pool=False):
        self.optimizer.zero_grad()
        try:
            model = self.model.module
        except:
            model = self.model

        # obtain slots
        if match_pool:
            # use pool's slots
            with torch.no_grad():
                slots = model.obtain_q(inputs)  # [bs, k20, h64]
            slots = model.prompt.match_pool(slots)
        else:
            slots = model.obtain_q(inputs)  # [bs, k20, h64]

        # forward all slots without grad to obtain loss matrix used to select minimal-5 for grads.
        # for l in self.e_layers:
        # with torch.no_grad():
        mo_matrix, _ = self.obtain_mo_matrix(
            None, slots=slots,
            train=True,
            samples=inputs,
            labels=targets,
            group_by_labels=False,
            return_features=True,
        )  # [bs, 20]

        if self.debug_mode:
            print(f'mo_matrix {mo_matrix.shape}')

        '''log'''
        for obj_idx in range(mo_matrix.shape[0]):
            for pop_idx in range(mo_matrix.shape[1]):
                self.epoch_log['mo']['Tag'].append('loss')
                self.epoch_log['mo']['Pop_id'].append(pop_idx)
                self.epoch_log['mo']['Obj_id'].append(obj_idx)
                self.epoch_log['mo']['Epoch_id'].append(self.epoch)
                self.epoch_log['mo']['Inner_id'].append(0)
                self.epoch_log['mo']['Value'].append(mo_matrix[obj_idx, pop_idx].item())

        # # select n_opt_slots minimal slots for each img
        # indexs = torch.sort(mo_matrix, dim=1)[1][:, :self.n_opt_slots]
        # indexs = torch.stack([indexs for _ in range(slots.shape[-1])], dim=-1)      # [bs, 5, 64]
        # slots = torch.gather(slots, dim=1, index=indexs)  # [bs, 5, h64]
        # # slots = torch.stack([slots[idx, indexs[idx]] for idx in range(indexs.shape[0])])  # [bs, 5, h64]
        #
        # mo_matrix, features = self.obtain_mo_matrix(
        #     None, slots=slots,
        #     train=True,
        #     samples=inputs,
        #     labels=targets,
        #     group_by_labels=False,
        #     return_features=True,
        # )  # [bs, 5]
        # # features: [bs, 5, 768]

        mo_matrix = torch.sort(mo_matrix, dim=1)[0][:, :self.n_opt_slots]   # [bs, 5]
        loss = torch.mean(mo_matrix, dim=1)        # [bs]
        loss = torch.mean(loss)

        loss.backward()

        if self.debug_mode:
            print(f'loss: {loss.item()}')

        self.epoch_log['scaler']['Tag'].append('loss/mo_loss')
        self.epoch_log['scaler']['Idx'].append(self.epoch)
        self.epoch_log['scaler']['Value'].append(loss.item())

        # # predict according to logits on all objs
        # # logits: [bs, 21, 100] -> [bs, 100] => mean, max
        logits = None

        # step
        self.optimizer.step()

        return loss.detach(), logits

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

    def obtain_mo_matrix(self, hard_l=None, slots=None, addition=None,
                         check=False, train=True,
                         samples=None, labels=None,
                         return_nui_labels=False,
                         group_by_labels=False,
                         return_features=False,
                         ):
        """Return mo_matrix: Torch tensor [obj, pop]
        Obj: samples; Pop: prompts
        If addition is not None: [num, h64], append and return [obj, pop+num]
        """
        if self.n_opt_slots <= 0:
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
            samples, labels = sampling(2, min_samples=num_sample_per_obj)
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

        ncc_losses = None
        features = None
        out = None
        if hard_l is None or hard_l in self.e_layers:       # None for all layer to use specific prompt

            if slots is None:
                # obtain slots
                try:
                    model = self.model.module
                except:
                    model = self.model
                slots = model.obtain_q(samples)  # [bs, k20, h64]

            if addition is not None:
                # add to slots
                addition = torch.stack([addition for _ in range(n_samples)])        # [bs, num, h64]
                slots = torch.cat([slots, addition], dim=1)        # [bs, k20+num, h64]

            len_slots = slots.shape[1]
            stack_slots = slots.reshape(n_samples * len_slots, slots.shape[-1])  # [bs*k, h]

            # rearrange samples labels and hard_obj_idx
            stack_samples = torch.stack([samples for _ in range(len_slots)],
                                        dim=1).reshape(n_samples*len_slots, *samples.shape[1:])
            # [bs * n_slots, 3, 224, 224]

            if self.debug_mode:
                print(n_samples, len_slots, '\nmo samples:', stack_samples.shape, 'slots:', stack_slots.shape)

            # pen: penultimate features; train: same forward as batch training.
            out, features = self.model(
                stack_samples, q=stack_slots,
                pen=True, train=train,
                # register_blk=hard_l,
                debug_mode=self.debug_mode)
            # features: [bs*n_slots, 768]
            # out is logits: [bs*n_slots, 100]

            ## pair-wise sim loss
            out = out.reshape(n_samples, len_slots, out.shape[-1])   # [bs, n_slots, 100]
            features = features.reshape(n_samples, len_slots, features.shape[-1])   # [bs, n_slots, 768]

            ncc_losses = self.obtain_loss(out, labels, mode='last', group_by_labels=group_by_labels)
            # [bs, n_slots]

        if return_nui_labels:
            return ncc_losses, nui_labels
        if return_features:
            return ncc_losses, features, out

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

    def process_concepts(self, concepts, num_prompts):
        # # from [bs, 1, 2] -> [bs, num_prompts]  multi-hot float
        # concepts = concepts[:, 0]       # [bs, 2]
        # concept_labels = F.one_hot(concepts, num_prompts)
        # concept_labels = torch.sum(concept_labels, dim=1).float()

        # # from [bs, 1, num_prompts] -> [bs, num_prompts]  multi-hot float
        concept_labels = concepts[:, 0]       # [bs, 21]

        return concept_labels

    def validation(self, dataloader, model=None, task_in=None, task_metric='acc', verbal=True, task_global=False):
        # return 0
        # pass task to forward if task-awareness
        if model is None:
            model = self.model

        # This function doesn't distinguish tasks.
        batch_timer = Timer()
        acc = AverageMeter()
        batch_timer.tic()

        orig_mode = model.training
        model.eval()

        # # load statistics for evaluating
        # self.load_statistics()

        for i, sample in enumerate(dataloader):
            concepts = None
            if len(sample) == 3:
                (input, target, task) = sample
            else:   # contain concepts
                (input, target, concepts, task) = sample

            if self.debug_mode and i == 0:
                print(
                    f'eval batch{i}: \nlen: {len(target)} target:{(target.min(), target.max())} '
                    f'task:{(task.min(), task.max())}')
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
                    _, _, out = self.obtain_mo_matrix(
                        None,
                        train=False,
                        samples=input,
                        labels=target,
                        group_by_labels=False,
                        return_features=True,
                    )
                    # out: [bs, 20, 100]

                    # # predict based on cls_stats
                    # # [bs, 20, 768] -> [bs, 768]
                    # output = self.predict_mo(features)[:, :self.valid_out_dim]        # [bs, n_cls]

                    # voting [bs, 20, 100] -> [bs, 100]
                    bs, n_slots, n_cls = out.shape
                    out = torch.argmax(out, dim=-1)     # [bs, 20]
                    output = torch.zeros(bs, n_cls).to(out.device)
                    for cls_id in range(n_cls):
                        output[:, cls_id] = torch.sum(out == cls_id, dim=-1)

                    output = output[:, :self.valid_out_dim]        # [bs, n_cls]

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
                        _, _, out = self.obtain_mo_matrix(
                            None,
                            train=False,
                            samples=input,
                            labels=target,
                            group_by_labels=False,
                            return_features=True,
                        )

                        # # predict
                        # # [bs, 21, 768] -> [bs, 100]
                        # output = self.predict_mo(features)        # [bs, n_cls]

                        # voting [bs, 20, 100] -> [bs, 100]
                        bs, n_slots, n_cls = out.shape
                        out = torch.argmax(out, dim=-1)  # [bs, 20]
                        output = torch.zeros(bs, n_cls).to(out.device)
                        for cls_id in range(n_cls):
                            output[:, cls_id] = torch.sum(out == cls_id, dim=-1)

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
        """predict based on cls_stats"""
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

