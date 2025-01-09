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
        out = self.model(inputs, train=True)
        if len(out) == 2:
            logits, prompt_loss = out
            prompt_loss = prompt_loss.sum()
        else:
            logits = out
            prompt_loss = 0
        logits = logits[:,:self.valid_out_dim]

        # # debug
        # print(f'prompt_loss: {prompt_loss}')

        # ce with heuristic
        if self.memory_size == 0:       # replay-based will have old tasks which may cause inf loss
            logits[:,:self.last_valid_out_dim] = -float('inf')
            # logits[:,:self.last_valid_out_dim] = logits[:, :self.last_valid_out_dim].detach().clone()
        
        dw_cls = self.dw_k[targets.long()]
        # dw_cls = self.dw_k[-1 * torch.ones(targets.size()).long()]
        total_loss = self.criterion(logits, targets.long(), dw_cls)

        # # debug
        # print(f'classification loss: {total_loss}')

        # ce loss
        total_loss = total_loss + prompt_loss

        # cheating reg on logits
        concept_similar_reg = torch.zeros(1).mean().to(total_loss.device)
        if self.concept_weight:
            concept_similar_reg = self._concept_similar_reg(None, logits, targets)
            self.epoch_log['scaler']['Tag'].append(f'loss/concept_similar_reg')
            self.epoch_log['scaler']['Idx'].append(self.epoch)
            self.epoch_log['scaler']['Value'].append(concept_similar_reg.item())

            # cal current_coeff
            coeff = self.config['concept_similar_reg_coeff']
            sen = self.config['concept_similar_reg_coeff_sensitivity']
            last_coeff = ((10 / self.n_cls) ** sen) * coeff
            if self.config['args'].dynamic_concept_similar_reg_coeff:
                current_coeff = last_coeff * (self.epoch+1) / self.epochs
            else:
                current_coeff = last_coeff
            self.epoch_log['scaler']['Tag'].append(f'coeff/concept_similar_reg/t{self.t}')
            self.epoch_log['scaler']['Idx'].append(self.epoch)
            self.epoch_log['scaler']['Value'].append(current_coeff)

            total_loss = total_loss + current_coeff * concept_similar_reg

        # step
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        loss_dict = {}
        if self.concept_weight:
            loss_dict['concept_similar_reg'] = torch.round(concept_similar_reg.detach()*100).item() / 100

        return (total_loss.detach(), logits, loss_dict)

    # sets model optimizers
    def init_optimizer(self, target=None, schedule=None):
        if schedule is None:
            schedule = self.schedule

        if type(self.config['lr']) is float:
            lr = self.config['lr']
        else:
            lr = self.config['lr'][0]

        lr_decreace_ratio = self.config['lr_decreace_ratio']
        larger_prompt_lr = self.config['args'].larger_prompt_lr

        # parse optimizer args
        # Multi-GPU
        if len(self.config['gpuid']) > 1:
            last = self.model.module.last
            prompt = self.model.module.prompt
        else:
            last = self.model.last
            prompt = self.model.prompt

        params_to_opt_p, names_p = [], []
        params_to_opt_l, names_l = [], []
        if self.config['mode'] in ['sys', 'pro', 'sub', 'non', 'noc']:
            # if fewshot testing self.config['mode'], only learn classifier: model.last
            for k, p in self.model.named_parameters():
                if 'last' in k and p.requires_grad:
                    params_to_opt_l.append(p)
                    names_l.append(k)
        elif target == 'last':
            for k, p in self.model.named_parameters():
                if 'last' in k and p.requires_grad:
                    params_to_opt_l.append(p)
                    names_l.append(k)
        elif target == 'prompt':
            for k, p in self.model.named_parameters():
                if 'prompt' in k and p.requires_grad:
                    params_to_opt_p.append(p)
                    names_p.append(k)
        elif target == 'p':
            for k, p in self.model.named_parameters():
                if 'e_p_' in k and p.requires_grad:
                    params_to_opt_p.append(p)
                    names_p.append(k)
                elif 'last' in k and p.requires_grad:
                    params_to_opt_l.append(p)
                    names_l.append(k)
        elif target == 'ka':
            for k, p in self.model.named_parameters():
                if ('e_k_' in k or 'e_a_' in k) and p.requires_grad:
                    params_to_opt_p.append(p)
                    names_p.append(k)
                elif 'last' in k and p.requires_grad:
                    params_to_opt_l.append(p)
                    names_l.append(k)
        else:
            for k, p in self.model.named_parameters():
                if 'prompt' in k and p.requires_grad:
                    params_to_opt_p.append(p)
                    names_p.append(k)
                elif 'last' in k and p.requires_grad:
                    params_to_opt_l.append(p)
                    names_l.append(k)

        print('******************* init optimizer **********************')
        print(f'optimizer params: {"all" if target is None else target} '
              f'len {[len(params_to_opt_p), len(params_to_opt_l)]}')
        print(f'{names_p}')
        print(f'{names_l}')

        if larger_prompt_lr:
            lrs = [lr, 0.1 * lr]
        else:       #
            lrs = [lr_decreace_ratio * lr, lr]
        params = [params_to_opt_p, params_to_opt_l]
        print(f'lrs: {lrs}')

        opt_args = []
        for idx in range(len(lrs)):
            _lr = lrs[idx]
            _params = params[idx]
            if len(_params) > 0:
                optimizer_arg = {'params':_params,
                                 'lr':_lr,
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
                opt_args.append(optimizer_arg)

        # create optimizers
        self.optimizer = torch.optim.__dict__[self.config['optimizer']](opt_args, lr=lr)    # default lr
        # self.optimizer = torch.optim.__dict__[self.config['optimizer']](**optimizer_arg)

        # create schedules
        if self.schedule_type == 'cosine':
            self.scheduler = CosineSchedule(self.optimizer, K=schedule[-1])
        elif self.schedule_type == 'decay':
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=schedule, gamma=0.1)
        else:       # no change
            self.scheduler = type('empty_scheduler', (), {})()
            self.scheduler.step = lambda x=0: None       # empty object scheduler with empty step() func.

    def create_model(self):
        # create vit-baseline without prompt
        cfg = self.config
        model = models.__dict__[cfg['model_type']].__dict__[cfg['model_name']](out_dim=self.out_dim, prompt_flag = 'None',prompt_param=self.prompt_param)
        return model

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


class CODAPromptCond(Prompt):

    def __init__(self, learner_config):
        self.given_prompts = False       # for cheating before cal super
        super(CODAPromptCond, self).__init__(learner_config)

        self.use_concept_labels = True
        self.use_concept_labels_as_aqk = True       # for cheating
        self.num_prompts = int(self.prompt_param[1][0])     # 21

        try:
            prompt = self.model.module.prompt
            last = self.model.module.last
        except:
            prompt = self.model.prompt
            last = self.model.last
        self.model_prompt = prompt
        self.model_last = last

    def create_model(self):
        cfg = self.config
        model = models.__dict__[cfg['model_type']].__dict__[cfg['model_name']](out_dim=self.out_dim, prompt_flag = 'coda_cond',prompt_param=self.prompt_param, use_vit_emb=False)
        if self.given_prompts:
            # load oracle prompts
            model_name = 'coda-cond-FPS21-normalattn-oracle-epoch5-cheating-mtl-1'
            filename = '/'.join(cfg['log_dir'].split('/')[:-1]) + '/' + model_name + '/models/repeat-1/task-1/'
            print(f'load prompts from {filename}')
            state_dict = torch.load(filename + 'class.pth')
            for key in list(state_dict.keys()):
                # complete with/without module.
                if 'module' in key:
                    state_dict[key[7:]] = state_dict[key]
                else:
                    state_dict[f'module.{key}'] = state_dict[key]
                # only remain with key 'e_p_'
                if 'e_p_' not in key:
                    del state_dict[key]
                    del state_dict[f'module.{key}']
            model.load_state_dict(state_dict, strict=False)
            for key, param in state_dict.items():
                print(f'{key}: {param.shape}')
            self.log('=> Load Done')
        return model

    def learn_batch(self, train_loader, train_dataset, model_save_dir, val_loader=None):

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
            # self.init_optimizer(target='ka' if self.given_prompts else None)
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

                    concepts = None
                    if hasattr(train_dataset, "return_concepts") and train_dataset.return_concepts:
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
                            concepts = concepts.cuda()  # [bs, 224, 224]

                    # # debug
                    # print(f'x shape: {x.shape}, y: {y}, task: {task}')

                    # model update
                    if not self.use_concept_labels:
                        concepts = None
                    loss, output, selection_loss = self.update_model(x, y, concepts=concepts)

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
                    ' * Loss {loss.avg:.3f} | S Loss {selection_loss:.3f} | '
                    'Train Acc {acc.avg:.3f} | '
                    'Time {time.avg:.3f}*{i}'.format(
                        loss=losses, selection_loss=selection_loss if selection_loss else 0, acc=acc, time=batch_time,
                        i=len(train_loader)))

                # reset
                losses = AverageMeter()
                acc = AverageMeter()

                # validation
                if val_loader is not None:
                    val_acc = self.validation(val_loader)

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

    def update_model(self, inputs, targets, concepts=None):
        if concepts is not None:
            # concepts: [bs, 1, 21] -> multi-hot float [bs, 21]
            concepts = self.process_concepts(concepts, self.num_prompts)
        logits, prompt_loss, aq_k_list = self.model(inputs, train=True, return_aqk=True,
                                                    concepts=concepts if self.use_concept_labels_as_aqk else None)
        logits = logits[:,:self.valid_out_dim]

        # # debug
        # print(f'prompt_loss: {prompt_loss}')

        # ce with heuristic
        if self.memory_size == 0:       # replay-based will have old tasks which may cause inf loss
            logits[:,:self.last_valid_out_dim] = -float('inf')
            # logits[:,:self.last_valid_out_dim] = logits[:, :self.last_valid_out_dim].detach().clone()
        dw_cls = self.dw_k[-1 * torch.ones(targets.size()).long()]
        total_loss = self.criterion(logits, targets.long(), dw_cls)

        # # debug
        # print(f'classification loss: {total_loss}')

        # ce loss
        total_loss = total_loss + prompt_loss.sum()

        # obtain grad on prompts
        self.optimizer.zero_grad()
        total_loss.backward(retain_graph=True)
        # remove grad on KA
        for key, param in self.model_prompt.named_parameters():
            if 'e_k_' in key or 'e_a_' in key:
                param.grad = None

        # loss on KA -> aq_k_list
        selection_loss = None
        if self.use_concept_labels and not self.use_concept_labels_as_aqk:      # explicitly use concept as selection.
            # logits, aq_k_list: [12*[bs, num_prompt]]
            # num_prompts = aq_k_list[0].shape[-1]

            selection_loss = []
            selection_criterion = nn.BCELoss(reduction='none')  # should be huber loss 0.5(x-y)^2 for regression
            for aq_k in aq_k_list:
                # selection_loss.append(selection_criterion(aq_k, concepts).mean())
                selection_loss.append(selection_criterion(aq_k, concepts).sum(dim=-1).mean())
                # sample-wise mean # amplify num_prompts times
            selection_loss = torch.sum(torch.stack(selection_loss))     # sum over layers
            selection_loss.backward()

        # step
        # self.optimizer.zero_grad()
        # total_loss.backward()
        self.optimizer.step()

        if selection_loss is not None:
            return total_loss.detach(), logits, selection_loss.detach()
        else:
            return total_loss.detach(), logits, selection_loss

    def process_concepts(self, concepts, num_prompts):
        # # from [bs, 1, 2] -> [bs, num_prompts]  multi-hot float
        # concepts = concepts[:, 0]       # [bs, 2]
        # concept_labels = F.one_hot(concepts, num_prompts)
        # concept_labels = torch.sum(concept_labels, dim=1).float()

        # # from [bs, 1, num_prompts] -> [bs, num_prompts]  multi-hot float
        concept_labels = concepts[:, 0]       # [bs, 21]

        return concept_labels

    def validation(self, dataloader, model=None, task_in=None, task_metric='acc', verbal=True, task_global=False):
        """Different: if use concept for forwarding"""
        # pass task to forward if task-awareness
        if model is None:
            model = self.model

        # This function doesn't distinguish tasks.
        batch_timer = Timer()
        acc = AverageMeter()
        batch_timer.tic()

        orig_mode = model.training
        model.eval()
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
                    if concepts is not None:
                        concepts = concepts.cuda()  # [bs, 1, 21]
            if concepts is not None:
                concepts = self.process_concepts(concepts, self.num_prompts)        # [bs, 21]
            if task_in is None:
                # output = model.forward(input, task_id=task[0].item())[:, :self.valid_out_dim]
                output = model.forward(input,
                                       concepts=concepts if self.use_concept_labels_as_aqk else None
                                       )[:, :self.valid_out_dim]

                # if self.debug_mode:
                #     print(f'batch{i}: \noutput:{output}')

                acc = accumulate_acc(output, target, task, acc, topk=(self.top_k,))
            else:
                mask = target >= task_in[0]
                mask_ind = mask.nonzero().view(-1)
                input, target, concepts = input[mask_ind], target[mask_ind], concepts[mask_ind]

                mask = target <= task_in[-1]
                mask_ind = mask.nonzero().view(-1)
                input, target, concepts = input[mask_ind], target[mask_ind], concepts[mask_ind]

                if len(target) > 1:
                    if task_global:
                        # output = model.forward(input, task_id=task[0].item())[:, :self.valid_out_dim]
                        output = model.forward(input,
                                               concepts=concepts if self.use_concept_labels_as_aqk else None
                                               )[:, :self.valid_out_dim]
                        acc = accumulate_acc(output, target, task, acc, topk=(self.top_k,))
                    else:
                        # output = model.forward(input, task_id=task[0].item())[:, task_in]
                        output = model.forward(input,
                                               concepts=concepts if self.use_concept_labels_as_aqk else None
                                               )[:, task_in]
                        acc = accumulate_acc(output, target - task_in[0], task, acc, topk=(self.top_k,))

        model.train(orig_mode)

        if verbal:
            self.log(' * Val Acc {acc.avg:.3f}, Total time {time:.2f}'
                     .format(acc=acc, time=batch_timer.toc()))
        return acc.avg


class PATCHPrompt(CODAPromptCond):

    def __init__(self, learner_config):
        super(PATCHPrompt, self).__init__(learner_config)

    def create_model(self):
        cfg = self.config
        model = models.__dict__[cfg['model_type']].__dict__[cfg['model_name']](out_dim=self.out_dim, prompt_flag = 'patch',prompt_param=self.prompt_param, use_vit_emb=False)
        return model

    def update_model(self, inputs, targets, concepts=None):
        """Difference: selection loss becomes patch-wise"""
        if concepts is not None:
            # concepts: [bs, 224, 224] -> [bs, 197, 21]
            concepts = self.process_concepts(concepts, self.num_prompts)
        logits, prompt_loss, aq_k_list = self.model(inputs, train=True, return_aqk=True,
                                                    concepts=concepts if self.use_concept_labels_as_aqk else None)
        logits = logits[:,:self.valid_out_dim]

        # # debug
        # print(f'prompt_loss: {prompt_loss}')

        # ce with heuristic
        if self.memory_size == 0:       # replay-based will have old tasks which may cause inf loss
            logits[:,:self.last_valid_out_dim] = -float('inf')
            # logits[:,:self.last_valid_out_dim] = logits[:, :self.last_valid_out_dim].detach().clone()
        dw_cls = self.dw_k[-1 * torch.ones(targets.size()).long()]
        total_loss = self.criterion(logits, targets.long(), dw_cls)

        # # debug
        # print(f'classification loss: {total_loss}')

        # ce loss
        total_loss = total_loss + prompt_loss.sum()

        # obtain grad on prompts
        self.optimizer.zero_grad()
        total_loss.backward(retain_graph=True)
        # remove grad on KA
        for key, param in self.model_prompt.named_parameters():
            if 'e_k_' in key or 'e_a_' in key:
                param.grad = None

        # loss on KA -> aq_k_list
        selection_loss = None
        if self.use_concept_labels:      # explicitly use concept as selection.
            # logits, aq_k_list: [12*[bs, 197, num_prompt]]
            # num_prompts = aq_k_list[0].shape[-1]

            selection_loss = []
            selection_criterion = nn.BCELoss(reduction='none')
            for aq_k in aq_k_list:
                # remove cls_token      [bs, 196, num_prompt]
                selection_loss.append(selection_criterion(aq_k[:, 1:, :], concepts[:, 1:, :]).mean())
                # selection_loss.append(selection_criterion(aq_k, concept_labels).sum(dim=2).mean())
                # patch-wise mean # amplify 10 times
            selection_loss = torch.mean(torch.stack(selection_loss))
            selection_loss.backward()

        # step
        # self.optimizer.zero_grad()
        # total_loss.backward()
        self.optimizer.step()

        if selection_loss is not None:
            return total_loss.detach(), logits, selection_loss.detach()
        else:
            return total_loss.detach(), logits, selection_loss

    def process_concepts(self, concepts, num_prompts, add_zero_cls_token=True):
        # from [bs, 224, 224] -> [bs, 197, num_prompts]
        try:
            model = self.model.module
        except:
            model = self.model
        patch_size = model.feat.patch_embed.patch_size  # (16, 16)

        concept_labels = concepts.unfold(1, *patch_size).unfold(2, *patch_size)
        # [bs, n_H, n_W, patch_size, patch_size]
        # max pooling: generally all elements in one patch have same mask value  # [bs, 196]
        concept_labels = torch.max(torch.max(concept_labels, dim=-1)[0], dim=-1)[0].flatten(1)
        concept_labels = F.one_hot(concept_labels, num_classes=num_prompts + 1)[:, :, :num_prompts].float()
        # blank's label is num_prompts, thus do not use all prompts

        # add zero for cls-token
        if add_zero_cls_token:
            concept_labels = torch.cat([torch.zeros_like(concept_labels[:, :1, :]), concept_labels], dim=1)

        return concept_labels


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
                for i, sample in enumerate(train_loader):

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