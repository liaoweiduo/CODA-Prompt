from __future__ import print_function
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
import sys
import copy
import pandas as pd
from utils.schedulers import CosineSchedule

class NormalNN(nn.Module):
    '''
    Normal Neural Network with SGD for classification
    '''
    def __init__(self, learner_config):

        super(NormalNN, self).__init__()
        self.debug_mode = learner_config['debug_mode']
        self.log = print
        self.config = learner_config
        self.out_dim = learner_config['out_dim']
        self.model = self.create_model()
        self.reset_optimizer = True
        self.overwrite = learner_config['overwrite']
        self.batch_size = learner_config['batch_size']
        self.tasks = learner_config['tasks']
        self.top_k = learner_config['top_k']
        self.seed = learner_config['seed']
        self.concept_weight = self.config['concept_weight']  # True to use concept
        self.target_concept_id = learner_config.get('target_concept_id')    # can be None for CFST

        # cls statistics
        self.cls_stats = {}
        self.cls_stats_n = {}

        # replay memory parameters
        self.memory_size = self.config['memory']
        self.task_count = 0

        # class balancing
        self.dw = self.config['DW']
        if self.memory_size <= 0:
            self.dw = False

        # supervised criterion
        self.criterion_fn = nn.CrossEntropyLoss(reduction='none')
        
        # cuda gpu
        if learner_config['gpuid'][0] >= 0:
            self.cuda()
            self.gpu = True
        else:
            self.gpu = False
        
        # highest class index from past task
        self.last_valid_out_dim = 0 

        # highest class index from current task
        self.valid_out_dim = 0

        # set up schedules
        self.schedule_type = self.config['schedule_type']
        self.schedule = self.config['schedule']

        # initialize optimizer
        self.init_optimizer()

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

    ##########################################
    #           MODEL TRAINING               #
    ##########################################

    def collect_statistics(self, train_loader, train_dataset, model=None, refresh=False):
        if model is None:
            model = self.model

        # refresh cls_stats for different tasks
        if refresh:
            self.cls_stats = {}
            self.cls_stats_n = {}

        batch_timer = Timer()
        batch_timer.tic()

        orig_mode = model.training
        model.eval()
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

            with torch.no_grad():
                _, features = model.forward(x, pen=True, forward_last=False)

                # label-wise collecting prototypes
                labels = torch.unique(y)
                for label in labels:
                    label = label.item()

                    # collect slots with this label
                    labeled_features = features[y == label]  # [n_img, 768]
                    n_img = labeled_features.size(0)
                    avg_features = torch.mean(labeled_features, dim=0)

                    if label in self.cls_stats.keys():
                        prev_features = self.cls_stats[label]
                        prev_n_img = self.cls_stats_n[label]
                        avg_features = (prev_features * prev_n_img + avg_features * n_img) / (prev_n_img + n_img)
                        n_img = prev_n_img + n_img

                    self.cls_stats[label] = avg_features
                    self.cls_stats_n[label] = n_img

        model.train(orig_mode)

        self.log(' * Collect statistics: Total time {time:.2f}'
                 .format(time=batch_timer.toc()))

    def learn_batch(self, train_loader, train_dataset, model_save_dir, val_loader=None):
        self.init_train_log()

        self.train_dataset = train_dataset
        self.t = train_dataset.t
        self.n_cls = len(self.tasks[self.t])     # tasks: [[0,1,...,49], [50,...,59], ...]
        print(f'num of classes: {self.n_cls}.')
        
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
            self.epochs = self.config['schedule'][-1]

            for epoch in range(self.config['schedule'][-1]):
                self.epoch=epoch

                if epoch > 0: self.scheduler.step()
                for param_group in self.optimizer.param_groups:
                    self.log('LR:', param_group['lr'])
                batch_timer.tic()
                loss_dict = {}
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

                    # # debug
                    # print(f'x shape: {x.shape}, y: {y}, task: {task}')

                    # debug:
                    # if self.debug_mode:
                    fake_img = torch.ones_like(x)       # bs, 3, 224, 224
                    fake_y = torch.ones_like(y).long()
                    # model update
                    loss, output, loss_dict = self.update_model(fake_img, fake_y)
                    print(f'debug: NormalNN, learn_batch, loss: {loss.item():.4f}; output {output.shape}: {output[0].detach().cpu().numpy()}')
                    raise Exception('stop')

                    # model update
                    loss, output, loss_dict = self.update_model(x, y)

                    # measure elapsed time
                    batch_time.update(batch_timer.toc())  
                    batch_timer.tic()
                    
                    # measure accuracy and record loss
                    y = y.detach()
                    accumulate_acc(output, y, task, acc, topk=(self.top_k,))
                    losses.update(loss,  y.size(0)) 
                    batch_timer.tic()

                # eval update
                self.log('Epoch:{epoch:.0f}/{total:.0f}'.format(epoch=self.epoch+1,total=self.config['schedule'][-1]))
                self.log(
                    ' * Loss {loss.avg:.3f} | '
                    'Train Acc {acc.avg:.3f} | '
                    '{loss_dict} | '
                    'Time {time.avg:.3f}*{i}'.format(
                        loss=losses, acc=acc, time=batch_time, i=len(train_loader),
                        loss_dict=loss_dict))

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

    def criterion(self, logits, targets, data_weights):
        loss_supervised = (self.criterion_fn(logits, targets.long()) * data_weights).mean()
        return loss_supervised 

    def _concept_similar_reg(self, features, logits, targets):
        """Cheating on concept-aware to decrease distance between two imgs that share the same concept"""
        # features: [bs 768]; logits with full range: [bs, 100]; targets: [bs]

        # preprocess logits
        logits = logits[:, self.last_valid_out_dim:self.valid_out_dim]      # [bs, 10]

        # collect concepts
        # self.label_concepts: [100, 2]
        concepts_batch = self.label_concepts[targets.cpu().numpy()]   # [bs, 2]
        concepts = np.unique(concepts_batch.flatten())

        # for each concept
        losses = []
        dist = nn.PairwiseDistance(p=2)     # l2-distance
        for concept in concepts:
            involved_img_inds = [idx for idx, img_concepts in enumerate(concepts_batch) if concept in img_concepts]
            if len(involved_img_inds) > 1:
                involved_logits = logits[involved_img_inds]    # [n_img, 10]

                # distance
                # [n_img, n_img] -> []
                loss = dist(involved_logits.unsqueeze(0), involved_logits.unsqueeze(1)).mean()
                losses.append(loss)
        losses = torch.stack(losses).mean()

        return losses

    def update_model(self, inputs, targets, target_scores = None, dw_force = None, kd_index = None):
        
        dw_cls = self.dw_k[-1 * torch.ones(targets.size()).long()]
        logits = self.forward(inputs)
        total_loss = self.criterion(logits, targets.long(), dw_cls)

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        return total_loss.detach(), logits, {}

    def validation(self, dataloader, model=None, task_in = None, task_metric='acc',  verbal = True, task_global=False,
                   **kwargs):
        # pass task to forward if task-awareness
        if model is None:
            model = self.model

        # This function doesn't distinguish tasks.
        batch_timer = Timer()
        acc = AverageMeter()
        batch_timer.tic()

        orig_mode = model.training
        model.eval()

        if len(self.cls_stats) != 0:
            cls_stats = torch.stack([self.cls_stats[label] for label in range(len(self.cls_stats))])
            # [n_cls, 768]    # will raise exception if label is not from 0->n_cls-1
            cls_stats = nn.functional.normalize(cls_stats, dim=1)
        else:
            cls_stats = None

        for i, sample in enumerate(dataloader):
            concepts = None
            if len(sample) == 3:
                (input, target, task) = sample
            else:   # contain concepts
                (input, target, concepts, task) = sample

            if self.debug_mode:
                print(f'batch{i}: \nlen: {len(target)} target:{(target.min(), target.max())} task:{(task.min(), task.max())}')
            if self.gpu:
                with torch.no_grad():
                    input = input.cuda()
                    target = target.cuda()
                    if concepts is not None:
                        concepts = concepts.cuda()  # [bs, 224, 224]        # do not use
            if task_in is None:
                # output = model.forward(input, task_id=task[0].item())[:, :self.valid_out_dim]
                with torch.no_grad():
                    output, features = model.forward(input, pen=True, **kwargs)

                # if self.debug_mode:
                #     print(f'batch{i}: \noutput:{output}')

                if len(self.cls_stats) == 0:
                    output = output[:, :self.valid_out_dim]
                else:
                    features = nn.functional.normalize(features, dim=1)
                    output = torch.einsum('bd,cd->bc', features, cls_stats)

                acc = accumulate_acc(output, target, task, acc, topk=(self.top_k,))
            else:
                mask = target >= task_in[0]
                mask_ind = mask.nonzero().view(-1) 
                input, target = input[mask_ind], target[mask_ind]

                mask = target <= task_in[-1]
                mask_ind = mask.nonzero().view(-1) 
                input, target = input[mask_ind], target[mask_ind]
                
                if len(target) > 1:
                    with torch.no_grad():
                        output, features = model.forward(input, pen=True, **kwargs)

                    if len(self.cls_stats) != 0:
                        features = nn.functional.normalize(features, dim=1)
                        output = torch.einsum('bd,cd->bc', features, cls_stats)

                    if task_global:
                        # output = model.forward(input, task_id=task[0].item())[:, :self.valid_out_dim]
                        output = output[:, :self.valid_out_dim]
                        acc = accumulate_acc(output, target, task, acc, topk=(self.top_k,))
                    else:
                        # output = model.forward(input, task_id=task[0].item())[:, task_in]
                        output = output[:, task_in]
                        acc = accumulate_acc(output, target-task_in[0], task, acc, topk=(self.top_k,))

        model.train(orig_mode)

        if verbal:
            self.log(' * Val Acc {acc.avg:.3f}, Total time {time:.2f}'
                    .format(acc=acc, time=batch_timer.toc()))
        return acc.avg

    ##########################################
    #             MODEL UTILS                #
    ##########################################

    # data weighting
    def data_weighting(self, dataset, num_seen=None):
        if self.concept_weight:
            self.label_concepts = np.array(dataset.get_concepts())  # [n_cls * [list of concepts: e.g., 1, 10]]
            self.num_concepts = dataset.num_concepts

        self.dw_k = torch.tensor(np.ones(self.valid_out_dim + 1, dtype=np.float32))
        # if hasattr(dataset, 'return_concepts') and dataset.return_concepts:
        if hasattr(dataset, 'target_sample_info') and dataset.target_sample_info is not None and self.config['mode'] == 'continual':
            concepts = dataset.get_concepts()   # [n_cls * [list of concepts: e.g., 1, 10]]
            target_concept = self.target_concept_id 
            for cls_id in range(self.valid_out_dim): 
                if target_concept in concepts[cls_id]: 
                    self.dw_k[cls_id] = 2
            
        # cuda
        if self.cuda:
            self.dw_k = self.dw_k.cuda()

    def save_model(self, filename):
        model_state = self.model.state_dict()
        for key in model_state.keys():  # Always save it to cpu
            model_state[key] = model_state[key].cpu()
        self.log('=> Saving class model to:', filename)
        torch.save(model_state, filename + 'class.pth')
        self.log('=> Save Done')

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
        self.log(f'=> Load Done from {filename}')
        # self.log(f'=> Load Done with params {list(state_dict.keys())}')

        if freeze:
            self.log('=> Freeze backbone')     # on CFST
            for k, p in self.model.named_parameters():
                if 'last' not in k:
                    p.requires_grad = False

        if self.gpu:
            self.model = self.model.cuda()
        self.model.eval()

    def load_model_other(self, filename, model):
        model.load_state_dict(torch.load(filename + 'class.pth'))
        if self.gpu:
            model = model.cuda()
        return model.eval()

    # sets model optimizers
    def init_optimizer(self):
        lr = self.config['lr']
        if type(lr) is list:
            lr = lr[-1]
        print(f'init_optimizer: lr: {lr}')
        print(f'num parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}')
        # parse optimizer args
        optimizer_arg = {'params':self.model.parameters(),
                         'lr':lr,
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
        
        # create schedules
        if self.schedule_type == 'cosine':
            self.scheduler = CosineSchedule(self.optimizer, K=self.schedule[-1])
        elif self.schedule_type == 'decay':
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.schedule, gamma=0.1)
        else:       # no change
            self.scheduler = type('empty_scheduler', (), {})()
            self.scheduler.step = lambda x=0: None       # empty object scheduler with empty step() func.

    def create_model(self):
        cfg = self.config

        # Define the backbone (MLP, LeNet, VGG, ResNet ... etc) of model
        model = models.__dict__[cfg['model_type']].__dict__[cfg['model_name']](out_dim=self.out_dim)

        return model

    def print_model(self):
        self.log(self.model)
        self.log('#parameter of model:', self.count_parameter())
    
    def reset_model(self):
        self.model.apply(weight_reset)

    def forward(self, x):
        return self.model.forward(x)[:, :self.valid_out_dim]

    def predict(self, inputs):
        self.model.eval()
        out = self.forward(inputs)
        return out
    
    def add_valid_output_dim(self, dim=0):
        # This function is kind of ad-hoc, but it is the simplest way to support incremental class learning
        self.log('Incremental class: Old valid output dimension:', self.valid_out_dim)
        self.valid_out_dim += dim
        self.log('Incremental class: New Valid output dimension:', self.valid_out_dim)
        return self.valid_out_dim

    def count_parameter(self):
        return sum(p.numel() for p in self.model.parameters())   

    def count_memory(self, dataset_size):
        return self.count_parameter() + self.memory_size * dataset_size[0]*dataset_size[1]*dataset_size[2]

    def cuda(self):
        torch.cuda.set_device(self.config['gpuid'][0])
        self.model = self.model.cuda()
        self.criterion_fn = self.criterion_fn.cuda()
        # Multi-GPU
        if len(self.config['gpuid']) > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.config['gpuid'], output_device=self.config['gpuid'][0])
        return self

    def _get_device(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.log("Running on:", device)
        return device

    def pre_steps(self):
        pass

class FinetunePlus(NormalNN):

    def __init__(self, learner_config):
        super(FinetunePlus, self).__init__(learner_config)

    def update_model(self, inputs, targets, target_KD = None):

        # get output
        logits = self.forward(inputs)

        # standard ce
        logits[:,:self.last_valid_out_dim] = -float('inf')
        dw_cls = self.dw_k[-1 * torch.ones(targets.size()).long()]
        total_loss = self.criterion(logits, targets.long(), dw_cls)

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        return total_loss.detach(), logits

def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()

def accumulate_acc(output, target, task, meter, topk):
    meter.update(accuracy(output, target, topk), len(target))
    return meter