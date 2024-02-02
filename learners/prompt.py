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
from .default import NormalNN, weight_reset, accumulate_acc
import copy
import torchvision
from utils.schedulers import CosineSchedule
from torch.autograd import Variable, Function

from .pmo_utils import Pool, available_setting


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
            logits[:,:self.last_valid_out_dim] = -float('inf')
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
    def init_optimizer(self):

        # parse optimizer args
        # Multi-GPU
        if len(self.config['gpuid']) > 1:
            if self.config['mode'] in ['sys', 'pro', 'sub', 'non', 'noc']:
                # if fewshot testing self.config['mode'], only learn classifier: model.last
                params_to_opt = list(self.model.module.last.parameters())
            else:
                params_to_opt = list(self.model.module.prompt.parameters()) + list(self.model.module.last.parameters())
        else:
            if self.config['mode'] in ['sys', 'pro', 'sub', 'non', 'noc']:
                params_to_opt = list(self.model.last.parameters())
            else:
                params_to_opt = list(self.model.prompt.parameters()) + list(self.model.last.parameters())
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
        
        # create schedules
        if self.schedule_type == 'cosine':
            self.scheduler = CosineSchedule(self.optimizer, K=self.schedule[-1])
        elif self.schedule_type == 'decay':
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.schedule, gamma=0.1)

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


# Our PMO (Pool & Multi-Objective)
class PMOPrompt(Prompt):
    def __init__(self, learner_config):
        super(PMOPrompt, self).__init__(learner_config)
        self.pool_size = self.prompt_param[1][3]        # =0 if do not enable pool hv loss
        self.pool = Pool(self.pool_size, self.seed)

        self.train_dataset = None

    def create_model(self):
        cfg = self.config
        model = models.__dict__[cfg['model_type']].__dict__[cfg['model_name']](out_dim=self.out_dim, prompt_flag = 'pmo',prompt_param=self.prompt_param)
        return model

    def learn_batch(self, train_loader, train_dataset, model_save_dir, val_loader=None):
        """Difference:
        Update_pool from dataset task.
        Cal model.(module).prompt.bind_pool
        """
        self.train_dataset = train_dataset

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
            # update pool
            if self.pool_size > 0:
                train_dataset.update_pool(self.pool_size, train_dataset.t, self.pool)
                # flash pool to prompt, if prompt needs it.
                try:
                    self.model.module.prompt.bind_pool(self.pool)
                except:
                    self.model.prompt.bind_pool(self.pool)

            if self.debug_mode:
                print(f'Pool shape: {len(self.pool.length())}.')

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
        If len(pool) > 0 (use hv loss):
            sample 1 task from current task, 1 task from a selected old task, 2 mixed tasks,
            cal mo loss matrix and hv loss.
        """

        # logits
        logits, prompt_loss = self.model(inputs, train=True)
        logits = logits[:,:self.valid_out_dim]

        if self.debug_mode:
            print(f'logits: {logits}')

            print(f'prompt_loss: {prompt_loss}')

        # ce with heuristic
        if self.memory_size == 0:       # replay-based will have old tasks which may cause inf loss
            logits[:,:self.last_valid_out_dim] = -float('inf')
        dw_cls = self.dw_k[-1 * torch.ones(targets.size()).long()]
        total_loss = self.criterion(logits, targets.long(), dw_cls)

        if self.debug_mode:
            print(f'classification loss: {total_loss} and {total_loss.grad_fn}')

        # ce loss
        total_loss = total_loss + prompt_loss.sum()

        # hv loss
        hv_loss = 0
        if self.train_dataset.t > 0:        # start from the second task
            # hv loss calculation without affecting RNG state
            state = np.random.get_state()
            np.random.seed(self.seed)
            mo_matrix = self.obtain_mo_matrix([self.train_dataset.t])



            np.random.set_state(state)

        total_loss = total_loss + hv_loss

        if self.debug_mode:
            print(f'hv loss: {hv_loss}')

        # step
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return total_loss.detach(), logits

    def obtain_mo_matrix(self, must_include_clusters=None):
        """"""
        '''multiple mo sampling'''
        num_imgs_clusters = [np.array([cls[1] for cls in classes]) for classes in self.pool.current_classes()]
        ncc_losses_mo = dict()  # f'p{task_idx}_o{obj_idx}'

        if len(num_imgs_clusters) == 0:
            return

        '''choose n_obj tasks as targets'''
        if must_include_clusters is None:
            must_include_clusters = []
        must_include_clusters = np.sort(must_include_clusters)
        if len(must_include_clusters) < self.config['n_obj']:
            # sample extra tasks
            num_choices = self.config['n_obj'] - len(must_include_clusters)
            task_list = [cluster[0] for cluster in self.pool.current_clusters()]
            candidates = list(filter(lambda x: x not in must_include_clusters, task_list))
            selected_tasks = np.random.choice(candidates, num_choices, replace=False)
            must_include_clusters = np.sort(np.concatenate([must_include_clusters, selected_tasks]))

        if self.debug_mode:
            print(f'must_include_clusters: {must_include_clusters}')

        '''choose fewshot setting for selected tasks'''
        n_way, n_shot, n_query = available_setting(num_imgs_clusters, self.config['mo_task_type'],
                                                   min_available_clusters=self.config['n_obj'],
                                                   must_include_clusters=must_include_clusters)
        if n_way == -1:  # not enough samples
            print(f"==>> pool has not enough samples. skip MO")
            break

        available_cluster_idxs = check_available(num_imgs_clusters, n_way, n_shot, n_query)

        selected_cluster_idxs = sorted(np.random.choice(
            available_cluster_idxs, args['train.n_obj'], replace=False))

        torch_tasks = []
        '''sample pure tasks from clusters in selected_cluster_idxs'''
        for cluster_idx in selected_cluster_idxs:
            pure_task = pool.episodic_sample(cluster_idx, n_way, n_shot, n_query, d=device)
            torch_tasks.append(pure_task)

            numpy_sample = task_to_device(pure_task, 'numpy')
            numpy_samples.append(numpy_sample)
            sample_domain_names.append(f'cluster_{cluster_idx}')

        '''sample mix tasks by mixer'''
        for mix_id in range(args['train.n_mix']):
            numpy_mix_task, _ = mixer.mix(
                task_list=[pool.episodic_sample(idx, n_way, n_shot, n_query)
                           for idx in selected_cluster_idxs],
                mix_id=mix_id
            )
            torch_tasks.append(task_to_device(numpy_mix_task, device))

            numpy_samples.append(numpy_mix_task)
            sample_domain_names.append(f'mix_{selected_cluster_idxs}')

        '''obtain ncc loss multi-obj matrix'''
        for task_idx, task in enumerate(torch_tasks):

            '''use url with pa'''
            model = url
            with torch.no_grad():
                task_features = pmo.embed(
                    torch.cat([task['context_images'], task['target_images']]))
                context_features = model.embed(task['context_images'])
                context_labels = task['context_labels']

            selection, selection_info = pmo.selector(
                task_features, gumbel=False, hard=False)
            vartheta = [torch.mm(selection, pmo.pas.detach().flatten(1)).view(512, 512, 1, 1)]
            # detach from pas to only train clustering and not train pas
            selected_features = apply_selection(context_features, vartheta)
            inner_loss, _, _ = prototype_loss(
                selected_features, context_labels, selected_features, context_labels,
                distance=args['test.distance'])
            inner_lr = 1
            grad = torch.autograd.grad(inner_loss, vartheta, create_graph=True)
            selection_params = list(map(lambda p: p[1] - inner_lr * p[0], zip(grad, vartheta)))
            # selection_params = pa_iterator(context_features, context_labels, max_iter=max_iter, lr=inner_lr,
            #                       distance=args['test.distance'],
            #                       vartheta_init=[vartheta, torch.optim.Adadelta(vartheta, lr=inner_lr)],
            #                       create_graph=True)
            '''forward to get mo matrix'''
            for obj_idx in range(len(selected_cluster_idxs)):  # 2
                obj_context_images = torch_tasks[obj_idx]['context_images']
                obj_target_images = torch_tasks[obj_idx]['target_images']
                obj_context_labels = torch_tasks[obj_idx]['context_labels']
                obj_target_labels = torch_tasks[obj_idx]['target_labels']

                obj_context_features = apply_selection(model.embed(obj_context_images),
                                                       selection_params)
                obj_target_features = apply_selection(model.embed(obj_target_images),
                                                      selection_params)

                obj_loss, stats_dict, _ = prototype_loss(
                    obj_context_features, obj_context_labels, obj_target_features, obj_target_labels,
                    distance=args['test.distance'])
                if f'p{task_idx}_o{obj_idx}' in ncc_losses_mo.keys():  # collect n_mo data
                    ncc_losses_mo[f'p{task_idx}_o{obj_idx}'].append(obj_loss)
                else:
                    ncc_losses_mo[f'p{task_idx}_o{obj_idx}'] = [obj_loss]

                epoch_log['mo_df'] = pd.concat([
                    epoch_log['mo_df'], pd.DataFrame.from_records([
                        {'Tag': 'loss', 'Pop_id': task_idx, 'Obj_id': obj_idx, 'Inner_id': 0,
                         'Value': stats_dict['loss']},
                        {'Tag': 'acc', 'Pop_id': task_idx, 'Obj_id': obj_idx, 'Inner_id': 0,
                         'Value': stats_dict['acc']}])])

    def obtain_hv_loss(self, mo_matrix): # todo: put to pmo_utils.py
        """"""
        '''calculate HV loss for n_mo matrix'''
        hv_loss = 0
        if len(ncc_losses_mo) > 0:
            ref = args['train.ref']
            ncc_losses_multi_obj = torch.stack([torch.stack([
                torch.mean(torch.stack(ncc_losses_mo[f'p{task_idx}_o{obj_idx}']))
                for task_idx in range(args['train.n_obj'] + args['train.n_mix'])
            ]) for obj_idx in range(args['train.n_obj'])])  # [2, 4]
            hv_loss = cal_hv_loss(ncc_losses_multi_obj, ref)
            epoch_log['scaler_df'] = pd.concat([
                epoch_log['scaler_df'], pd.DataFrame.from_records([{
                    'Tag': 'loss/hv_loss', 'Idx': 0, 'Value': hv_loss.item()}])])

    def obtain_entangle_loss(self):
        """"""
        '''entanglement within one cluster'''
        ncc_losses_et = []
        for cluster_idx, cluster in enumerate(pool.clusters):
            if len(cluster) > 0:
                for et_train_idx in range(args['train.n_et_cond']):
                    '''check pool has enough samples and generate 1 setting'''
                    n_way, n_shot, n_query = available_setting(
                        [num_imgs_clusters[cluster_idx]], args['train.mo_task_type'])
                    if n_way == -1:  # not enough samples
                        print(f"==>> pool has not enough samples. skip MO")
                        break

                    '''sample a task to condition model'''
                    et_task = pool.episodic_sample(cluster_idx, n_way, n_shot, n_query, d=device)

                    '''use url with pa'''
                    model = url
                    with torch.no_grad():
                        task_features = pmo.embed(
                            torch.cat([et_task['context_images'], et_task['target_images']]))
                        context_features = model.embed(et_task['context_images'])
                        context_labels = et_task['context_labels']

                    selection, selection_info = pmo.selector(
                        task_features, gumbel=False, hard=False)
                    vartheta = [torch.mm(selection, pmo.pas.detach().flatten(1)).view(512, 512, 1, 1)]
                    # detach from pas to only train clustering and not train pas
                    selected_features = apply_selection(context_features, vartheta)
                    inner_loss, _, _ = prototype_loss(
                        selected_features, context_labels, selected_features, context_labels,
                        distance=args['test.distance'])
                    inner_lr = 1
                    grad = torch.autograd.grad(inner_loss, vartheta, create_graph=True)
                    selection_params = list(map(lambda p: p[1] - inner_lr * p[0], zip(grad, vartheta)))

                    '''sample train.n_et_update tasks for ncc losses'''
                    for et_update_idx in range(args['train.n_et_update']):
                        n_way, n_shot, n_query = available_setting(
                            [num_imgs_clusters[cluster_idx]], args['train.mo_task_type'])
                        up_task = pool.episodic_sample(cluster_idx, n_way, n_shot, n_query, d=device)

                        obj_context_images = up_task['context_images']
                        obj_target_images = up_task['target_images']
                        obj_context_labels = up_task['context_labels']
                        obj_target_labels = up_task['target_labels']

                        obj_context_features = apply_selection(model.embed(obj_context_images),
                                                               selection_params)
                        obj_target_features = apply_selection(model.embed(obj_target_images),
                                                              selection_params)

                        obj_loss, stats_dict, _ = prototype_loss(
                            obj_context_features, obj_context_labels, obj_target_features, obj_target_labels,
                            distance=args['test.distance'])
                        ncc_losses_et.append(obj_loss)
                        epoch_log['scaler_df'] = pd.concat([
                            epoch_log['scaler_df'], pd.DataFrame.from_records([
                                {'Tag': 'et/loss', 'Idx': 0, 'Value': stats_dict['loss']},
                                {'Tag': 'et/acc', 'Idx': 0, 'Value': stats_dict['acc']}])])

        '''average ncc_losses_et'''
        if len(ncc_losses_et) > 0:
            et_loss = torch.mean(torch.stack(ncc_losses_et))
            scaler = et_loss.item()
        else:
            et_loss = 0
            scaler = 0
        epoch_log['scaler_df'] = pd.concat([
            epoch_log['scaler_df'], pd.DataFrame.from_records([{
                'Tag': 'loss/et_loss', 'Idx': 0, 'Value': scaler}])])


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