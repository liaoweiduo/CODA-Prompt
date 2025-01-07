import copy
import yaml
import json
import os
import pickle
from collections import OrderedDict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn

from typing import List, Dict, Any, Optional
from torch.utils.tensorboard import SummaryWriter

from learners.pmo_utils import Pool, draw_heatmap, draw_objs, cal_hv, cal_min_crowding_distance


class Debugger:
    def __init__(self, level='DEBUG', args=None, exp_path=None, name='[Default]'):
        """args need to be dict"""
        assert (args is not None or exp_path is not None
                ), "args and exp_path should not be both None."

        self.levels = ['DEBUG', 'INFO']
        self.level = self.levels.index(level)   # 0 or 1
        self.args = args
        self.exp_path = exp_path
        if self.exp_path is not None:
            self.load_args(exp_path=self.exp_path)
        else:
            self.exp_path = self.args['log_dir']
        args = self.args
        self.name = name

        for k, v in self.args.items():       # list to str
            if type(v) is list:
                self.args[k] = str(v)

        self.dataset = self.args['dataset']         # CIFAR100, CGQA,...

        self.storage = {}

    def collect_results(self, max_task=-1):
        # collect results
        self.storage['results'] = {}
        self.collect_AA_CA_FF(max_task)
        self.collect_CFST()

    def default_columns(self):
        # default params
        output_args = [
            'max_task', 'lr', 'prompt_param', 'larger_prompt_lr']
        if self.args['lr_decreace_ratio'] != 1.0:
            output_args.append('lr_decrease_ratio')

        if self.args['learner_name'] == 'SLOTPrompt':
            # slot params
            output_args.extend(['n_slots', 'n_iters', 'slot_temp', 's2p_mode'])
            if self.args.get('use_intra_consistency_reg', False):
                output_args.extend(['intra_consistency_reg_coeff'])
            if self.args.get('use_slot_ortho_reg', False):
                output_args.extend(['slot_ortho_reg_mode', 'slot_ortho_reg_coeff'])

        # prompt param
        if self.args.get('use_weight_reg', False):
            output_args.extend(['weight_reg_mode', 'weight_reg_coeff'])
        if self.args.get('use_selection_onehot_reg', False):
            output_args.extend(['selection_onehot_reg_mode', 'selection_onehot_reg_coeff'])
        if self.args.get('use_selection_slot_similar_reg', False):
            output_args.extend(['selection_slot_similar_reg_mode', 'selection_slot_similar_reg_coeff'])
        if self.args.get('use_prompt_concept_alignment_reg', False):
            output_args.extend(['prompt_concept_alignment_reg_coeff'])
        if self.args.get('concept_weight', False):
            output_args.extend(['concept_similar_reg_mode', 'concept_similar_reg_coeff'])
        if self.args.get('use_old_samples_for_reg', False):
            output_args.extend(['use_old_samples_for_reg'])
        if self.args.get('use_slot_logit_similar_reg', False):
            output_args.extend(['slot_logit_similar_reg_mode', 'slot_logit_similar_reg_coeff'])

        columns = []
        columns.extend(['AA', 'l-AA', 'CA', 'l-CA', 'FF', 'l-FF'])
        if self.dataset == 'CGQA':
            columns.extend(['sys', 'pro', 'sub', 'Hn', 'non', 'noc', 'Hr', 'Ha'])
        else:
            columns.extend(['sys', 'pro', 'Hn', 'non', 'noc', 'Hr', 'Ha'])   # no sub

        return output_args, columns

    def generate_df(self, column_info=None):
        """args and storage to value"""
        # form dict
        if column_info is None:
            column_info = self.default_columns()
        output_args, columns = column_info

        row = OrderedDict()
        row['name'] = self.name
        row['exp_path'] = self.exp_path

        for output_arg in output_args:
            row[output_arg] = self.args.get(output_arg, '-')
        for res in columns:
            target = self.storage['results'].get(res, None)
            if target:
                row[res] = target['Mean']
                row[f'{res}(str)'] = F"{target['Mean']:.2f}$\pm${target['CI95']:.2f}({target['Std']:.2f})"

        df = pd.Series(data=row).to_frame().T

        return df

    def collect_AA_CA_FF(self, max_task=-1):
        # pt
        file = os.path.join(self.exp_path, 'results-acc', 'pt.yaml')
        try:
            data_yaml = yaml.load(open(file, 'r'), Loader=yaml.Loader)
            data = np.array(data_yaml['history'])  # [n_tsk, n_tsk, n_run]
        except:
            if self.check_level('INFO'):
                print(f'File not find: {file}.')
            return
            # data = np.zeros((2,2,2))
        # example: data[:,:,0]
        # [[94.2 84.4 78.7 69.3 69.  66.5 62.  60.6 49.4 44.1]
        #  [ 0.  80.7 74.6 70.1 65.2 63.2 57.1 57.2 51.9 44.8]
        #  [ 0.   0.  79.1 74.2 74.2 66.7 61.9 61.3 55.9 48.3]
        #  [ 0.   0.   0.  75.8 72.4 62.4 59.7 57.5 55.3 50.9]
        #  [ 0.   0.   0.   0.  58.6 54.1 53.  48.6 45.  40.1]
        #  [ 0.   0.   0.   0.   0.  70.  66.3 64.9 61.  56. ]
        #  [ 0.   0.   0.   0.   0.   0.  74.5 67.9 66.5 64.4]
        #  [ 0.   0.   0.   0.   0.   0.   0.  62.3 55.5 51.7]
        #  [ 0.   0.   0.   0.   0.   0.   0.   0.  78.2 75.4]
        #  [ 0.   0.   0.   0.   0.   0.   0.   0.   0.  55. ]]

        if max_task == -1:
            max_task = data.shape[1]
        self.args['max_task'] = max_task        # visually change the number of finished tasks

        if max_task > data.shape[1]:    # haven't finish yet
            return

        AA = data[:max_task, max_task - 1].mean(axis=0)    # [n_tsk, n_run]
        data_cu = np.array([data[:, i].sum(axis=0) / (i + 1) for i in range(max_task)])  # [n_tsk, n_run]
        CA = data_cu.mean(axis=0)
        data_ff = np.array([data[i, i] - data[i, -1] for i in range(max_task - 1)])  # [n_tsk, n_run]
        FF = data_ff.mean(axis=0)
        self.storage['results']['AA'] = {
            'Details': AA, 'Mean': AA.mean(), 'Std': AA.std(), 'CI95': 1.96 * (AA.std() / np.sqrt(len(AA)))}
        self.storage['results']['CA'] = {
            'Details': CA, 'Mean': CA.mean(), 'Std': CA.std(), 'CI95': 1.96 * (CA.std() / np.sqrt(len(CA)))}
        self.storage['results']['FF'] = {
            'Details': FF, 'Mean': FF.mean(), 'Std': FF.std(), 'CI95': 1.96 * (FF.std() / np.sqrt(len(FF)))}

        # local pt
        try:
            file = os.path.join(self.exp_path, 'results-acc', 'pt-local.yaml')
            data_yaml = yaml.load(open(file, 'r'), Loader=yaml.Loader)
            data = np.array(data_yaml['history'])  # [n_tsk, n_tsk, n_run]
        except:
            if self.check_level('INFO'):
                print(f'File not find: {file}.')
            return
            # data = np.zeros((2,2,2))
        AA = data[:, max_task - 1].mean(axis=0)    # [n_tsk, n_run]
        data_cu = np.array([data[:, i].sum(axis=0) / (i + 1) for i in range(max_task)])  # [10, n_run]
        CA = data_cu.mean(axis=0)
        data_ff = np.array([data[i, i] - data[i, -1] for i in range(max_task - 1)])  # [10, n_run]
        FF = data_ff.mean(axis=0)
        self.storage['results']['l-AA'] = {
            'Details': AA, 'Mean': AA.mean(), 'Std': AA.std(), 'CI95': 1.96 * (AA.std() / np.sqrt(len(AA)))}
        self.storage['results']['l-CA'] = {
            'Details': CA, 'Mean': CA.mean(), 'Std': CA.std(), 'CI95': 1.96 * (CA.std() / np.sqrt(len(CA)))}
        self.storage['results']['l-FF'] = {
            'Details': FF, 'Mean': FF.mean(), 'Std': FF.std(), 'CI95': 1.96 * (FF.std() / np.sqrt(len(FF)))}

    def collect_CFST(self):
        dataset = self.dataset
        mean_datas = {}
        for target in ['sys', 'pro', 'sub', 'non', 'noc'] if dataset == 'CGQA' else ['sys', 'pro', 'non', 'noc']:
            file = os.path.join(self.exp_path, 'results-acc', f'global-{target}.yaml')
            try:
                data_yaml = yaml.load(open(file, 'r'), Loader=yaml.Loader)
                data = np.array(data_yaml['mean'])  # [50]
            except:
                if self.check_level('INFO'):
                    print(f'File not find: {file}.')
                return
                # data = np.zeros((2))

            mean_data = data.mean()
            mean_datas[target] = mean_data
            std_data = data.std()
            ci95_data = 1.96 * (std_data / np.sqrt(len(data)))

            # print(f'{target}: {mean_data:.2f}$\pm${ci95_data:.2f}, std:{std_data:.2f}')
            self.storage['results'][target] = {'Details': data, 'Mean': mean_data, 'Std': std_data, 'CI95': ci95_data}

        if dataset == 'CGQA':
            hn = 3 / (1 / mean_datas['sys'] + 1 / mean_datas['pro'] + 1 / mean_datas['sub'])
            hr = 2 / (1 / mean_datas['non'] + 1 / mean_datas['noc'])
            ha = 5 / (1 / mean_datas['sys'] + 1 / mean_datas['pro'] + 1 / mean_datas['sub'] + 1 / mean_datas[
                'non'] + 1 / mean_datas['noc'])
        else:
            hn = 2 / (1 / mean_datas['sys'] + 1 / mean_datas['pro'])
            hr = 2 / (1 / mean_datas['non'] + 1 / mean_datas['noc'])
            ha = 4 / (1 / mean_datas['sys'] + 1 / mean_datas['pro'] + 1 / mean_datas['non'] + 1 / mean_datas['noc'])
        # print(f"Hn: {hn:.2f}; Hr: {hr:.2f}; Ha: {ha:.2f}")
        self.storage['results']['Hn'] = {'Details': [hn], 'Mean': hn, 'Std': 0, 'CI95': 0}
        self.storage['results']['Hr'] = {'Details': [hr], 'Mean': hr, 'Std': 0, 'CI95': 0}
        self.storage['results']['Ha'] = {'Details': [ha], 'Mean': ha, 'Std': 0, 'CI95': 0}

    def load_args(self, exp_path):
        print(f'Load args from {exp_path}.')
        self.exp_path = exp_path
        path = os.path.join(exp_path, 'args.yaml')
        args = yaml.load(open(path, 'r'), Loader=yaml.Loader)
        self.args = args

    def check_level(self, level):
        thres = self.levels.index(level)
        return thres >= self.level

    def print_prototype_change(self, model: nn.Module, i, writer: Optional[SummaryWriter] = None):
        """

        Args:
            model: target model contain prototypes
            writer: None or obj: writer
            i: iter index

        Returns:

        """
        level = self.levels.index('DEBUG')
        if level < self.level:
            return

        proto = model.selector.prototypes.detach().cpu().numpy()
        if 'proto' in self.storage:
            old_proto = self.storage['proto']
        else:
            old_proto = 0
        dif = np.linalg.norm((proto - old_proto).ravel(), 2)  # l2 distance
        self.storage['proto'] = proto

        print(f'proto diff (l2) is {dif}.\nmov_avg_alpha is {model.selector.mov_avg_alpha.item()}.')
        if writer is not None:
            writer.add_scalar('params/cluster-centers-dif', dif, i + 1)
            writer.add_scalar('params/mov_avg_alpha', model.selector.mov_avg_alpha.item(), i + 1)

    def print_grad(self, model: nn.Module, key=None, prefix=''):
        """

        Args:
            model: target model
            key: for parameter name and also for storage name for diff
            prefix: print prefix

        Returns:

        """
        level = self.levels.index('DEBUG')
        if level < self.level:
            return

        vs = []
        with torch.no_grad():
            for k, v in model.named_parameters():
                if (key is None) or (key in k):
                    if v.grad is not None:
                        vs.append(v.grad.flatten())
            vs = torch.cat(vs).detach().cpu().numpy()
            if key in self.storage:
                dif = vs - self.storage[key]
            else:
                dif = vs
            self.storage[key] = vs
            dif = np.abs(dif)
            print(f'{prefix}mean abs grad diff for {key} is {np.mean(dif)}.')

    def save_log(self, log, log_name):
        """

        Args:
            log: dict
            log_name: path to save log
        """
        print('=> Saving log to:', log_name)
        with open(log_name, 'wb') as f:
            pickle.dump(log, f)
        print('=> Save Done')

    def write_pool(self, pool: Pool, i, writer: Optional[SummaryWriter] = None, prefix='pool'):
        """

        Args:
            pool: pool after call buffer2cluster()
            i: iter index
            writer: None or obj: writer
            prefix:

        Returns:

        """
        level = self.levels.index('INFO')
        if level < self.level:
            return

        print(f'iter {i}: {prefix} num_cls info: '
              f'{[f"{idx}[{len(sim)}]" for idx, sim in enumerate(pool.current_similarities())]}.')

        if writer is not None:
            '''all images and img_sims and class_sims '''
            images = pool.current_images(single_image=True)
            for cluster_id, cluster in enumerate(images):
                if len(cluster) > 0:
                    writer.add_image(f"{prefix}-img/{cluster_id}", cluster, i + 1)
            similarities = pool.current_similarities(image_wise=True)
            class_similarities = pool.current_similarities()
            for cluster_id, (img_sim, cls_sim) in enumerate(zip(similarities, class_similarities)):
                if len(img_sim) > 0:
                    # img_sim [num_cls * [num_img, 8]]; cls_sim [num_cls * [8]]
                    sim = np.concatenate([
                        np.concatenate([img_sim[cls_idx],
                                        *[cls_sim[cls_idx][np.newaxis, :]] * max(10, len(img_sim[cls_idx]) // 2)])
                        for cls_idx in range(len(img_sim))
                    ])
                    figure = draw_heatmap(sim, verbose=False)
                    writer.add_figure(f"{prefix}-img-sim/{cluster_id}", figure, i + 1)

    def write_scaler(self, df, key, i, writer: Optional[SummaryWriter] = None, prefix='', inner=True):
        """

        Args:
            df: [Tag, Idx, Value]
            key: in Tag
            i:
            writer:
            prefix:
            inner: True, write image based on Idx

        Returns: last Idx's avg_value or -1

        """
        level = self.levels.index('INFO')
        if level < self.level:
            return

        if key == 'all':
            keys = list(set(df.Tag))
        else:
            keys = [key]

        values = []
        for key in keys:
            t_df = df[df.Tag == key]
            value = -1
            for idx in sorted(set(t_df.Idx)):
                value = t_df[t_df.Idx == idx].Value.mean()
                value = np.nan_to_num(value)
                if not inner:
                    writer.add_scalar(f'{prefix}{key}/{idx}', value, i + 1)

            print(f'{prefix}{key}: {value:.5f}.')
            values.append(value)

            if inner:
                self.write_inner(df, key=key, i=i, writer=writer, prefix=prefix)

        return keys, values

    def write_inner(self, df, key, i, writer: Optional[SummaryWriter] = None, prefix=''):
        """

        Args:
            df: [Tag, Idx, Value]
            key: in Tag
            i:
            writer:
            prefix:

        Returns:

        """
        level = self.levels.index('INFO')
        if level < self.level:
            return

        t_df = df[df.Tag == key]

        fig, ax = plt.subplots()
        ax.grid(True)
        sns.lineplot(t_df, x='Idx', y='Value', ax=ax)

        writer.add_figure(f"{prefix}{key}", fig, i + 1)

    def write_hv(self, mo_dict, i, ref=0, writer: Optional[SummaryWriter] = None, target='acc', norm=False,
                 prefix='hv'):
        """

        Args:
            mo_dict: dataframe ['Tag', 'Pop_id', 'Obj_id', 'Inner_id', 'Value']
            ref: ref for cal hv
            writer:
            target: also for mo_dict's Tag selector.
            norm: either to do normalization for n_inner > 1
            prefix:
            i: indicate x axis

        Returns:

        """
        level = self.levels.index('INFO')
        if level < self.level:
            return

        t_df = mo_dict[mo_dict.Tag == target]
        n_pop = len(set(t_df.Pop_id))
        n_inner = len(set(t_df.Inner_id))
        n_obj = len(set(t_df.Obj_id))
        objs = np.array([[[
            t_df[(t_df.Pop_id == pop_idx) & (t_df.Obj_id == obj_idx) & (
                        t_df.Inner_id == inner_idx)].Value.mean()
            for pop_idx in range(n_pop)] for obj_idx in range(n_obj)] for inner_idx in range(n_inner)
        ])  # [n_inner, n_obj, n_pop]
        objs = np.nan_to_num(objs)

        '''for normalization'''
        if norm:
            min_objs = np.min(np.min(objs, axis=2, keepdims=True), axis=0, keepdims=True) - 1e-10
            max_objs = np.max(np.max(objs, axis=2, keepdims=True), axis=0, keepdims=True)
            objs = (objs - min_objs) / (max_objs - min_objs)

        '''cal hv for each inner mo'''
        hv = -1
        if ref == 'relative':
            ref = np.mean(objs[0], axis=-1).tolist()  # [n_obj]   to be list
        for inner_step in range(n_inner):
            hv = cal_hv(objs[inner_step], ref, target='min' if target == 'loss' else 'max')
            writer.add_scalar(f'{prefix}_details/{target}/{i+1}', hv, inner_step + 1)
        writer.add_scalar(f'{prefix}/{target}', hv, i + 1)

        print(f"==>> {prefix}: {target} {hv:.3f}.")

    def write_avg_span(self, mo_dict, i, writer: Optional[SummaryWriter] = None, target='acc', norm=False,
                       prefix='avg_span'):
        """
        E_i(max(f_i) - min(f_i))
        Args:
            mo_dict: dataframe ['Tag', 'Pop_id', 'Obj_id', 'Inner_id', 'Value']
            writer:
            target: also for mo_dict's Tag selector.
            norm: either to norm obj space (use for n_inner > 1)
            prefix:
            i: indicate x axis

        Returns:

        """
        level = self.levels.index('INFO')
        if level < self.level:
            return

        t_df = mo_dict[mo_dict.Tag == target]
        n_pop = len(set(t_df.Pop_id))
        n_inner = len(set(t_df.Inner_id))
        n_obj = len(set(t_df.Obj_id))
        objs = np.array([[[
            t_df[(t_df.Pop_id == pop_idx) & (t_df.Obj_id == obj_idx) & (
                        t_df.Inner_id == inner_idx)].Value.mean()
            for pop_idx in range(n_pop)] for obj_idx in range(n_obj)] for inner_idx in range(n_inner)
        ])  # [n_inner, n_obj, n_pop]
        objs = np.nan_to_num(objs)

        '''for normalization'''
        if norm:
            min_objs = np.min(np.min(objs, axis=2, keepdims=True), axis=0, keepdims=True) - 1e-10
            max_objs = np.max(np.max(objs, axis=2, keepdims=True), axis=0, keepdims=True)
            objs = (objs - min_objs) / (max_objs - min_objs)

        '''cal avg span for each inner mo'''
        avg_span = -1
        for inner_step in range(n_inner):
            avg_span = np.mean(
                [np.max(objs[inner_step][obj_idx]) - np.min(objs[inner_step][obj_idx]) for obj_idx in
                 range(n_obj)])
            writer.add_scalar(f'{prefix}_details/{target}/{i+1}', avg_span, inner_step + 1)
        writer.add_scalar(f'{prefix}/{target}', avg_span, i + 1)

        print(f"==>> {prefix}: {target} {avg_span:.5f}.")

        return avg_span

    def write_min_crowding_distance(self, mo_dict, i, writer: Optional[SummaryWriter] = None, target='acc', norm=False,
                                    prefix='min_cd'):
        """
        only for nd solutions. if min cd is inf, use avg_span instead.
        Args:
            mo_dict: dataframe ['Tag', 'Pop_id', 'Obj_id', 'Inner_id', 'Value']
            writer:
            target: also for mo_dict's Tag selector.
            norm: either to norm obj space (use for n_inner > 1)
            prefix:
            i: indicate x axis

        Returns:

        """
        level = self.levels.index('INFO')
        if level < self.level:
            return

        t_df = mo_dict[mo_dict.Tag == target]
        n_pop = len(set(t_df.Pop_id))
        n_inner = len(set(t_df.Inner_id))
        n_obj = len(set(t_df.Obj_id))
        objs = np.array([[[
            t_df[(t_df.Pop_id == pop_idx) & (t_df.Obj_id == obj_idx) & (
                        t_df.Inner_id == inner_idx)].Value.mean()
            for pop_idx in range(n_pop)] for obj_idx in range(n_obj)] for inner_idx in range(n_inner)
        ])  # [n_inner, n_obj, n_pop]
        objs = np.nan_to_num(objs)

        '''for normalization'''
        if norm:
            min_objs = np.min(np.min(objs, axis=2, keepdims=True), axis=0, keepdims=True) - 1e-10
            max_objs = np.max(np.max(objs, axis=2, keepdims=True), axis=0, keepdims=True)
            objs = (objs - min_objs) / (max_objs - min_objs)

        '''cal min crowding distance for each inner mo (after nd sort)'''
        cd = -1
        for inner_step in range(n_inner):
            cd = cal_min_crowding_distance(objs[inner_step])
            if cd == np.inf:
                avg_span = np.mean(
                    [np.max(objs[inner_step][obj_idx]) - np.min(objs[inner_step][obj_idx]) for obj_idx in
                     range(n_obj)])
                cd = avg_span
            writer.add_scalar(f'{prefix}_details/{target}/{i+1}', cd, inner_step + 1)
        writer.add_scalar(f'{prefix}/{target}', cd, i + 1)

        print(f"==>> {prefix}: {target} {cd:.5f}.")

        return cd

    def write_mo(self, mo_dict, pop_labels, i, writer: Optional[SummaryWriter] = None, target='acc',
                 prefix='mo'):
        """
        draw mo graph for different inner step.
        Args:
            mo_dict: {pop_idx: {inner_idx: [n_obj]}} or
                dataframe ['Tag', 'Pop_id', 'Obj_id', 'Epoch_id', 'Inner_id', 'Value']
            pop_labels:
            i:
            writer:
            target: for mo_dict's Tag selector.
            prefix:

        Returns:

        """
        level = self.levels.index('INFO')
        if level < self.level:
            return

        if type(mo_dict) is dict:
            n_pop, n_inner, n_obj = len(mo_dict), len(mo_dict[0]), len(mo_dict[0][0])
            objs = np.array([
                [[mo_dict[pop_idx][inner_idx][obj_idx] for pop_idx in range(n_pop)]
                 for obj_idx in range(n_obj)] for inner_idx in range(n_inner)
            ])  # [n_inner, n_obj, n_pop]

            '''log objs figure'''
            if n_obj == 2:
                fig, ax = plt.subplots(1, 1)
            else:       # n_obj > 2
                fig, ax = plt.subplots(1, 1, subplot_kw={'polar': True})
            draw_objs(objs, pop_labels, ax=ax)
            writer.add_figure(f"{prefix}/objs_{target}", fig, i + 1)

            with open(os.path.join(writer.log_dir, f'{prefix}_mo_dict_{target}.json'), 'w') as f:
                json.dump(mo_dict, f)
        else:
            # for exp in set(mo_dict.Exp):
            #     for inner_lr in set(mo_dict.Inner_lr):
            #         for logit_scale in set(mo_dict.Logit_scale):
            #             t_df = mo_dict[(mo_dict.Tag == target) &
            #                            (mo_dict.Exp == exp) & (mo_dict.Inner_lr == inner_lr) &
            #                            (mo_dict.Logit_scale == logit_scale)]
            t_df = mo_dict[mo_dict.Tag == target]
            n_pop = len(set(t_df.Pop_id))
            n_epoch = len(set(t_df.Epoch_id))
            n_inner = len(set(t_df.Inner_id))
            n_obj = len(set(t_df.Obj_id))
            objs = np.array([[[[
                t_df[(t_df.Pop_id == pop_idx) & (t_df.Obj_id == obj_idx) & (
                            t_df.Inner_id == inner_idx) & (t_df.Epoch_id == epoch_idx)].Value.iloc[0]    # .mean()
                for pop_idx in range(n_pop)] for obj_idx in range(n_obj)]
                for inner_idx in range(n_inner)] for epoch_idx in range(n_epoch)])
            # .mean() is average over epoch, thus, not correct mo. just visualize the first one in this epoch
            # [n_epoch, n_inner, n_obj, n_pop]
            objs = np.nan_to_num(objs)
            if len(objs.shape) != 2:        # no log
                return

            '''log objs figure along epoch for last inner step'''
            if n_obj == 2:
                fig, ax = plt.subplots(1, 1)
            else:       # n_obj > 2
                fig, ax = plt.subplots(1, 1, subplot_kw={'polar': True})
            draw_objs(objs[:, -1], pop_labels, ax=ax)      # [:,-1,:2,:] for the first 2 axes
            # writer.add_figure(f"objs_{target}_{exp}_innerlr_{inner_lr}{prefix}/logit_scale_{logit_scale}",
            #                   figure, i + 1)
            writer.add_figure(f"{prefix}/{target}_epoch", fig, i + 1)

            '''log objs figure along inner step for all epoch'''
            for e in range(n_epoch):
                if n_obj == 2:
                    fig, ax = plt.subplots(1, 1)
                else:       # n_obj > 2
                    fig, ax = plt.subplots(1, 1, subplot_kw={'polar': True})
                draw_objs(objs[e], pop_labels, ax=ax)   # [e, :, :2, :] for the first 2 axes
                writer.add_figure(f"{prefix}_inner/{target}_t{i+1}", fig, e)

    def write_task(self, pmo, task: dict, task_title, i, writer: Optional[SummaryWriter] = None, prefix='task'):
        """

        Args:
            pmo:
            task: ['context_images', 'target_images', 'context_labels', 'target_labels']
            task_title:
            i:
            writer:
            prefix:

        Returns:

        """
        level = self.levels.index('DEBUG')
        if level < self.level:
            return

        '''write images'''
        imgs = torch.cat([task['context_images'], task['target_images']]).cuda()
        numpy_imgs = imgs.cpu().numpy()
        writer.add_images(f"{prefix}/{task_title}", numpy_imgs, i+1)

        '''log img sim in the task'''
        with torch.no_grad():
            img_features = pmo.embed(imgs)
            _, selection_info = pmo.selector(img_features, gumbel=False, hard=False, average=False)
            img_sim = selection_info['y_soft']  # [img_size, 10]
            _, selection_info = pmo.selector(img_features, gumbel=False, hard=False)
            tsk_sim = selection_info['y_soft']  # [1, 10]
        sim = torch.cat([img_sim, *[tsk_sim] * (img_sim.shape[0] // 10)]).cpu().numpy()
        figure = draw_heatmap(sim, verbose=False)
        writer.add_figure(f"{prefix}/{task_title}/sim", figure, i + 1)
