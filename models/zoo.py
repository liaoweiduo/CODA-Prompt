import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torchvision.models as models
from torch.autograd import Variable
from .vit import VisionTransformer
import numpy as np
import copy


# Our method!
class CodaPrompt(nn.Module):
    def __init__(self, emb_d, n_tasks, prompt_param, key_dim=768):
        """Difference:
        ortho init for prompts
        """
        super().__init__()
        self.task_count = 0
        self.emb_d = emb_d
        self.key_d = key_dim
        self.n_tasks = n_tasks

        # prompt basic param
        self.e_pool_size = int(prompt_param[0])  # 100
        self.e_p_length = int(prompt_param[1])  # 8
        self.e_layers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        # [] for no prompt.

        # strenth of ortho penalty
        self.ortho_mu = prompt_param[2]  # 0.0

        # trigger fixed prompt size (FPS)
        self.FPS = False         # set to False to use origin coda-p

        # e prompt init
        for e in self.e_layers:
            # for model saving/loading simplicity, we init the full paramaters here
            # however, please note that we reinit the new components at each task
            # in the "spirit of continual learning", as we don't know how many tasks
            # we will encounter at the start of the task sequence
            #
            # in the original paper, we used ortho init at the start - this modification is more 
            # fair in the spirit of continual learning and has little affect on performance
            e_l = self.e_p_length
            p = tensor_prompt(self.e_pool_size, e_l, emb_d)  # [100, 8, 768]
            k = tensor_prompt(self.e_pool_size, self.key_d)  # [100, 768]
            a = tensor_prompt(self.e_pool_size, self.key_d)
            p = self.gram_schmidt(p)
            k = self.gram_schmidt(k)
            a = self.gram_schmidt(a)
            setattr(self, f'e_p_{e}', p)
            setattr(self, f'e_k_{e}', k)
            setattr(self, f'e_a_{e}', a)

    def process_task_count(self):
        self.task_count += 1

        if not self.FPS:
            # in the spirit of continual learning, we will reinit the new components
            # for the new task with Gram Schmidt
            #
            # in the original paper, we used ortho init at the start - this modification is more
            # fair in the spirit of continual learning and has little affect on performance
            #
            # code for this function is modified from:
            # https://github.com/legendongary/pytorch-gram-schmidt/blob/master/gram_schmidt.py
            for e in self.e_layers:
                K = getattr(self, f'e_k_{e}')
                A = getattr(self, f'e_a_{e}')
                P = getattr(self, f'e_p_{e}')
                k = self.gram_schmidt(K)
                a = self.gram_schmidt(A)
                p = self.gram_schmidt(P)
                setattr(self, f'e_p_{e}', p)
                setattr(self, f'e_k_{e}', k)
                setattr(self, f'e_a_{e}', a)

    # code for this function is modified from:
    # https://github.com/legendongary/pytorch-gram-schmidt/blob/master/gram_schmidt.py
    def gram_schmidt(self, vv):  # 施密特正交化

        def projection(u, v):
            denominator = (u * u).sum()

            if denominator < 1e-8:
                return None
            else:
                return (v * u).sum() / denominator * u

        # check if the tensor is 3D and flatten the last two dimensions if necessary
        is_3d = len(vv.shape) == 3
        if is_3d:
            shape_2d = copy.deepcopy(vv.shape)
            vv = vv.view(vv.shape[0], -1)

        # swap rows and columns
        vv = vv.T

        # process matrix size
        nk = vv.size(1)
        uu = torch.zeros_like(vv, device=vv.device)

        # get starting point
        if self.FPS:        # use all prompts
            s = 0
            f = self.e_pool_size
        else:
            pt = int(self.e_pool_size / (self.n_tasks))
            s = int(self.task_count * pt)
            f = int((self.task_count + 1) * pt)
        if s > 0:
            uu[:, 0:s] = vv[:, 0:s].clone()
        for k in range(s, f):
            redo = True
            while redo:
                redo = False
                vk = torch.randn_like(vv[:, k]).to(vv.device)
                uk = 0
                for j in range(0, k):
                    if not redo:
                        uj = uu[:, j].clone()
                        proj = projection(uj, vk)
                        if proj is None:
                            redo = True
                            print('restarting!!!')
                        else:
                            uk = uk + proj
                if not redo: uu[:, k] = vk - uk
        for k in range(s, f):
            uk = uu[:, k].clone()
            uu[:, k] = uk / (uk.norm())

        # undo swapping of rows and columns
        uu = uu.T

        # return from 2D
        if is_3d:
            uu = uu.view(shape_2d)

        return torch.nn.Parameter(uu)

    def handle_x_querry(self, x_querry, x_block, l):
        if x_querry is None:
            raise ValueError('x_querry is None')
        return x_querry

    def forward(self, x_querry, l, x_block, train=False, task_id=None):
        x_querry = self.handle_x_querry(x_querry, x_block, l)
        # e prompts
        e_valid = False
        if l in self.e_layers:
            e_valid = True
            B, C = x_querry.shape  # [bs, 768]

            K = getattr(self, f'e_k_{l}')  # [100, 768]
            A = getattr(self, f'e_a_{l}')  # [100, 768]
            p = getattr(self, f'e_p_{l}')  # [100, 8, 768]
            if self.FPS:        # use all prompts
                s = 0
                f = self.e_pool_size
            else:
                pt = int(self.e_pool_size / (self.n_tasks))  # 100/10=10
                s = int(self.task_count * pt)  # 10 prompts for one task
                f = int((self.task_count + 1) * pt)

            # freeze/control past tasks
            if train:
                if self.task_count > 0:
                    K = torch.cat((K[:s].detach().clone(), K[s:f]), dim=0)
                    A = torch.cat((A[:s].detach().clone(), A[s:f]), dim=0)
                    p = torch.cat((p[:s].detach().clone(), p[s:f]), dim=0)
                else:
                    K = K[s:f]
                    A = A[s:f]
                    p = p[s:f]
            else:
                K = K[0:f]
                A = A[0:f]
                p = p[0:f]

            # b = bs, d = 768, k = 100, l=8
            # with attention and cosine sim
            # (b x 1 x d) * soft([1 x k x d]) = (b x k x d) -> attention = k x d
            a_querry = torch.einsum('bd,kd->bkd', x_querry, A)
            # # (b x k x d) - [1 x k x d] = (b x k) -> key = k x d
            n_K = nn.functional.normalize(K, dim=1)
            q = nn.functional.normalize(a_querry, dim=2)
            aq_k = torch.einsum('bkd,kd->bk', q, n_K)  # aq_k is alpha (cosine similarity) [bs, 100]

            # aq_k = torch.ones((B, f)).to(p.device)      # just use all prompts with 1; un-condition type

            # (b x 1 x k x 1) * [1 x plen x k x d] = (b x plen x d) -> prompt = plen x k x d
            P_ = torch.einsum('bk,kld->bld', aq_k, p)

            # select prompts
            i = int(self.e_p_length / 2)  # 8 / 2
            Ek = P_[:, :i, :]
            Ev = P_[:, i:, :]

            # ortho penalty
            if train and self.ortho_mu > 0:
                loss = ortho_penalty(K) * self.ortho_mu
                loss += ortho_penalty(A) * self.ortho_mu
                loss += ortho_penalty(p.view(p.shape[0], -1)) * self.ortho_mu
            else:
                loss = 0
        else:
            loss = 0

        # combine prompts for prefix tuning
        if e_valid:
            p_return = [Ek, Ev]
        else:
            p_return = None

        # return
        return p_return, loss, x_block


def ortho_penalty(t):
    return ((t @ t.T - torch.eye(t.shape[0]).cuda()) ** 2).mean()


class CodaPromptCond(CodaPrompt):
    """i-Prompt-based conditioning"""
    def __init__(self, emb_d, n_tasks, prompt_param, key_dim=768):
        super(CodaPromptCond, self).__init__(emb_d, n_tasks, prompt_param, key_dim=key_dim)

    def handle_x_querry(self, x_querry, x_block, l):
        # x_block shape: [bs, 197, 768]
        return self.handle_x_querry_single_x(x_querry, x_block, l)

    def handle_x_querry_avg_x(self, x_querry, x_block, l):
        # use x_block to drive x_querry
        # x_querry shape: [bs, 768]
        x_querry = torch.mean(x_block, dim=1)  # average over cls_token and other patches.

        return x_querry

    def handle_x_querry_single_x(self, x_querry, x_block, l):
        # x_querry shape: [bs, 197, 768]
        return x_block

    def forward(self, x_querry, l, x_block, train=False, task_id=None):
        """Differences:
            cal Prompt for each patch
        """
        x_querry = self.handle_x_querry(x_querry, x_block, l)   # [bs, 197, 768]
        # e prompts
        e_valid = False
        task_id = self.task_count
        # if task_id is None:
        #     task_id = self.task_count

        if l in self.e_layers:
            e_valid = True
            B, pp, C = x_querry.shape  # [bs, 197, 768]

            K = getattr(self, f'e_k_{l}')  # [100, 768]
            A = getattr(self, f'e_a_{l}')  # [100, 768]
            p = getattr(self, f'e_p_{l}')  # [100, 8, 768]

            if self.FPS:  # use all prompts
                s = 0
                f = self.e_pool_size
            else:
                pt = int(self.e_pool_size / (self.n_tasks))  # 100/10=10
                s = int(self.task_count * pt)  # 10 prompts for one task
                f = int((self.task_count + 1) * pt)
            # s = int(self.task_count * pt)  # 10 prompts for one task
            # f = int((self.task_count + 1) * pt)

            # freeze/control past tasks
            if train:
                if self.task_count > 0:
                    K = torch.cat((K[:s].detach().clone(), K[s:f]), dim=0)
                    A = torch.cat((A[:s].detach().clone(), A[s:f]), dim=0)
                    p = torch.cat((p[:s].detach().clone(), p[s:f]), dim=0)
                else:
                    K = K[s:f]
                    A = A[s:f]
                    p = p[s:f]
            else:
                K = K[0:f]
                A = A[0:f]
                p = p[0:f]
            # K [100, 768] A [100, 768] p [100, 8, 768]

            # rearrange KAp, according to obj for each x
            # K A -> [bs, 10->100, 768], p -> [bs, 10->100, 8, 768]
            K = torch.stack([K for _ in range(B)])
            A = torch.stack([A for _ in range(B)])
            p = torch.stack([p for _ in range(B)])

            # b = bs, p=197, d = 768, k = 100, l=8, o=1 or s
            # (b x p x 1 x d) * soft([b x 1 x ot x d]) = (b x p x ot x d) -> attention = b x ot x d
            a_querry = torch.einsum('bpd,bod->bpod', x_querry, A)
            n_K = nn.functional.normalize(K, dim=-1)
            q = nn.functional.normalize(a_querry, dim=-1)
            # sum((b x p x ot x d) - [b x 1 x ot x d]) = (b x p x ot) -> key = b x ot x d
            aq_k = torch.einsum('bpod,bod->bpo', q, n_K)
            # aq_k is alpha (cosine similarity) [bs, 197, ot]

            # (b x p x ot x 1 x 1) * [b x 1 x ot x l x d] = (b x p x l x d) -> prompt = b x ot x l x d
            P_ = torch.einsum('bpo,bold->bpld', aq_k, p)

            # select prompts
            i = int(self.e_p_length / 2)  # 8 / 2
            Ek = P_[:, :, :i, :]        # 2-nd dim (197), contain cls-token
            Ev = P_[:, :, i:, :]

            # ortho penalty
            if train and self.ortho_mu > 0:
                loss = ortho_penalty(K) * self.ortho_mu
                loss += ortho_penalty(A) * self.ortho_mu
                loss += ortho_penalty(p.view(p.shape[0], -1)) * self.ortho_mu
            else:
                loss = 0
        else:
            loss = 0

        # combine prompts for prefix tuning
        if e_valid:
            p_return = [Ek, Ev]
        else:
            p_return = None

        # return
        return p_return, loss, x_block


class PmoPrompt(CodaPromptCond):
    def __init__(self, emb_d, n_tasks, prompt_param, key_dim=768):
        super(PmoPrompt, self).__init__(emb_d, n_tasks, prompt_param[:3], key_dim=key_dim)

        if self.FPS:
            self.n_prompt_per_task = int(self.e_pool_size)  # num of prompts
        else:
            self.n_prompt_per_task = int(self.e_pool_size / (self.n_tasks))  # num of prompts for 1 task.
        self.n_obj = self.n_prompt_per_task     # here, n_obj is number of prompt per task
        # self.n_prompt_per_task or prompt_param[3] or specify how many prompt used for 1 obj

        # # dataset with pool
        # self.pool_size = prompt_param[3]
        # self.pool = None        # init using self.bind_pool()   to pass pool from learner: PMOPrompt

        # self.updated_weights = None     # temp for inner update

    # def gram_schmidt(self, vv):  # disable gram schmidt to use uniform init
    #
    #     return vv

    def KAPselection(self, l, pre_learn, train):

        # if self.updated_weights is not None and fast_weights:
        #     # put to the same device to support computation
        #     # print(f"{self.updated_weights[f'e_k_{l}'].shape}, {self.updated_weights[f'e_k_{l}'].device}")
        #     device = getattr(self, f'e_k_{l}').device
        #     K = self.updated_weights[f'e_k_{l}'].to(device)
        #     A = self.updated_weights[f'e_a_{l}'].to(device)
        #     p = self.updated_weights[f'e_p_{l}'].to(device)
        # else:
        K = getattr(self, f'e_k_{l}')  # [100, 768]
        A = getattr(self, f'e_a_{l}')  # [100, 768]
        p = getattr(self, f'e_p_{l}')  # [100, 8, 768]

        if self.FPS:  # use all prompts
            s = 0
            f = self.e_pool_size
        else:
            pt = int(self.e_pool_size / (self.n_tasks))  # 100/10=10
            s = int(self.task_count * pt)  # 10 prompts for one task
            f = int((self.task_count + 1) * pt)
        # s = int(self.task_count * pt)  # 10 prompts for one task
        # f = int((self.task_count + 1) * pt)

        # freeze/control past tasks
        if pre_learn and self.task_count > 0 and not self.FPS:  # other tasks, only use old prompt
            K = K[:s].detach().clone()
            A = A[:s].detach().clone()
            p = p[:s].detach().clone()
        elif train:
            if self.task_count > 0:
                K = torch.cat((K[:s].detach().clone(), K[s:f]), dim=0)
                A = torch.cat((A[:s].detach().clone(), A[s:f]), dim=0)
                p = torch.cat((p[:s].detach().clone(), p[s:f]), dim=0)
            else:
                K = K[s:f]
                A = A[s:f]
                p = p[s:f]
        else:
            K = K[0:f]
            A = A[0:f]
            p = p[0:f]

        return K, A, p, s, f

    def forward(self, x_querry, l, x_block, train=False, task_id=None,
                hard_obj_idx=None, hard_l=None, mask=None, mask_mode='use',
                pre_learn=False,
                debug_mode=False, **kwargs):
        """Differences:
            Use hard_obj_idx and hard_l to locate mask for prompt.
            hard_obj_idx can be -1 to select all old prompts.
            Use mask to determine whether to mask out the prompt (True) or only select the prompt (False)
            mask_mode: 'maskout' or 'use'
            pre_learn: True to only use ViT or old prompt
        """
        return self.forward_patch_wise(x_querry, l, x_block, train, task_id,
                                       hard_obj_idx, hard_l, mask, mask_mode,
                                       pre_learn, debug_mode, **kwargs)

        x_querry = self.handle_x_querry(x_querry, x_block, l)   # [bs, 768]
        # e prompts
        e_valid = False
        task_id = self.task_count
        # if task_id is None:
        #     task_id = self.task_count

        if pre_learn and (self.task_count == 0 or self.FPS):      # first task, use vit
            return None, 0, x_block     # p_return, loss, x_block

        if l in self.e_layers:
            e_valid = True
            # B, C = x_querry.shape  # [bs, 768]
            K, A, p, s, f = self.KAPselection(l, pre_learn, train)

            # b = bs, d = 768, k = 100, l=8
            # with attention and cosine sim
            # (b x 1 x d) * soft([1 x k x d]) = (b x k x d) -> attention = k x d
            a_querry = torch.einsum('bd,kd->bkd', x_querry, A)
            # (b x k x d) - [1 x k x d] = (b x k) -> key = k x d
            n_K = nn.functional.normalize(K, dim=1)
            q = nn.functional.normalize(a_querry, dim=2)
            aq_k = torch.einsum('bkd,kd->bk', q, n_K)  # aq_k is alpha (cosine similarity) [bs, 10->100]

            # mask according to obj idx
            # if hard_l is None, do for all layer
            if hard_obj_idx is not None and (hard_l == l or (hard_l is None and (s > 0 or hard_obj_idx > -1))):
                ot = int(self.n_prompt_per_task / self.n_obj)  # number of prompts for one obj

                # aq_k [bs, f] modification
                if mask_mode == 'maskout':
                    if hard_obj_idx == -1:      # mask out all old prompt
                        if type(mask) is float or type(mask) is int:  # constant prompt
                            aq_k[:, :s] = 1
                        elif type(mask) is str:  # random prompt
                            aq_k[:, :s] = 1
                        else:  # mask is None: only maskout this prompt
                            aq_k = torch.cat((
                                torch.zeros_like(aq_k[:, :s]),  # detach and mask 0
                                aq_k[:, s:],                    # mask 1
                            ), dim=1)
                    elif type(mask) is float or type(mask) is int:  # constant prompt
                        aq_k[:, (hard_obj_idx * ot):((hard_obj_idx + 1) * ot)] = 1
                    elif type(mask) is str:  # random prompt
                        aq_k[:, (hard_obj_idx * ot):((hard_obj_idx + 1) * ot)] = 1
                    else:  # mask is None: only select this prompt
                        aq_k = torch.cat((
                            torch.zeros_like(aq_k[:, :(hard_obj_idx * ot)]),  # detach and mask 0
                            aq_k[:, (hard_obj_idx * ot):((hard_obj_idx + 1) * ot)],
                            # torch.ones_like(aq_k[:, (hard_obj_idx * ot):((hard_obj_idx + 1) * ot)]),  # mask 1
                            torch.zeros_like(aq_k[:, ((hard_obj_idx + 1) * ot):]),  # detach and mask 0
                        ), dim=1)
                elif mask_mode == 'use':
                    if hard_obj_idx == -1:      # use all old prompt
                        if type(mask) is float or type(mask) is int:  # constant prompt
                            aq_k[:, s:] = 1
                        elif type(mask) is str:  # random prompt
                            aq_k[:, s:] = 1
                        else:  # mask is None: only select this prompt
                            aq_k = torch.cat((
                                aq_k[:, :s],                    # mask 1
                                torch.zeros_like(aq_k[:, s:]),  # detach and mask 0
                            ), dim=1)
                    elif type(mask) is float or type(mask) is int:  # constant prompt
                        aq_k[:, :(hard_obj_idx * ot)] = 1
                        aq_k[:, ((hard_obj_idx + 1) * ot):] = 1
                    elif type(mask) is str:  # random prompt
                        aq_k[:, :(hard_obj_idx * ot)] = 1
                        aq_k[:, ((hard_obj_idx + 1) * ot):] = 1
                    else:  # mask is None: only select this prompt, other layer also do not use prompt
                        aq_k = torch.cat((
                            torch.zeros_like(aq_k[:, :(hard_obj_idx * ot)]),  # detach and mask 0
                            aq_k[:, (hard_obj_idx * ot):((hard_obj_idx + 1) * ot)],
                            # torch.ones_like(aq_k[:, (hard_obj_idx * ot):((hard_obj_idx + 1) * ot)]),  # mask 1
                            torch.zeros_like(aq_k[:, ((hard_obj_idx + 1) * ot):]),  # detach and mask 0
                        ), dim=1)
                else:
                    raise Exception(f'Unknown mask comb: {mask_mode}, {mask} for obj{hard_obj_idx}')

                # p [f, 8, 768] modification
                # if train use torch random seed, else use numpy random seed (since it can be fixed)
                if mask_mode == 'maskout':
                    if hard_obj_idx == -1:      # mask out all old prompt
                        if type(mask) is float or type(mask) is int:  # constant prompt
                            p = torch.cat((
                                torch.fill_(torch.empty_like(p[:s]), mask),
                                p[s:]
                            ), dim=0)
                        elif mask == 'randn':  # random prompt
                            p = torch.cat((
                                torch.randn_like(p[:s]) if train else
                                torch.from_numpy(np.random.randn(s, *p.shape[1:])).float().to(p.device),
                                p[s:]
                            ), dim=0)
                        elif mask == 'uniform':  # uniform random prompt
                            p = torch.cat((
                                torch.nn.init.uniform_(
                                    p[:s].detach().clone()) if train else
                                torch.from_numpy(np.random.uniform(size=(s, *p.shape[1:]))).float().to(p.device),
                                p[s:]
                            ), dim=0)
                        elif mask == 'ortho':  # ortho random prompt
                            p = torch.cat((
                                torch.nn.init.orthogonal_(
                                    p[:s].detach().clone()) if train else
                                torch.from_numpy(ortho_random(size=(s, *p.shape[1:]))).float().to(p.device),
                                p[s:]
                            ), dim=0)
                        else:  # mask is None: only select this prompt
                            assert mask is None, f'mask `{mask}` is not None but unrecognized str'
                    elif type(mask) is float or type(mask) is int:  # constant prompt
                        p = torch.cat((
                            p[:(hard_obj_idx * ot)],
                            torch.fill_(torch.empty_like(p[(hard_obj_idx * ot):((hard_obj_idx + 1) * ot)]),
                                        mask),
                            p[((hard_obj_idx + 1) * ot):]
                        ), dim=0)
                    elif mask == 'randn':  # random prompt
                        p = torch.cat((
                            p[:(hard_obj_idx * ot)],
                            torch.randn_like(p[(hard_obj_idx * ot):((hard_obj_idx + 1) * ot)]) if train else
                            torch.from_numpy(np.random.randn(ot, *p.shape[1:])).float().to(p.device),
                            p[((hard_obj_idx + 1) * ot):]
                        ), dim=0)
                    elif mask == 'uniform':  # uniform random prompt
                        p = torch.cat((
                            p[:(hard_obj_idx * ot)],
                            torch.nn.init.uniform_(
                                p[(hard_obj_idx * ot):((hard_obj_idx + 1) * ot)].detach().clone()) if train else
                            torch.from_numpy(np.random.uniform(size=(ot, *p.shape[1:]))).float().to(p.device),
                            p[((hard_obj_idx + 1) * ot):]
                        ), dim=0)
                    elif mask == 'ortho':  # ortho random prompt
                        p = torch.cat((
                            p[:(hard_obj_idx * ot)],
                            torch.nn.init.orthogonal_(
                                p[(hard_obj_idx * ot):((hard_obj_idx + 1) * ot)].detach().clone()) if train else
                            torch.from_numpy(ortho_random(size=(ot, *p.shape[1:]))).float().to(p.device),
                            p[((hard_obj_idx + 1) * ot):]
                        ), dim=0)
                    else:  # mask is None: only select this prompt
                        assert mask is None, f'mask `{mask}` is not None but unrecognized str'
                elif mask_mode == 'use':
                    if hard_obj_idx == -1:      # use all old prompt
                        if type(mask) is float or type(mask) is int:  # constant prompt
                            p = torch.cat((
                                p[:s],
                                torch.fill_(torch.empty_like(p[s:]), mask),
                            ), dim=0)
                        elif mask == 'randn':  # random prompt
                            p_ = torch.randn_like(p) if train else (
                                torch.from_numpy(np.random.randn(*p.shape)).float().to(p.device))
                            p_[:s] = p[:s]
                            p = p_
                        elif mask == 'uniform':  # uniform random prompt
                            p_ = torch.nn.init.uniform_(p.detach().clone()) if train else (
                                torch.from_numpy(np.random.uniform(size=p.shape)).float().to(p.device))
                            p_[:s] = p[:s]
                            p = p_
                        elif mask == 'ortho':  # ortho random prompt
                            p_ = torch.nn.init.orthogonal_(p.detach().clone()) if train else (
                                torch.from_numpy(ortho_random(size=p.shape)).float().to(p.device))
                            p_[:s] = p[:s]
                            p = p_
                        else:  # mask is None: only select this prompt
                            pass
                    elif type(mask) is float or type(mask) is int:  # constant prompt
                        p = torch.cat((
                            torch.fill_(torch.empty_like(p[:(hard_obj_idx * ot)]), mask),
                            p[(hard_obj_idx * ot):((hard_obj_idx + 1) * ot)],
                            torch.fill_(torch.empty_like(p[((hard_obj_idx + 1) * ot):]), mask),
                        ), dim=0)
                    elif mask == 'randn':  # random prompt
                        p_ = torch.randn_like(p) if train else (
                            torch.from_numpy(np.random.randn(*p.shape)).float().to(p.device))
                        p_[(hard_obj_idx * ot):((hard_obj_idx + 1) * ot)
                        ] = p[(hard_obj_idx * ot):((hard_obj_idx + 1) * ot)]
                        p = p_
                    elif mask == 'uniform':  # uniform random prompt
                        p_ = torch.nn.init.uniform_(p.detach().clone()) if train else (
                            torch.from_numpy(np.random.uniform(size=p.shape)).float().to(p.device))
                        p_[(hard_obj_idx * ot):((hard_obj_idx + 1) * ot)
                        ] = p[(hard_obj_idx * ot):((hard_obj_idx + 1) * ot)]
                        p = p_
                    elif mask == 'ortho':  # ortho random prompt
                        p_ = torch.nn.init.orthogonal_(p.detach().clone()) if train else (
                            torch.from_numpy(ortho_random(size=p.shape)).float().to(p.device))
                        p_[(hard_obj_idx * ot):((hard_obj_idx + 1) * ot)
                        ] = p[(hard_obj_idx * ot):((hard_obj_idx + 1) * ot)]
                        p = p_
                    else:  # mask is None: only select this prompt
                        pass

            elif hard_obj_idx is not None and mask is None:
                # do not use prompts for all other layers if only select this prompt (mask: None)
                aq_k = torch.zeros_like(aq_k)
                e_valid = False  # make p_return to be None

            if debug_mode:
                print(f'aq_k in layer{l}: {aq_k[0]}')
                print(f'p in layer{l}: {p[:, 0, 0]}')

            # (b x 1 x k x 1) * [1 x plen x k x d] = (b x plen x d) -> prompt = plen x k x d
            P_ = torch.einsum('bk,kld->bld', aq_k, p)

            # select prompts
            i = int(self.e_p_length / 2)  # 8 / 2
            Ek = P_[:, :i, :]
            Ev = P_[:, i:, :]

            # ortho penalty
            if train and self.ortho_mu > 0:
                loss = ortho_penalty(K) * self.ortho_mu
                loss += ortho_penalty(A) * self.ortho_mu
                loss += ortho_penalty(p.view(p.shape[0], -1)) * self.ortho_mu
            else:
                loss = 0
        else:
            loss = 0

        # combine prompts for prefix tuning
        if e_valid:
            p_return = [Ek, Ev]
        else:
            p_return = None

        # return
        return p_return, loss, x_block

    def forward_patch_wise(self, x_querry, l, x_block, train=False, task_id=None,
                           hard_obj_idx=None, hard_l=None, mask=None, mask_mode='use',
                           pre_learn=False,
                           debug_mode=False, **kwargs):
        """Differences:
            cal Prompt for each patch
            hard_obj_idx can be a list without -1, or scaler -1
        """
        x_querry = self.handle_x_querry(x_querry, x_block, l)   # [bs, 197, 768]
        # e prompts
        e_valid = False
        task_id = self.task_count
        # if task_id is None:
        #     task_id = self.task_count

        if pre_learn and (self.task_count == 0 or self.FPS):      # first task, use vit
            return None, 0, x_block     # p_return, loss, x_block

        if l in self.e_layers:
            e_valid = True
            B, pp, C = x_querry.shape  # [bs, 197, 768]
            K, A, p, s, f = self.KAPselection(l, pre_learn, train)
            # K [100, 768] A [100, 768] p [100, 8, 768]

            # rearrange KAp, according to obj for each x
            if hard_obj_idx is not None and (hard_l == l or hard_l is None):
                assert ((type(hard_obj_idx) is int and hard_obj_idx == -1) or len(hard_obj_idx) == len(x_querry)
                        ), f"hard_obj_idx: {hard_obj_idx}; x_querry: {x_querry.shape}"
                ot = int(self.n_prompt_per_task / self.n_obj)  # number of prompts for one obj
                # K A -> [bs, ot, 768], p -> [bs, ot, 8, 768]
                if type(hard_obj_idx) is int and hard_obj_idx == -1:
                    if s == 0:      # first task no conditioning
                        return None, 0, x_block     # p_return, loss, x_block
                    K = torch.stack([K[:s] for _ in range(B)])
                    A = torch.stack([A[:s] for _ in range(B)])
                    p = torch.stack([p[:s] for _ in range(B)])
                else:
                    K = torch.stack([K[(idx * ot): ((idx+1) * ot)] for idx in hard_obj_idx])
                    A = torch.stack([A[(idx * ot): ((idx+1) * ot)] for idx in hard_obj_idx])
                    p = torch.stack([p[(idx * ot): ((idx+1) * ot)] for idx in hard_obj_idx])
            else:       # use all prompts
                # K A -> [bs, 10->100, 768], p -> [bs, 10->100, 8, 768]
                K = torch.stack([K for _ in range(B)])
                A = torch.stack([A for _ in range(B)])
                p = torch.stack([p for _ in range(B)])

            # b = bs, p=197, d = 768, k = 100, l=8, o=1 or s
            # (b x p x 1 x d) * soft([b x 1 x ot x d]) = (b x p x ot x d) -> attention = b x ot x d
            a_querry = torch.einsum('bpd,bod->bpod', x_querry, A)
            n_K = nn.functional.normalize(K, dim=-1)
            q = nn.functional.normalize(a_querry, dim=-1)
            # sum((b x p x ot x d) - [b x 1 x ot x d]) = (b x p x ot) -> key = b x ot x d
            aq_k = torch.einsum('bpod,bod->bpo', q, n_K)
            # aq_k is alpha (cosine similarity) [bs, 197, ot]

            if debug_mode:
                print(f'aq_k in layer{l}: {aq_k.shape} \n{aq_k[0, 0]}')
                print(f'p in layer{l}: {p.shape} \n{p[:, 0, 0, 0]}')

            # (b x p x ot x 1 x 1) * [b x 1 x ot x l x d] = (b x p x l x d) -> prompt = b x ot x l x d
            P_ = torch.einsum('bpo,bold->bpld', aq_k, p)

            # select prompts
            i = int(self.e_p_length / 2)  # 8 / 2
            Ek = P_[:, :, :i, :]        # 2-nd dim (197), contain cls-token
            Ev = P_[:, :, i:, :]

            # ortho penalty
            if train and self.ortho_mu > 0:
                loss = ortho_penalty(K) * self.ortho_mu
                loss += ortho_penalty(A) * self.ortho_mu
                loss += ortho_penalty(p.view(p.shape[0], -1)) * self.ortho_mu
            else:
                loss = 0
        else:
            loss = 0

        # combine prompts for prefix tuning
        if e_valid:
            p_return = [Ek, Ev]
        else:
            p_return = None

        # return
        return p_return, loss, x_block


def ortho_random(size):
    """Generated by ChatGPT3.5"""

    def gram_schmidt(A):
        Q, R = np.linalg.qr(A)
        return Q

    reverse = False
    if size[-2] < size[-1]:
        reverse = True
        size = (*size[:-2], size[-1], size[-2])
    # 生成一个随机矩阵
    random_matrix = np.random.rand(*size)

    # 对随机矩阵执行Gram-Schmidt正交化
    orthogonal_matrix = gram_schmidt(random_matrix)

    if reverse:
        orthogonal_matrix = np.swapaxes(orthogonal_matrix, -1, -2)

    return orthogonal_matrix


# CODA-Prompt with memory replay
class CodaPromptR(CodaPrompt):
    def __init__(self, emb_d, n_tasks, prompt_param, key_dim=768):
        super(CodaPromptR, self).__init__(emb_d, n_tasks, prompt_param, key_dim=768)

        # self.batch_task_ids = None      # gpu tensor [bs]

    # def gram_schmidt(self, vv):  # 施密特正交化
    #
    #     return vv

    def forward(self, x_querry, l, x_block, train=False, task_id=None):
        """difference:
        Replay-based needs task_id for each sample to support selecting correct prompt
        task_id w.r.t. x_querry, self.task_count w.r.t. current task
        treat different task differently during training.
        """
        # # debug
        # print(f'in CodaPromptR forward: task_id={task_id}; '
        #       f'self.task_count={self.task_count}.')

        if not train:
            # for evaluation, use super()
            return super().forward(x_querry, l, x_block, train=train, task_id=task_id)

        x_querry_ = x_querry
        B, C = x_querry.shape  # [bs, 768]
        assert (len(task_id) == B
                ), f'B: {B}, len(task_id): {len(task_id)}'

        # e prompts
        e_valid = False
        loss = 0
        num_samples = 0
        if l in self.e_layers:
            e_valid = True

            E_breaker = int(self.e_p_length / 2)  # 8 / 2
            K_ = getattr(self, f'e_k_{l}')  # [100, 768]
            A_ = getattr(self, f'e_a_{l}')  # [100, 768]
            p_ = getattr(self, f'e_p_{l}')  # [100, 8, 768]

            pt = int(self.e_pool_size / (self.n_tasks))  # 100/10=10

            P_ = torch.ones([B, *p_.shape[1:]]).to(p_.device)  # [bs, 8, 768]

            # apart x_querry according to batch_task_ids
            batch_task_ids = task_id
            unique_task_ids = torch.unique(batch_task_ids)
            for task in unique_task_ids:
                mask = batch_task_ids == task
                # # debug
                # print(f'mask: {mask}')

                x_querry = x_querry_[mask]  # mask input
                num_samples_ = len(x_querry)
                num_samples += num_samples_

                # use all involved prompts (self.task_count) # task -> corresponding task id
                s = int(self.task_count * pt)
                f = int((self.task_count + 1) * pt)
                if task < self.task_count:  # old task just use prompts but not train
                    K = K_[0:f].detach().clone()
                    A = A_[0:f].detach().clone()
                    p = p_[0:f].detach().clone()
                else:  # new task: task == self.task_count
                    # do not freeze old prompts
                    # K = K_[0:f]
                    # A = A_[0:f]
                    # p = p_[0:f]
                    # # or freeze old prompts
                    if self.task_count > 0:  # current task is not the first task
                        K = torch.cat((K_[:s].detach().clone(), K_[s:f]), dim=0)
                        A = torch.cat((A_[:s].detach().clone(), A_[s:f]), dim=0)
                        p = torch.cat((p_[:s].detach().clone(), p_[s:f]), dim=0)
                    else:  # first task
                        K = K_[s:f]
                        A = A_[s:f]
                        p = p_[s:f]

                # b = bs, d = 768, k = 100, l=8
                # with attention and cosine sim
                # (b x 1 x d) * soft([1 x k x d]) = (b x k x d) -> attention = k x d
                a_querry = torch.einsum('bd,kd->bkd', x_querry, A)
                # # (b x k x d) - [1 x k x d] = (b x k) -> key = k x d
                n_K = nn.functional.normalize(K, dim=1)
                q = nn.functional.normalize(a_querry, dim=2)
                aq_k = torch.einsum('bkd,kd->bk', q, n_K)
                # aq_k is alpha (cosine similarity) [bs, 100]

                # # debug
                # print(f'aq_k: {aq_k}')

                # (b x 1 x k x 1) * [1 x plen x k x d] = (b x plen x d) -> prompt = plen x k x d
                P__ = torch.einsum('bk,kld->bld', aq_k, p)

                # # debug
                # print(f'P__ shape: {P__.shape}')      # [bs, 8, 768]

                # move P__ to P_
                P_[mask] = P__

                # ortho penalty
                if train and self.ortho_mu > 0:
                    loss += ortho_penalty(K) * self.ortho_mu * num_samples_
                    loss += ortho_penalty(A) * self.ortho_mu * num_samples_
                    loss += ortho_penalty(p.view(p.shape[0], -1)) * self.ortho_mu * num_samples_

            if train and self.ortho_mu > 0:
                loss = loss / num_samples  # to average

            # select prompts
            Ek = P_[:, :E_breaker, :]  # [bs, 4, 768]
            Ev = P_[:, E_breaker:, :]  # [bs, 4, 768]

        # combine prompts for prefix tuning
        p_return = None
        if e_valid:
            p_return = [Ek, Ev]

        # return
        return p_return, loss, x_block


# @article{wang2022dualprompt,
#   title={DualPrompt: Complementary Prompting for Rehearsal-free Continual Learning},
#   author={Wang, Zifeng and Zhang, Zizhao and Ebrahimi, Sayna and Sun, Ruoxi and Zhang, Han and Lee,
#       Chen-Yu and Ren, Xiaoqi and Su, Guolong and Perot, Vincent and Dy, Jennifer and others},
#   journal={European Conference on Computer Vision},
#   year={2022}
# }
class DualPrompt(nn.Module):
    def __init__(self, emb_d, n_tasks, prompt_param, key_dim=768):
        super().__init__()
        self.task_count = 0
        self.emb_d = emb_d
        self.key_d = key_dim
        self.n_tasks = n_tasks
        self._init_smart(emb_d, prompt_param)

        # g prompt init
        for g in self.g_layers:
            p = tensor_prompt(self.g_p_length, emb_d)
            setattr(self, f'g_p_{g}', p)

        # e prompt init
        for e in self.e_layers:
            p = tensor_prompt(self.e_pool_size, self.e_p_length, emb_d)
            k = tensor_prompt(self.e_pool_size, self.key_d)
            setattr(self, f'e_p_{e}', p)
            setattr(self, f'e_k_{e}', k)

    def _init_smart(self, emb_d, prompt_param):

        self.top_k = 1
        self.task_id_bootstrap = True

        # prompt locations
        self.g_layers = [0, 1]
        self.e_layers = [2, 3, 4]

        # prompt pool size
        self.g_p_length = int(prompt_param[2])
        self.e_p_length = int(prompt_param[1])
        self.e_pool_size = int(prompt_param[0])

    def process_task_count(self):
        self.task_count += 1

    def forward(self, x_querry, l, x_block, train=False, task_id=None):

        # e prompts
        e_valid = False
        if l in self.e_layers:
            e_valid = True
            B, C = x_querry.shape
            K = getattr(self, f'e_k_{l}')  # 0 based indexing here
            p = getattr(self, f'e_p_{l}')  # 0 based indexing here

            # cosine similarity to match keys/querries
            n_K = nn.functional.normalize(K, dim=1)
            q = nn.functional.normalize(x_querry, dim=1).detach()
            cos_sim = torch.einsum('bj,kj->bk', q, n_K)

            if train and task_id < self.n_tasks:  # for continual training tasks
                # dual prompt during training uses task id
                if self.task_id_bootstrap:
                    loss = (1.0 - cos_sim[:, task_id]).sum()
                    P_ = p[task_id].expand(len(x_querry), -1, -1)
                else:
                    top_k = torch.topk(cos_sim, self.top_k, dim=1)
                    k_idx = top_k.indices
                    loss = (1.0 - cos_sim[:, k_idx]).sum()
                    P_ = p[k_idx]
            else:  # inference or fewshot tasks
                top_k = torch.topk(cos_sim, self.top_k, dim=1)
                k_idx = top_k.indices
                loss = 0
                P_ = p[k_idx]

            # select prompts
            if train and self.task_id_bootstrap and task_id < self.n_tasks:
                # for continual training tasks
                i = int(self.e_p_length / 2)
                Ek = P_[:, :i, :].reshape((B, -1, self.emb_d))
                Ev = P_[:, i:, :].reshape((B, -1, self.emb_d))
            else:
                i = int(self.e_p_length / 2)
                Ek = P_[:, :, :i, :].reshape((B, -1, self.emb_d))
                Ev = P_[:, :, i:, :].reshape((B, -1, self.emb_d))

        # g prompts
        g_valid = False
        if l in self.g_layers:
            g_valid = True
            j = int(self.g_p_length / 2)
            p = getattr(self, f'g_p_{l}')  # 0 based indexing here
            P_ = p.expand(len(x_querry), -1, -1)
            Gk = P_[:, :j, :]
            Gv = P_[:, j:, :]

        # combine prompts for prefix tuning
        if e_valid and g_valid:
            Pk = torch.cat((Ek, Gk), dim=1)
            Pv = torch.cat((Ev, Gv), dim=1)
            p_return = [Pk, Pv]
        elif e_valid:
            p_return = [Ek, Ev]
        elif g_valid:
            p_return = [Gk, Gv]
            loss = 0
        else:
            p_return = None
            loss = 0

        # return
        if train:
            return p_return, loss, x_block
        else:
            return p_return, 0, x_block


# @inproceedings{wang2022learning,
#   title={Learning to prompt for continual learning},
#   author={Wang, Zifeng and Zhang, Zizhao and Lee, Chen-Yu and Zhang, Han and Sun, Ruoxi
#       and Ren, Xiaoqi and Su, Guolong and Perot, Vincent and Dy, Jennifer and Pfister, Tomas},
#   booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
#   pages={139--149},
#   year={2022}
# }
class L2P(DualPrompt):
    def __init__(self, emb_d, n_tasks, prompt_param, key_dim=768):
        super().__init__(emb_d, n_tasks, prompt_param, key_dim)

    def _init_smart(self, emb_d, prompt_param):
        self.top_k = 5
        self.task_id_bootstrap = False

        # prompt locations
        self.g_layers = []
        if prompt_param[2] > 0:
            self.e_layers = [0, 1, 2, 3, 4]
        else:
            self.e_layers = [0]

        # prompt pool size
        self.g_p_length = -1
        self.e_p_length = int(prompt_param[1])
        self.e_pool_size = int(prompt_param[0])


# note - ortho init has not been found to help l2p/dual prompt
def tensor_prompt(a, b, c=None, ortho=False):
    if c is None:
        p = torch.nn.Parameter(torch.FloatTensor(a, b), requires_grad=True)
    else:
        p = torch.nn.Parameter(torch.FloatTensor(a, b, c), requires_grad=True)
    if ortho:
        nn.init.orthogonal_(p)
    else:
        nn.init.uniform_(p)
    return p


class ViTZoo(nn.Module):
    def __init__(self, num_classes=10, pt=False, prompt_flag=False, prompt_param=None, use_vit_emb=True):
        super(ViTZoo, self).__init__()

        # get last layer
        self.last = nn.Linear(512, num_classes)
        self.prompt_flag = prompt_flag
        self.task_id = None
        self.use_vit_emb = use_vit_emb

        # get feature encoder
        zoo_model = None
        if pt:
            zoo_model = VisionTransformer(img_size=224, patch_size=16, embed_dim=768, depth=12,
                                          num_heads=12, ckpt_layer=0,
                                          drop_path_rate=0
                                          )
            try:
                from timm.models import vit_base_patch16_224
                load_dict = vit_base_patch16_224(pretrained=True).state_dict()
            except:
                print(f'Load vit_base_patch16_224 from local file: '
                      f'{os.path.abspath("../checkpoints/vit_base_patch16_224.pth")}')
                load_dict = torch.load("../checkpoints/vit_base_patch16_224.pth")

            del load_dict['head.weight'];
            del load_dict['head.bias']
            zoo_model.load_state_dict(load_dict)

        # classifier
        self.last = nn.Linear(768, num_classes)

        # create prompting module
        if self.prompt_flag == 'l2p':
            self.prompt = L2P(768, prompt_param[0], prompt_param[1])
        elif self.prompt_flag == 'dual':
            self.prompt = DualPrompt(768, prompt_param[0], prompt_param[1])
        elif self.prompt_flag == 'coda':
            self.prompt = CodaPrompt(768, prompt_param[0], prompt_param[1])
        elif self.prompt_flag == 'coda_r':
            self.prompt = CodaPromptR(768, prompt_param[0], prompt_param[1])
        elif self.prompt_flag == 'coda_cond':
            self.prompt = CodaPromptCond(768, prompt_param[0], prompt_param[1])
        elif self.prompt_flag == 'pmo':
            self.prompt = PmoPrompt(768, prompt_param[0], prompt_param[1])
        else:
            self.prompt = None

        # feature encoder changes if transformer vs resnet
        self.feat = zoo_model

        # freeze feature encoder
        for param in self.feat.parameters():
            param.requires_grad = False

    # pen: get penultimate features    
    def forward(self, x, register_blk=-1, task_id=None, pen=False, train=False,
                cond_x=None, **kwargs):
        # kwargs for prompt
        if task_id is None:
            task_id = self.task_id

        if self.prompt is not None:
            if self.use_vit_emb:
                with torch.no_grad():
                    if cond_x is not None:  # condition model
                        q, _ = self.feat(cond_x)
                    else:
                        q, _ = self.feat(x)
                    q = q[:, 0, :]
            else:
                q = None
            out, prompt_loss = self.feat(x, prompt=self.prompt, q=q, train=train, task_id=task_id,
                                         register_blk=register_blk,
                                         **kwargs)
            out = out[:, 0, :]
        else:
            out, _ = self.feat(x, register_blk=register_blk)
            out = out[:, 0, :]
        out = out.view(out.size(0), -1)
        if not pen:
            out = self.last(out)
        if self.prompt is not None and train:
            return out, prompt_loss
        else:
            return out


def vit_pt_imnet(out_dim, block_division=None, prompt_flag='None', prompt_param=None, **kwargs):
    return ViTZoo(num_classes=out_dim, pt=True, prompt_flag=prompt_flag, prompt_param=prompt_param, **kwargs)
