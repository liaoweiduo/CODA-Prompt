import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torchvision.models as models
from torch.autograd import Variable
from timm.models.layers import trunc_normal_
from .vit import VisionTransformer
import numpy as np
import copy


class SlotAttention(nn.Module):
    def __init__(self, emb_d, n_slots, num_patches=196, key_dim=128, n_iter=5, temp=1.):
        super().__init__()
        self.emb_d = emb_d          # emb for representation 768
        self.key_d = key_dim        # emb for slot: Dslot 64
        self.num_patches = num_patches      # warning: need to change for other backbone

        # slot basic param
        self.n_slots = n_slots  # 5   number of slots
        self.n_iter = n_iter     # T
        self.temp = temp
        self.attn_epsilon = 1e-8
        self.gru_d = self.key_d

        # slot related modules
        self.ln_input = nn.LayerNorm(self.emb_d)
        self.ln_slot = nn.LayerNorm(self.key_d)
        self.ln_output = nn.LayerNorm(self.key_d)
        self.mu = init_tensor(1, 1, self.key_d)             # slot gaussian distribution mu
        self.log_sigma = init_tensor(1, 1, self.key_d)          # slot gaussian distribution sigma
        self.k = nn.Linear(emb_d, key_dim, bias=False)
        self.q = nn.Linear(key_dim, key_dim, bias=False)
        self.v = nn.Linear(emb_d, key_dim, bias=False)
        # self.gru = nn.GRU(self.key_d, self.gru_d, batch_first=True)
        self.gru = nn.GRUCell(self.key_d, self.gru_d)
        self.mlp = nn.Sequential(
            nn.Linear(self.key_d, self.key_d, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(self.key_d, self.key_d, bias=True)
        )

        # slot decoder
        # self.ln_decoder = nn.LayerNorm(self.key_d)      # ln for
        self.decoder_pos_emb = nn.Parameter(torch.zeros(1, num_patches, 1, key_dim))
        trunc_normal_(self.decoder_pos_emb, std=.02)
        self.decoder = nn.Sequential(
            nn.Linear(self.key_d, self.key_d * 2, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(self.key_d * 2, self.emb_d, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(self.emb_d, self.emb_d, bias=True),
        )

    def forward(self, features, temp=None, n_iter=None):
        # features: [bs, n196, 768]
        slots, attn, _ = self.forward_slots(features, temp=temp, n_iter=n_iter)
        # slots [bs, k20, d64], attn [bs, n196, k20]

        # recon
        # slot_features = self.ln_decoder(slots)
        # broadcast slots into shape [bs, n, k, d]
        slot_features = slots.unsqueeze(1)      # [bs, 1, k, d]
        slot_features = slot_features + self.decoder_pos_emb    # apply pos emb on n -> [bs, n, k, d]
        slot_features = self.decoder(slot_features)     # [bs, n196, k20, 768]
        slot_features = torch.einsum('bnkd,bnk->bnd', slot_features, attn)       # [bs, n196, 768]

        # recon loss
        # features = self.ln_input(features)      # reconstruct features after ln
        recon_loss = F.mse_loss(slot_features, features, reduction='none')      # [bs, n196, 768]
        recon_loss = torch.mean(torch.mean(recon_loss, dim=-1), dim=-1)     # [bs]

        return slots, attn, recon_loss

    def forward_slots(self, features, temp=None, n_iter=None):
        # features [bs, 196, 768]
        bs = features.shape[0]

        n_iter = self.n_iter if n_iter is None else n_iter
        temp = self.temp if temp is None else temp
        iter_slots = []
        iter_attn_vis = []

        # init
        features = self.ln_input(features)
        slots = torch.randn(bs, self.n_slots, self.key_d, device=self.log_sigma.device) * torch.exp(self.log_sigma) + self.mu
        # [bs, k, 64]

        # iter
        k = self.k(features)    # [bs, 196, 64]
        v = self.v(features)    # [bs, 196, 64]
        k = (self.key_d ** (-0.5) * temp) * k

        attn_vis = None
        for t in range(n_iter):
            slots_prev = slots.clone()
            slots = self.ln_slot(slots)
            q = self.q(slots)       # [bs, k, 64]

            # b = bs, n = 196, k = 5, d = 64
            ## softmax(KQ^T/sqrt(d), dim='slots')
            # sum((b x n x 1 x d) * [b x 1 x k x d]) = (b x n x k)
            attn = torch.einsum('bnd,bkd->bnk', k, q)
            # attn = attn * (self.key_d ** -0.5)
            # softmax over slots
            attn_vis = F.softmax(attn, dim=-1)      # [b, n, k]

            ## updates = WeightedMean(attn+epsilon, v)
            attn = attn_vis + self.attn_epsilon
            attn = attn / torch.sum(attn, dim=-2, keepdim=True)
            # sum((b x n x k x 1) * (b x n x 1 x d)) = (b x k x d)
            updates = torch.einsum('bnk,bnd->bkd', attn, v)

            ## slots = GRU(state=slots_prev[b,k,d], inputs=updates[b,k,d])  (for each slot)

            slots = self.gru(updates.view(-1, self.key_d),               # [b*k, d]
                             slots_prev.reshape(-1, self.key_d))         # [b*k, d]
            # slots = self.gru(updates.view(-1, 1, self.key_d).contiguous(),       # [b*k, 1, d]
            #                  slots_prev.view(1, -1, self.key_d).contiguous()            # [1, b*k, d]
            #                  )[0]        # out: [b*k, 1, d]
            slots = slots.view(bs, self.n_slots, self.key_d)        # [b, k, d]

            ## slots += MLP(LayerNorm(slots))
            slots = slots + self.mlp(self.ln_output(slots))

            iter_slots.append(slots.detach().clone())
            iter_attn_vis.append(attn_vis.detach().clone())

        return slots, attn_vis, {'slots': iter_slots, 'attns': iter_attn_vis}


def init_tensor(a, b=None, c=None, d=None, ortho=False):
    if b is None:
        p = torch.nn.Parameter(torch.FloatTensor(a), requires_grad=True)
    elif c is None:
        p = torch.nn.Parameter(torch.FloatTensor(a, b), requires_grad=True)
    elif d is None:
        p = torch.nn.Parameter(torch.FloatTensor(a, b, c), requires_grad=True)
    else:
        p = torch.nn.Parameter(torch.FloatTensor(a, b, c, d), requires_grad=True)
    if ortho:
        nn.init.orthogonal_(p)
    elif b is None:         # for bias
        nn.init.constant_(p, 0)
    else:               # for weight
        nn.init.xavier_uniform_(p)
    return p


class Slot2Prompt(nn.Module):
    def __init__(self, emb_d, n_tasks, e_pool_size, e_p_length, e_layers, FPS, key_dim=128, temp=1., mode=1):
        super().__init__()
        self.task_count = 0
        self.emb_d = emb_d
        self.key_d = key_dim
        self.n_tasks = n_tasks

        # prompt basic param
        self.e_pool_size = e_pool_size  # 100
        self.e_p_length = e_p_length    # 8
        self.e_layers = e_layers        # [0, 1, 2, 3, 4, 5]
        self.FPS = FPS          # or can be True all the time?
        self.select_slot_temp = temp
        self.select_prompt_temp = 1.0
        self.selector_mode = None
        if 'attn' in mode:
            self.selector_mode = 'attn'
        elif 'mlp' in mode:
            self.selector_mode = 'mlp'
        elif 'gate' in mode:
            self.selector_mode = 'gate'

        self.cond_mode = None
        if 'sig' in mode:
            self.cond_mode = 'sig'
        elif 'soft' in mode:
            self.cond_mode = 'soft'
        elif 'cos' in mode:
            self.cond_mode = 'cos'
        elif 'avg' in mode:
            self.cond_mode = 'avg'
        elif 'hard' in mode:
            self.cond_mode = 'hard'
        else:
            raise Exception(f'Un-implemented {mode}.')

        if 'ortho' in mode:
            self.ortho = True
        else:
            self.ortho = False

        # if 'coda' in mode:
        #     self.coda = True
        # else:
        #     self.coda = False

        print(f'Initial s2p in mode {self.selector_mode} with cond {self.cond_mode}, FPS {self.FPS}.')

        if self.selector_mode == 'gate' or self.selector_mode == 'mlp':
            self.slot_map = nn.ModuleList([
                # nn.Sequential(nn.Linear(key_dim, key_dim), nn.ReLU(inplace=True), nn.Linear(key_dim, key_dim)),
                nn.Linear(key_dim, 1) if self.selector_mode == 'gate'
                else nn.Linear(key_dim, key_dim),
            ])
            self.prompt_map = nn.ModuleList([
                nn.Sequential(nn.Linear(key_dim, 2*key_dim), nn.ReLU(inplace=True),
                              nn.Linear(2*key_dim, len(self.e_layers) * self.e_p_length * self.emb_d))   # [64 -> 12*8*768]
            ])

            # for k, p in self.s2p.named_parameters():
            #     if 'weight' in k:
            #         nn.init.kaiming_uniform_(p, nonlinearity='linear')
            #     if 'bias' in k:
            #         nn.init.constant_(p, 0)
        elif self.selector_mode == 'attn':
            self.slot_attn_slot_ln = nn.LayerNorm(key_dim)
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
                p = init_tensor(self.e_pool_size, e_l, emb_d, ortho=self.ortho)   # [100, 8, 768]
                k = init_tensor(self.e_pool_size, self.key_d, ortho=self.ortho)   # [100, 128]
                a = init_tensor(self.e_pool_size, self.key_d, ortho=self.ortho)   # [100, 128]
                if self.ortho:
                    p = self.gram_schmidt(p)
                    k = self.gram_schmidt(k)
                    a = self.gram_schmidt(a)
                setattr(self, f'e_p_{e}', p)
                setattr(self, f'e_k_{e}', k)
                setattr(self, f'e_a_{e}', a)

            # task query
            self.slot_attn_task_key = init_tensor(self.key_d)       # [128] or [self.n_tasks, 128]
            self.slot_attn_slot_selection_w = init_tensor(self.key_d, self.key_d)   # [128, 128] or [self.n_tasks, 128, 128]
            self.slot_attn_slot_selection_b = init_tensor(self.key_d)   # [128] or [self.n_tasks, 128]
            # self.slot_ln2 = nn.LayerNorm(key_dim)
        else:
            raise NotImplementedError

    def new_task(self):

        self.task_count += 1

        if not self.FPS:
            if self.selector_mode == 'gate' or self.selector_mode == 'mlp':
                self.slot_map.append(
                    nn.Linear(self.key_d, 1) if self.selector_mode == 'gate'
                    else nn.Linear(self.key_d, self.key_d))
                self.prompt_map.append(
                    nn.Sequential(nn.Linear(self.key_d, 2*self.key_d), nn.ReLU(inplace=True),
                                  nn.Linear(2*self.key_d, len(self.e_layers) * self.e_p_length * self.emb_d)))
            elif self.ortho:
                for e in self.e_layers:
                    K = getattr(self, f'e_k_{e}')
                    P = getattr(self, f'e_p_{e}')
                    A = getattr(self, f'e_a_{e}')
                    A = self.gram_schmidt(A)
                    K = self.gram_schmidt(K)
                    P = self.gram_schmidt(P)
                    setattr(self, f'e_p_{e}', P)
                    setattr(self, f'e_k_{e}', K)
                    setattr(self, f'e_a_{e}', A)

    def forward(self, slots, s2p=None, train=False, phase='new', select_mode='coda'):
        """phase reuse: only use detached old prompts; new: use learnable new prompts"""
        # train control the detach of old K and p
        # slots [bs, n20, h64]
        bs, n, h = slots.shape
        if s2p is None:
            s2p = self

        w = torch.zeros(bs, n).to(slots.device)
        w_slots = None
        selections = None
        if self.selector_mode == 'gate':
            slot_map = s2p.slot_map[-1]          # [self.key_d -> self.key_d] or -> 1
            prompt_map = s2p.prompt_map[-1]      # [self.key_d -> len(self.e_layers) * self.e_p_length * self.emb_d]
            weights = F.sigmoid(slot_map(slots))        # -> [bs, k, 1]
            weighted_slots = torch.sum(weights * slots, dim=1)     # -> [bs, h]
            prompts = prompt_map(weighted_slots).reshape(bs, 1, len(self.e_layers), self.e_p_length, self.emb_d)
            # [bs, 1, e, l, d]
        elif self.selector_mode == 'mlp':       # use dense
            slot_map = s2p.slot_map[-1]          # [self.key_d -> self.key_d] or -> 1
            prompt_map = s2p.prompt_map[-1]      # [self.key_d -> len(self.e_layers) * self.e_p_length * self.emb_d]
            weighted_slots = slot_map(slots)
            weighted_slots = torch.mean(weighted_slots, dim=1)   # mean over K
            prompts = prompt_map(weighted_slots).reshape(bs, 1, len(self.e_layers), self.e_p_length, self.emb_d)
            # [bs, 1, e, l, d]
        else:
            slots = self.slot_attn_slot_ln(slots)  # apply layernorm to alleviate shifting in slots
            avg_slots = slots.mean(dim=1)       # [b, n10]
            # # or min max scale to [-1, 1]
            # slots = slots.reshape(bs*n, h)
            # slots = slots - slots.min(dim=0)[0]  # norm on each axis
            # slots = slots / slots.max(dim=0)[0]
            # slots = slots * (1 - -1) + -1   # from [0, 1] to [-1, 1]
            # slots = slots.reshape(bs, n, h)

            if self.cond_mode in ['sig', 'soft', 'cos', 'hard']:
                # learn to weights slots as inputs to select prompt
                slot_selection_w = self.slot_attn_slot_selection_w  # [128, 128] or [self.n_tasks, 128, 128]
                slot_selection_b = self.slot_attn_slot_selection_b  # [128] or [self.n_tasks, 128]
                # [bs, n10, h128] @ [h128, d128] -> [bs, n10, d128]
                mapped_slots = torch.einsum('bnh,hd->bnd', slots, slot_selection_w)
                mapped_slots = mapped_slots + slot_selection_b
                # mapped_slots = self.slot_ln2(mapped_slots)
                mapped_slots = torch.tanh(mapped_slots)
                task_key = self.slot_attn_task_key  # [128] or [self.n_tasks, 128]
                if self.cond_mode == 'sig':     # sig(1/sqrt(D) S_m@K_t)
                    w = torch.einsum('bnd,d->bn', mapped_slots, task_key)
                    w = w * (task_key.shape[-1] ** -0.5)
                    w = w * self.select_slot_temp
                    w = torch.sigmoid(w)
                elif self.cond_mode == 'hard':
                    w = torch.einsum('bnd,d->bn', mapped_slots, task_key)
                    w = w * (task_key.shape[-1] ** -0.5)
                    w = w * self.select_slot_temp
                    torch.sign(w) + w - w.detach()      # re-param
                elif self.cond_mode == 'soft':      # softmax(1/sqrt(D) S_m@K_t)
                    w = torch.einsum('bnd,d->bn', mapped_slots, task_key)
                    w = w * (task_key.shape[-1] ** -0.5)
                    w = w * self.select_slot_temp
                    w = F.softmax(w, dim=-1)
                elif self.cond_mode == 'cos':   # cos(S_m, K_t)
                    n_m_s = nn.functional.normalize(mapped_slots, dim=-1)
                    n_k_t = nn.functional.normalize(task_key, dim=-1)
                    w = torch.einsum('bnd,d->bn', n_m_s, n_k_t)
                else:
                    raise Exception(f'Un-implemented {self.cond_mode}.')

                # # or cosine sim
                # # normalization
                # n_mapped_slots = nn.functional.normalize(mapped_slots, dim=-1)
                # n_task_key = nn.functional.normalize(task_key, dim=-1)
                # w = torch.einsum('bnd,d->bn', n_mapped_slots, n_task_key)
                # w = w * self.select_slot_temp
                # w = torch.sigmoid(w)

                w_slots = torch.einsum('bnh,bn->bh', mapped_slots, w)  # weighted slots

            prompts = []
            selections = []
            for l in self.e_layers:
                K = getattr(s2p, f'e_k_{l}')  # [100, h]
                A = getattr(s2p, f'e_a_{l}')  # [100, h]
                p = getattr(s2p, f'e_p_{l}')  # [100, 8, 768]
                if s2p.FPS:  # use all prompts
                    s = 0
                    f = self.e_pool_size
                else:       # train task-specific prompts and check phase
                    pt = int(self.e_pool_size / (self.n_tasks))  # 100/10=10
                    s = int(self.task_count * pt)  # 10 prompts for one task
                    f = int((self.task_count + 1) * pt)

                    if phase == 'reuse' and self.task_count > 0:      # only use detached old prompts for new tasks
                        f = s

                # freeze/control past tasks
                if train:
                    if self.task_count > 0:
                        # K = K[0:f]
                        # A = A[0:f]
                        if phase == 'reuse':        # reuse learn prompt selection for new samples
                            K = K[0:f]
                            A = A[0:f]
                        else:
                            K = torch.cat((K[:s].detach().clone(), K[s:f]), dim=0)
                            A = torch.cat((A[:s].detach().clone(), A[s:f]), dim=0)
                        p = torch.cat((p[:s].detach().clone(), p[s:f]), dim=0)
                    else:       # first task, s=0
                        K = K[s:f]
                        A = A[s:f]
                        p = p[s:f]
                else:
                    K = K[0:f]
                    A = A[0:f]
                    p = p[0:f]

                # b = bs, n = 10 (# slots), h=128, d = 768, k = 30 (# prompts), l=8
                # with attention and cosine sim

                # # without A
                # K = nn.functional.normalize(K, dim=-1)
                # slots_ = nn.functional.normalize(slots, dim=-1)
                # aq_k = torch.einsum('bnh,kh->bnk', slots_, K)  # aq_k [bs, n10, k30]

                # # with A
                # # print(f'slots: {slots_.shape}; A: {A.shape}; K: {K.shape}')
                # slots_ = torch.einsum('bnh,kh->bnkh', slots, A)      # attended slots
                # K = nn.functional.normalize(K, dim=-1)
                # slots_ = nn.functional.normalize(slots_, dim=-1)
                # aq_k = torch.einsum('bnkh,kh->bnk', slots_, K)  # aq_k [bs, n10, k30]
                #
                # # apply temp
                # temp = self.select_prompt_temp
                # # aq_k = ((self.key_d ** (-0.5)) * aq_k) * temp
                # aq_k = aq_k * temp
                # # over slot pool, thus each slot sharply select one slot in the pool
                # if select_mode == 'sigmoid':
                #     aq_k = torch.sigmoid(aq_k)
                #     aq_k_repa = aq_k
                # elif select_mode == 'softmax':
                #     aq_k = torch.softmax(aq_k, dim=-1)
                #     if train:
                #         aq_k_repa = aq_k
                #     else:
                #         # Reparametrization trick.
                #         index = aq_k.max(-1, keepdim=True)[1]
                #         aq_k_repa = torch.zeros_like(aq_k, memory_format=torch.legacy_contiguous_format
                #                                      ).scatter_(-1, index, 1.0)
                #         aq_k_repa = aq_k_repa - aq_k.detach() + aq_k
                # else:   # 'coda'
                #     aq_k_repa = aq_k
                #
                # slot-wise selection for prompt using aq_k
                # # aq_k = torch.ones((B, f)).to(p.device)      # just use all prompts with 1; un-condition type
                # P = torch.einsum('bnk,kld->bnld', aq_k_repa, p)   # wei-sum over k -> bnld

                # image-wise selection for prompt
                if 'avg' in self.cond_mode:
                    # use average slots as inputs to select prompt
                    slots_ = torch.einsum('bh,kh->bkh', avg_slots, A)      # attended slots
                    w_slots = avg_slots
                elif self.cond_mode in ['sig', 'soft', 'cos', 'hard']:
                    slots_ = torch.einsum('bh,kh->bkh', w_slots, A)      # attended slots
                else:
                    raise Exception(f'Un-implemented {self.cond_mode}.')

                slots_ = nn.functional.normalize(slots_, dim=-1)
                n_K = nn.functional.normalize(K, dim=-1)
                aq_k = torch.einsum('bkh,kh->bk', slots_, n_K)  # aq_k [bs, k30]
                aq_k = aq_k * self.select_prompt_temp   # 1
                if select_mode == 'sigmoid':
                    aq_k = torch.sigmoid(aq_k)
                elif select_mode == 'softmax':
                    aq_k = torch.softmax(aq_k, dim=-1)
                else:   # 'coda'
                    pass
                P = torch.einsum('bk,kld->bld', aq_k, p)
                P = P.unsqueeze(1)      # make shape consistent: [bld] -> [bnld]
                aq_k = aq_k.unsqueeze(1)      # make shape consistent: [bk] -> [bnk]

                prompts.append(P)
                selections.append(aq_k)
            prompts = torch.stack(prompts, dim=2)       # [bs, n10, e, l, d]   n is num_slots
            selections = torch.stack(selections, dim=2)     # [bs, n10, e, k30]

        # prompt_map = getattr(self, f's2p')  # [h64, e12, p8, d768]
        # # [bs, k20, h64] @ [h64, e12, p8, d768] -> [bs, k20, e12, p8, d768]
        # prompts = torch.einsum('bkh,hepd->bkepd', slots, prompt_map)

        # w: [bs, n10]
        return prompts, selections, w, w_slots

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

    # def prompt_map_init(self, task_id):
    #     for e in self.e_layers:
    #         prompt_map = init_tensor(self.key_d, self.e_p_length, self.emb_d)  # [64, 8, 768]
    #         # setattr(self, f's2p_{task_id}_{e}', prompt_map)       # [bs, 64] @ [64, 8, 768] -> [bs, 8, 768]
    #         setattr(self, f's2p_{task_id}_{e}', prompt_map)

    # def prompt_map_freeze(self, task_id):
    #     for e in self.e_layers:
    #         prompt_map = getattr(self, f's2p_{task_id}_{e}')
    #         prompt_map.requires_grad = False
    #         setattr(self, f's2p_{task_id}_{e}', prompt_map)
