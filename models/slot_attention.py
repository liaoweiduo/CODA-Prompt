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


class SlotAttention(nn.Module):
    def __init__(self, emb_d, n_slots, key_dim=128, n_iter=5):
        super().__init__()
        self.emb_d = emb_d          # emb for representation 768
        self.key_d = key_dim        # emb for slot: Dslot 64

        # slot basic param
        self.n_slots = n_slots  # 5   number of slots
        self.n_iter = n_iter     # T
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
        self.ln_decoder = nn.LayerNorm(self.key_d)
        self.decoder = nn.Sequential(
            nn.Linear(self.key_d, self.key_d * 2, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(self.key_d * 2, self.emb_d, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(self.emb_d, self.emb_d, bias=True),
        )

    def forward(self, features):
        # features: [bs, n196, 768]
        slots, attn = self.forward_slots(features)
        # slots [bs, k20, d64], attn [bs, n196, k20]

        slot_features = self.ln_decoder(slots)
        slot_features = self.decoder(slot_features)     # [bs, k20, 768]
        slot_features = torch.einsum('bkd,bnk->bnd', slot_features, attn)       # [bs, n196, 768]

        # recon loss
        recon_loss = F.mse_loss(slot_features, features)

        return slots, attn, recon_loss

    def forward_slots(self, features):
        # features [bs, 196, 768]
        bs = features.shape[0]

        # init
        features = self.ln_input(features)
        slots = torch.randn(bs, self.n_slots, self.key_d, device=self.log_sigma.device) * torch.exp(self.log_sigma) + self.mu
        # [bs, k, 64]
        attn_vis = None

        # iter
        k = self.k(features)    # [bs, 196, 64]
        v = self.v(features)    # [bs, 196, 64]
        k = (self.key_d ** (-0.5)) * k

        for t in range(self.n_iter):
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

        return slots, attn_vis


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
    else:
        nn.init.xavier_uniform_(p)
    return p


class Slot2Prompt(nn.Module):
    def __init__(self, emb_d, n_tasks, e_pool_size, e_p_length, e_layers, FPS, key_dim=128):
        super().__init__()
        self.task_count = 0
        self.emb_d = emb_d
        self.key_d = key_dim
        self.n_tasks = n_tasks

        # prompt basic param
        self.e_pool_size = e_pool_size  # 100
        self.e_p_length = e_p_length    # 8
        self.e_layers = e_layers        # [0, 1, 2, 3, 4, 5]
        self.FPS = FPS

        self.selector_mode = 'pk'     # [gate, mlp, attn]
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
                k = tensor_prompt(self.e_pool_size, self.key_d)  # [100, 128]
                # a = tensor_prompt(self.e_pool_size, self.key_d)
                p = self.gram_schmidt(p)
                k = self.gram_schmidt(k)
                # a = self.gram_schmidt(a)
                setattr(self, f'e_p_{e}', p)
                setattr(self, f'e_k_{e}', k)
                # setattr(self, f'e_a_{e}', a)
        else:
            raise NotImplementedError

    def new_task(self):
        if not self.FPS:
            if self.selector_mode == 'gate' or self.selector_mode == 'mlp':
                self.slot_map.append(
                    nn.Linear(self.key_d, 1) if self.selector_mode == 'gate'
                    else nn.Linear(self.key_d, self.key_d))
                self.prompt_map.append(
                    nn.Sequential(nn.Linear(self.key_d, 2*self.key_d), nn.ReLU(inplace=True),
                                  nn.Linear(2*self.key_d, len(self.e_layers) * self.e_p_length * self.emb_d)))
            else:
                for e in self.e_layers:
                    K = getattr(self, f'e_k_{e}')
                    P = getattr(self, f'e_p_{e}')
                    k = self.gram_schmidt(K)
                    p = self.gram_schmidt(P)
                    setattr(self, f'e_p_{e}', p)
                    setattr(self, f'e_k_{e}', k)

    def forward(self, slots, s2p=None, train=False):
        # slots [bs, n20, h64]
        bs, n, h = slots.shape
        if s2p is None:
            s2p = self
        slot_map = s2p.slot_map[-1]          # [self.key_d -> self.key_d] or -> 1
        prompt_map = s2p.prompt_map[-1]      # [self.key_d -> len(self.e_layers) * self.e_p_length * self.emb_d]

        if self.selector_mode == 'gate':
            weights = F.sigmoid(slot_map(slots))        # -> [bs, k, 1]
            weighted_slots = torch.sum(weights * slots, dim=1)     # -> [bs, h]
            prompts = prompt_map(weighted_slots).reshape(bs, len(self.e_layers), self.e_p_length, self.emb_d)
            # [bs, e, l, d]
        elif self.selector_mode == 'mlp':       # use dense
            weighted_slots = slot_map(slots)
            weighted_slots = torch.mean(weighted_slots, dim=1)   # mean over K
            prompts = prompt_map(weighted_slots).reshape(bs, len(self.e_layers), self.e_p_length, self.emb_d)
            # [bs, e, l, d]
        else:
            prompts = []
            for l in self.e_layers:
                K = getattr(self, f'e_k_{l}')  # [100, h]
                p = getattr(self, f'e_p_{l}')  # [100, 8, 768]
                if self.FPS:  # use all prompts
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
                        p = torch.cat((p[:s].detach().clone(), p[s:f]), dim=0)
                    else:
                        K = K[s:f]
                        p = p[s:f]
                else:
                    K = K[0:f]
                    p = p[0:f]

                # b = bs, n = 10 (# slots), h=128, d = 768, k = 30 (# prompts), l=8
                # with attention and cosine sim
                n_K = nn.functional.normalize(K, dim=1)
                q = nn.functional.normalize(slots, dim=2)
                aq_k = torch.einsum('bnh,kh->bnk', q, n_K)  # aq_k is cosine similarity [bs, n10, k30]
                # aq_k = torch.ones((B, f)).to(p.device)      # just use all prompts with 1; un-condition type
                P = torch.einsum('bnk,kld->bld', aq_k, p)   # wei-sum over k -> bnld -> sum over n -> bld
                prompts.append(P)
            prompts = torch.stack(prompts, dim=1)       # [bs, e, l, d]

        # prompt_map = getattr(self, f's2p')  # [h64, e12, p8, d768]
        # # [bs, k20, h64] @ [h64, e12, p8, d768] -> [bs, k20, e12, p8, d768]
        # prompts = torch.einsum('bkh,hepd->bkepd', slots, prompt_map)

        return prompts

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
    #         prompt_map = tensor_prompt(self.key_d, self.e_p_length, self.emb_d)  # [64, 8, 768]
    #         # setattr(self, f's2p_{task_id}_{e}', prompt_map)       # [bs, 64] @ [64, 8, 768] -> [bs, 8, 768]
    #         setattr(self, f's2p_{task_id}_{e}', prompt_map)

    # def prompt_map_freeze(self, task_id):
    #     for e in self.e_layers:
    #         prompt_map = getattr(self, f's2p_{task_id}_{e}')
    #         prompt_map.requires_grad = False
    #         setattr(self, f's2p_{task_id}_{e}', prompt_map)


# note - ortho init has not been found to help l2p/dual prompt
def tensor_prompt(a, b=None, c=None, d=None, ortho=False) -> torch.nn.Parameter:
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
    else:
        nn.init.uniform_(p)
    return p
