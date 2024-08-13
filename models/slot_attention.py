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
    def __init__(self, emb_d, n_slots, key_dim=64, n_iter=3, e_p_length=8, e_layers=None):
        super().__init__()
        if e_layers is None:
            e_layers = [0, 1, 2, 3, 4]
        self.emb_d = emb_d          # emb for representation 768
        self.key_d = key_dim        # emb for slot: Dslot 64

        # slot basic param
        self.n_slots = n_slots  # 5   number of slots
        self.n_iter = n_iter     # T
        self.attn_epsilon = 1e-8
        self.gru_d = self.key_d
        self.e_p_length = e_p_length
        self.e_layers = e_layers

        # slot related modules
        self.ln_input = nn.LayerNorm(self.emb_d)
        self.ln_slot = nn.LayerNorm(self.key_d)
        self.ln_output = nn.LayerNorm(self.key_d)
        self.mu = init_tensor(self.key_d)             # slot gaussian distribution mu
        self.sigma = init_tensor(self.key_d)          # slot gaussian distribution sigma
        self.k = nn.Linear(emb_d, key_dim, bias=False)
        self.q = nn.Linear(key_dim, key_dim, bias=False)
        self.v = nn.Linear(emb_d, key_dim, bias=False)
        self.gru = nn.GRU(self.key_d, self.gru_d, batch_first=True)
        # self.gru = nn.GRUCell(self.key_d, self.gru_d)
        self.mlp = nn.Sequential(
            nn.Linear(self.key_d, self.key_d, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(self.key_d, self.key_d, bias=True)
        )

        prompt_map = init_tensor(self.key_d, len(self.e_layers), self.e_p_length, self.emb_d)  # [64, 12,  8, 768]
        # # [bs, 64] @ [64, 12, 8, 768] -> [bs, 12, 8, 768]
        setattr(self, f's2p', prompt_map)

    def forward(self, features, return_slots=False):
        # features [bs, 196, 768]
        bs = features.shape[0]

        # init
        features = self.ln_input(features)
        slots = torch.randn(bs, self.n_slots, self.key_d, device=self.sigma.device) * self.sigma + self.mu
        # [bs, k, 64]

        # iter
        for t in range(self.n_iter):
            slots_prev = slots.clone()
            slots = self.ln_slot(slots)
            k = self.k(features)    # [bs, 196, 64]
            q = self.q(slots)       # [bs, k, 64]
            v = self.v(features)    # [bs, 196, 64]

            # b = bs, n = 196, k = 5, d = 64
            ## softmax(KQ^T/sqrt(d), dim='slots')
            # sum((b x n x 1 x d) * [b x 1 x k x d]) = (b x n x k)
            attn = torch.einsum('bnd,bkd->bnk', k, q)
            attn = attn * (self.key_d ** -0.5)
            # softmax over slots
            attn = F.softmax(attn, dim=-1)

            ## updates = WeightedMean(attn+epsilon, v)
            attn = attn + self.attn_epsilon
            attn = attn / torch.sum(attn, dim=-2, keepdim=True)
            # sum((b x n x k x 1) * (b x n x 1 x d)) = (b x k x d)
            updates = torch.einsum('bnk,bnd->bkd', attn, v)

            ## slots = GRU(state=slots_prev[b,k,d], inputs=updates[b,k,d])  (for each slot)
            slots = self.gru(updates.view(-1, 1, self.key_d).contiguous(),       # [b*k, 1, d]
                             slots_prev.view(1, -1, self.key_d).contiguous()            # [1, b*k, d]
                             )[0]        # out: [b*k, 1, d]
            slots = slots.view(bs, self.n_slots, self.key_d)        # [b, k, d]

            ## slots += MLP(LayerNorm(slots))
            slots = slots + self.mlp(self.ln_output(slots))

        prompts = self.slot2prompt(slots)

        if return_slots:
            return prompts, slots
        return prompts

    def slot2prompt(self, slots):
        prompt_map = getattr(self, f's2p')  # [h64, e12, p8, d768]
        # [bs, k20, h64] @ [h64, e12, p8, d768] -> [bs, k20, e12, p8, d768]
        prompts = torch.einsum('bkh,hepd->bkepd', slots, prompt_map)

        return prompts


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
        nn.init.uniform_(p)
    return p
