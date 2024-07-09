import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torchvision.models as models
from torch.autograd import Variable
from .vit import VisionTransformer
from .slot_attention import SlotAttention
import numpy as np
import copy


class SlotDecoder(nn.Module):
    def __init__(self, emb_d, n_tasks, prompt_param, key_dim=64):
        super().__init__()
        self.task_count = 0
        self.emb_d = emb_d          # emb for representation
        self.key_d = key_dim        # emb for slot: Dslot
        self.n_tasks = n_tasks

        # slot basic param
        self.slot_attn = SlotAttention(emb_d, n_slots=int(prompt_param[0]))
        # 5   number of slots

        # last
        self.out_dim = int(prompt_param[1])       # 21 dim before classifier
        self.outmlp = nn.Linear(self.key_d, self.out_dim, bias=True)

    def process_task_count(self):
        self.task_count += 1

    def forward(self, features, train=False, task_id=None, **kwargs):
        # features [bs, 196, 768]
        slots = self.slot_attn(features)

        # outmlp
        out = self.outmlp(slots)        # [b, k, 21]
        out = F.relu(out)

        return out, slots


def init_tensor(a, b=None, c=None, ortho=False):
    if b is None:
        p = torch.nn.Parameter(torch.FloatTensor(a), requires_grad=True)
    elif c is None:
        p = torch.nn.Parameter(torch.FloatTensor(a, b), requires_grad=True)
    else:
        p = torch.nn.Parameter(torch.FloatTensor(a, b, c), requires_grad=True)
    if ortho:
        nn.init.orthogonal_(p)
    else:
        nn.init.uniform_(p)
    return p


class ViTDecoder(nn.Module):
    def __init__(self, num_classes=10, pt=False, flag='None', params=None, **kwargs):
        super(ViTDecoder, self).__init__()

        # get last layer
        self.flag = flag
        self.task_id = None
        self.pen_dim = int(params[1][1])     # change emb dim from 768 to pen dim

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

        # create decoder module
        if self.flag == 'slot':
            self.decoder = SlotDecoder(768, params[0], params[1], key_dim=64)
        else:
            self.decoder = None

        # classifier
        if self.decoder is not None:
            self.last = nn.Linear(self.pen_dim, num_classes)
        else:
            self.last = nn.Linear(768, num_classes)

        # feature encoder changes if transformer vs resnet
        self.feat = zoo_model

        # freeze feature encoder
        for param in self.feat.parameters():
            param.requires_grad = False

    # pen: get penultimate features    
    def forward(self, x, register_blk=-1, task_id=None, pen=False, train=False,
                cond_x=None, **kwargs):
        # kwargs for decoder
        if task_id is None:
            task_id = self.task_id

        with torch.no_grad():
            if cond_x is not None:  # condition model
                out, _, _ = self.feat(cond_x, register_blk=register_blk)
            else:
                out, _, _ = self.feat(x, register_blk=register_blk)
            features = out[:, 1:, :]    # [bs, 196, 768]
            cls_token = out[:, 0, :]    # [bs, 768]

        decoder_out = None
        if self.decoder is not None:
            out, slots = self.decoder(features, train=train, **kwargs)
            # out [bs, k5, 21]  slot [bs, k5, d64]

            if not pen:
                out = torch.sum(out, dim=1)     # [bs, 21]
                out = self.last(out)            # [bs, 100]

        else:       # use vit as feature extractor and last as classifier
            # out = out.view(out.size(0), -1)
            out = cls_token
            if not pen:
                out = self.last(out)            # [bs, 100]
        if train:
            return out, cls_token
        else:
            return out


def vit_decoder_imnet(out_dim, block_division=None, prompt_flag='None', prompt_param=None, **kwargs):
    return ViTDecoder(num_classes=out_dim, pt=True, flag=prompt_flag, params=prompt_param, **kwargs)
