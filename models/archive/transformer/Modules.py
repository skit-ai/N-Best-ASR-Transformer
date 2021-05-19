import torch
import torch.nn as nn
import numpy as np

__author__ = "Yu-Hsiang Huang"

'''
Modified by chen.liu
@ 2019.09.29

Multiply attention matrix by some weights
'''

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1, rel_pos=False, rel_pos_clip=None, d_k=None, score_util='np', nh=4):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)
        self.rel_pos = rel_pos
        self.rel_pos_clip = rel_pos_clip
        self.score_util = score_util

        if rel_pos:
            # version 1
            self.relpos_emb_k = nn.Embedding(rel_pos_clip * 2 + 1, d_k)
            self.relpos_emb_v = nn.Embedding(rel_pos_clip * 2 + 1, d_k)

            # version 2
            # self.relpos_emb_k = nn.Embedding(4, d_k)
            # self.relpos_emb_v = nn.Embedding(4, d_k)

        if self.score_util == 'pp':
            self.nh = nh
            self.score_lambda = nn.Parameter(torch.randn(nh, 1, 1, 1))  # (1, nh, 1, 1)
            nn.init.normal_(self.score_lambda, 1, 0.1)

    def max_clip(self, x):
        clip = self.rel_pos_clip
        return max(-clip, min(x, clip))

    def max_clip_tensor(self, x):
        clip = torch.tensor([self.rel_pos_clip]).to(x.device)
        return torch.max(-clip, torch.min(x, clip))

    def forward(self, q, k, v, mask=None, attn_w=None, pos_seqs=None):
        '''
        q/k/v: (b*nh, l, d)
        pos_seqs: (b*nh, l)
        '''
        attn = torch.bmm(q, k.transpose(1, 2))
        # print('--attn', attn.max(), attn.min())

        if self.rel_pos:
            bnh, l, d = k.size()
            device = q.device

            # WARNING: too slow !
            # table = torch.tensor([[
            #     [self.max_clip(j - i) + self.rel_pos_clip for j in seq] for i in seq]
            #     for seq in pos_seqs.tolist()
            # ]).to(device)  # (b*nh, l, l); range: [0, 2*clip]

            ##########################################################################
            # version 1: relative position gap with max-clipping
            ##########################################################################
            psa = pos_seqs.unsqueeze(1).expand(-1, l, -1)  # (b*nh, l, l)
            psb = pos_seqs.unsqueeze(2)  # (b*nh, l, 1)
            table = (self.max_clip_tensor(psa - psb) + self.rel_pos_clip).to(device)  # (b*nh, l, l)

            ##########################################################################
            # version 2: adjacency relation, range {-1, 0, 1} -> emb idx{1, 2, 3}
            # idx  relation
            # 0    i and j are not adjacent
            # 1    i is left adjacent to j
            # 2    i and j are in same position
            # 3    i is right adjacent to j
            ##########################################################################
            # psa = pos_seqs.unsqueeze(1).expand(-1, l, -1)  # (b*nh, l, l)
            # psb = pos_seqs.unsqueeze(2)  # (b*nh, l, 1)
            # table = psa.eq(psb - 1) * 1 + psa.eq(psb) * 2 + psa.eq(psb + 1) * 3
            # table = table.long().to(device)


            e_k = self.relpos_emb_k(table)  # (b*nh, l, l, d)

            # rel_attn_k = torch.tensor(np.einsum('ijm, ijkm -> ijk', q, e_k)).to(device)  # (b*nh, l, l)
            rel_attn_k = torch.einsum('ijm, ijkm -> ijk', q, e_k)  # (b*nh, l, l)
            attn += rel_attn_k

        # should be before masking
        if attn_w is not None:
            assert attn_w.size() == attn.size()
            if self.score_util == 'pp':
                bnh, l1, l2 = attn_w.size()
                bs = int(bnh / self.nh)
                attn_w = attn_w.view(self.nh, bs, l1, l2)
                posterior_scores = torch.mul(self.score_lambda, attn_w.data)
                attn = attn + posterior_scores.contiguous().view(bnh, l1, l2)
            elif self.score_util == 'mul':
                attn = attn * attn_w  # element-wise multiplication
            elif self.score_util == 'np':
                attn = attn + attn_w  # element-wise plus
            elif self.score_util == 'none':
                pass

        attn = attn / self.temperature
        # print('--attn_t', attn.max(), attn.min())

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)
            # print('--attn_m', attn.max(), attn.min(), attn[-1])
            # input()

        attn = self.softmax(attn)
        # print('--attn_sm', attn.max(), attn.min())
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)  # (b*nh, l, d)

        if self.rel_pos:
            e_v = self.relpos_emb_v(table)  # (b*nh, l, l, d)
            # rel_attn_v = torch.tensor(np.einsum('ijk, ijkm -> ijm', attn, e_v)).to(device)  # (b*nh, l, d)
            rel_attn_v = torch.einsum('ijk, ijkm -> ijm', attn, e_v)  # (b*nh, l, d)
            output += rel_attn_v

        return output, attn

