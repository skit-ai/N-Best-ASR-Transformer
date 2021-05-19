''' Define the sublayers in encoder/decoder layer '''
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from models.transformer.Modules import ScaledDotProductAttention

__author__ = "Yu-Hsiang Huang"

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1, rel_pos=False, rel_pos_clip=None, ex_mask=None,
            rm_qkv=True, rm_fc=True):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.rm_qkv = rm_qkv
        self.rm_fc = rm_fc

        if self.rm_qkv or self.rm_fc:
            assert d_model == n_head * d_k == n_head * d_v

        if not self.rm_qkv:
            self.w_qs = nn.Linear(d_model, n_head * d_k)
            self.w_ks = nn.Linear(d_model, n_head * d_k)
            self.w_vs = nn.Linear(d_model, n_head * d_v)

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5),
            rel_pos=rel_pos, rel_pos_clip=rel_pos_clip, d_k=d_k, nh=n_head)
        self.layer_norm = nn.LayerNorm(d_model)

        if not self.rm_fc:
            self.fc = nn.Linear(n_head * d_v, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None, pos_seqs=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        if not self.rm_qkv:
            q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
            k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
            v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
        else:
            q = q.view(sz_b, len_q, n_head, d_k)
            k = k.view(sz_b, len_k, n_head, d_k)
            v = v.view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n*b) x lv x dv

        if pos_seqs is not None:
            assert pos_seqs.size() == (sz_b, len_q)
            pos_seqs = pos_seqs.repeat(n_head, 1)  # (n*b) x lq

        mask = mask.repeat(n_head, 1, 1) # (n*b) x .. x ..
        output, attn = self.attention(q, k, v, mask=mask, pos_seqs=pos_seqs)
        # print('-o', output.max(), output.min())

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)

        if not self.rm_fc:
            output = self.dropout(self.fc(output))
        else:
            output = self.dropout(output)

        output = self.layer_norm(output + residual)

        return output, attn

class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, 1) # position-wise
        self.w_2 = nn.Conv1d(d_hid, d_in, 1) # position-wise
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output
