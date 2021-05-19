import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, q_dim, kv_dim, dropout, device, attn_type):
        super(Attention, self).__init__()
        self.device = device
        self.q_dim = q_dim
        self.kv_dim = kv_dim
        self.attn_type = attn_type

        if attn_type == 'mlp':
            self.Wa = nn.Linear(q_dim + kv_dim, kv_dim, bias=False)
            self.Va = nn.Linear(kv_dim, 1, bias=False)
        elif attn_type == 'dot':
            assert q_dim == kv_dim
        elif attn_type == 'general':
            self.Wa = nn.Linear(q_dim, kv_dim, bias=False)
        self.dropout_layer = nn.Dropout(p=dropout)
        self.init_weight()

    def init_weight(self, init_range=0.2):
        if self.attn_type == 'mlp':
            nn.init.uniform_(self.Wa.weight, a=-init_range, b=init_range)
            nn.init.uniform_(self.Va.weight, a=-init_range, b=init_range)
        elif self.attn_type == 'general':
            nn.init.uniform_(self.Wa.weight, a=-init_range, b=init_range)

    def forward(self, q_seq, kv_seq, mask=None):
        '''
        q_seq: (b, ql, qd)
        kv_seq: (b, kvl, kvd)
        '''

        if self.attn_type == 'mlp':
            ql = q_seq.size(1)
            kvl = kv_seq.size(1)
            q_seq_ = q_seq.unsqueeze(2).expand(-1, -1, kvl, -1)  # (b, ql, kvl, qd)
            kv_seq_ = kv_seq.unsqueeze(1).expand(-1, ql, -1, -1)  # (b, ql, kvl, kvd)
            e = self.Wa(torch.cat((q_seq_, kv_seq_), dim=3))  # (b, ql, kvl, kvd)
            e = self.Va(torch.tanh(e)).squeeze(3)  # (b, ql, kvl)
        elif self.attn_type == 'dot':
            e = torch.bmm(q_seq, kv_seq.transpose(1, 2))  # (b, ql, kvl)
        elif self.attn_type == 'general':
            q_seq_ = self.Wa(q_seq)  # (b, ql, kvd)
            e = torch.bmm(q_seq_, kv_seq.transpose(1, 2))  # (b, ql, kvl)

        # mask = [[0] * l + [1] * (max(seq_lens) - l) for l in seq_lens]
        # mask = torch.ByteTensor(mask).to(self.device)
        if mask is not None:
            e.masked_fill_(mask, -float('inf'))
        attn = F.softmax(e, dim=-1)
        attn = self.dropout_layer(attn)  # (b, ql, kvl)

        ctx = torch.bmm(attn, kv_seq)  # (b, ql, kvd)

        return ctx, attn


class SimpleSelfAttention(nn.Module):
    def __init__(self, dim, dropout, device):
        super(SimpleSelfAttention, self).__init__()
        self.device = device
        self.dim = dim

        self.U = nn.Linear(dim, dim)

        self.Wa = nn.Linear(dim, 1, bias=False)
        self.dropout_layer = nn.Dropout(p=dropout)
        self.init_weight()

    def init_weight(self, init_range=0.2):
        nn.init.uniform_(self.Wa.weight, a=-init_range, b=init_range)

    def forward(self, seq, seq_lens):
        '''
        seq: (b, l, d)
        out: (b, d)
        '''

        # bug!
        # e = self.Wa(torch.tanh(seq)).squeeze(2)  # (b, l, 1) -> (b, l)

        proj = self.U(seq)  # (b, l, d)
        e = self.Wa(torch.tanh(proj)).squeeze(2)  # (b, l, 1) -> (b, l)

        mask = [[0] * l + [1] * (max(seq_lens) - l) for l in seq_lens]
        mask = torch.ByteTensor(mask).to(self.device)
        e.masked_fill_(mask, -float('inf'))
        e = F.softmax(e, dim=1)
        e = self.dropout_layer(e)  # (b, l)

        ctx = torch.bmm(e.unsqueeze(1), seq).squeeze(1)  # (b, 1, d) -> (b, d)

        return ctx
