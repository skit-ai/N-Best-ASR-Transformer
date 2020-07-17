''' Define the Transformer model '''
import torch
import torch.nn as nn
import numpy as np
import utils.Constants as Constants
from models.transformer.Layers import EncoderLayer, DecoderLayer

__author__ = "Yu-Hsiang Huang"

'''
Modified by chen.liu
@ 2019.09.29

* Handle the case where position_idx > max_pos
* Add score utilities
'''


def get_non_pad_mask(seq):
    assert seq.dim() == 2
    return seq.ne(Constants.PAD).type(torch.float).unsqueeze(-1)


def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    ''' Sinusoid position encoding table '''

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    return torch.FloatTensor(sinusoid_table)


def get_sinusoid_encoding(pos, d_hid):
    ''' Get sinusoid encoding vector for a certain value '''

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    vec = np.array(get_posi_angle_vec(pos))
    vec[0::2] = np.sin(vec[0::2])
    vec[1::2] = np.cos(vec[1::2])

    return torch.FloatTensor(vec)


def get_attn_key_pad_mask(seq_k, seq_q):
    ''' For masking out the padding part of key sequence. '''

    # Expand to fit the shape of key query attention matrix.
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(Constants.PAD)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk

    return padding_mask


def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''

    sz_b, len_s = seq.size()
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls

    return subsequent_mask


def get_score_indices(score, n_part):
    ''' score: (batch, seq) tensor'''
    pad_mask = score.lt(0)
    zero_p_mask = score.eq(0)
    score_indices = torch.ceil(score * n_part).long()  # {1, 2, ..., n_part}
    score_indices.masked_fill_(pad_mask, Constants.PAD)
    score_indices.masked_fill_(zero_p_mask, 1)
    return score_indices


def get_slf_attn_w(score_seq, pos_seq, score_util):
    assert score_seq.size() == pos_seq.size()
    b, l = score_seq.size()
    attn_w = score_seq.unsqueeze(1).expand(-1, l, -1)  # b x l x l

    if score_util in ['scl_attn', 'enc_scl_attn']:
        ### version 1: divide attention weight by a scaling factor
        # psa = pos_seq.unsqueeze(1).expand(-1, l, -1)  # (b*nh, l, l)
        # psb = pos_seq.unsqueeze(2)  # (b*nh, l, 1)
        # pos_gap = torch.abs(psa - psb)  # >= 0
        # # scale = torch.log(pos_gap.float() + 1) + 1  # l+log(gap+1) range: [1, +infy)
        # scale = pos_gap.float() + 1  # gap+1
        # attn_w = attn_w / scale
        ### version 2: mask attention
        psa = pos_seq.unsqueeze(1).expand(-1, l, -1)  # (b*nh, l, l)
        psb = pos_seq.unsqueeze(2)  # (b*nh, l, 1)
        mask = psa.ne(psb).float()
        attn_w = attn_w * mask

    return attn_w


def get_pos_adj_mask(pos_seq):
    ''' mask matrix showing the neighboring relation '''
    b, l = pos_seq.size()
    adj_mask = torch.zeros(b, l, l).to(pos_seq.device)

    for i, seq in enumerate(pos_seq):
        for j in range(l):
            if seq[j] == Constants.PAD:
                continue
            adj_idxs = (seq == seq[j] + 1).nonzero().squeeze(1).tolist() + [j]
            adj_mask[i][adj_idxs, j] = 1

    # expand lower-triangular matrix to a full one
    adj_mask += adj_mask.transpose(1, 2)

    return adj_mask


class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self,
            n_src_vocab, len_max_seq, d_word_vec,
            n_layers, n_head, d_k, d_v,
            d_model, d_inner, dropout=0.1, score_util=None,
            pos_emb=False, rel_pos=False, rel_pos_clip=None,
            ex_mask=None):

        super().__init__()

        self.n_position = len_max_seq + 1
        self.d_word_vec = d_word_vec
        self.score_util = score_util
        self.pos_emb = pos_emb

        self.score_util_enc = (score_util is not None) and ('enc' in score_util)
        self.score_util_attn = (score_util is not None) and ('attn' in score_util)
        self.score_util_catenc = (score_util is not None) and ('catenc' in score_util)

        self.src_word_emb = nn.Embedding(
            n_src_vocab, d_word_vec, padding_idx=Constants.PAD)

        self.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(self.n_position, d_word_vec, padding_idx=0),
            freeze=True)

        # if score_util in ['enc', 'enc_attn', 'enc_scl_attn', 'catenc', 'catenc_attn']:
        if self.score_util_enc:
            # self.n_part = 100
            self.n_part = 10
            # self.n_part = 5
            self.score_enc = nn.Embedding(
                self.n_part + 1, d_word_vec, padding_idx=Constants.PAD)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout,
                rel_pos=rel_pos, rel_pos_clip=rel_pos_clip, ex_mask=ex_mask)
            for _ in range(n_layers)])

    def forward(self, inputs, slf_attn_mask=None, non_pad_mask=None, return_attns=False):

        enc_slf_attn_list = []

        src_seq, src_pos, src_score = \
            inputs['tokens'], inputs['positions'], inputs['scores']

        # -- Prepare masks
        if slf_attn_mask is None:
            slf_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=src_seq)
        if non_pad_mask is None:
            non_pad_mask = get_non_pad_mask(src_seq)

        # -- Forward
        enc_output = self.src_word_emb(src_seq)
        if self.pos_emb:
            enc_output += self.position_enc_wrapper(src_pos)

        # applying scores to an additional encoding layer
        if self.score_util_enc:
            score_indices = get_score_indices(src_score, self.n_part)
            score_emb = self.score_enc(score_indices)
            if self.score_util_catenc:
                enc_output = torch.cat((enc_output, score_emb), dim=-1)
            else:
                enc_output += self.score_enc(score_indices)

        # applying scores in attention weights
        if self.score_util_attn:
            slf_attn_w = get_slf_attn_w(src_score, src_pos, self.score_util)
            # slf_attn_w = get_slf_attn_w(src_score) * get_pos_adj_mask(src_pos)
        else:
            slf_attn_w = None

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask,
                slf_attn_w=slf_attn_w,
                pos_seqs=src_pos)
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output

    def position_enc_wrapper(self, inp):
        ''' combine both regular and irregular cases '''

        device = inp.device

        # inp: (batch, seq)
        exceeded_pos_mask = inp.ge(self.n_position)

        # shortcut: all entries are valid (mask all zero)
        if not exceeded_pos_mask.any():
            return self.position_enc(inp)

        # valid_inp: if pos < n_position, keep the value; else change to 0
        valid_inp = inp.masked_fill(exceeded_pos_mask, 0)  # out-of-place assignment
        valid_pos_emb = self.position_enc(valid_inp)

        # exceeded_inp: select all position indices and store as a 1-D tensor (suppose size=n)
        exceeded_inp = inp.masked_select(exceeded_pos_mask).tolist()  # 1-D tensor
        exceeded_pos_emb = torch.stack(
            [get_sinusoid_encoding(pos, self.d_word_vec) for pos in exceeded_inp]).to(device)  # (n, emb)

        valid_pos_emb[exceeded_pos_mask] = exceeded_pos_emb  # (batch, seq, emb)

        return valid_pos_emb


class Decoder(nn.Module):
    ''' A decoder model with self attention mechanism. '''

    def __init__(
            self,
            n_tgt_vocab, len_max_seq, d_word_vec,
            n_layers, n_head, d_k, d_v,
            d_model, d_inner, dropout=0.1):

        super().__init__()
        n_position = len_max_seq + 1

        self.tgt_word_emb = nn.Embedding(
            n_tgt_vocab, d_word_vec, padding_idx=Constants.PAD)

        self.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(n_position, d_word_vec, padding_idx=0),
            freeze=True)

        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, tgt_seq, tgt_pos, src_seq, enc_output, return_attns=False):

        dec_slf_attn_list, dec_enc_attn_list = [], []

        # -- Prepare masks
        non_pad_mask = get_non_pad_mask(tgt_seq)

        slf_attn_mask_subseq = get_subsequent_mask(tgt_seq)
        slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=tgt_seq, seq_q=tgt_seq)
        slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)

        dec_enc_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=tgt_seq)

        # -- Forward
        dec_output = self.tgt_word_emb(tgt_seq) + self.position_enc(tgt_pos)

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output, enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask,
                dec_enc_attn_mask=dec_enc_attn_mask)

            if return_attns:
                dec_slf_attn_list += [dec_slf_attn]
                dec_enc_attn_list += [dec_enc_attn]

        if return_attns:
            return dec_output, dec_slf_attn_list, dec_enc_attn_list
        return dec_output,

class Transformer(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(
            self,
            n_src_vocab, n_tgt_vocab, len_max_seq,
            d_word_vec=512, d_model=512, d_inner=2048,
            n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1,
            tgt_emb_prj_weight_sharing=True,
            emb_src_tgt_weight_sharing=True):

        super().__init__()

        self.encoder = Encoder(
            n_src_vocab=n_src_vocab, len_max_seq=len_max_seq,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            dropout=dropout)

        self.decoder = Decoder(
            n_tgt_vocab=n_tgt_vocab, len_max_seq=len_max_seq,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            dropout=dropout)

        self.tgt_word_prj = nn.Linear(d_model, n_tgt_vocab, bias=False)
        nn.init.xavier_normal_(self.tgt_word_prj.weight)

        assert d_model == d_word_vec, \
        'To facilitate the residual connections, \
         the dimensions of all module outputs shall be the same.'

        if tgt_emb_prj_weight_sharing:
            # Share the weight matrix between target word embedding & the final logit dense layer
            self.tgt_word_prj.weight = self.decoder.tgt_word_emb.weight
            self.x_logit_scale = (d_model ** -0.5)
        else:
            self.x_logit_scale = 1.

        if emb_src_tgt_weight_sharing:
            # Share the weight matrix between source & target word embeddings
            assert n_src_vocab == n_tgt_vocab, \
            "To share word embedding table, the vocabulary size of src/tgt shall be the same."
            self.encoder.src_word_emb.weight = self.decoder.tgt_word_emb.weight

    def forward(self, src_seq, src_pos, tgt_seq, tgt_pos):

        tgt_seq, tgt_pos = tgt_seq[:, :-1], tgt_pos[:, :-1]

        enc_output, *_ = self.encoder(src_seq, src_pos)
        dec_output, *_ = self.decoder(tgt_seq, tgt_pos, src_seq, enc_output)
        seq_logit = self.tgt_word_prj(dec_output) * self.x_logit_scale

        return seq_logit.view(-1, seq_logit.size(2))
