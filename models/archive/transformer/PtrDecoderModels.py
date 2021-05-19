''' Define the Transformer model '''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import utils.Constants as Constants
from models.transformer.Layers import EncoderLayer, DecoderLayer

__author__ = "Yu-Hsiang Huang"

'''
Modified by chen.liu
@ 2019.09.24

To handle the case where position_idx > max_pos
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


def get_cross_attn_w(score_seq, len_tgt):
    bs, len_src = score_seq.size()
    attn_w = score_seq.unsqueeze(1).expand(-1, len_tgt, -1)  # b x len_tgt x len_src
    return attn_w


class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self,
            n_src_vocab, len_max_seq, d_word_vec,
            n_layers, n_head, d_k, d_v,
            d_model, d_inner, dropout=0.1):

        super().__init__()

        self.n_position = len_max_seq + 1
        self.d_word_vec = d_word_vec

        self.src_word_emb = nn.Embedding(
            n_src_vocab, d_word_vec, padding_idx=Constants.PAD)

        self.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(self.n_position, d_word_vec, padding_idx=0),
            freeze=True)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, src_seq, src_pos, return_attns=False):

        enc_slf_attn_list = []

        # -- Prepare masks
        slf_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=src_seq)
        non_pad_mask = get_non_pad_mask(src_seq)

        # -- Forward
        # enc_output = self.src_word_emb(src_seq) + self.position_enc(src_pos)
        enc_output = self.src_word_emb(src_seq) + self.position_enc_wrapper(src_pos)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output,

    def position_enc_wrapper(self, inp):
        ''' combine both regular and irregular cases '''

        device = inp.device

        # inp: (batch, seq)
        exceeded_pos_mask = (inp >= self.n_position)

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
            d_model, d_inner, dropout=0.1,
            with_ptr=True, d_ext_fea=0,
            dec_score_util=None, decoder_tied=False):

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

        self.n_head = n_head

        # external features
        self.d_ext_fea = d_ext_fea
        self.ext_fea_proj = nn.Linear(d_model + d_ext_fea, d_model)
        self.non_lin = nn.ReLU()

        self.tgt_word_prj = nn.Linear(d_model, n_tgt_vocab)
        self.dropout_layer = nn.Dropout(dropout)

        # pointer mechanism
        self.with_ptr = with_ptr
        if with_ptr:
            self.pointer_lin = nn.Linear(d_model * 3, 1)
            self.attn_head_weight = nn.Parameter(torch.ones(n_head))

        # score utilities in decoder
        self.dec_score_util = dec_score_util

        if decoder_tied:
            self.tgt_word_prj.weight = self.tgt_word_emb.weight

    def forward(self, tgt_seq, tgt_pos, src_seq, src_score, enc_output, ext_fea=None, return_attns=False,
            extra_zeros=None, extend_idx=None):
        '''
            specially for ONE input uttrance
            - batch_size (bs) here means the number of triples
            - len_tgt: len of values, often very small (around 2)
            - len_src/len_src': len of source seq with/without padding
        '''

        bs, len_tgt = tgt_seq.size()
        len_src = src_seq.size(1)
        no_pad_len_src = extend_idx.size(1)
        n_head = self.n_head

        dec_slf_attn_list, dec_enc_attn_list = [], []

        # -- Prepare masks
        non_pad_mask = get_non_pad_mask(tgt_seq)

        slf_attn_mask_subseq = get_subsequent_mask(tgt_seq)
        slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=tgt_seq, seq_q=tgt_seq)
        slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)

        dec_enc_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=tgt_seq)

        # -- Forward
        dec_output = self.tgt_word_emb(tgt_seq) + self.position_enc(tgt_pos)  # (bs, len_tgt, d_model)

        raw_dec_inp = dec_output.clone()
        if ext_fea is not None:  # ext_fea -> (bs, 1, d_ext)
            assert ext_fea.size(-1) == self.d_ext_fea
            ext_fea = ext_fea.expand(-1, dec_output.size(1), -1)
            dec_output = self.ext_fea_proj(self.dropout_layer(torch.cat([dec_output, ext_fea], dim=2)))
            dec_output = self.non_lin(dec_output)

        # re-scaled decoder-encoder attention
        if self.dec_score_util != 'none':
            attn_w = get_cross_attn_w(src_score, len_tgt)
        else:
            attn_w = None

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn, self_ctx, cross_ctx = dec_layer(
                dec_output, enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask,
                dec_enc_attn_mask=dec_enc_attn_mask,
                dec_enc_attn_w=attn_w)
            # dec_slf_attn -> (bs * nhead, len_tgt, len_tgt)
            # dec_enc_attn -> (bs * nhead, len_tgt, len_src)
            # self_ctx -> (bs, len_tgt, d_model)
            # cross_ctx -> (bs, len_tgt, d_model)

            if return_attns:
                dec_slf_attn_list += [dec_slf_attn]
                dec_enc_attn_list += [dec_enc_attn]

        # classifier  # TODO: add pointer
        seq_logit = self.tgt_word_prj(self.dropout_layer(dec_output))
        scores = torch.softmax(seq_logit, dim=-1)  # (bs, len_tgt, dec_vocab_size)
        if self.with_ptr:
            pointer_input = torch.cat([raw_dec_inp, self_ctx, cross_ctx], dim=-1)
            pointer_input = self.dropout_layer(pointer_input)
            pointer_ratio = torch.sigmoid(self.pointer_lin(pointer_input))  # (bs, len_tgt, 1)

            last_dec_enc_attn = dec_enc_attn_list[-1].view(n_head, bs, len_tgt, len_src)
            attn_head_weight = F.softmax(self.attn_head_weight, dim=-1).view(n_head, 1, 1, 1)
            attn_dist = (attn_head_weight * last_dec_enc_attn).sum(0)  # (bs, len_tgt, len_src)

            vocab_dist_v = pointer_ratio * scores  # (bs, len_tgt, dec_vocab_size)
            attn_dist_v = (1 - pointer_ratio) * attn_dist  # (bs, len_tgt, len_src)
            attn_dist_v = attn_dist_v[:, :, :no_pad_len_src]  # (bs, len_tgt, len_src')

            if extra_zeros is not None:
                extra_zeros = extra_zeros.unsqueeze(0).expand(bs, len_tgt, -1)  # (1, #oov) -> (bs, len_tgt, #oov)
                vocab_dist_v = torch.cat([vocab_dist_v, extra_zeros], -1)

            extend_idx = extend_idx.unsqueeze(1).expand(bs, len_tgt, -1)  # (bs, len_tgt, len_src')
            final_dist = vocab_dist_v.scatter_add(2, extend_idx, attn_dist_v)
            final_dist = torch.log(final_dist + 1e-12)  # (bs, len_tgt, dec_vocab_size + #oov)
        else:
            final_dist = torch.log(scores + 1e-12)

        if return_attns:
            return final_dist, dec_slf_attn_list, dec_enc_attn_list
        return final_dist,

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
