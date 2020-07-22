import utils.Constants as Constants
import torch


def get_non_pad_mask(seq):
    assert seq.dim() == 2
    return seq.ne(Constants.PAD).type(torch.float).unsqueeze(-1)


def get_attn_key_pad_mask(seq_k, seq_q):
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(Constants.PAD)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk
    return padding_mask


def prepare_mask(inputs):
    masks = {}
    tokens = inputs['tokens']
    self_mask = get_attn_key_pad_mask(tokens, tokens)
    non_pad_mask = get_non_pad_mask(tokens)
    masks['self_mask'] = self_mask
    masks['non_pad_mask'] = non_pad_mask
    return masks

