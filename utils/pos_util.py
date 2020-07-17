import torch
import utils.Constants as Constants


def get_sequential_pos(in_seqs):
    bs, l = in_seqs.size()
    device = in_seqs.device
    pos_seqs = torch.arange(1, l + 1).unsqueeze(0).expand(bs, -1).to(device)

    pad_mask = in_seqs.eq(Constants.PAD).to(device)
    pos_seqs = pos_seqs.masked_fill(pad_mask, Constants.PAD)
    return pos_seqs
