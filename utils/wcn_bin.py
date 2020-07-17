import torch
import torch.nn as nn
import torch.nn.functional as F


def bin_merger(enc_out, pos_seqs, score_seqs, src_score_scaler=None):
    '''
    merge word-level outputs into bin-level representation
    - enc_out: (b, seq, dm)
    - pos_seqs: (b, seq)  0: PAD
    - score_seqs: (b, seq) or None(1best)
    - score_scaler: (b, seq) -1: PAD
    '''

    device = enc_out.device
    b, l, d = enc_out.size()
    if score_seqs is not None:
        if src_score_scaler is not None:
            score_seqs *= src_score_scaler
        enc_out = enc_out * score_seqs.unsqueeze(2)  # multiplied by score in advance

    # version 1
    # batch_pos_indices = [
    #     [seq.eq(i).nonzero().view(-1) for i in range(1, max(seq)+1)]
    #     for seq in pos_seqs
    # ]  # b x [[...] [...] [...]]

    # bin_outs = [
    #     torch.stack([enc_out[i].index_select(dim=0, index=idx).mean(0)
    #         for idx in batch_pos_indices[i]])
    #     for i in range(b)
    # ]  # b x tensor(seq' x dm)

    # lens = [t.size(0) for t in bin_outs]
    # max_len = max(lens)

    # # padding zeros: (maxlen - seq') x d
    # # final: (b, maxlen, d)
    # final = torch.stack([torch.cat((t, torch.zeros(max_len-t.size(0), d).to(device)), 0)
    #     for t in bin_outs])

    # version 2
    M = torch.zeros(b, pos_seqs.max() + 1, l).to(device)  # (b, seq'+1, l)
    M[torch.arange(b).unsqueeze(1).expand(-1, l), pos_seqs, torch.arange(l)] = 1
    # M = F.normalize(M, p=1, dim=2)
    final = torch.bmm(M, enc_out)[:, 1:, :]  # (b, seq'+1, l) x (b, l, d) -> (b, seq'+1, d) -> ...
    lens = [max(s) for s in pos_seqs.tolist()]

    return final, lens


def length_reorder(tensor, lens):
    '''
    reorder by descending order of lens, for LSTM input
    tensor: (b, seq, d)
    lens: (b, ), list
    '''
    device = tensor.device
    lens = torch.LongTensor(lens)
    sorted_lens, sort_indices = torch.sort(lens, descending=True)
    sorted_tensor = torch.index_select(tensor, dim=0, index=sort_indices.to(device))
    sorted_lens = sorted_lens.tolist()

    return sorted_tensor, sorted_lens, sort_indices


def length_order_back(sorted_tensor, sort_indices):
    '''
    order back
    sorted_tensor: (b, seq, d')
    sort_indices: (b, )
    '''
    ori_tensor = torch.empty_like(sorted_tensor)
    ori_tensor[sort_indices] = sorted_tensor

    return ori_tensor
