import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

import utils.Constants as Constants


def read_wcn_data(fn):
    '''
    * fn: wcn data file name
    * line format - word:parent:sibling:type ... \t<=>\tword:pos:score word:pos:score ... \t<=>\tlabel1;label2...
    * system act <=> utterance <=> labels
    '''
    in_seqs = []
    pos_seqs = []
    score_seqs = []
    sa_seqs = []
    sa_parent_seqs = []
    sa_sib_seqs = []
    sa_type_seqs = []
    labels = []
    with open(fn, 'r') as fp:
        lines = fp.readlines()
        for line in lines:
            sa, inp, lbl = line.strip('\n\r').split('\t<=>\t')
            inp_list = inp.strip().split(' ')
            in_seq, pos_seq, score_seq = zip(*[item.strip().split(':') for item in inp_list])
            in_seqs.append(list(in_seq))
            pos_seqs.append(list(map(int, pos_seq)))
            score_seqs.append(list(map(float, score_seq)))
            sa_list = sa.strip().split(' ')
            sa_seq, pa_seq, sib_seq, ty_seq = zip(*[item.strip().split(':') for item in sa_list])
            sa_seqs.append(list(sa_seq))
            sa_parent_seqs.append(list(map(int, pa_seq)))
            sa_sib_seqs.append(list(map(int, sib_seq)))
            sa_type_seqs.append(list(map(int, ty_seq)))

            if len(lbl) == 0:
                labels.append([])
            else:
                labels.append(lbl.strip().split(';'))

    return in_seqs, pos_seqs, score_seqs, \
        sa_seqs, sa_parent_seqs, sa_sib_seqs, sa_type_seqs, labels


def prepare_wcn_dataloader(data, memory, batch_size, max_seq_len, device, shuffle_flag=False):
    dataset = WCN_Dataset(data)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        collate_fn=lambda batch, memory=memory, maxsl=max_seq_len, device=device: \
            collate_fn(batch, memory, max_seq_len, device)
    )

    return dataloader


def collate_fn(batch, memory, maxsl, device):
    '''
    * batch: list of tuples (in_seq, pos_seq, score_seq, sa_seq, sa_parent_seq, sa_sib_seq, sa_type_seq, label)
    '''

    word2idx, label2idx, sysact2idx = memory['word2idx'], memory['label2idx'], memory['sysact2idx']

    #################### processing utterances ####################
    # add <cls> at the beginning of seq
    cls = True

    # cut seq that is too long
    if maxsl is not None:
        batch = [
            (item[0][:maxsl], item[1][:maxsl], item[2][:maxsl], item[3], item[4], item[5], item[6], item[7])
            for item in batch
        ]
    in_seqs, pos_seqs, score_seqs, sa_seqs, sa_parent_seqs, sa_sib_seqs, sa_type_seqs, label_lists = zip(*batch)

    max_len = max(len(item[0]) for item in batch)

    in_idx_seqs = [
        [word2idx[w] if w in word2idx else Constants.UNK for w in seq]
        for seq in in_seqs
    ]

    # padding seqs
    batch_in = np.array([
        [Constants.CLS] * cls +  seq + [Constants.PAD] * (max_len - len(seq))
        for seq in in_idx_seqs
    ])

    # if cls: pos of <cls> = 1; others plus 1
    # else: seq remains unchanged
    batch_pos = np.array([
        [1] * cls + [p + int(cls) for p in seq] + [Constants.PAD] * (max_len - len(seq))
        for seq in pos_seqs
    ])

    batch_score = np.array([
        [1] * cls + seq + [-1] * (max_len - len(seq))
        for seq in score_seqs
    ])

    #################### processing system acts ####################

    max_sa_len = max(len(item[3]) for item in batch)

    sa_idx_seqs = [
        [sysact2idx[w] if w in sysact2idx else Constants.UNK for w in seq]
        for seq in sa_seqs
    ]
    batch_sa = [
        seq + [Constants.PAD] * (max_sa_len - len(seq))
        for seq in sa_idx_seqs
    ]

    # parent & sibling & type: padded with -2
    batch_sa_parent = [
        seq + [-2] * (max_sa_len - len(seq))
        for seq in sa_parent_seqs
    ]
    batch_sa_sib = [
        seq + [-2] * (max_sa_len - len(seq))
        for seq in sa_sib_seqs
    ]
    batch_sa_type = [
        seq + [-2] * (max_sa_len - len(seq))
        for seq in sa_type_seqs
    ]

    #################### processing labels ####################
    label_idx_lists = [
        [label2idx[l] if l in label2idx else Constants.UNK for l in label_list]
        for label_list in label_lists
    ]
    # filling label map
    labels_map = torch.zeros(len(batch), len(label2idx))
    for i, lbl in enumerate(label_idx_lists):
        for idx in lbl:
            labels_map[i][idx] = 1

    # final processing
    batch_in = torch.LongTensor(batch_in).to(device)
    batch_pos = torch.LongTensor(batch_pos).to(device)
    batch_score = torch.FloatTensor(batch_score).to(device)
    batch_sa = torch.LongTensor(batch_sa).to(device)
    batch_sa_parent = torch.LongTensor(batch_sa_parent).to(device)
    batch_sa_sib = torch.LongTensor(batch_sa_sib).to(device)
    batch_sa_type = torch.LongTensor(batch_sa_type).to(device)
    batch_labels = labels_map.float().to(device)

    return batch_in, batch_pos, batch_score, \
            batch_sa, batch_sa_parent, batch_sa_sib, batch_sa_type, batch_labels, \
            list(in_seqs), list(sa_seqs), list(label_lists)


class WCN_Dataset(Dataset):
    def __init__(self, data):
        super(WCN_Dataset, self).__init__()
        self.in_seqs, self.pos_seqs, self.score_seqs, \
            self.sa_seqs, self.sa_parent_seqs, self.sa_sib_seqs, \
            self.sa_type_seqs, self.labels = data

    def __len__(self):
        return len(self.in_seqs)

    def __getitem__(self, index):
        in_seq = self.in_seqs[index]
        pos_seq = self.pos_seqs[index]
        score_seq = self.score_seqs[index]
        sa_seq = self.sa_seqs[index]
        sa_parent_seq = self.sa_parent_seqs[index]
        sa_sib_seq = self.sa_sib_seqs[index]
        sa_type_seq = self.sa_type_seqs[index]
        label = self.labels[index]
        return in_seq, pos_seq, score_seq, sa_seq, sa_parent_seq, sa_sib_seq, sa_type_seq, label



