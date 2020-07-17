from collections import defaultdict

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
    * batch: list of tuples (in_seq, pos_seq, score_seq, label)
    '''

    word2idx, label2idx, sysact2idx = memory['enc2idx'], memory['label2idx'], memory['sysact2idx']
    act2idx, slot2idx, value2idx = \
        memory['act2idx'], memory['slot2idx'], memory['dec2idx']

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
    label_idx_lists = [
        [label2idx[l] if l in label2idx else Constants.UNK for l in label_list]
        for label_list in label_lists
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

    # filling label map
    labels_map = torch.zeros(len(batch), len(label2idx))
    for i, lbl in enumerate(label_idx_lists):
        for idx in lbl:
            labels_map[i][idx] = 1

    # final processing
    batch_in = torch.LongTensor(batch_in).to(device)
    batch_pos = torch.LongTensor(batch_pos).to(device)
    batch_score = torch.FloatTensor(batch_score).to(device)
    batch_labels = labels_map.float().to(device)

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

    batch_sa = torch.LongTensor(batch_sa).to(device)
    batch_sa_parent = torch.LongTensor(batch_sa_parent).to(device)
    batch_sa_sib = torch.LongTensor(batch_sa_sib).to(device)
    batch_sa_type = torch.LongTensor(batch_sa_type).to(device)

    #################### processing inputs & outputs for hierarchical decoding ####################
    # oov processing
    oov_lists = []
    extend_ids = []
    for seq in in_seqs:
        seq = [Constants.CLS_WORD] * cls + seq
        ids, oov = seq2extend_ids(seq, value2idx)
        extend_ids.append(torch.tensor(ids).view(1, -1).to(device))
        oov_lists.append(oov)

    # act predictor labels
    acts = [[item.strip().split('-')[0] for item in label_list]
        for label_list in label_lists]
    acts_indices = [[act2idx[a] for a in acts_utt]
        for acts_utt in acts]
    acts_map = torch.zeros(len(batch), len(act2idx)).to(device)
    for i, a in enumerate(acts_indices):
        for idx in a:
            acts_map[i][idx] = 1

    # slot predictor inputs and labels
    double_acts_labels = [
        [item for item in label_list if len(item.strip().split('-')) > 1]
        for label_list in label_lists
    ]
    act_slot_dict = [defaultdict(list) for _ in range(len(batch))]
    for i, labels_utt in enumerate(double_acts_labels):
        for l in labels_utt:
            ll = l.strip().split('-')
            act_slot_dict[i][ll[0]].append(ll[1])

    act_inputs = [list(dic.keys()) for dic in act_slot_dict]
    act_inputs = [[act2idx[a] for a in acts_utt] for acts_utt in act_inputs]
    # max_act_len = max([len(a) for a in act_inputs])
    # act_inputs = [acts_utt + (max_act_len - len(acts_utt)) * [Constants.PAD]
    #     for acts_utt in act_inputs]
    # act_inputs = torch.tensor(act_inputs)  # (b, #double_acts)
    act_inputs = [torch.tensor(acts_utt).view(-1, 1).to(device) if len(acts_utt) > 0 else None
        for acts_utt in act_inputs]  # list: batch x tensor(#acts, 1)

    # slots_map = torch.zeros(len(batch), max_act_len, len(slot2idx))
    slots_map = []  # list: batch x tensor(#double_acts, #slots)
    for i, dic in enumerate(act_slot_dict):
        if len(dic) == 0:
            slots_map.append(None)
        else:
            tmp = torch.zeros(len(dic), len(slot2idx)).to(device)
            for j, (a, slots) in enumerate(dic.items()):
                for s in slots:
                    if s in slot2idx:
                        tmp[j][slot2idx[s]] = 1
                    else:
                        tmp[j][Constants.PAD] = 1
            slots_map.append(tmp)

    # value decoder inputs and labels
    triple_acts_labels = [
        [item for item in label_list if len(item.strip().split('-')) > 2]
        for label_list in label_lists
    ]
    asv_dict = [{} for _ in range(len(batch))]
    for i, labels_utt in enumerate(triple_acts_labels):
        for l in labels_utt:
            ll = l.strip().split('-')
            a_s = '-'.join(ll[:2])
            asv_dict[i][a_s] = ll[2]

    act_slot_ids = []  # list: batch x tensor(#triple_act_slots, 2)
    value_inp_ids = []
    value_out_ids = []
    for i, dic in enumerate(asv_dict):
        if len(dic) == 0:
            act_slot_ids.append(None)
            value_inp_ids.append(None)
            value_out_ids.append(None)
        else:
            tmp = []
            tmp_v_inp, tmp_v_out = [], []
            for j, (a_s, value) in enumerate(dic.items()):
                act_slot = a_s.strip().split('-')
                a_id = act2idx[act_slot[0]]
                s_id = slot2idx[act_slot[1]] \
                    if act_slot[1] in slot2idx else Constants.PAD
                tmp.append([a_id, s_id])
                inp_ids = value2ids(value.strip().split(), value2idx)
                out_ids = value2extend_ids(value.strip().split(), value2idx, oov_lists[i])
                tmp_v_inp.append(torch.tensor([Constants.BOS] + inp_ids).view(1, -1).to(device))
                tmp_v_out.append(torch.tensor(out_ids + [Constants.EOS]).to(device))
            act_slot_ids.append(torch.tensor(tmp).to(device))
            value_inp_ids.append(tmp_v_inp)
            value_out_ids.append(tmp_v_out)

    return batch_in, batch_pos, batch_score, \
            batch_sa, batch_sa_parent, batch_sa_sib, batch_sa_type, batch_labels, \
            list(in_seqs), list(sa_seqs), list(label_lists), \
            acts_map, act_inputs, slots_map, act_slot_ids, value_inp_ids, value_out_ids, \
            extend_ids, oov_lists


def seq2extend_ids(lis, word2idx):
    ids = []
    oovs = []
    for w in lis:
        if w in word2idx:
            ids.append(word2idx[w])
        else:
            if w not in oovs:
                oovs.append(w)
            oov_num = oovs.index(w)
            ids.append(len(word2idx) + oov_num)
    return ids, oovs


def value2ids(lis, word2idx):
    ids = []
    for w in lis:
        if w in word2idx:
            ids.append(word2idx[w])
        else:
            ids.append(Constants.UNK)
    return ids


def value2extend_ids(lis, word2idx, oovs):
    ids = []
    for w in lis:
        if w in word2idx:
            ids.append(word2idx[w])
        else:
            if w in oovs:
                ids.append(len(word2idx) + oovs.index(w))
            else:
                ids.append(Constants.UNK)
    return ids


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

