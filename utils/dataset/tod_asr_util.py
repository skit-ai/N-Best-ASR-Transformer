import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd

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
            inp, lbl = line.strip('\n\r').split('\t<=>\t')
            inp_list = inp.strip().split(' ')
            in_seqs.append(inp_list)
            if len(lbl) == 0:
                labels.append([])
            else:
                labels.append(lbl.strip().split(';'))

    return in_seqs, labels


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

    in_seqs, label_lists = zip(*batch)

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
    batch_labels = labels_map.float().to(device)

    return batch_labels,list(in_seqs), list(label_lists)


class WCN_Dataset(Dataset):
    def __init__(self, data):
        super(WCN_Dataset, self).__init__()
        self.in_seqs, self.labels = data

    def __len__(self):
        return len(self.in_seqs)

    def __getitem__(self, index):
        in_seq = self.in_seqs[index]
        label = self.labels[index]
        return in_seq,label



def observability_lens(eic, epoch, dataset_type, output_dir, extra_name):

    total_length = len(raw_inputs)
    epochs_list = [epoch]*total_length
    dataset_type_list = [dataset_type]*total_length
    mean_loss_list = [eic.mean_loss]*total_length
    precision_list = [eic.precision]*total_length
    recall_list = [eic.recall]*total_length
    f1_list = [eic.f1]*total_length
    acc_list = [eic.acc]*total_length

    epoch_df = pd.DataFrame(
        list(zip(epochs_list, dataset_type_list, mean_loss_list, precision_list, recall_list, f1_list, acc_list, eic.raw_inputs, eic.whole_pred_classes, eic.true_golds)),
        columns=["epoch", "dataset", "mean_loss", "precision", "recall", "f1", "acc", "raw_inputs", "pred_classes", "gold"]
        )

    epoch_df.to_csv(os.path.join(output_dir, f"epoch_{epoch}_observe_{extra_name}.csv"), index=False)


class EpochInfoCollector:

    def __init__(
        self, 
        raw_inputs, whole_pred_classes, true_golds,
        mean_loss, precision, recall, f1, acc
        ):
        self.raw_inputs = raw_inputs
        self.whole_pred_classes = whole_pred_classes
        self.true_golds = true_golds 
        self.mean_loss = mean_loss
        self.precision = precision
        self.recall = recall
        self.f1 = f1
        self.acc = acc

