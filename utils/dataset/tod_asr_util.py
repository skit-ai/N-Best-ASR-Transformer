import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from tabulate import tabulate

import os
import utils.Constants as Constants


def _get_stratified_sampled_data(asr_in_seqs,trans_in_seqs,labels,coverage):
    data = pd.DataFrame({"asr_in_seqs":asr_in_seqs,
                        "trans_in_seqs":trans_in_seqs,
                        "labels":labels})
    total_sample_count = data.shape[0]

    data["labels_tuple"] = data.labels.apply(lambda x: tuple(x))   
    unique_data = data.drop_duplicates(subset=['labels_tuple'], keep='first')

    
    unique_sample_count = unique_data.shape[0]

    print("coverage",coverage)
    print("total_sample_count",total_sample_count)
    print("unique_sample_count",unique_sample_count)
    
    rem_sample_count = int(np.round(abs((float(coverage)*total_sample_count) - unique_sample_count)))
    data = data[~data.isin(unique_data)].dropna()

    print("rem_sample_count",rem_sample_count)
    
    rem_data = data.sample(n = rem_sample_count, random_state = 42).reset_index(drop=True)
    
    sampled_data = pd.concat([unique_data, rem_data], ignore_index=True)

    
    print("sampled_data",sampled_data.shape)
    return sampled_data
    


def read_wcn_data(fn,coverage=None):
    '''
    * fn: wcn data file name
    * line format - word:parent:sibling:type ... \t<=>\tword:pos:score word:pos:score ... \t<=>\tlabel1;label2...
    * system act <=> utterance <=> labels
    '''
    asr_in_seqs = []
    trans_in_seqs = []
    labels = []
    with open(fn, 'r') as fp:
        lines = fp.readlines()
        for line in lines:
            asr_inp,trans_inp,lbl = line.strip('\n\r').split('\t<=>\t')
            asr_inp_list = asr_inp.strip().split(' ')
            trans_inp_list = trans_inp.strip().split(' ')
            asr_in_seqs.append(asr_inp_list)
            trans_in_seqs.append(trans_inp_list)
            if len(lbl) == 0:
                labels.append([])
            else:
                labels.append(lbl.strip().split(';'))

    if coverage:
        sampled_data = _get_stratified_sampled_data(asr_in_seqs,trans_in_seqs,labels,coverage)
        asr_in_seqs = sampled_data.asr_in_seqs.values
        trans_in_seqs = sampled_data.trans_in_seqs.values
        labels = sampled_data.labels.values
        
    return asr_in_seqs,trans_in_seqs,labels


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

    in_seqs,trans_in_seqs,label_lists = zip(*batch)

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

    return batch_labels,list(in_seqs),list(trans_in_seqs),list(label_lists)


class WCN_Dataset(Dataset):
    def __init__(self, data):
        super(WCN_Dataset, self).__init__()
        self.in_seqs,self.trans_in_seqs,self.labels = data

    def __len__(self):
        return len(self.in_seqs)

    def __getitem__(self, index):
        in_seq = self.in_seqs[index]
        trans_in_seq = self.trans_in_seqs[index]
        label = self.labels[index]
        return in_seq,trans_in_seq,label


def classification_report(df):
    """
    assuming no duplicates in the gold and prediction
    also skipping out values where it exists in prediction and not in ground truth (hierarchial)
    """
    
    unique_ground_truths = df["gold"].explode().unique()
    set_ground_truths = set(unique_ground_truths)
    
    y_true_labels = {label: [] for label in set_ground_truths}
    y_pred_labels = {label: [] for label in set_ground_truths}
    
    for idx, row in df.iterrows():
        
        set_gold = set(row["gold"])
        set_pred = set(row["pred_classes"])
        
        for label in set_gold:
            if label in set_pred:
                y_true_labels[label].append(1)
                y_pred_labels[label].append(1)
            else:
                y_true_labels[label].append(1)
                y_pred_labels[label].append(0)
        
        
        for label in ((set_pred - set_gold) & set_ground_truths):
            y_true_labels[label].append(0)
            y_pred_labels[label].append(1)
            
    metrics_accrued = {}
    
    for label in set_ground_truths:
        
        precision, recall, fscore, _ = precision_recall_fscore_support(
            y_true_labels[label], y_pred_labels[label], average="binary", zero_division=0
        )
                
        support = y_true_labels[label].count(1)
        
        metrics_accrued[label] = (round(precision, 2), round(recall, 2), round(fscore, 2), support)
            
    # only python 3.7+
    metrics_accrued = dict(sorted(metrics_accrued.items()))

    table = [[label, precision, recall, fscore, support] for label,(precision, recall, fscore, support) in metrics_accrued.items()]
    headers = ["label", "precision", "recall", "f1-score", "support"]

    return tabulate(table, headers)



def observability_lens(eic, epoch, dataset_type, output_dir, extra_name):

    total_length = len(eic.raw_inputs)
    epochs_list = [epoch]*total_length
    dataset_type_list = [dataset_type]*total_length
    mean_loss_list = [eic.mean_loss]*total_length
    precision_list = [eic.precision]*total_length
    recall_list = [eic.recall]*total_length
    f1_list = [eic.f1]*total_length
    acc_list = [eic.acc]*total_length

    epoch_df = pd.DataFrame(
        list(zip(epochs_list, dataset_type_list, mean_loss_list, precision_list, recall_list, f1_list, acc_list, eic.raw_inputs, eic.whole_pred_classes, eic.true_golds, eic.matches)),
        columns=["epoch", "dataset", "mean_loss", "precision", "recall", "f1", "acc", "raw_inputs", "pred_classes", "gold", "matches"]
        )

    epoch_df.to_csv(os.path.join(output_dir, f"epoch_{epoch}_for_{dataset_type}_observe_{extra_name}.csv"), index=False)

    metrics_string = classification_report(epoch_df)

    with open(os.path.join(output_dir, f"classification_report_epoch_{epoch}_for_{dataset_type}.txt"), "w") as fp:
        fp.write(metrics_string)


class EpochInfoCollector:

    def __init__(
        self, 
        raw_inputs, whole_pred_classes, true_golds, matches,
        mean_loss, precision, recall, f1, acc
        ):
        self.raw_inputs = raw_inputs
        self.whole_pred_classes = whole_pred_classes
        self.true_golds = true_golds
        self.matches = matches
        self.mean_loss = mean_loss
        self.precision = precision
        self.recall = recall
        self.f1 = f1
        self.acc = acc

