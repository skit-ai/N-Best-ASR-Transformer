import os
import sys
import json
import re
import time
import random
import argparse
from datetime import timedelta

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from transformers import BertTokenizer,BertModel,RobertaTokenizer,RobertaModel,XLMRobertaTokenizer, XLMRobertaModel, get_linear_schedule_with_warmup
from transformers.optimization import AdamW
from transformers import *
install_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(install_path)

from utils.util import make_logger, get_exp_dir_bert
from utils.fscore import update_f1, compute_f1
from utils.dataset.tod_asr_util import read_wcn_data, prepare_wcn_dataloader, observability_lens, EpochInfoCollector
from utils.gpu_selection import auto_select_gpu
from utils.bert_xlnet_inputs import prepare_inputs_for_bert_xlnet_one_seq, prepare_inputs_for_bert_xlnet,prepare_inputs_for_bert_xlnet_seq_base, prepare_inputs_for_bert_xlnet_seq_ids,prepare_inputs_for_roberta
from utils.pos_util import get_sequential_pos
from utils.mask_util import prepare_mask
from utils.STC_util import convert_labels, reverse_top2bottom, onehot_to_scalar
from models.model import make_model
from models.optimization import BertAdam
import utils.Constants as Constants


MODEL_CLASSES = {
    "bert": (BertModel,BertTokenizer,'bert-base-uncased'),
    "roberta": (RobertaModel,RobertaTokenizer,'roberta-base'),
    "xlm-roberta": (XLMRobertaModel,XLMRobertaTokenizer,'xlm-roberta-base'),
}

def parse_arguments():
    parser = argparse.ArgumentParser()

    ######################### model structure #########################
    parser.add_argument('--emb_size', type=int, default=256, help='word embedding dimension')
    parser.add_argument('--hidden_size', type=int, default=512, help='hidden layer dimension')
    parser.add_argument('--max_seq_len', type=int, default=None, help='max sequence length')
    parser.add_argument('--n_layers', type=int, default=6, help='#transformer layers')
    parser.add_argument('--n_head', type=int, default=4, help='#attention heads')
    parser.add_argument('--d_k', type=int, default=64, help='dimension of k in attention')
    parser.add_argument('--d_v', type=int, default=64, help='dimension of v in attention')
    parser.add_argument('--score_util', default='pp', choices=['none', 'np', 'pp', 'mul'],
        help='how to utilize scores in Transformer & BERT: np-naiveplus; pp-paramplus')
    parser.add_argument('--sent_repr', default='bin_sa_cls',
        choices=['cls', 'maxpool', 'attn', 'bin_lstm', 'bin_sa', 'bin_sa_cls', 'tok_sa_cls'],
        help='sentence level representation')
    parser.add_argument('--cls_type', default='stc', choices=['nc', 'tf_hd', 'stc'], help='classifier type')

    ######################### data & vocab #########################
    parser.add_argument('--dataset', required=True, help='<domain>')
    parser.add_argument('--dataroot', required=True, help='path to dataset')
    parser.add_argument('--train_file', default='train', help='base file name of train dataset')
    parser.add_argument('--valid_file', default='valid', help='base file name of valid dataset')
    parser.add_argument('--test_file', default='test', help='base file name of test dataset')
    parser.add_argument('--ontology_path', default=None, help='ontology')

    ######################## pretrained model (BERT) ########################
    parser.add_argument('--bert_model_name', default='bert-base-uncased',
        choices=['bert-base-uncased', 'bert-base-cased', 'bert-large-uncased', 'bert-large-cased'])
    parser.add_argument('--fix_bert_model', action='store_true')

    ######################### training & testing options #########################
    parser.add_argument('--testing', action='store_true', help=' test your model (default is training && testing)')
    parser.add_argument('--deviceId', type=int, default=-1, help='train model on ith gpu. -1:cpu, 0:auto_select')
    parser.add_argument('--random_seed', type=int, default=999, help='initial random seed')
    parser.add_argument('--l2', type=float, default=0, help='weight decay')
    parser.add_argument('--dropout', type=float, default=0., help='dropout rate at each non-recurrent layer')
    parser.add_argument('--bert_dropout', type=float, default=0.1, help='dropout rate for BERT')
    parser.add_argument('--batchSize', type=int, default=16, help='training batch size')
    parser.add_argument('--max_norm', type=float, default=5.0, help='threshold of gradient clipping (2-norm)')
    parser.add_argument('--max_epoch', type=int, default=50, help='max number of epochs to train')
    parser.add_argument('--experiment', default='exp', help='experiment directories for storing models and logs')
    parser.add_argument('--optim_choice', default='bertadam', choices=['adam', 'adamw', 'bertadam'], help='optimizer choice')
    parser.add_argument('--lr', default=5e-4, type=float, help='learning rate')
    parser.add_argument('--bert_lr', default=1e-5, type=float, help='learning rate for bert')
    parser.add_argument('--warmup_proportion', type=float, default=0.1, help='warmup propotion')
    parser.add_argument('--init_type', default='uf', choices=['uf', 'xuf', 'normal'], help='init type')
    parser.add_argument('--init_range', type=float, default=0.2, help='init range, for naive uniform')

    ######################## system act #########################
    parser.add_argument('--with_system_act', action='store_true', help='whether to include the last system act')

    ####################### Loss function setting ###############

    parser.add_argument('--add_l2_loss',type=bool,default=False , help='whether to add l2 loss between pure and asr transcripts')

    ###################### Pre-trained model config ##########################


    parser.add_argument('--pre_trained_model',help = 'pre-trained model name to use among bert,roberta,xlm-roberta')
    parser.add_argument('--tod_pre_trained_model',help = 'tod_pre_trained model checkpoint path')

    opt = parser.parse_args()
    
    ######################### option verification & adjustment #########################
    # device definition
    if opt.deviceId >= 0:
        if opt.deviceId > 0:
            opt.deviceId, gpu_name, valid_gpus = auto_select_gpu(assigned_gpu_id=opt.deviceId - 1)
        elif opt.deviceId == 0:
            opt.deviceId, gpu_name, valid_gpus = auto_select_gpu()
        print('Valid GPU list: %s ; GPU %d (%s) is auto selected.' % (valid_gpus, opt.deviceId, gpu_name))
        torch.cuda.set_device(opt.deviceId)
        opt.device = torch.device('cuda')
    else:
        print('CPU is used.')
        opt.device = torch.device('cpu')

    # random seed set
    random.seed(opt.random_seed)
    np.random.seed(opt.random_seed)
    torch.manual_seed(opt.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(opt.random_seed)

    # d_model: just equals embedding size
    opt.d_model = opt.emb_size

    # ontology
    opt.ontology = None if opt.ontology_path is None else \
        json.load(open(opt.ontology_path))

    return opt


def cal_ce_loss(bottom_scores_dict, batch_labels, top2bottom_dict, opt):
    ce_losses = []
    lin_keys = [k for k, v in top2bottom_dict.items() if len(v) > 1]
    for k in lin_keys:
        k_str = 'lin_%s' % k
        bottom_indices = top2bottom_dict[k]
        bottom_labels = batch_labels[:, bottom_indices]
        bottom_scalar_labels = onehot_to_scalar(bottom_labels)
        bottom_scores = bottom_scores_dict[k_str]
        bottom_scores = torch.log(bottom_scores + 1e-12)
        ce_loss = opt.ce_loss_function(bottom_scores, bottom_scalar_labels)
        ce_losses.append(ce_loss)
    return sum(ce_losses) / len(ce_losses)


def cal_total_loss(top_scores, bottom_scores_dict, batch_preds, batch_labels, memory, opt,asr_hidden_state=None,transcription_hidden_state=None):
    batch_size= top_scores.size(0)

    loss_record = 0.
    total_loss = 0.

    # MSE loss 
    if opt.add_l2_loss and (asr_hidden_state is not None) and (transcription_hidden_state is not None):
        mse_loss = opt.mse_loss_function(asr_hidden_state,transcription_hidden_state)
        loss_record += mse_loss.item() / batch_size
        print("MSE loss",mse_loss.item())
        total_loss += mse_loss

    # bottom-label BCE loss
    bottom_loss_flag = True
    if bottom_loss_flag:
        bottom_loss = opt.class_loss_function(batch_preds, batch_labels)
        loss_record += bottom_loss.item() / batch_size
        total_loss += bottom_loss

    # top-label BCE loss
    top_loss_flag = True
    if top_loss_flag:
        batch_top_labels = convert_labels(batch_labels, memory['bottom2top_mat'])
        top_loss = opt.class_loss_function(top_scores, batch_top_labels)
        loss_record += top_loss.item() / batch_size
        total_loss += top_loss

    # bottom-label CE loss for each top-label
    ce_loss_flag = True
    if ce_loss_flag:
        ce_loss = cal_ce_loss(bottom_scores_dict, batch_labels,
            memory['top2bottom_dict'], opt)
        loss_record += ce_loss.item() / batch_size
        total_loss += ce_loss

    return loss_record, total_loss


def pred_one_sample(i, ts, bottom_scores_dict, memory, opt):
    # ts: top scores
    # i: index of the sample in a batch
    pred_classes = []
    top_ids = [j for j, p in enumerate(ts) if p > 0.5]
    for ti in top_ids:
        bottom_ids = memory['top2bottom_dict'][ti]
        if len(bottom_ids) == 1:
            pred_classes.append(memory['idx2label'][bottom_ids[0]])
        else:
            bs = bottom_scores_dict['lin_%d' % ti][i]
            lbl_idx_in_vector = bs.data.cpu().numpy().argmax(axis=-1)
            real_lbl_idx = bottom_ids[lbl_idx_in_vector]
            lbl = memory['idx2label'][real_lbl_idx]
            if not lbl.endswith('NONE'):
                pred_classes.append(lbl)

    return pred_classes


def filter_informative(labels, ontology):
    # filter tuples by whether they are informative according to ontology
    new_labels = []
    for lbl in labels:
        tup = lbl.split('-')
        if len(tup) == 3 :
            act, slot, value = tup
            if slot == "this" or (slot in ontology["informable"] and len(ontology["informable"][slot]) > 1) :
                new_labels.append(lbl)
        else :
            new_labels.append(lbl)
    return new_labels


def train_epoch(model, data, opt, memory):
    '''Epoch operation in training phase'''

    model.train()
    opt.optimizer.zero_grad()

    TP, FP, FN = 0, 0, 0
    corr, tot = 0, 0
    losses = []

    for step, batch in enumerate(data):
        # prepare data
        batch_labels,raw_in,raw_trans_in,raw_labels = batch

        # prepare inputs for BERT/XLNET
        inputs = {}
         #pretrained_inputs,input_lens=prepare_inputs_for_bert_xlnet_seq_base(raw_in,opt.tokenizer,device=opt.device)
        input_ids,input_lens=prepare_inputs_for_roberta(raw_in,opt.tokenizer,device=opt.device,opt)
        trans_input_ids,trans_input_lens=prepare_inputs_for_roberta(raw_trans_in,opt.tokenizer,device=opt.device)
        # forward
        top_scores, bottom_scores_dict, batch_preds,asr_hidden_rep,trans_hidden_rep = model(input_ids,trans_input_ids,classifier_input_type="transcript")
        # top_scores -> (batch, #top_classes)
        # batch_preds -> (batch, #bottom_classes)  # not used in this case
        # bottom_scores_dict -> 'lin_i': (batch, #bottom_classes_per_top_label)
        #        in which 'i' is the index of top-label

        # backward
        loss_record, total_loss = cal_total_loss(top_scores, bottom_scores_dict, batch_preds, batch_labels, memory, opt,asr_hidden_rep,trans_hidden_rep)
        losses.append(loss_record)
        total_loss.backward()

        if (step + 1) % opt.n_accum_steps == 0:
            # clip gradient
            if opt.optim_choice.lower() != 'bertadam' and opt.max_norm > 0:
                params = list(filter(lambda p: p.requires_grad, list(model.parameters())))
                torch.nn.utils.clip_grad_norm_(params, opt.max_norm)

            # update parameters
            if opt.optim_choice.lower() in ['adam', 'bertadam']:
                opt.optimizer.step()
            elif opt.optim_choice.lower() == 'adamw':
                opt.optimizer.step()
                opt.scheduler.step()

            # clear gradients
            opt.optimizer.zero_grad()

        # calculate performance
        for i, (ts, gold) in enumerate(zip(top_scores.tolist(), raw_labels)):
            pred_classes = pred_one_sample(i, ts, bottom_scores_dict, memory, opt)
            TP, FP, FN = update_f1(pred_classes, gold, TP, FP, FN)
            tot += 1
            if set(pred_classes) == set(gold):
                corr += 1

    mean_loss = np.mean(losses)
    p, r, f = compute_f1(TP, FP, FN)
    acc = corr / tot * 100

    return mean_loss, (p, r, f), acc


def eval_epoch(model, data, opt, memory, fp, efp):
    '''Epoch operation in evaluating phase'''

    model.eval()

    # sake of observability
    raw_inputs = []
    whole_pred_classes = []
    true_golds = []
    matches = []

    TP, FP, FN = 0, 0, 0
    corr, tot = 0, 0
    losses = []

    all_cases = []
    err_cases = []
    utt_id = 0

    for j, batch in enumerate(data):
        # prepare data
        batch_labels,raw_in,raw_trans_in,raw_labels = batch

        # prepare inputs for BERT/XLNET
        inputs = {}
        input_ids,input_lens=prepare_inputs_for_roberta(raw_in,opt.tokenizer,device=opt.device)
        trans_input_ids,trans_input_lens=prepare_inputs_for_roberta(raw_trans_in,opt.tokenizer,device=opt.device)
        # forward
        top_scores, bottom_scores_dict, batch_preds,asr_hidden_rep,trans_hidden_rep = model(input_ids)
        
        #top_scores, bottom_scores_dict, batch_preds = model(inputs, masks, return_attns=False)
        loss, _ = cal_total_loss(top_scores, bottom_scores_dict, batch_preds, batch_labels, memory, opt)
        losses.append(loss)

        # calculate performance
        batch_pred_classes = []
        batch_ids = []

        for i, (ts, gold, raw) in enumerate(zip(top_scores.tolist(), raw_labels, raw_in)):
            pred_classes = pred_one_sample(i, ts, bottom_scores_dict, memory, opt)

            # ontology filter
            if opt.ontology is not None:
                pred_classes = filter_informative(pred_classes, opt.ontology)
                gold = filter_informative(gold, opt.ontology)

            TP, FP, FN = update_f1(pred_classes, gold, TP, FP, FN)

            tot += 1
            if set(pred_classes) == set(gold):
                corr += 1

            batch_pred_classes.append(pred_classes)

            batch_ids.append(utt_id)
            utt_id += 1

            # keep intermediate results
            res_info = '%s\t<=>\t%s\t<=>\t%s\n' % (
                ' '.join(raw), ';'.join(pred_classes), ';'.join(gold))
            fp.write(res_info)
            if set(pred_classes) != set(gold):
                efp.write(res_info)
                err_cases.append((raw, pred_classes, gold))
            all_cases.append((raw, pred_classes, gold))

            # noting inputs, labels and predictions
            raw_inputs.append(" ".join(raw))
            whole_pred_classes.append(pred_classes)
            true_golds.append(gold)
            matches.append(True if set(pred_classes) == set(gold) else False)

    mean_loss = np.mean(losses)
    p, r, f = compute_f1(TP, FP, FN)
    try:
        acc = corr / tot * 100
    except:
        acc = 0    

    # err_analysis(err_cases)

    # collecting overall useful values
    eic = EpochInfoCollector(raw_inputs, whole_pred_classes, true_golds, matches, mean_loss, p, r, f, acc)


    if opt.testing:
        return mean_loss, (p, r, f), acc, all_cases, eic
    else:
        return mean_loss, (p, r, f), acc, eic


def train(model, train_dataloader, valid_dataloader, test_dataloader, opt, memory):
    '''Start training'''

    logger = make_logger(os.path.join(opt.exp_dir, 'log.train'))
    t0 = time.time()
    logger.info('Training starts at %s' % (time.asctime(time.localtime(time.time()))))
    export_csv_model_name = "tod_asr_bert_stc"

    best = {'epoch': 0, 'vf': 0., 'tef': 0.}

    for i in range(opt.max_epoch):
        # evaluating train set
        start = time.time()
        train_loss, (trp, trr, trf), tr_acc = train_epoch(model, train_dataloader, opt, memory)
        logger.info('[Train]\tEpoch: %02d\tTime: %.2f\tLoss: %.2f\t(p/r/f): (%.2f/%.2f/%.2f)\tAcc: %.2f' %
            (i, time.time()-start, train_loss, trp, trr, trf, tr_acc))

        # evaluating valid set
        with open(os.path.join(opt.exp_dir, 'valid.iter%d'%i), 'w') as fp, \
                open(os.path.join(opt.exp_dir, 'valid.iter%d.err'%i), 'w') as efp:
            start = time.time()
            valid_loss, (vp, vr, vf), v_acc, v_eic = eval_epoch(model, valid_dataloader, opt, memory, fp, efp)
            logger.info('[Valid]\tEpoch: %02d\tTime: %.2f\tLoss: %.2f\t(p/r/f): (%.2f/%.2f/%.2f)\tAcc: %.2f' %
                (i, time.time()-start, valid_loss, vp, vr, vf, v_acc))
            observability_lens(v_eic, i, "valid", opt.exp_dir, export_csv_model_name)

        # evaluating test set
        with open(os.path.join(opt.exp_dir, 'test.iter%d'%i), 'w') as fp, \
                open(os.path.join(opt.exp_dir, 'test.iter%d.err'%i), 'w') as efp:
            start = time.time()
            test_loss, (tep, ter, tef), te_acc, te_eic = eval_epoch(model, test_dataloader, opt, memory, fp, efp)
            logger.info('[Test]\tEpoch: %02d\tTime: %.2f\tLoss: %.2f\t(p/r/f): (%.2f/%.2f/%.2f)\tAcc: %.2f' %
                (i, time.time()-start, test_loss, tep, ter, tef, te_acc))
            observability_lens(te_eic, i, "test", opt.exp_dir, export_csv_model_name)

        # save model
        if vf > best['vf']:
            best['epoch'] = i
            best['vf'] = vf
            best['tef'] = tef
            best['v_acc'] = v_acc
            best['te_acc'] = te_acc
            model.save_model(os.path.join(opt.exp_dir, 'model.pt'))
            logger.info('NEW BEST:\tEpoch: %02d\tvalid F1/Acc: %.2f/%.2f\ttest F1/Acc: %.2f/%.2f' % (
                i, vf, v_acc, tef, te_acc))

    logger.info('Done training. Elapsed time: %s' % timedelta(seconds=time.time() - t0))
    logger.info('BEST RESULT:\tEpoch: %02d\tBest valid F1/Acc: %.2f/%.2f\ttest F1/Acc: %.2f/%.2f' % (
        best['epoch'], best['vf'], best['v_acc'], best['tef'], best['te_acc']))


def test(model, train_dataloader, valid_dataloader, test_dataloader, opt, memory):
    '''Start testing'''

    logger = make_logger(os.path.join(opt.exp_dir, 'log.test'))
    t0 = time.time()
    logger.info('Testing starts at %s' % (time.asctime(time.localtime(time.time()))))

    # evaluating train set
    with open(os.path.join(opt.exp_dir, 'train.eval'), 'w') as fp, \
            open(os.path.join(opt.exp_dir, 'train.eval.err'), 'w') as efp:
        start = time.time()
        train_loss, (trp, trr, trf), tr_acc, train_all_cases = eval_epoch(model, train_dataloader, opt, memory, fp, efp)
        logger.info('[Train]\tTime: %.2f\tLoss: %.2f\t(p/r/f): (%.2f/%.2f/%.2f)\tAcc: %.2f' %
            (time.time()-start, train_loss, trp, trr, trf, tr_acc))

    # evaluating valid set
    with open(os.path.join(opt.exp_dir, 'valid.eval'), 'w') as fp, \
            open(os.path.join(opt.exp_dir, 'valid.eval.err'), 'w') as efp:
        start = time.time()
        valid_loss, (vp, vr, vf), v_acc, valid_all_cases = eval_epoch(model, valid_dataloader, opt, memory, fp, efp)
        logger.info('[Valid]\tTime: %.2f\tLoss: %.2f\t(p/r/f): (%.2f/%.2f/%.2f)\tAcc: %.2f' %
            (time.time()-start, valid_loss, vp, vr, vf, v_acc))

    # evaluating test set
    with open(os.path.join(opt.exp_dir, 'test.eval'), 'w') as fp, \
            open(os.path.join(opt.exp_dir, 'test.eval.err'), 'w') as efp:
        start = time.time()
        test_loss, (tep, ter, tef), te_acc, test_all_cases = eval_epoch(model, test_dataloader, opt, memory, fp, efp)
        logger.info('[Test]\tTime: %.2f\tLoss: %.2f\t(p/r/f): (%.2f/%.2f/%.2f)\tAcc: %.2f' %
            (time.time()-start, test_loss, tep, ter, tef, te_acc))

    logger.info('Done testing. Elapsed time: %s' % timedelta(seconds=time.time() - t0))


if __name__ == '__main__':
    #tod-bert-models/ToD-BERT-jnt
    opt = parse_arguments()
    print('Karthik is a good boy')
    if opt.tod_pre_trained_model:
        opt.tokenizer = AutoTokenizer.from_pretrained(opt.custom_pre_trained_model)
        opt.pretrained_model = AutoModel.from_pretrained(opt.custom_pre_trained_model)
    else:
        if MODEL_CLASSES.get(opt.pre_trained_model):
            pre_trained_model,pre_trained_tokenizer,model_name = MODEL_CLASSES.get(opt.pre_trained_model) 
            opt.tokenizer = pre_trained_model.from_pretrained(model_name)
            opt.pretrained_model = pre_trained_tokenizer.get(opt.pre_trained_model).from_pretrained(model_name)
    # memory
    memory = torch.load(os.path.join(opt.dataroot, 'memory.pt'))
    opt.word_vocab_size = opt.tokenizer.vocab_size  # subword-level
    if opt.with_system_act:
        opt.sysact_vocab_size = len(memory['sysact2idx'])
    opt.label_vocab_size = len(memory['label2idx'])
    opt.top_label_vocab_size = len(memory['toplabel2idx'])
    opt.top2bottom_dict = memory['top2bottom_dict']
    memory['bottom2top_mat'] = reverse_top2bottom(memory['top2bottom_dict'])
    print('word vocab size:', opt.word_vocab_size)
    if opt.with_system_act:
        print('system act vocab size:', opt.sysact_vocab_size)
    print('#labels:', opt.label_vocab_size)
    print('#top-labels:', opt.top_label_vocab_size)
    print(opt)

    # exp dir
    opt.exp_dir = get_exp_dir_bert(opt)
    if not opt.testing and not os.path.exists(opt.exp_dir):
        os.makedirs(opt.exp_dir)

    # model definition & num of params
    model = make_model(opt)
    model = model.to(opt.device)

    trainable_parameters = list(filter(lambda p: p[1].requires_grad, model.named_parameters()))
    n_params = sum([np.prod(p.size()) for n, p in trainable_parameters])
    bert_parameters = list(filter(lambda n_p: 'bert_encoder' in n_p[0], trainable_parameters))
    n_bert_params = sum([np.prod(p.size()) for n, p in bert_parameters])
    print(model)
    print('num params: {}'.format(n_params))
    print('num bert params: {}, {}%'.format(n_bert_params, 100 * n_bert_params / n_params))

    # dataloader preparation
    opt.n_accum_steps = 4 if opt.n_layers == 12 else 1

    train_data = read_wcn_data(os.path.join(opt.dataroot, opt.train_file))
    valid_data = read_wcn_data(os.path.join(opt.dataroot, opt.valid_file))
    test_data = read_wcn_data(os.path.join(opt.dataroot, opt.test_file))
    train_dataloader = prepare_wcn_dataloader(train_data, memory, int(opt.batchSize / opt.n_accum_steps),
        opt.max_seq_len, opt.device, shuffle_flag=True)
    valid_dataloader = prepare_wcn_dataloader(valid_data, memory, int(opt.batchSize / opt.n_accum_steps),
        opt.max_seq_len, opt.device, shuffle_flag=False)
    test_dataloader = prepare_wcn_dataloader(test_data, memory, int(opt.batchSize / opt.n_accum_steps),
        opt.max_seq_len, opt.device, shuffle_flag=False)

    # optimizer
    params = list(model.parameters())
    params = list(filter(lambda p: p.requires_grad, params))
    named_params = list(model.named_parameters())
    named_params = list(filter(lambda p: p[1].requires_grad, named_params))

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    is_bert = lambda name: 'bert_encoder' in name
    is_decay = lambda name: not any(nd in name for nd in no_decay)

    optimizer_grouped_parameters = []
    for n, p in named_params:
        params_group = {}
        params_group['params'] = p
        params_group['weight_decay'] = 0.01 if is_decay(n) else 0
        params_group['lr'] = opt.bert_lr if is_bert(n) else opt.lr
        optimizer_grouped_parameters.append(params_group)


    if opt.optim_choice == 'adam':
        opt.optimizer = optim.Adam(params, lr=opt.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=opt.l2)
    elif opt.optim_choice.lower() == 'bertadam':
        num_train_optimization_steps = (len(train_dataloader.dataset) // opt.batchSize + 1) * opt.max_epoch
        opt.optimizer = BertAdam(
            optimizer_grouped_parameters,
            lr=opt.lr, warmup=opt.warmup_proportion,
            t_total=num_train_optimization_steps
        )
    elif opt.optim_choice.lower() == 'adamw':
        num_train_optimization_steps = (len(train_dataloader.dataset) // opt.batchSize + 1) * opt.max_epoch
        opt.optimizer = AdamW(optimizer_grouped_parameters, lr=opt.lr, correct_bias=False)
        opt.scheduler = get_linear_schedule_with_warmup(
            opt.optimizer,
            num_warmup_steps=int(opt.warmup_proportion * num_train_optimization_steps),
            num_training_steps=num_train_optimization_steps
        )  # PyTorch scheduler

    # loss function
    opt.class_loss_function = nn.BCELoss(reduction='sum')
    opt.ce_loss_function = nn.NLLLoss(reduction='sum')
    opt.mse_loss_function = nn.MSELoss()

    # training or testing
    if opt.testing:
        model.load_model(os.path.join(opt.exp_dir, 'model.pt'))
        test(model, train_dataloader, valid_dataloader, test_dataloader, opt, memory)
    else:
        train(model, train_dataloader, valid_dataloader, test_dataloader, opt, memory)
