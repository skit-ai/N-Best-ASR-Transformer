import os
import sys
import logging


def make_logger(fn, noStdout=False):
    logFormatter = logging.Formatter('%(message)s')
    logger = logging.getLogger('mylogger')
    logger.setLevel(logging.DEBUG)
    fileHandler = logging.FileHandler(fn, mode='w')
    fileHandler.setFormatter(logFormatter)
    logger.addHandler(fileHandler)
    if not noStdout:
        consoleHandler = logging.StreamHandler(sys.stdout)
        consoleHandler.setFormatter(logFormatter)
        logger.addHandler(consoleHandler)
    return logger


def get_exp_dir_bert(opt, old_version=False):

    dataset_path = 'data_%s' % (opt.dataset)

    exp_dir_list = []
    if 'n_dec_layers' in opt:
        exp_dir_list.append('nl_%s_%s' % (opt.n_layers, opt.n_dec_layers))
    else:
        exp_dir_list.append('nl_%s' % (opt.n_layers))
    exp_dir_list.append('nh_%s' % (opt.n_head))
    exp_dir_list.append('dk_%s' % (opt.d_k))
    exp_dir_list.append('dv_%s' % (opt.d_v))
    exp_dir_list.append('bs_%s' % (opt.batchSize))
    exp_dir_list.append('dp_%s_%s' % (opt.dropout, opt.bert_dropout))
    lr_str = '%s_%s' % (opt.lr, opt.bert_lr)
    if 'finetune_lr' in opt:
        lr_str += '_%s_%s' % (opt.finetune_lr, opt.finetune_bert_lr)
    exp_dir_list.append('opt_%s_%s_%s' % (
        opt.optim_choice, opt.warmup_proportion, lr_str))
    exp_dir_list.append('mn_%s' % (opt.max_norm))
    exp_dir_list.append('me_%s' % (opt.max_epoch))
    exp_dir_list.append('seed_%s' % (opt.random_seed))

    if old_version:
        exp_dir_list.append('score_attn')
    else:
        exp_dir_list.append('score_%s' % (opt.score_util))
    if old_version:
        exp_dir_list.append('cls_%s' % (opt.cls_type))
    else:
        exp_dir_list.append('repr_%s' % (opt.sent_repr))
        exp_dir_list.append('cls_%s' % (opt.cls_type))

    exp_name = '__'.join(exp_dir_list)

    return os.path.join(opt.experiment, dataset_path, exp_name)


def merge_vocabs(v1, v2):
    for word in v2.keys():
        if word not in v1:
            idx = len(v1)
            v1[word] = idx
    return v1


def extend_vocab_with_sep(word2idx):
    sep_token = '<sep>'
    sep_id = len(word2idx)
    word2idx[sep_token] = sep_id
    return word2idx
