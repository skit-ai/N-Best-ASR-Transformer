#!/bin/bash

#################### model structure ####################
es=768  # also d_model; normal: 100 / bert: 768
hs=128
msl=512  # max_seq_len
nl=12
nh=12; dk=64; dv=64  # bert
score_util='pp'  # none/np/pp/mul
sent_repr='bin_sa_cls'
cls_type='nc'

#################### data & vocab dirs ####################
dataset="dstc2"
dataroot="dstc2_data/processed_data"
exp_path="exp/exp_NC_BERT/"

#################### pretrained embedding ####################
fix_bert_model=false
if ! ${fix_bert_model}; then unset fix_bert_model; fi
bert_model_name='bert-base-uncased'  # bert-base-uncased/xlnet-base-cased

#################### training & testing options ####################
device=0
l2=1e-8
dp=0.3
bert_dp=0.1  # dropout rate for BERT
bs=32
mn=5.0
me=50
optim="bertadam"  # bertadam/adamw
lr=5e-5  # for non-bert params
bert_lr=5e-5
wp=0.1  #warmup
init_type='uf'  # uf/xuf
init_range=0.02
seed=999

#################### cmd ####################

python3 WCN_BERT_NC.py \
    --testing \
    --emb_size ${es} --hidden_size ${hs} ${msl:+--max_seq_len ${msl}} \
    --n_layers ${nl} --n_head ${nh} --d_k ${dk} --d_v ${dv} \
    --score_util ${score_util} --sent_repr ${sent_repr} --cls_type ${cls_type} \
    --dataset ${dataset} --dataroot ${dataroot} \
    --bert_model_name ${bert_model_name} ${fix_bert_model:+--fix_bert_model} \
    --deviceId ${device} --random_seed ${seed} --l2 ${l2} --dropout ${dp} --bert_dropout ${bert_dp} \
    --optim_choice ${optim} --lr ${lr} --bert_lr ${bert_lr} --warmup_proportion ${wp} \
    --init_type ${init_type} --init_range ${init_range} \
    --batchSize ${bs} --max_norm ${mn} --max_epoch ${me} \
    --experiment ${exp_path}

