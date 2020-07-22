#!/bin/bash

#################### model structure ####################
es=768  # also d_model; normal: 100 / bert: 768
hs=128
msl=512  # max_seq_len
nl=12
n_dec_layers=1
nh=12; dk=64; dv=64  # bert
score_util='pp'  # none/np/pp/mul
sent_repr='bin_sa_cls'  # bin_sa_cls/cls/tok_sa_cls
cls_type='tf_hd'

dec_tied=true
if ! ${dec_tied}; then unset dec_tied; fi

#################### HD ####################
with_ptr=true
if ! ${with_ptr}; then unset with_ptr; fi

#################### data & vocab dirs ####################
dataset="dstc2"
dataroot="dstc2_data/processed_data"
exp_path="exp/exp_TFHD_BERT/"

#################### pretrained embedding ####################
fix_bert_model=false
if ! ${fix_bert_model}; then unset fix_bert_model; fi
bert_model_name='bert-base-uncased'

#################### training & testing options ####################
device=0
l2=1e-8
dp=0.3
bert_dp=0.1
bs=32
mn=5.0
me=50
optim="bertadam"  # bertadam/adamw
lr=3e-5  # for non-bert params
bert_lr=3e-5
wp=0.1  # warmup propotion
init_type='uf'  # uf/xuf
init_range=0.02
seed=999

#################### cmd ####################

python3 WCN_BERT_TFHD.py \
    --testing \
    --emb_size ${es} --hidden_size ${hs} ${msl:+--max_seq_len ${msl}} \
    --n_layers ${nl} --n_head ${nh} --d_k ${dk} --d_v ${dv} \
    --n_dec_layers ${n_dec_layers}\
    --score_util ${score_util} --sent_repr ${sent_repr} --cls_type ${cls_type} \
    ${dec_tied:+--decoder_tied} ${with_ptr:+--with_ptr} \
    --dataset ${dataset} --dataroot ${dataroot} \
    --bert_model_name ${bert_model_name} \
    ${fix_bert_model:+--fix_bert_model} \
    --deviceId ${device} --random_seed ${seed} --l2 ${l2} --dropout ${dp} --bert_dropout ${bert_dp} \
    --optim_choice ${optim} --lr ${lr} --bert_lr ${bert_lr} --warmup_proportion ${wp} \
    --init_type ${init_type} --init_range ${init_range} \
    --batchSize ${bs} --max_norm ${mn} --max_epoch ${me} \
    --experiment ${exp_path}

