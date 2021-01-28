#!/bin/bash

#################### model structure ####################
es=768  # also d_model; normal: 100 / bert: 768
hs=128
msl=512  # max_seq_len
nl=12
nh=12; dk=64; dv=64  # bert
score_util='pp'  # none/np/pp/mul
sent_repr='bin_sa_cls'
cls_type='stc'

#################### data & vocab dirs ####################
dataset="dstc2"
dataroot="dstc2_data/transcription_added_data/processed_data/raw"
exp_path="exp/exp_Transcript_L2_XLM_ROBERT/"

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
lr=3e-5  # for non-bert params
bert_lr=3e-5
wp=0.1  #warmup
init_type='uf'  # uf/xuf
init_range=0.02
seed=999

#################### cmd ####################

################### loss function ###########
add_l2_loss=true

################## pre - trained mode ########

pre_trained_model = 'xlm-roberta'
tod_pre_trained_model = 'tod-bert-models/ToD-BERT-jnt'

python3 TOD_ASR_BERT_STC.py \
    --dataset ${dataset} --dataroot ${dataroot} \
    --bert_model_name ${bert_model_name} ${fix_bert_model:+--fix_bert_model} \
    --deviceId ${device} --random_seed ${seed} --l2 ${l2} --dropout ${dp} --bert_dropout ${bert_dp} \
    --optim_choice ${optim} --lr ${lr} --bert_lr ${bert_lr} --warmup_proportion ${wp} \
    --init_type ${init_type} --init_range ${init_range} \
    --batchSize ${bs} --max_norm ${mn} --max_epoch ${me} \
    --experiment ${exp_path} \
    --add_l2_loss ${add_l2_loss}
    --pre_trained_model ${pre_trained_model}
    --tod_pre_trained_model ${tod_pre_trained_model}

