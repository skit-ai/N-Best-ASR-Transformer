#!/bin/bash
'''
Magic bash script to train your model.
'''

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

#You can keep this fixed 
dataroot="dstc2_data/processed_data/raw"

#Pass path to export checkpoint and logs
exp_path="exp/exp_bert_sep_segment_ids/"

#################### pretrained embedding ####################
fix_bert_model=false
if ! ${fix_bert_model}; then unset fix_bert_model; fi
bert_model_name='bert-base-uncased'  #  bert_model_name = {bert-base-uncased, xlnet-base-cased}

#################### training & testing options ####################
device=0
l2=1e-8
dp=0.3
bert_dp=0.1  # dropout rate for BERT
bs=16
mn=5.0
me=50
optim="bertadam"  # bertadam/adamw
lr=3e-5  # for non-bert params
bert_lr=3e-5
wp=0.1  #warmup
init_type='uf'  # uf/xuf
init_range=0.02
seed=999

#################### Sample Complexity Arguments ####################
# coverage = (0,1] 
# where, coverage = 1 means you are including the whole data set for training. 
# and, coverage < 1 refers to the percentage of samples you want to consider for training your model. 
# For our work we test our model for sample complexity coverage of {0.05, 0.10, 0.20, 0.50}
# NOTE- Based on coverage percentage stratified data samples will be picked as a training set.
coverage=1.0 

################## pre - trained mode ########
#pass pre-trained model name if you want to use a pre-trained model from Transformers package like bert,roberta,xlm-roberta
pre_trained_model='bert'

#pass the checkpoint path to tod pre-trained model if you want to use model 
#NOTE : You need to pass this value parameter to --tod_pre_trained_model if you wish you use this model. 
tod_pre_trained_model='multi-seq-asr-bert/ToD-BERT-jnt'

python3 multi_seq_asr_bert_stc.py \
    --dataset ${dataset} --dataroot ${dataroot} \
    --bert_model_name ${bert_model_name} ${fix_bert_model:+--fix_bert_model} \
    --deviceId ${device} --random_seed ${seed} --l2 ${l2} --dropout ${dp} --bert_dropout ${bert_dp} \
    --optim_choice ${optim} --lr ${lr} --bert_lr ${bert_lr} --warmup_proportion ${wp} \
    --init_type ${init_type} --init_range ${init_range} \
    --batchSize ${bs} --max_norm ${mn} --max_epoch ${me} \
    --experiment ${exp_path} \
    --pre_trained_model ${pre_trained_model} \
    --coverage ${coverage} \
    --add_segment_ids    #flag to add or remove segment ids  
    #--without_system_act #Uncomment this if you wish to remove system act 
    #--add_l2_loss \ #Uncomment this if you wish to use l2 loss 
    #--tod_pre_trained_model ${pre_trained_model} #Uncomment this if you wish to multi-seq-asr-bert  
