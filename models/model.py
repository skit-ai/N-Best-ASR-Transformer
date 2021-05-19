import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules.hierarchical_classifier import HierarchicalClassifier

def make_model(opt):

    return TOD_ASR_Transformer_STC(opt)

class TOD_ASR_Transformer_STC(nn.Module):
    '''TOD ASR Transformer Semantic Tuple Classifier'''
    def __init__(self, opt):

        super(TOD_ASR_Transformer_STC, self).__init__()

        #self.pretrained_model_opts = pretrained_model_opts

        self.bert_encoder = opt.pretrained_model  
        # encoders

        self.dropout_layer = nn.Dropout(opt.dropout)

        self.device = opt.device
        self.score_util = opt.score_util
        self.sent_repr = opt.sent_repr
        self.cls_type = opt.cls_type

        # feature dimension
        fea_dim = 768

        self.clf = HierarchicalClassifier(opt.top2bottom_dict, fea_dim, opt.label_vocab_size, opt.dropout)


<<<<<<< HEAD
=======

>>>>>>> main
    def forward(self,opt,input_ids,trans_input_ids=None,seg_ids=None,trans_seg_ids=None,return_attns=False,classifier_input_type="asr"):
        
        #linear input to fed to downstream classifier 
        lin_in=None 

        # encoder on asr out
        #If XLM-Roberta don't pass token type ids 
        if opt.pre_trained_model and opt.pre_trained_model=="xlm-roberta": 
            outputs = self.bert_encoder(input_ids=input_ids,attention_mask=input_ids>0)
        else:
            outputs = self.bert_encoder(input_ids=input_ids,attention_mask=input_ids>0,token_type_ids=seg_ids)    
        sequence_output = outputs[0]
        asr_lin_in = sequence_output[:, 0, :]

        #encoder on manual transcription
        trans_lin_in = None
        if trans_input_ids is not None:
            #If XLM-Roberta don't pass token type ids 
            if opt.pre_trained_model and opt.pre_trained_model=="xlm-roberta":
                trans_outputs = self.bert_encoder(input_ids=trans_input_ids,attention_mask=trans_input_ids>0)
            else:
                trans_outputs = self.bert_encoder(input_ids=trans_input_ids,attention_mask=trans_input_ids>0,token_type_ids=trans_seg_ids)  
            trans_sequence_output = trans_outputs[0]
            trans_lin_in = trans_sequence_output[:, 0, :]
        
        if classifier_input_type=="transcript":
            lin_in = trans_lin_in
        else:
            lin_in = asr_lin_in    

        # decoder / classifier
        if self.cls_type == 'stc':
            top_scores, bottom_scores_dict, final_scores = self.clf(lin_in)


        if return_attns:
            return top_scores, bottom_scores_dict, final_scores, attns,asr_lin_in,trans_lin_in
        else:
            return top_scores, bottom_scores_dict, final_scores,asr_lin_in,trans_lin_in

    def load_model(self, load_dir):
        if self.device.type == 'cuda':
            self.load_state_dict(torch.load(open(load_dir, 'rb')))
        else:
            self.load_state_dict(torch.load(open(load_dir, 'rb'),
                map_location=lambda storage, loc: storage))

    def save_model(self, save_dir):
        torch.save(self.state_dict(), open(save_dir, 'wb'))

