import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils.Constants as Constants
from models.transformer.BertEmb_PtrDecoderModels import Decoder as BertEmb_Decoder
from models.transformer.PtrDecoderModels import Decoder
import utils.bert_xlnet_inputs as bert_xlnet_inputs


class MultiClassifier(nn.Module):
    def __init__(self, input_dim, inter_dim, class_size, dropout):
        super(MultiClassifier, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.proj_layer = nn.Linear(input_dim, inter_dim)
        self.out_layer = nn.Linear(inter_dim, class_size)
        self.non_lin = nn.ReLU()

    def forward(self, inp):
        out = self.proj_layer(self.dropout(inp))
        out = self.non_lin(out)
        out = self.out_layer(self.dropout(out))
        scores = torch.sigmoid(out)
        return scores


class TransHierarDecoder_withBERT(nn.Module):
    def __init__(self, dec_word_vocab_size, act_vocab_size, slot_vocab_size,
            emb_size, d_model, d_inner, max_seq_len,
            n_layers, n_head, d_k, d_v, dropout, with_ptr=True,
            decoder_tied=False,
            bert_emb=None):

        super(TransHierarDecoder_withBERT, self).__init__()

        # whether to use bert embedding for encoding acts and slots
        # self.act_slot_bert = act_slot_bert
        # self.value_bert = value_bert

        assert bert_emb is not None
        self.act_emb = self.slot_emb = bert_emb # including position & segment

        self.act_clf = MultiClassifier(2 * d_model, emb_size, act_vocab_size, dropout)
        self.slot_clf = MultiClassifier(2 * d_model + emb_size, emb_size, slot_vocab_size, dropout)

        self.decoder = BertEmb_Decoder(
            dec_word_vocab_size, max_seq_len, emb_size, n_layers, n_head,
            d_k, d_v, d_model, d_inner,
            dropout=dropout, with_ptr=with_ptr, d_ext_fea=2 * emb_size,
            decoder_tied=decoder_tied,
            bert_emb=bert_emb)
        self.with_ptr = with_ptr

    def slot_predict(self, fea, acts):
        act_nums = len(acts)
        act_embs = [self.act_emb(act_in.unsqueeze(0)) for act_in in acts]  # list of tensor(1, n, emb_size), n = length of sub-words
        act_embs_mean = [emb.mean(dim=1) for emb in act_embs]  # list of tensor(1, emb_size,)
        act_embs_final = torch.stack(act_embs_mean, dim=1).squeeze(0)  # (1, #acts, emb) -> (#acts, emb)
        inputs = torch.cat([act_embs_final, fea.expand(act_nums, -1)], dim=1)  # (#acts, emb + fea_dim)
        slot_scores = self.slot_clf(inputs)
        return slot_scores

    def forward(self, src_seq, src_score, enc_out, sent_level_fea, acts, act_slot_pairs,
            values, values_pos, values_seg,
            extra_zeros, enc_batch_extend_vocab_idx):
        '''
        specially for batch_size = 1
        BERT input format:
        - src_seq: WITH pad -> (1, seq)
        - src_score: WITH pad -> (1, seq)
        - tgt_seq: list of value tensors, each tensor begins with idx[<s>]
        - enc_out: with_pad -> (1, seq, d_model)
        - sent_level_fea: for act & slot classification -> (1, 2 * d_model)
        - acts: act inputs -> list of tensor (len,), len(act_inputs) = #acts
        - act_slot_pairs -> list of tuple(tensor(len_act,), tensor(len_slot,))
        - values: with pad -> tensor(#values, len_value)
        - extra_zeros -> (1, #oov), TODO
        - enc_batch_extend_vocab_idx: WITHOUT pad -> (1, seq), TODO
        For non-BERT inputs, refer to transformer_hd.py.
        '''

        # act prediction
        act_scores = self.act_clf(sent_level_fea)

        if acts is None:
            return act_scores, None, None

        # slot prediction
        slot_scores = self.slot_predict(sent_level_fea, acts)

        if act_slot_pairs is None:
            return act_scores, slot_scores, None

        # value prediction
        assert len(act_slot_pairs) == len(values)
        bs = len(values)

        # incorporate act & slot info
        ext_acts = [tup[0] for tup in act_slot_pairs]
        ext_slots = [tup[1] for tup in act_slot_pairs]
        act_embs = [self.act_emb(act_in.unsqueeze(0)) for act_in in ext_acts]  # list of tensor(1, n, emb_size), n = length of sub-words
        act_embs_mean = [emb.mean(dim=1) for emb in act_embs]  # list of tensor(1, emb_size,)
        act_embs_stack = torch.stack(act_embs_mean, dim=1).squeeze(0)  # (1, #acts, emb) -> (#acts, emb)
        slot_embs = [self.slot_emb(slot_in.unsqueeze(0)) for slot_in in ext_slots]
        slot_embs_mean = [emb.mean(dim=1) for emb in slot_embs]
        slot_embs_stack = torch.stack(slot_embs_mean, dim=1).squeeze(0)  # (1, #values, emb) -> (#values, emb)
        ext_fea = torch.cat([act_embs_stack, slot_embs_stack], 1).unsqueeze(1)  # (#values, 2*emb) -> (#values, 1, 2*emb)

        src_seq = src_seq.expand(bs, -1)  #(#values, src_seq_len)
        src_score = src_score.expand(bs, -1)  #(#values, src_seq_len)
        enc_out = enc_out.expand(bs, -1, -1)  # (#values, src_seq_len, d_model)

        value_scores, *_ = self.decoder(values, values_pos, values_seg, src_seq, src_score, enc_out, ext_fea=ext_fea,
            return_attns=True, extra_zeros=extra_zeros, extend_idx=enc_batch_extend_vocab_idx)
        # value_scores -> (#values, value_len, #n_class)

        return act_scores, slot_scores, value_scores

    def decode(self, src_seq, src_score, enc_out, sent_level_fea, extra_zeros, enc_batch_extend_vocab_idx,
            memory, oov_list, device, tokenizer=None):
        ''' specially for one utterance'''
        utt_triples = []

        # act prediction
        act_scores = self.act_clf(sent_level_fea)
        act_scores = act_scores.data.cpu().view(-1,).numpy()
        pred_acts = [j for j, p in enumerate(act_scores) if p > 0.5]
        act_pairs = [(j, memory['idx2act'][j]) for j in pred_acts]
        remain_acts = []
        for act in act_pairs:
            if act[1] == 'pad':
                continue
            elif act[1] in memory['single_acts']:
                utt_triples.append(act[1])
            else:
                remain_acts.append(act)

        if len(remain_acts) == 0:
            return utt_triples

        # slot prediction
        remain_act_slots = []
        num_acts = len(remain_acts)  # #double_acts
        act_input = torch.tensor([act[0] for act in remain_acts]).view(num_acts, 1).to(device)
        act_inputs = bert_xlnet_inputs.prepare_inputs_for_bert_xlnet_act([act_input], memory, tokenizer, device)
        act_input = act_inputs[0]  # for batch=1, take the only one out
        slot_scores = self.slot_predict(sent_level_fea, act_input)  # (#double_acts, slot_vocab_size)
        for i in range(num_acts):
            act = remain_acts[i]
            pred_slots = [j for j, p in enumerate(slot_scores[i]) if p > 0.5]
            slot_pairs = [(j, memory['idx2slot'][j]) for j in pred_slots]
            if act[1] in memory['double_acts']:
                for slot in slot_pairs:
                    utt_triples.append('-'.join([act[1], slot[1]]))
            else:
                for slot in slot_pairs:
                    if slot[1] != 'pad':
                        remain_act_slots.append(list(zip(act, slot)))
        if len(remain_act_slots) == 0:
            return utt_triples

        # value_prediction
        bs = len(remain_act_slots)  # #values
        # src_seq = src_seq.expand(bs, -1)  #(#values, src_seq_len)
        # enc_out = enc_out.expand(bs, -1, -1)  # (#values, src_seq_len, d_model)

        act_slot_pairs = torch.tensor([act_slot[0] for act_slot in remain_act_slots]).to(device)  # (#values, 2)
        act_slot_pairs_batch = bert_xlnet_inputs.prepare_inputs_for_bert_xlnet_slot(
            [act_slot_pairs], memory, tokenizer, device)
        act_slot_pairs = act_slot_pairs_batch[0]  # take the only one out

        ext_acts = [tup[0] for tup in act_slot_pairs]
        ext_slots = [tup[1] for tup in act_slot_pairs]
        act_embs = [self.act_emb(act_in.unsqueeze(0)) for act_in in ext_acts]  # list of tensor(1, n, emb_size), n = length of sub-words
        act_embs_mean = [emb.mean(dim=1) for emb in act_embs]  # list of tensor(1, emb_size,)
        act_embs_stack = torch.stack(act_embs_mean, dim=1).squeeze(0)  # (1, #acts, emb) -> (#acts, emb)
        slot_embs = [self.slot_emb(slot_in.unsqueeze(0)) for slot_in in ext_slots]
        slot_embs_mean = [emb.mean(dim=1) for emb in slot_embs]
        slot_embs_stack = torch.stack(slot_embs_mean, dim=1).squeeze(0)  # (1, #values, emb) -> (#values, emb)
        ext_fea = torch.cat([act_embs_stack, slot_embs_stack], 1).unsqueeze(1)  # (#values, 2*emb) -> (#values, 1, 2*emb)

        values = self.decode_value_greedy(src_seq, src_score, enc_out, ext_fea,
            extra_zeros, enc_batch_extend_vocab_idx, memory['idx2dec'], oov_list,
            device=device, tokenizer=tokenizer)
        assert len(values) == len(remain_act_slots)
        for act_slot, value in zip(remain_act_slots, values):
            if value is not None:
                utt_triples.append('-'.join(list(act_slot[1]) + [value]))

        return utt_triples

    def decode_value_greedy(self, src_seq, src_score, enc_out, ext_fea,
            extra_zeros, enc_batch_extend_vocab_idx, idx2value, oov_list,
            device=None, max_seq_len=5, tokenizer=None):

        bs = ext_fea.size(0)
        value_vocab_size = len(tokenizer)

        all_values = []

        bos_token = tokenizer.cls_token_id
        eos_token = tokenizer.sep_token_id

        for i in range(bs):
            dec_seq = torch.empty(1, 1).fill_(bos_token).long().to(device)  # [CLS] as begin of sentence
            value_ids = []
            for j in range(max_seq_len - 1):
                dec_pos = torch.arange(1, j + 2).unsqueeze(0).to(device)
                dec_seg = torch.zeros(1, j + 1).long().to(device)
                word_prob, *_ = self.decoder(dec_seq, dec_pos, dec_seg, src_seq, src_score, enc_out, ext_fea=ext_fea[i:i+1],
                    return_attns=True, extra_zeros=extra_zeros, extend_idx=enc_batch_extend_vocab_idx)
                next_word_prob = word_prob[:, -1, :]  # (1, vocab_size)
                _, next_word_id = torch.max(next_word_prob, dim=1)  # (1,)

                # print('next:', next_word_id.item(), tokenizer._convert_id_to_token(next_word_id.item()))
                if next_word_id.item() == eos_token:  # [SEP] as end of sequence
                    # print('SEP encoutered. Break!')
                    break

                value_ids.append(next_word_id.item())
                in_vocab_value_id = next_word_id.item() if next_word_id.item() < value_vocab_size else Constants.UNK
                dec_seq = torch.cat([dec_seq, torch.empty(1, 1).fill_(in_vocab_value_id).long().to(device)], dim=-1)  # (1, j+2)

            if value_ids == []:
                value = None
            else:
                if self.with_ptr:
                    value_lis = [tokenizer._convert_id_to_token(vid) if vid < value_vocab_size else oov_list[vid - value_vocab_size]
                        for vid in value_ids]
                else:
                    value_lis = [tokenizer._convert_id_to_token(vid) for vid in value_ids]
                value = tokenizer.convert_tokens_to_string(value_lis)
                if value == 'dont care':  # merge dontcare
                    value = 'dontcare'
            all_values.append(value)

        return all_values


