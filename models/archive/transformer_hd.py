import torch
import torch.nn as nn
import torch.nn.functional as F

import utils.Constants as Constants
from models.transformer.PtrDecoderModels import Decoder
from models.transformer.Beam import Beam

# class MultiClassifier(nn.Module):
#     def __init__(self, input_dim, class_size, dropout):
#         super(MultiClassifier, self).__init__()
#         self.dropout = nn.Dropout(dropout)
#         self.lin = nn.Linear(input_dim, class_size)
# 
#     def forward(self, vec):
#         vec = self.dropout(vec)
#         logits = self.lin(vec)
#         scores = torch.sigmoid(logits)
#         return scores


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


class TransHierarDecoder(nn.Module):
    def __init__(self, dec_word_vocab_size, act_vocab_size, slot_vocab_size,
            emb_size, d_model, d_inner, max_seq_len,
            n_layers, n_head, d_k, d_v, dropout, with_ptr=True,
            dec_score_util=None, decoder_tied=False):

        super(TransHierarDecoder, self).__init__()

        self.act_emb = nn.Embedding(act_vocab_size, emb_size, padding_idx=Constants.PAD)
        self.slot_emb = nn.Embedding(slot_vocab_size, emb_size, padding_idx=Constants.PAD)

        self.act_clf = MultiClassifier(2 * d_model, emb_size, act_vocab_size, dropout)
        self.slot_clf = MultiClassifier(2 * d_model + emb_size, emb_size, slot_vocab_size, dropout)

        self.decoder = Decoder(
            dec_word_vocab_size, max_seq_len, emb_size, n_layers, n_head,
            d_k, d_v, d_model, d_inner,
            dropout=dropout, with_ptr=with_ptr, d_ext_fea=2 * emb_size,
            dec_score_util=dec_score_util, decoder_tied=decoder_tied)
        self.with_ptr = with_ptr

        if decoder_tied:
            self.act_clf.out_layer.weight = self.act_emb.weight
            self.slot_clf.out_layer.weight = self.slot_emb.weight

    def slot_predict(self, fea, acts):
        act_nums = acts.size(0)  # acts -> (act_num, 1)
        act_embs = self.act_emb(acts).squeeze(1)  # (#acts, 1, emb) -> (#acts, emb)
        inputs = torch.cat([act_embs, fea.expand(act_nums, -1)], dim=1)  # (#acts, emb + fea_dim)
        slot_scores = self.slot_clf(inputs)
        return slot_scores

    def forward(self, src_seq, src_score, enc_out, sent_level_fea, acts, act_slot_pairs, values, extra_zeros, enc_batch_extend_vocab_idx):
        '''
        specially for batch_size = 1
        - src_seq: WITH pad -> (1, seq)
        - src_score: WITH pad -> (1, seq)
        - tgt_seq: list of value tensors, each tensor begins with idx[<s>]
        - enc_out: with_pad -> (1, seq, d_model)
        - sent_level_fea: for act & slot classification -> (1, 2 * d_model)
        - acts: act inputs -> (#acts, 1)
        - extra_zeros -> (1, #oov)
        - enc_batch_extend_vocab_idx: WITHOUT pad -> (1, seq)
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
        # value inputs pre-processing
        assert len(act_slot_pairs) == len(values)
        bs = len(values)
        max_value_len = max([v.size(1) for v in values])
        padded_values = [F.pad(v, (0, max_value_len-v.size(1)), 'constant', Constants.PAD)
            for v in values]
        tgt_seq = torch.cat(padded_values, dim=0)  # (#values, value_len)
        tgt_pos = torch.arange(1, max_value_len+1).expand(bs, -1).to(tgt_seq.device)  # (#values, value_len)
        src_seq = src_seq.expand(bs, -1)  #(#values, src_seq_len)
        src_score = src_score.expand(bs, -1)  #(#values, src_seq_len)
        enc_out = enc_out.expand(bs, -1, -1)  # (#values, src_seq_len, d_model)

        # incorporate act & slot info
        pair_num = act_slot_pairs.size(0)  # act_slot_pairs -> (#values, 2)
        ext_acts = act_slot_pairs[:, 0].unsqueeze(1)
        ext_slots = act_slot_pairs[:, 1].unsqueeze(1)
        acts_emb = self.act_emb(ext_acts)
        slots_emb = self.slot_emb(ext_slots)
        ext_fea = torch.cat([acts_emb, slots_emb], 2)  # (#values, 1, 2*emb_size)

        value_scores, *_ = self.decoder(tgt_seq, tgt_pos, src_seq, src_score, enc_out, ext_fea=ext_fea,
            return_attns=True, extra_zeros=extra_zeros, extend_idx=enc_batch_extend_vocab_idx)
        # value_scores -> (#values, value_len, #n_class)

        return act_scores, slot_scores, value_scores

    def decode(self, src_seq, src_score, enc_out, sent_level_fea, extra_zeros, enc_batch_extend_vocab_idx,
            memory, oov_list, device, beam_size, tokenizer=None):
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
        ext_acts = act_slot_pairs[:, 0].unsqueeze(1)
        ext_slots = act_slot_pairs[:, 1].unsqueeze(1)
        acts_emb = self.act_emb(ext_acts)
        slots_emb = self.slot_emb(ext_slots)
        ext_fea = torch.cat([acts_emb, slots_emb], 2)  # (#values, 1, 2*emb_size)

        # hyp, scores = self.decode_value_beam_search(
        #     src_seq, enc_out, ext_fea, extra_zeros, enc_batch_extend_vocab_idx,
        #     n_bm=beam_size, device=device)
        values = self.decode_value_greedy(src_seq, src_score, enc_out, ext_fea,
            extra_zeros, enc_batch_extend_vocab_idx, memory['idx2dec'], oov_list,
            device=device)
        assert len(values) == len(remain_act_slots)
        for act_slot, value in zip(remain_act_slots, values):
            if value is not None:
                utt_triples.append('-'.join(list(act_slot[1]) + [value]))

        return utt_triples

    def decode_value_greedy(self, src_seq, src_score, enc_out, ext_fea,
            extra_zeros, enc_batch_extend_vocab_idx, idx2value, oov_list,
            device=None, max_seq_len=5):

        bs = ext_fea.size(0)
        value_vocab_size = len(idx2value)

        all_values = []

        for i in range(bs):
            dec_seq = torch.empty(1, 1).fill_(Constants.BOS).long().to(device)
            value_ids = []
            for j in range(max_seq_len - 1):
                dec_pos = torch.arange(1, j + 2).unsqueeze(0).to(device)
                word_prob, *_ = self.decoder(dec_seq, dec_pos, src_seq, src_score, enc_out, ext_fea=ext_fea[i:i+1],
                    return_attns=True, extra_zeros=extra_zeros, extend_idx=enc_batch_extend_vocab_idx)
                next_word_prob = word_prob[:, -1, :]  # (1, vocab_size)
                _, next_word_id = torch.max(next_word_prob, dim=1)  # (1,)

                if next_word_id.item() == Constants.EOS:
                    break

                value_ids.append(next_word_id.item())
                in_vocab_value_id = next_word_id.item() if next_word_id.item() < value_vocab_size else Constants.UNK
                dec_seq = torch.cat([dec_seq, torch.empty(1, 1).fill_(in_vocab_value_id).long().to(device)], dim=-1)  # (1, j+2)

            if value_ids == []:
                value = None
            else:
                if self.with_ptr:
                    value_lis = [idx2value[vid] if vid < value_vocab_size else oov_list[vid - value_vocab_size]
                        for vid in value_ids]
                else:
                    value_lis = [idx2value[vid] for vid in value_ids]
                value = ' '.join(value_lis)
            all_values.append(value)

        return all_values


    # bug unsolved
    def decode_value_beam_search(self, src_seq, src_enc,
            ext_fea, extra_zeros, enc_batch_extend_vocab_idx, n_bm=5, device=None):

        def get_inst_idx_to_tensor_position_map(inst_idx_list):
            ''' Indicate the position of an instance in a tensor. '''
            return {inst_idx: tensor_position for tensor_position, inst_idx in enumerate(inst_idx_list)}

        def collect_active_part(beamed_tensor, curr_active_inst_idx, n_prev_active_inst, n_bm):
            ''' Collect tensor parts associated to active instances. '''

            _, *d_hs = beamed_tensor.size()
            n_curr_active_inst = len(curr_active_inst_idx)
            new_shape = (n_curr_active_inst * n_bm, *d_hs)

            beamed_tensor = beamed_tensor.view(n_prev_active_inst, -1)
            beamed_tensor = beamed_tensor.index_select(0, curr_active_inst_idx)
            beamed_tensor = beamed_tensor.view(*new_shape)

            return beamed_tensor

        def collate_active_info(
                src_seq, src_enc, ext_fea, inst_idx_to_position_map, active_inst_idx_list, device):
            # Sentences which are still active are collected,
            # so the decoder will not run on completed sentences.
            n_prev_active_inst = len(inst_idx_to_position_map)
            active_inst_idx = [inst_idx_to_position_map[k] for k in active_inst_idx_list]
            active_inst_idx = torch.LongTensor(active_inst_idx).to(device)

            active_src_seq = collect_active_part(src_seq, active_inst_idx, n_prev_active_inst, n_bm)
            active_src_enc = collect_active_part(src_enc, active_inst_idx, n_prev_active_inst, n_bm)
            active_ext_fea = collect_active_part(ext_fea, active_inst_idx, n_prev_active_inst, n_bm)
            active_inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(active_inst_idx_list)

            return active_src_seq, active_src_enc, active_ext_fea, active_inst_idx_to_position_map

        def beam_decode_step(
                inst_dec_beams, len_dec_seq, src_seq, enc_output, inst_idx_to_position_map, n_bm, device,
                ext_fea, extra_zeros, enc_batch_extend_vocab_idx):
            ''' Decode and update beam status, and then return active beam idx '''

            def prepare_beam_dec_seq(inst_dec_beams, len_dec_seq):
                dec_partial_seq = [b.get_current_state() for b in inst_dec_beams if not b.done]
                dec_partial_seq = torch.stack(dec_partial_seq).to(device)
                dec_partial_seq = dec_partial_seq.view(-1, len_dec_seq)
                return dec_partial_seq

            def prepare_beam_dec_pos(len_dec_seq, n_active_inst, n_bm, device):
                dec_partial_pos = torch.arange(1, len_dec_seq + 1, dtype=torch.long, device=device)
                dec_partial_pos = dec_partial_pos.unsqueeze(0).repeat(n_active_inst * n_bm, 1)
                return dec_partial_pos

            def predict_word(dec_seq, dec_pos, src_seq, enc_output, n_active_inst, n_bm,
                    ext_fea, extra_zeros, enc_batch_extend_vocab_idx):
                word_prob, *_ = self.decoder(dec_seq, dec_pos, src_seq, enc_output, ext_fea=ext_fea,
                    return_attns=True, extra_zeros=extra_zeros, extend_idx=enc_batch_extend_vocab_idx)
                word_prob = word_prob[:, -1, :]  # Pick the last step: (bs * n_bm) * vocab_size
                word_prob = word_prob.view(n_active_inst, n_bm, -1)
                return word_prob

            def collect_active_inst_idx_list(inst_beams, word_prob, inst_idx_to_position_map):
                active_inst_idx_list = []
                for inst_idx, inst_position in inst_idx_to_position_map.items():
                    is_inst_complete = inst_beams[inst_idx].advance(word_prob[inst_position])
                    if not is_inst_complete:
                        active_inst_idx_list += [inst_idx]

                return active_inst_idx_list

            n_active_inst = len(inst_idx_to_position_map)

            dec_seq = prepare_beam_dec_seq(inst_dec_beams, len_dec_seq)
            dec_pos = prepare_beam_dec_pos(len_dec_seq, n_active_inst, n_bm, device)
            word_prob = predict_word(dec_seq, dec_pos, src_seq, enc_output, n_active_inst, n_bm,
                ext_fea, extra_zeros, enc_batch_extend_vocab_idx)

            # Update the beam with predicted word prob information and collect incomplete instances
            active_inst_idx_list = collect_active_inst_idx_list(
                inst_dec_beams, word_prob, inst_idx_to_position_map)

            return active_inst_idx_list

        def collect_hypothesis_and_scores(inst_dec_beams, n_best):
            all_hyp, all_scores = [], []
            for inst_idx in range(len(inst_dec_beams)):
                scores, tail_idxs = inst_dec_beams[inst_idx].sort_scores()
                all_scores += [scores[:n_best]]

                hyps = [inst_dec_beams[inst_idx].get_hypothesis(i) for i in tail_idxs[:n_best]]
                all_hyp += [hyps]
            return all_hyp, all_scores

        with torch.no_grad():
            #-- Repeat data for beam search
            n_inst, len_s, d_h = src_enc.size()
            d_ext = ext_fea.size(-1)
            # NOTE: to ensure order of repeat!
            src_seq = src_seq.repeat(1, n_bm).view(n_inst * n_bm, len_s)
            src_enc = src_enc.repeat(1, n_bm, 1).view(n_inst * n_bm, len_s, d_h)
            ext_fea = ext_fea.repeat(1, n_bm, 1).view(n_inst * n_bm, 1, d_ext)

            #-- Prepare beams
            inst_dec_beams = [Beam(n_bm, device=device) for _ in range(n_inst)]

            #-- Bookkeeping for active or not
            active_inst_idx_list = list(range(n_inst))
            inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(active_inst_idx_list)

            #-- Decode
            max_token_seq_len = 5
            for len_dec_seq in range(1, max_token_seq_len + 1):

                active_inst_idx_list = beam_decode_step(
                    inst_dec_beams, len_dec_seq, src_seq, src_enc, inst_idx_to_position_map, n_bm, device,
                    ext_fea, extra_zeros, enc_batch_extend_vocab_idx)

                if not active_inst_idx_list:
                    break  # all instances have finished their path to <EOS>

                src_seq, src_enc, ext_fea, inst_idx_to_position_map = collate_active_info(
                    src_seq, src_enc, ext_fea, inst_idx_to_position_map, active_inst_idx_list, device)

        n_best = 1
        batch_hyp, batch_scores = collect_hypothesis_and_scores(inst_dec_beams, n_best)

        return batch_hyp, batch_scores


