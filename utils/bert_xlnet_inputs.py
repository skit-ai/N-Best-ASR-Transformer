import torch
import itertools

import utils.Constants as Constants
from helpers.act_slot_split_map import SPLIT_MAP

'''
largely borrowed from su.zhu
'''

def prepare_inputs_for_bert_xlnet(sentences, word_lengths, tokenizer, padded_position_ids, padded_scores=None,
        cls_token_at_end=False, pad_on_left=False, cls_token='[CLS]', sep_token='[SEP]',
        pad_token=0, sequence_a_segment_id=0, cls_token_segment_id=1, pad_token_segment_id=0, device=None):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """
    """ output: {
        'tokens': tokens_tensor,                # input_ids
        'segments': segments_tensor,            # token_type_ids
        'positions': positions_tensor,          # position_ids
        'scores': scores_tensor,                # scores
        'scores_scaler': scores_scaler_tensor,  # scores_scaler
        'mask': input_mask,                     # attention_mask
        'selects': selects_tensor,              # original_word_to_token_position
        'copies': copies_tensor                 # original_word_position
        }
    """

    pad_pos = 0
    pad_score = -1

    ## sentences are sorted by sentence length
    max_length_of_sentences = max(word_lengths)
    tokens = []
    position_ids = []
    if padded_scores is not None:
        scores = []
        scores_scaler = []
    segment_ids = []
    selected_indexes = []
    batch_size = len(sentences)
    for i in range(batch_size):
        ws = sentences[i]
        # ps = padded_position_ids[i].tolist()[:word_lengths[i]]  # BUG! 位置应从idx=1开始截取，因为数据处理时idx=0为CLS
        ps = padded_position_ids[i].tolist()[1:word_lengths[i]+1]
        if padded_scores is not None:
            # ss = padded_scores[i].tolist()[:word_lengths[i]]  # 同position
            ss = padded_scores[i].tolist()[1:word_lengths[i]+1]
        else:
            ss = [None] * word_lengths[i]
        selected_index = []
        ts, tok_ps, tok_ss, tok_sc = [], [], [], []
        for w, pos, score in zip(ws, ps, ss):
            if cls_token_at_end:
                selected_index.append(len(ts))
            else:
                selected_index.append(len(ts) + 1)
            tok_w = tokenizer.tokenize(w)
            ts += tok_w
            tok_ps += [pos] * len(tok_w)  # shared position
            if padded_scores is not None:
                tok_ss += [score] * len(tok_w)   # shared score
            tok_sc += [1.0 / len(tok_w)] * len(tok_w)
        ts += [sep_token]
        tok_ps += [max(tok_ps) + 1]
        if padded_scores is not None:
            tok_ss += [1.0]
        tok_sc += [1.0]
        si = [sequence_a_segment_id] * len(ts)
        if cls_token_at_end:
            ts = ts + [cls_token]
            # tok_ps = tok_ps + [max(tok_ps) + 1]  # BUG! 同上，若CLS在末尾，将原pos-1
            tok_ps = [x - 1 for x in tok_ps] + [max(tok_ps)]
            if padded_scores is not None:
                tok_ss = tok_ss + [1.0]
            tok_sc = tok_sc + [1.0]
            si = si + [cls_token_segment_id]
        else:
            ts = [cls_token] + ts
            # tok_ps = [1] + [x + 1 for x in tok_ps]  # BUG! 同上，tok_ps从2开始，不需要再加1
            tok_ps = [1] + tok_ps
            if padded_scores is not None:
                tok_ss = [1.0] + tok_ss
            tok_sc = [1.0] + tok_sc
            si = [cls_token_segment_id] + si
        tokens.append(ts)
        position_ids.append(tok_ps)
        if padded_scores is not None:
            scores.append(tok_ss)
        scores_scaler.append(tok_sc)
        segment_ids.append(si)
        selected_indexes.append(selected_index)
    token_lens = [len(tokenized_text) for tokenized_text in tokens]
    max_length_of_tokens = max(token_lens)
    #if not cls_token_at_end: # bert
    #    assert max_length_of_tokens <= model_bert.config.max_position_embeddings
    padding_lengths = [max_length_of_tokens - len(tokenized_text) for tokenized_text in tokens]
    if pad_on_left:
        input_mask = [[0] * padding_lengths[idx] + [1] * len(tokenized_text) for idx,tokenized_text in enumerate(tokens)]
        indexed_tokens = [[pad_token] * padding_lengths[idx] + tokenizer.convert_tokens_to_ids(tokenized_text) for idx,tokenized_text in enumerate(tokens)]
        padded_tok_positions = [[pad_pos] * padding_lengths[idx] + p for idx, p in enumerate(position_ids)]
        if padded_scores is not None:
            padded_tok_scores = [[pad_score] * padding_lengths[idx] + s for idx, s in enumerate(scores)]
            padded_tok_scores_scaler = [[pad_score] * padding_lengths[idx] + sc for idx, sc in enumerate(scores_scaler)]
        segments_ids = [[pad_token_segment_id] * padding_lengths[idx] + si for idx,si in enumerate(segment_ids)]
        selected_indexes = [[padding_lengths[idx] + i + idx * max_length_of_tokens for i in selected_index] for idx,selected_index in enumerate(selected_indexes)]
    else:
        input_mask = [[1] * len(tokenized_text) + [0] * padding_lengths[idx] for idx,tokenized_text in enumerate(tokens)]
        indexed_tokens = [tokenizer.convert_tokens_to_ids(tokenized_text) + [pad_token] * padding_lengths[idx] for idx,tokenized_text in enumerate(tokens)]
        padded_tok_positions = [p + [pad_pos] * padding_lengths[idx] for idx, p in enumerate(position_ids)]
        if padded_scores is not None:
            padded_tok_scores = [s + [pad_score] * padding_lengths[idx] for idx, s in enumerate(scores)]
            padded_tok_scores_scaler = [sc + [pad_score] * padding_lengths[idx] for idx, sc in enumerate(scores_scaler)]
        segments_ids = [si + [pad_token_segment_id] * padding_lengths[idx] for idx,si in enumerate(segment_ids)]
        selected_indexes = [[0 + i + idx * max_length_of_tokens for i in selected_index] for idx,selected_index in enumerate(selected_indexes)]
    copied_indexes = [[i + idx * max_length_of_sentences for i in range(length)] for idx,length in enumerate(word_lengths)]

    input_mask = torch.tensor(input_mask, dtype=torch.long, device=device)
    tokens_tensor = torch.tensor(indexed_tokens, dtype=torch.long, device=device)
    positions_tensor = torch.tensor(padded_tok_positions, dtype=torch.long, device=device)
    if padded_scores is not None:
        scores_tensor = torch.tensor(padded_tok_scores, dtype=torch.float, device=device)
        scores_scaler_tensor = torch.tensor(padded_tok_scores_scaler, dtype=torch.float, device=device)
    segments_tensor = torch.tensor(segments_ids, dtype=torch.long, device=device)
    selects_tensor = torch.tensor(list(itertools.chain.from_iterable(selected_indexes)), dtype=torch.long, device=device)
    copies_tensor = torch.tensor(list(itertools.chain.from_iterable(copied_indexes)), dtype=torch.long, device=device)

    if padded_scores is not None:
        return {'tokens': tokens_tensor, 'token_lens': token_lens, 'positions': positions_tensor, 'scores': scores_tensor, 'scores_scaler': scores_scaler_tensor,
                'segments': segments_tensor, 'selects': selects_tensor, 'copies': copies_tensor, 'mask': input_mask}
    else:
        return {'tokens': tokens_tensor, 'token_lens': token_lens, 'positions': positions_tensor,
                'segments': segments_tensor, 'selects': selects_tensor, 'copies': copies_tensor, 'mask': input_mask}


def prepare_inputs_for_bert_xlnet_sysact(sentences, word_lengths, tokenizer, padded_position_ids,
        padded_sa_parents, padded_sa_sibs,
        cls_token_at_end=False, pad_on_left=False, cls_token='[CLS]', sep_token='[SEP]',
        pad_token=0, sequence_a_segment_id=0, cls_token_segment_id=1, pad_token_segment_id=0, device=None):
    ''' ONLY for BERT '''
    assert not cls_token_at_end and not pad_on_left
    assert cls_token_segment_id == 0

    pad_pos = 0
    pad_parent = -2

    ## sentences are sorted by sentence length
    max_length_of_sentences = max(word_lengths)
    tokens = []
    position_ids = []
    sa_parents = []
    sa_sibs = []
    segment_ids = []
    selected_indexes = []
    batch_size = len(sentences)
    for i in range(batch_size):
        ws = sentences[i]
        ps = padded_position_ids[i].tolist()[:word_lengths[i]]
        sps = padded_sa_parents[i].tolist()[:word_lengths[i]]
        ssi = padded_sa_sibs[i].tolist()[:word_lengths[i]]
        selected_index = []
        ts, tok_ps, tok_sps, tok_ssi = [], [], [], []
        index_map = {-1: [-1]}  # from word index to sub-word index
        for j, (w, pos, parent, sib) in enumerate(zip(ws, ps, sps, ssi)):
            selected_index.append(len(ts))
            # NOTE: IMPORTANT!!
            if w == Constants.CLS_WORD:
                w = cls_token
            tok_w = tokenizer.tokenize(w)
            ts += tok_w

            # update index map
            index_map[j] = list(range(len(ts) - len(tok_w), len(ts)))

            tok_ps += [pos] * len(tok_w)  # shared position
            tok_sps += [index_map[parent][0]] * len(tok_w)  # shared parent
            if sib == 0:
                first_token_id = selected_index[-1]
                tok_ssi += ([0] + [first_token_id] * (len(tok_w) - 1))
            else:
                before_first_token_id = selected_index[-1] - 1
                tok_ssi += [before_first_token_id] * len(tok_w)
        ts += [sep_token]
        tok_ps += [max(tok_ps) + 1]
        # tok_sps += [-1]  # no parent for SEP
        tok_sps += [0]  # parent is CLS for SEP
        tok_ssi += [0]  # no sibling for SEP
        si = [sequence_a_segment_id] * len(ts)

        tokens.append(ts)
        position_ids.append(tok_ps)
        sa_parents.append(tok_sps)
        sa_sibs.append(tok_ssi)
        segment_ids.append(si)
        selected_indexes.append(selected_index)
    token_lens = [len(tokenized_text) for tokenized_text in tokens]
    max_length_of_tokens = max(token_lens)
    #if not cls_token_at_end: # bert
    #    assert max_length_of_tokens <= model_bert.config.max_position_embeddings
    padding_lengths = [max_length_of_tokens - len(tokenized_text) for tokenized_text in tokens]
    input_mask = [[1] * len(tokenized_text) + [0] * padding_lengths[idx] for idx,tokenized_text in enumerate(tokens)]
    indexed_tokens = [tokenizer.convert_tokens_to_ids(tokenized_text) + [pad_token] * padding_lengths[idx] for idx,tokenized_text in enumerate(tokens)]
    padded_tok_positions = [p + [pad_pos] * padding_lengths[idx] for idx, p in enumerate(position_ids)]
    padded_tok_parents = [p + [pad_parent] * padding_lengths[idx] for idx, p in enumerate(sa_parents)]
    padded_tok_sibs = [s + [pad_parent] * padding_lengths[idx] for idx, s in enumerate(sa_sibs)]
    segments_ids = [si + [pad_token_segment_id] * padding_lengths[idx] for idx,si in enumerate(segment_ids)]
    selected_indexes = [[0 + i + idx * max_length_of_tokens for i in selected_index] for idx,selected_index in enumerate(selected_indexes)]
    copied_indexes = [[i + idx * max_length_of_sentences for i in range(length)] for idx,length in enumerate(word_lengths)]

    input_mask = torch.tensor(input_mask, dtype=torch.long, device=device)
    tokens_tensor = torch.tensor(indexed_tokens, dtype=torch.long, device=device)
    positions_tensor = torch.tensor(padded_tok_positions, dtype=torch.long, device=device)
    sa_parents_tensor = torch.tensor(padded_tok_parents, dtype=torch.long, device=device)
    sa_sibs_tensor = torch.tensor(padded_tok_sibs, dtype=torch.long, device=device)
    segments_tensor = torch.tensor(segments_ids, dtype=torch.long, device=device)
    selects_tensor = torch.tensor(list(itertools.chain.from_iterable(selected_indexes)), dtype=torch.long, device=device)
    copies_tensor = torch.tensor(list(itertools.chain.from_iterable(copied_indexes)), dtype=torch.long, device=device)

    return {'tokens': tokens_tensor, 'token_lens': token_lens, 'positions': positions_tensor,
            'parents': sa_parents_tensor, 'sibs': sa_sibs_tensor,
            'segments': segments_tensor, 'selects': selects_tensor, 'copies': copies_tensor, 'mask': input_mask}


# Regard utt & sysact as one sequence
# [CLS] utt [SEP] sysact [SEP]
def prepare_inputs_for_bert_xlnet_one_seq(utterances, utt_lens, padded_utt_pos_ids, padded_utt_scores,
        system_acts, sa_lens, padded_sa_pos_ids, padded_sa_parents, padded_sa_sibs, padded_sa_types, tokenizer,
        cls_token_at_end=False, pad_on_left=False, cls_token='[CLS]', sep_token='[SEP]',
        pad_token=0, sequence_a_segment_id=0, sequence_b_segment_id=1,
        cls_token_segment_id=1, pad_token_segment_id=0, device=None):

    ''' ONLY for BERT '''
    assert not cls_token_at_end and not pad_on_left
    assert cls_token_segment_id == 0

    pad_pos = 0
    pad_score = -1
    pad_parent = -2  # also pad sibling

    tokens, position_ids = [], []  # including utt & sysact
    scores, scores_scaler = [], []
    sa_parents, sa_sibs = [], []
    segment_ids = []
    selected_indexes = []
    utt_token_lens, sa_token_lens = [], []

    assert len(utterances) == len(system_acts)
    batch_size = len(utterances)

    for i in range(batch_size):
        selected_index = []
        ts, tok_ps, tok_ss, tok_sc, si = [], [], [], [], []
        tok_sps, tok_ssi = [], []

        # 1) tokenize utterance as sequence A
        utt_seq = utterances[i]
        utt_pos_seq = padded_utt_pos_ids[i].tolist()[1:utt_lens[i]+1]  # starts from 1
        utt_sco_seq = padded_utt_scores[i].tolist()[1:utt_lens[i]+1]
        for w, pos, score in zip(utt_seq, utt_pos_seq, utt_sco_seq):
            selected_index.append(len(ts) + 1)
            tok_w = tokenizer.tokenize(w)
            ts += tok_w
            tok_ps += [pos] * len(tok_w)  # shared position
            tok_ss += [score] * len(tok_w)   # shared score
            tok_sc += [1.0 / len(tok_w)] * len(tok_w)
            # for consistency
            tok_sps += [pad_parent] * len(tok_w)  # PAD parent for normal utt
            tok_ssi += [pad_parent] * len(tok_w)  # PAD sibling for normal utt
        # [SEP]
        ts += [sep_token]
        tok_ps += [max(tok_ps) + 1]
        tok_ss += [1.0]
        tok_sc += [1.0]
        si += [sequence_a_segment_id] * len(ts)
        # for consistency
        tok_sps += [pad_parent]  # PAD parent for SEP
        tok_ssi += [pad_parent]  # PAD sibling for SEP

        utt_token_lens.append(len(ts) + 1)  # +1 due to CLS

        # 2) tokenize system act as sequence B
        len_tokenized_a = len(ts)  # length of tokenized sequence A
        max_pos_a = max(tok_ps)  # max position id of tokenized sequence A
        sa_seq = system_acts[i]
        sa_pos_seq = padded_sa_pos_ids[i].tolist()[:sa_lens[i]]
        sa_par_seq = padded_sa_parents[i].tolist()[:sa_lens[i]]
        sa_sib_seq = padded_sa_sibs[i].tolist()[:sa_lens[i]]
        sa_typ_seq = padded_sa_types[i].tolist()[:sa_lens[i]]
        index_map = {-1: [-1], 0: [0]}  # from word index to sub-word index
        for j, (w, pos, parent, sib, typ) in enumerate(zip(sa_seq, sa_pos_seq, sa_par_seq, sa_sib_seq, sa_typ_seq)):
            if w == '<cls>':  # skip pre-defined <cls>
                continue
            selected_index.append(len(ts) + 1)
            tok_w = tokenizer.tokenize(w)
            ts += tok_w

            # update index map
            index_map[j] = list(range(len(ts) - len(tok_w) + 1, len(ts) + 1))  # +1 due to heading [CLS]

            cur_max_pos = max(tok_ps)
            tok_ps += list(range(cur_max_pos + 1, cur_max_pos + 1 + len(tok_w)))  # incremental
            tok_sps += [index_map[parent][0]] * len(tok_w)  # shared parent
            if sib == 0:
                first_token_id = selected_index[-1]
                tok_ssi += ([0] + [first_token_id] * (len(tok_w) - 1))
            else:
                before_first_token_id = selected_index[-1] - 1
                tok_ssi += [before_first_token_id] * len(tok_w)
            # for consistency
            tok_ss += [1.0] * len(tok_w)
            tok_sc += [1.0] * len(tok_w)
        # [SEP]
        ts += [sep_token]
        tok_ps += [max(tok_ps) + 1]
        # tok_sps += [-1]  # no parent for SEP
        tok_sps += [0]  # parent is CLS for SEP
        tok_ssi += [0]  # no sibling for SEP
        si += [sequence_b_segment_id] * (len(ts) - len_tokenized_a)  # sequence B
        # for consistency
        tok_ss += [1.0]
        tok_sc += [1.0]

        sa_token_lens.append(len(ts) - len_tokenized_a)

        # 3) [CLS]
        ts = [cls_token] + ts
        tok_ps = [1] + tok_ps  # original tok_ps start from 2
        tok_ss = [1.0] + tok_ss
        tok_sc = [1.0] + tok_sc
        tok_sps = [-1] + tok_sps
        tok_ssi = [0] + tok_ssi
        si = [cls_token_segment_id] + si

        tokens.append(ts)
        position_ids.append(tok_ps)
        scores.append(tok_ss)
        scores_scaler.append(tok_sc)
        sa_parents.append(tok_sps)
        sa_sibs.append(tok_ssi)
        segment_ids.append(si)
        selected_indexes.append(selected_index)

    token_lens = [len(tokenized_text) for tokenized_text in tokens]
    max_length_of_tokens = max(token_lens)

    # padding
    padding_lengths = [max_length_of_tokens - len(tokenized_text) for tokenized_text in tokens]
    input_mask = [[1] * len(tokenized_text) + [0] * padding_lengths[idx] for idx,tokenized_text in enumerate(tokens)]
    indexed_tokens = [tokenizer.convert_tokens_to_ids(tokenized_text) + [pad_token] * padding_lengths[idx] for idx,tokenized_text in enumerate(tokens)]
    padded_tok_positions = [p + [pad_pos] * padding_lengths[idx] for idx, p in enumerate(position_ids)]
    padded_tok_scores = [s + [pad_score] * padding_lengths[idx] for idx, s in enumerate(scores)]
    padded_tok_scores_scaler = [sc + [pad_score] * padding_lengths[idx] for idx, sc in enumerate(scores_scaler)]
    padded_tok_parents = [p + [pad_parent] * padding_lengths[idx] for idx, p in enumerate(sa_parents)]
    padded_tok_sibs = [s + [pad_parent] * padding_lengths[idx] for idx, s in enumerate(sa_sibs)]
    segments_ids = [si + [pad_token_segment_id] * padding_lengths[idx] for idx,si in enumerate(segment_ids)]
    selected_indexes = [[0 + i + idx * max_length_of_tokens for i in selected_index] for idx,selected_index in enumerate(selected_indexes)]

    input_mask = torch.tensor(input_mask, dtype=torch.long, device=device)
    tokens_tensor = torch.tensor(indexed_tokens, dtype=torch.long, device=device)
    positions_tensor = torch.tensor(padded_tok_positions, dtype=torch.long, device=device)
    scores_tensor = torch.tensor(padded_tok_scores, dtype=torch.float, device=device)
    scores_scaler_tensor = torch.tensor(padded_tok_scores_scaler, dtype=torch.float, device=device)
    sa_parents_tensor = torch.tensor(padded_tok_parents, dtype=torch.long, device=device)
    sa_sibs_tensor = torch.tensor(padded_tok_sibs, dtype=torch.long, device=device)
    segments_tensor = torch.tensor(segments_ids, dtype=torch.long, device=device)
    selects_tensor = torch.tensor(list(itertools.chain.from_iterable(selected_indexes)), dtype=torch.long, device=device)

    return {'tokens': tokens_tensor, 'token_lens': token_lens, 'positions': positions_tensor,
            'utt_token_lens': utt_token_lens, 'sa_token_lens': sa_token_lens,
            'scores': scores_tensor, 'scores_scaler': scores_scaler_tensor,
            'parents': sa_parents_tensor, 'sibs': sa_sibs_tensor,
            'segments': segments_tensor, 'selects': selects_tensor, 'mask': input_mask}


def prepare_inputs_for_bert_xlnet_act(act_inputs, memory, tokenizer, device):
    '''
    inputs:
        - act_inputs: list of {tensor(#acts, 1), None}
    outputs:
        - act_inputs_token_ids: list of {list of tensor(len,), None}
    '''
    act_inputs_token_ids = []

    for i in range(len(act_inputs)):
        ai = act_inputs[i]

        # act_inputs
        if ai is not None:
            ai_words = [memory['idx2act'][x[0]] for x in ai.tolist()]  # [str] * #acts
            tok_ai = [
                tokenizer.tokenize(
                    ' '.join(SPLIT_MAP[x]) if x in SPLIT_MAP else x
                )
                for x in ai_words
            ]
            tok_ai_ids = [
                torch.LongTensor(tokenizer.convert_tokens_to_ids(x)).to(device)
                for x in tok_ai
            ]  # list of tensor(len,)
            act_inputs_token_ids.append(tok_ai_ids)
        else:
            act_inputs_token_ids.append(None)

    return act_inputs_token_ids


def prepare_inputs_for_bert_xlnet_slot_enums(slot_inputs, memory, tokenizer, device):
    '''
    inputs:
        - slot_inputs: list of {tensor(#slots, 1), None}
    outputs:
        - slot_inputs_token_ids: list of {list of tensor(len,), None}
    '''
    slot_inputs_token_ids = []

    for i in range(len(slot_inputs)):
        si = slot_inputs[i]

        # slot_inputs
        if si is not None:
            si_words = [memory['idx2slot'][x[0]] for x in si.tolist()]  # [str] * #slots
            tok_si = [
                tokenizer.tokenize(
                    ' '.join(SPLIT_MAP[x]) if x in SPLIT_MAP else x
                )
                for x in si_words
            ]
            tok_si_ids = [
                torch.LongTensor(tokenizer.convert_tokens_to_ids(x)).to(device)
                for x in tok_si
            ]  # list of tensor(len,)
            slot_inputs_token_ids.append(tok_si_ids)
        else:
            slot_inputs_token_ids.append(None)

    return slot_inputs_token_ids


def prepare_inputs_for_bert_xlnet_slot(act_slot_pairs, memory, tokenizer, device):
    '''
    inputs:
        - act_slot_pairs: list of {tensor(#value_inps, 2), None}
    outputs:
        - act_slot_pairs_token_ids: list of {list of tuple(tensor(len_act,), tensor(len_slot)), None}
    '''
    act_slot_pairs_token_ids = []

    for i in range(len(act_slot_pairs)):
        asp = act_slot_pairs[i]

        # act_slot_pairs
        if asp is not None:
            pair_ids = []
            for pair in asp.tolist():
                # act
                pr_act_word = memory['idx2act'][pair[0]]
                pr_act_tokens = tokenizer.tokenize(
                    ' '.join(SPLIT_MAP[pr_act_word]) if pr_act_word in SPLIT_MAP else pr_act_word
                )
                pr_act_token_ids = torch.LongTensor(
                    tokenizer.convert_tokens_to_ids(pr_act_tokens)).to(device)
                # slot
                pr_slot_word = memory['idx2slot'][pair[1]]
                pr_slot_tokens = tokenizer.tokenize(
                    ' '.join(SPLIT_MAP[pr_slot_word]) if pr_slot_word in SPLIT_MAP else pr_slot_word
                )
                pr_slot_token_ids = torch.LongTensor(
                    tokenizer.convert_tokens_to_ids(pr_slot_tokens)).to(device)
                # pair
                pair_ids.append((pr_act_token_ids, pr_slot_token_ids))
            act_slot_pairs_token_ids.append(pair_ids)
        else:
            act_slot_pairs_token_ids.append(None)

    return act_slot_pairs_token_ids


def prepare_inputs_for_bert_xlnet_value(value_inps, value_outs, labels, memory, tokenizer,
        cls_token_at_end=False, pad_on_left=False, cls_token='[CLS]', sep_token='[SEP]',
        pad_token=0, sequence_a_segment_id=0, sequence_b_segment_id=1,
        cls_token_segment_id=1, pad_token_segment_id=0, device=None):
    '''
    inputs:
        - value_inps: list of {list of tensor(1, len), None}
        - value_outs: list of {list of tensor(len,), None}
        - labels: list of {list of str}
    outputs:
        - value_inps_token_ids: list of {tensor(#value_inps, len), None}, tensors are padded
        - value_outs_token_ids: list of {tensor(#value_inps, len), None}
        - value_inps_pos_ids
        - value_inps_seg_ids
    '''
    pad_pos = 0

    value_inps_token_ids, value_outs_token_ids = [], []
    value_inps_pos_ids, value_inps_seg_ids = [], []

    for i in range(len(value_inps)):
        vi = value_inps[i]
        vo = value_outs[i]
        lbl = labels[i]

        # value inputs/outputs
        if vi is not None:
            assert vo is not None

            triple_lbl = [item for item in lbl if len(item.strip().split('-')) > 2]
            triple_lbl_values = [item.strip().split('-')[2] for item in triple_lbl]
            # print(vi, len(vi))
            # print(triple_lbl_values, len(triple_lbl_values))
            assert len(vi) == len(triple_lbl_values), '{}; {}'.format(vi, triple_lbl_values)

            # replace dontcare with 'dont care'
            triple_lbl_values = [v.replace('dontcare', 'dont care') for v in triple_lbl_values]

            vi_tokens = [tokenizer.tokenize(tokenizer.cls_token + ' ' + v) for v in triple_lbl_values]
            vo_tokens = [tokenizer.tokenize(v + ' ' + tokenizer.sep_token) for v in triple_lbl_values]
            vi_token_ids = [tokenizer.convert_tokens_to_ids(x) for x in vi_tokens]
            vo_token_ids = [tokenizer.convert_tokens_to_ids(x) for x in vo_tokens]

            v_pos_ids = [list(range(1, len(v) + 1)) for v in vi_tokens]
            v_seg_ids = [[sequence_a_segment_id] * len(v) for v in vi_tokens]

            # padding both value inputs and outputs
            v_lens = [len(v) for v in vi_tokens]
            padding_lens = [max(v_lens) - len(v) for v in vi_tokens]

            padded_vi_token_ids = [v + [pad_token] * padding_lens[idx] for idx, v in enumerate(vi_token_ids)]
            padded_vo_token_ids = [v + [pad_token] * padding_lens[idx] for idx, v in enumerate(vo_token_ids)]
            padded_vi_pos_ids = [p + [pad_pos] * padding_lens[idx] for idx, p in enumerate(v_pos_ids)]
            padded_vi_seg_ids = [s + [pad_token_segment_id] * padding_lens[idx] for idx, s in enumerate(v_seg_ids)]

            vi_token_ids = torch.LongTensor(padded_vi_token_ids).to(device)
            vo_token_ids = torch.LongTensor(padded_vo_token_ids).to(device)
            vi_pos_ids = torch.LongTensor(padded_vi_pos_ids).to(device)
            vi_seg_ids = torch.LongTensor(padded_vi_seg_ids).to(device)

            value_inps_token_ids.append(vi_token_ids)
            value_outs_token_ids.append(vo_token_ids)
            value_inps_pos_ids.append(vi_pos_ids)
            value_inps_seg_ids.append(vi_seg_ids)

        else:
            value_inps_token_ids.append(None)
            value_outs_token_ids.append(None)
            value_inps_pos_ids.append(None)
            value_inps_seg_ids.append(None)

    return {'value_inps': value_inps_token_ids, 'value_outs': value_outs_token_ids,
            'positions': value_inps_pos_ids, 'segments': value_inps_seg_ids}


def prepare_inputs_for_bert_xlnet_act_slot_value(act_inputs, act_slot_pairs, value_inps, value_outs, labels, memory, tokenizer,
        cls_token_at_end=False, pad_on_left=False, cls_token='[CLS]', sep_token='[SEP]',
        pad_token=0, sequence_a_segment_id=0, sequence_b_segment_id=1,
        cls_token_segment_id=1, pad_token_segment_id=0, device=None):
    '''
    inputs:
        - act_inputs: list of {tensor(#acts, 1), None}
        - act_slot_pairs: list of {tensor(#value_inps, 2), None}
          - #value_inps == #value_outs
        - value_inps: list of {list of tensor(1, len), None}
        - value_outs: list of {list of tensor(len,), None}
        - labels: list of {list of str}
    outputs:
        - act_inputs_token_ids: list of {list of tensor(len,), None}
        - act_slot_pairs_token_ids: list of {list of tuple(tensor(len_act,), tensor(len_slot)), None}
          - #tuples == #value_inps
        - value_inps_token_ids: list of {tensor(#value_inps, len), None}, tensors are padded
        - value_outs_token_ids: list of {tensor(#value_inps, len), None}
        - value_inps_pos_ids
        - value_inps_seg_ids
    '''
    assert not cls_token_at_end and not pad_on_left
    assert cls_token_segment_id == 0

    assert len(act_inputs) == len(act_slot_pairs) == len(value_inps) == len(value_outs)

    act_inputs_token_ids = prepare_inputs_for_bert_xlnet_act(act_inputs, memory, tokenizer, device)
    act_slot_pairs_token_ids = prepare_inputs_for_bert_xlnet_slot(act_slot_pairs, memory, tokenizer, device)
    results = prepare_inputs_for_bert_xlnet_value(value_inps, value_outs, labels, memory, tokenizer,
        cls_token_at_end=cls_token_at_end, pad_on_left=pad_on_left, cls_token=cls_token, sep_token=sep_token,
        pad_token=pad_token, sequence_a_segment_id=sequence_a_segment_id, sequence_b_segment_id=sequence_b_segment_id,
        cls_token_segment_id=cls_token_segment_id, pad_token_segment_id=pad_token_segment_id, device=device)

    results['act_inputs'] = act_inputs_token_ids
    results['act_slot_pairs'] = act_slot_pairs_token_ids

    return results


def prepare_inputs_one_seq(utterances, utt_ids, utt_lens, padded_utt_pos_ids, padded_utt_scores,
        system_acts, sa_ids, sa_lens, padded_sa_pos_ids, padded_sa_parents, padded_sa_sibs,
        sep_id, device):

    pad_token = 0
    pad_pos = 0
    pad_score = -1
    pad_parent = -2  # also pad sibling

    tokens, position_ids = [], []  # including utt & sysact
    scores = []
    sa_parents, sa_sibs = [], []
    utt_token_lens, sa_token_lens = [], []

    assert utt_ids.size(0) == sa_ids.size(0)
    batch_size = utt_ids.size(0)

    for i in range(batch_size):
        # utt
        utt_seq = utterances[i]
        ts = utt_ids[i].tolist()[:utt_lens[i]]  # <cls> included
        tok_ps = padded_utt_pos_ids[i].tolist()[:utt_lens[i]]
        tok_ss = padded_utt_scores[i].tolist()[:utt_lens[i]]
        tok_sps = [pad_parent] * utt_lens[i]  # PAD parent for normal utt
        tok_ssi = [pad_parent] * utt_lens[i]  # PAD sibling for normal utt

        # sep 1
        ts.append(sep_id)
        tok_ps.append(max(tok_ps) + 1)
        tok_ss.append(1.0)
        tok_sps.append(pad_parent)
        tok_ssi.append(pad_parent)

        utt_tok_l = utt_lens[i] + 1  # +1 due to SEP
        utt_token_lens.append(utt_tok_l)

        # sysact
        sa_seq = system_acts[i]
        sa_id = sa_ids[i].tolist()[1:sa_lens[i]]  # start from 1 to skip pre-defined <cls>
        sa_par_seq = padded_sa_parents[i].tolist()[1:sa_lens[i]]
        sa_par_seq = [0 if p == 0 else p - 1 + utt_tok_l for p in sa_par_seq]  # modify: 0: keep; n(!=0): n - 1 + len_utt
        sa_sib_seq = padded_sa_sibs[i].tolist()[1:sa_lens[i]]
        sa_sib_seq = [0 if s == 0 else s - 1 + utt_tok_l for s in sa_sib_seq]  # modify

        ts += sa_id
        tok_ps += list(range(max(tok_ps) + 1, max(tok_ps) + 1 + sa_lens[i] - 1))  # incremental
        tok_ss += [1.0] * (sa_lens[i] - 1)
        tok_sps += sa_par_seq
        tok_ssi += sa_sib_seq

        # sep 2
        ts.append(sep_id)
        tok_ps.append(max(tok_ps) + 1)
        tok_ss.append(1.0)
        tok_sps.append(0)  # parent is CLS
        tok_ssi.append(0)  # no sibling

        sa_tok_l = sa_lens[i]  # +1: SEP; -1: skip pre-defined <cls>
        sa_token_lens.append(sa_tok_l)

        tokens.append(ts)
        position_ids.append(tok_ps)
        scores.append(tok_ss)
        sa_parents.append(tok_sps)
        sa_sibs.append(tok_ssi)

    token_lens = [len(ids) for ids in tokens]
    max_length_of_tokens = max(token_lens)

    # padding
    padding_lengths = [max_length_of_tokens - len(ids) for ids in tokens]
    padded_tokens = [ids + [pad_token] * padding_lengths[idx] for idx, ids in enumerate(tokens)]
    padded_tok_positions = [p + [pad_pos] * padding_lengths[idx] for idx, p in enumerate(position_ids)]
    padded_tok_scores = [s + [pad_score] * padding_lengths[idx] for idx, s in enumerate(scores)]
    padded_tok_parents = [p + [pad_parent] * padding_lengths[idx] for idx, p in enumerate(sa_parents)]
    padded_tok_sibs = [s + [pad_parent] * padding_lengths[idx] for idx, s in enumerate(sa_sibs)]

    tokens_tensor = torch.tensor(padded_tokens, dtype=torch.long, device=device)
    positions_tensor = torch.tensor(padded_tok_positions, dtype=torch.long, device=device)
    scores_tensor = torch.tensor(padded_tok_scores, dtype=torch.float, device=device)
    sa_parents_tensor = torch.tensor(padded_tok_parents, dtype=torch.long, device=device)
    sa_sibs_tensor = torch.tensor(padded_tok_sibs, dtype=torch.long, device=device)

    return {'tokens': tokens_tensor, 'token_lens': token_lens, 'positions': positions_tensor,
            'utt_token_lens': utt_token_lens, 'sa_token_lens': sa_token_lens,
            'scores': scores_tensor, 'parents': sa_parents_tensor, 'sibs': sa_sibs_tensor}


def prepare_inputs_for_bert_xlnet_wcn_base(raw_in, tokenizer):

    '''
    @ input:
    - raw_in: list of {list of [(id_1, s_1), (id_2, s_2), ...]}
    @ output:
    - bert_inputs: list of {list of [([tok_id1_1, tok_id1_2, ...], s_1), ...]}
      only replace id_1 with [tok_id1_1, tok_id1_2]
      NOTE: pad bins with (pad_id, 1.0)
    '''

    batch_size = len(raw_in)
    lens = [len(item) for item in raw_in]
    max_len = max(lens)

    bert_inputs = []

    for i in range(batch_size):
        bin_inputs = []
        inp = raw_in[i]
        for word_bin in inp:
            word_inputs = []
            for word, score in word_bin:
                tok_word = tokenizer.tokenize(word)
                tok_word_id = [tokenizer._convert_token_to_id(t) for t in tok_word]
                word_inputs.append((tok_word_id, score))
            bin_inputs.append(word_inputs)
        # padding
        bin_inputs = bin_inputs + [[([tokenizer.pad_token_id], 1.0)]] * (max_len - len(bin_inputs))
        bert_inputs.append(bin_inputs)

    return bert_inputs


def prepare_inputs_for_bert_xlnet_seq_base(raw_in, tokenizer, device):
    '''
    @ input:
    - raw_in: list of strings
    @ output:
    - bert_inputs: padded Tensor
    '''

    bert_inputs = []

    for seq in raw_in:
        tok_seq = []
        for word in seq:
            tok_word = tokenizer.tokenize(word)
            tok_seq += tok_word
        bert_inputs.append([tokenizer.cls_token] + tok_seq + [tokenizer.sep_token])

    input_lens = [len(seq) for seq in bert_inputs]
    max_len = max(input_lens)

    bert_input_ids = [tokenizer.convert_tokens_to_ids(seq) + [tokenizer.pad_token_id] * (max_len - len(seq))
        for seq in bert_inputs]
    bert_input_ids = torch.tensor(bert_input_ids, dtype=torch.long, device=device)

    return bert_input_ids, input_lens


