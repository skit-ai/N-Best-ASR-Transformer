''' Define the Bert-like Transformer model '''
import torch
import torch.nn as nn
import numpy as np
import math
import copy


def get_slf_attn_w(score_seq):
    _, l = score_seq.size()
    attn_w = score_seq.unsqueeze(1).expand(-1, l, -1)  # b x l x l
    return attn_w


def get_attn_key_pad_mask(seq_k, seq_q):
    ''' For masking out the padding part of key sequence. '''
    # Expand to fit the shape of key query attention matrix.
    len_q = seq_q.size(1)
    pad_id = 0
    padding_mask = seq_k.eq(pad_id)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk
    return padding_mask


class BertSelfAttention(nn.Module):
    def __init__(self, bert_config, bert_self_attention, score_util):
        super().__init__()

        self.rm_qkv = False
        self.score_util = score_util  # mul/np/pp/none

        if not self.rm_qkv:
            self.query = copy.deepcopy(bert_self_attention.query)
            self.key = copy.deepcopy(bert_self_attention.key)
            self.value = copy.deepcopy(bert_self_attention.value)

        self.num_attention_heads = bert_config.num_attention_heads
        self.attention_head_size = int(bert_config.hidden_size / bert_config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.dropout = nn.Dropout(bert_config.attention_probs_dropout_prob)

        if self.score_util == 'pp':
            self.score_lambda = nn.Parameter(torch.randn(1, bert_config.num_attention_heads, 1, 1))  # (1, nh, 1, 1)
            # (1) ones
            # nn.init.ones_(self.score_lambda)
            # (2) normal
            nn.init.normal_(self.score_lambda, 1, 0.1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask, head_mask, attn_w):

        if not self.rm_qkv:
            mixed_query_layer = self.query(hidden_states)
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)
        else:
            mixed_query_layer = hidden_states
            mixed_key_layer = hidden_states
            mixed_value_layer = hidden_states

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        # print('query', self.query.weight.size(), self.query.weight.max().data, self.query.weight.min().data, self.query.weight.mean().data)
        # print('hidden', hidden_states.size(), hidden_states.max().data, hidden_states.min().data, hidden_states.mean().data)
        # print('q layer', query_layer.size(), query_layer.max().data, query_layer.min().data, query_layer.mean().data)
        # print('k layer', key_layer.size(), key_layer.max().data, key_layer.min().data, key_layer.mean().data)
        # x = attention_scores[0, 4, :, :]
        # print('ori attn', x.size(), x.max().data, x.min().data, x.mean().data)
        # print(x)

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)  # (b, nh, l, l)

        if attn_w is not None:
            if self.score_util == 'pp':
                attn_w = attn_w.unsqueeze(1).expand(-1, self.num_attention_heads, -1, -1)
                posterior_scores = torch.mul(self.score_lambda, attn_w.data)
                attention_scores = attention_scores + posterior_scores
                # x2 = attn_w[0, 4, :, :]
                # y = self.score_lambda.data[0, 4, 0, 0]
                # print('attn weight', x2.size(), x2.max().data, x2.min().data, x2.mean().data)
                # print('lambda', y)
                # x3 = posterior_scores[0, 4, :, :]
                # print('lambda * weight', x3.size(), x3.max().data, x3.min().data, x3.mean().data)
                # input()
            elif self.score_util == 'mul':
                attn_w = attn_w.unsqueeze(1).expand(-1, self.num_attention_heads, -1, -1)
                attention_scores = attention_scores * attn_w.data
            elif self.score_util == 'np':
                attn_w = attn_w.unsqueeze(1).expand(-1, self.num_attention_heads, -1, -1)
                attention_scores = attention_scores + attn_w.data
            elif self.score_util == 'none':
                pass

        # pad mask
        attention_mask = attention_mask.unsqueeze(1).expand(-1, self.num_attention_heads, -1, -1)  # (b, nh, l, l)
        attention_scores = attention_scores.masked_fill(attention_mask, -np.inf)

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)

        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)  # (b, nh, l, dk)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()  # (b, l, nh, dk)
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)  # (b, l, d)

        return context_layer, attention_probs


class BertAttention(nn.Module):
    def __init__(self, bert_config, bert_attention, dp, score_util):
        super().__init__()
        self.output = copy.deepcopy(bert_attention.output)
        if dp != 0.1:
            self.output.dropout = nn.Dropout(dp)

        bert_self_attention = copy.deepcopy(bert_attention.self)
        self.self = BertSelfAttention(bert_config, bert_self_attention, score_util)

    def forward(self, hidden_states, attention_mask, head_mask, slf_attn_w):
        self_outputs = self.self(hidden_states, attention_mask, head_mask, slf_attn_w)
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class BertLayer(nn.Module):
    def __init__(self, bert_config, bert_layer, dp, score_util):
        super().__init__()
        self.is_decoder = False
        self.intermediate = copy.deepcopy(bert_layer.intermediate)
        self.output = copy.deepcopy(bert_layer.output)
        if dp != 0.1:
            self.output.dropout = nn.Dropout(dp)

        bert_attention = copy.deepcopy(bert_layer.attention)
        self.attention = BertAttention(bert_config, bert_attention, dp, score_util)

    def forward(self, hidden_states, attention_mask, head_mask, slf_attn_w):
        self_attention_outputs = self.attention(hidden_states, attention_mask, head_mask, slf_attn_w)
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output, outputs


class BertEncoder(nn.Module):
    def __init__(self, bert_config, pretrained_model_opts,
            score_util, n_layers=12):
        super().__init__()
        self.bert_config = bert_config
        self.pretrained_model_opts = pretrained_model_opts
        self.n_layers = n_layers
        self.score_util = score_util

        # load bert
        bert_model = pretrained_model_opts['model']
        dp = pretrained_model_opts['dp']

        self.embeddings = copy.deepcopy(bert_model.embeddings)
        if dp != 0.1:
            self.embeddings.dropout = nn.Dropout(dp)

        bert_layers = copy.deepcopy(bert_model.encoder.layer)
        self.layer_stack = nn.ModuleList([
            BertLayer(bert_config, bert_layers[i], dp, score_util)
            for i in range(self.n_layers)
        ])

    def forward(self, inputs, attention_mask=None, return_attns=False, default_pos=False):
        src_seq, src_pos, src_score, src_type = \
            inputs['tokens'], inputs['positions'], inputs['scores'], inputs['segments']

        # BERT default position embedding
        if default_pos:
            src_pos = None

        enc_output = self.embeddings(src_seq, src_type, src_pos)  # (b, l, d)

        if self.score_util != 'none':
            slf_attn_w = get_slf_attn_w(src_score)  # (b, l, l)
        else:
            slf_attn_w = None

        if attention_mask is None:
            attention_mask = get_attn_key_pad_mask(src_seq, src_seq)  # (b, l, l)

        enc_slf_attn_list = []
        for i, enc_layer in enumerate(self.layer_stack):
            enc_output, enc_slf_attn = enc_layer(
                enc_output, attention_mask, head_mask=None,
                slf_attn_w=slf_attn_w)
            if return_attns:
                enc_slf_attn_list.append(enc_slf_attn)

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output
