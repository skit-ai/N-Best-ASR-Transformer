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


class BertSelfAttention(nn.Module):
    def __init__(self, bert_config, bert_self_attention):
        super().__init__()

        self.rm_qkv = False

        if not self.rm_qkv:
            self.query = copy.deepcopy(bert_self_attention.query)
            self.key = copy.deepcopy(bert_self_attention.key)
            self.value = copy.deepcopy(bert_self_attention.value)

        self.num_attention_heads = bert_config.num_attention_heads
        self.attention_head_size = int(bert_config.hidden_size / bert_config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.dropout = nn.Dropout(bert_config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask, head_mask):

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
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)  # (b, nh, l, l)

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
    def __init__(self, bert_config, bert_attention, dp):
        super().__init__()
        self.output = copy.deepcopy(bert_attention.output)
        if dp != 0.1:
            self.output.dropout = nn.Dropout(dp)

        bert_self_attention = copy.deepcopy(bert_attention.self)
        self.self = BertSelfAttention(bert_config, bert_self_attention)

    def forward(self, hidden_states, attention_mask, head_mask):
        self_outputs = self.self(hidden_states, attention_mask, head_mask)
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class BertLayer(nn.Module):
    def __init__(self, bert_config, bert_layer, dp):
        super().__init__()
        self.is_decoder = False
        self.intermediate = copy.deepcopy(bert_layer.intermediate)
        self.output = copy.deepcopy(bert_layer.output)
        if dp != 0.1:
            self.output.dropout = nn.Dropout(dp)

        bert_attention = copy.deepcopy(bert_layer.attention)
        self.attention = BertAttention(bert_config, bert_attention, dp)

    def forward(self, hidden_states, attention_mask, head_mask):
        self_attention_outputs = self.attention(hidden_states, attention_mask, head_mask)
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output, outputs


class BertEncoder(nn.Module):
    def __init__(self, bert_config, pretrained_model_opts,
            n_layers):
        super().__init__()
        self.bert_config = bert_config
        self.pretrained_model_opts = pretrained_model_opts
        self.n_layers = n_layers

        # load bert
        bert_model = pretrained_model_opts['model']
        dp = pretrained_model_opts['dp']

        assert pretrained_model_opts['used_emb']
        self.embeddings = copy.deepcopy(bert_model.embeddings)
        if dp != 0.1:
            self.embeddings.dropout = nn.Dropout(dp)

        assert pretrained_model_opts['used_hid']
        bert_layers = copy.deepcopy(bert_model.encoder.layer)
        self.layer_stack = nn.ModuleList([
            BertLayer(bert_config, bert_layers[i], dp)
            for i in range(self.n_layers)
        ])

    def forward(self, inputs, attention_mask=None, return_attns=False):
        src_seq, src_pos, src_type = \
            inputs['tokens'], inputs['positions'], inputs['segments']

        enc_output = self.embeddings(src_seq, src_type, src_pos)  # (b, l, d)

        if attention_mask is None:
            attention_mask = get_attn_key_pad_mask(src_seq, src_seq)  # (b, l, l)

        enc_slf_attn_list = []
        for i, enc_layer in enumerate(self.layer_stack):
            enc_output, enc_slf_attn = enc_layer(
                enc_output, attention_mask, head_mask=None)
            if return_attns:
                enc_slf_attn_list.append(enc_slf_attn)

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output
