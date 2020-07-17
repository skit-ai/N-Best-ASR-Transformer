import torch
import torch.nn as nn
import torch.nn.functional as F


class HierarchicalClassifier(nn.Module):
    def __init__(self, top2bottom_dict, input_dim, n_bottom_labels, dropout):
        super(HierarchicalClassifier, self).__init__()

        self.top2bottom_dict = top2bottom_dict
        self.n_bottom_labels = n_bottom_labels

        # top classifier
        self.n_top_labels = len(top2bottom_dict)
        self.top_linear_layer = nn.Linear(input_dim, self.n_top_labels)

        # bottom classifiers
        linear_layers = {}
        for k, v in top2bottom_dict.items():
            # k: top-label idx
            # v: list of botton-label indices
            n_class = len(v)
            if n_class >= 2:
                linear_layers['lin_%d' % (k)] = nn.Linear(input_dim, n_class)
        self.linear_layers = nn.ModuleDict(linear_layers)

        self.dropout_layer = nn.Dropout(dropout)

    def init_weight(self, init_range=0.2):
        params = self.parameters()
        for param in params:
            nn.init.uniform_(param, a=-init_range, b=init_range)


    def forward(self, features):
        '''
        features: (b, dim)
        '''
        b = features.size(0)

        top_out = self.top_linear_layer(self.dropout_layer(features))  # (b, n_top)
        top_scores = torch.sigmoid(top_out)  # (b, n_top)

        bottom_scores_dict = {}
        for k in self.linear_layers.keys():
            out = self.linear_layers[k](self.dropout_layer(features))  # (b, n_bottom[k])
            scores = torch.softmax(out, dim=1)
            bottom_scores_dict[k] = scores

        final_scores = torch.empty(b, self.n_bottom_labels).to(features.device)
        for i in range(self.n_top_labels):
            k = 'lin_%d' % i
            bottom_ids = self.top2bottom_dict[i]
            if len(bottom_ids) >= 2:
                assert k in self.linear_layers.keys()
                final_scores[:, bottom_ids] = top_scores[:, i].unsqueeze(1) * bottom_scores_dict[k]
            else:
                final_scores[:, bottom_ids] = top_scores[:, i].unsqueeze(1)

        return top_scores, bottom_scores_dict, final_scores
