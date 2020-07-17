import torch


def convert_labels(bottom_labels, b2t_mat):
    b2t_mat = b2t_mat.to(bottom_labels.device)
    top_labels = torch.mm(bottom_labels, b2t_mat)
    return top_labels


def reverse_top2bottom(t2b):
    '''
    t2b: top-to-bottom dict
    '''
    b2t = {}
    for k, vs in t2b.items():
        for v in vs:
            if v in b2t:
                raise ValueError('map from bottom to top should be unique')
            b2t[v] = k

    tvs = len(t2b)  # top labels vocab size
    bvs = len(b2t)  # bottom labels vocab size
    b2t_list = [b2t[i] for i in range(bvs)]
    b2t_mat = torch.zeros(bvs, tvs)
    b2t_mat[range(bvs), b2t_list] = 1
    return b2t_mat


def onehot_to_scalar(bottom_labels):
    '''
    bottom_labels -> (b, #bottom_labels)
    '''
    # ensure that each top-label has ONLY ONE or NO bottom labels
    assert bottom_labels.sum(dim=1).le(1).all()

    scalar_labels = bottom_labels.max(dim=1)[1]
    # e.g.
    # bottom_labels = [[0., 0., 0.],
    #                  [0., 1., 0.],
    #                  [0., 0., 0.],
    #                  [0., 0., 0.],
    #                  [1., 0., 0.]]
    # scalar_labels = [2, 1, 2, 2, 0]
    #     NOTE: max()之后，第1/3/4行得到的index已经是2（#column-1）
    #           但以防万一，计算一个mask并手动计算该行的index

    none_mask = bottom_labels.sum(dim=1).eq(0)  # sum为0的行 => mask为1
    # none_mask = [1, 0, 1, 1, 0]
    scalar_labels = scalar_labels.masked_fill(none_mask, bottom_labels.size(1) - 1)

    return scalar_labels

