import torch
import numpy as np
import time


def read_glove_emb(file_path, device, word2idx):
    print('Read embedding from', file_path)
    start = time.time()
    word_to_vec = {}
    with open(file_path) as f:
        for i, line in enumerate(f):
            first_space_index = line.index(' ')
            word = line[:first_space_index]
            values = line[first_space_index + 1:]
            vector = np.fromstring(values, sep=' ', dtype=np.float)
            word_to_vec[word] = vector

    vec_list, ext_word2idx ,count = [], {}, -1
    for word in word_to_vec:
        if word in word2idx:
            count += 1
            vec_list.append(torch.tensor(word_to_vec[word], device=device))
            ext_word2idx[word] = count

    print('Done. Elapsed time: %s s' % (time.time() - start))

    return ext_word2idx, vec_list


def get_glove_emb_dim(file_path):
    '''only return vector dim for checking'''
    with open(file_path) as f:
        for line in f:
            first_space_index = line.index(' ')
            values = line[first_space_index + 1:]
            vector = np.fromstring(values, sep=' ', dtype=np.float)
            return len(vector)
