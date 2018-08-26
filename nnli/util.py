# -*- coding: utf-8 -*-

import numpy as np

from nnli.padding import pad_sequences


def make_batches(size, batch_size):
    """
    Returns a list of batch indices (tuples of indices).

    :param size: Size of the dataset (number of examples).
    :param batch_size: Batch size.
    :return: List of batch indices (tuples of indices).
    """
    nb_batch = int(np.ceil(size / float(batch_size)))
    res = [(i * batch_size, min(size, (i + 1) * batch_size)) for i in range(0, nb_batch)]
    return res


def stats(values):
    mean, std = np.mean(values), np.std(values)
    return '{0:.4f} ± {1:.4f}'.format(mean, std)


def semi_sort(sizes1, sizes2):
    batch_1 = np.logical_and(sizes1 <= 20, sizes2 <= 20)
    batch_2 = np.logical_and(np.logical_and(20 < sizes1, sizes1 < 50), np.logical_and(20 < sizes2, sizes2 < 50))
    batch_3 = np.logical_not(np.logical_or(batch_1, batch_2))
    batch_1_idx, = np.where(batch_1)
    batch_2_idx, = np.where(batch_2)
    batch_3_idx, = np.where(batch_3)
    return np.concatenate((batch_1_idx, batch_2_idx, batch_3_idx))


def to_tensors(instances, token_to_index, label_to_index,
               bos_idx=1, eos_idx=2, unk_idx=3):
    sentence1_idx_lst, sentence2_idx_lst, label_idx_lst = [], [], []

    for instance in instances:
        sentence1_idx = [bos_idx]
        sentence2_idx = [bos_idx]

        sentence1_idx += [token_to_index.get(token, unk_idx) for token in instance['sentence1_parse_tokens']]
        sentence2_idx += [token_to_index.get(token, unk_idx) for token in instance['sentence2_parse_tokens']]

        # sentence1_idx += [eos_idx]
        # sentence2_idx += [eos_idx]

        sentence1_idx_lst += [sentence1_idx]
        sentence2_idx_lst += [sentence2_idx]

        if 'gold_label' in instance:
            label = instance['gold_label']
            label_idx = label_to_index[label]
            label_idx_lst += [label_idx]

    sentence1_len_lst = [len(sen) for sen in sentence1_idx_lst]
    sentence2_len_lst = [len(sen) for sen in sentence2_idx_lst]

    name_to_tensor = None
    if len(instances) > 0:
        name_to_tensor = {
            'sequence1': pad_sequences(sentence1_idx_lst),
            'sequence1_length': np.array(sentence1_len_lst),
            'sequence2': pad_sequences(sentence2_idx_lst),
            'sequence2_length': np.array(sentence2_len_lst)
        }

        if len(label_idx_lst) > 0:
            name_to_tensor.update({
                'label': np.array(label_idx_lst)
            })

    return name_to_tensor
