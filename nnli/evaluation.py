# -*- coding: utf-8 -*-

import numpy as np

from nnli import util


def evaluate(session, tensors, placeholders, metric, batch_size=None):
    feed_dict = {
        placeholders[key]: tensors[key] for key in placeholders
    }

    if batch_size is None:
        res = session.run(metric, feed_dict=feed_dict)
    else:
        res_lst = []
        tensor_names = [name for name in tensors.keys() if name != 'dropout']
        tensor_name = tensor_names[0]
        nb_instances = tensors[tensor_name].shape[0]
        batches = util.make_batches(size=nb_instances, batch_size=batch_size)
        for batch_start, batch_end in batches:

            def get_batch(tensor):
                return tensor[batch_start:batch_end] if not isinstance(tensor, float) else tensor

            batch_feed_dict = {
                ph: get_batch(tensor) for ph, tensor in feed_dict.items()
            }

            batch_res = session.run(metric, feed_dict=batch_feed_dict)
            res_lst += batch_res.tolist()
        res = np.array(res_lst)
    return res
