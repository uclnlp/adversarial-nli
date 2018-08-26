# -*- coding: utf-8 -*-

import tensorflow as tf


def get_placeholders(prefix):
    sequence1_ph = tf.placeholder(dtype=tf.int32, shape=[None, None], name='{}_sequence1'.format(prefix))
    sequence1_len_ph = tf.placeholder(dtype=tf.int32, shape=[None], name='{}_sequence1_length'.format(prefix))

    sequence2_ph = tf.placeholder(dtype=tf.int32, shape=[None, None], name='{}_sequence2'.format(prefix))
    sequence2_len_ph = tf.placeholder(dtype=tf.int32, shape=[None], name='{}_sequence2_length'.format(prefix))

    placeholders = {
        'sequence1': sequence1_ph,
        'sequence1_length': sequence1_len_ph,
        'sequence2': sequence2_ph,
        'sequence2_length': sequence2_len_ph
    }

    return placeholders
