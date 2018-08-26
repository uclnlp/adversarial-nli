#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse

import os
import sys

import pickle

import numpy as np
import tensorflow as tf

from nnli.parser import SNLI

from nnli import util
from nnli import tfutil
from nnli import embeddings as E
from nnli import evaluation

from nnli.models import ConditionalBiLSTM
from nnli.models import FeedForwardDAM
from nnli.models import FeedForwardDAMP
from nnli.models import FeedForwardDAMS
from nnli.models import ESIM

import nnli.regularizers as R

from nnli.samplers import WithoutReplacementSampler
from nnli.generators import InstanceGenerator

import logging

logger = logging.getLogger(os.path.basename(sys.argv[0]))


def main(argv):
    logger.info('Command line: {}'.format(' '.join(arg for arg in argv)))

    def fmt(prog):
        return argparse.HelpFormatter(prog, max_help_position=100, width=200)

    argparser = argparse.ArgumentParser('Evaluating Neural NLI models via Adversarial Training', formatter_class=fmt)

    argparser.add_argument('--data', '-d', action='store', type=str, default='data/snli/snli_1.0_train.jsonl.gz')

    argparser.add_argument('--model', '-m', action='store', type=str, default='ff-dam',
                           choices=['cbilstm', 'ff-dam', 'esim'])

    argparser.add_argument('--embedding-size', '-e', action='store', type=int, default=300)
    argparser.add_argument('--representation-size', '-r', action='store', type=int, default=200)

    argparser.add_argument('--restore', action='store', type=str, default=None)

    args = argparser.parse_args(argv)

    # Command line arguments
    data_path = args.data

    model_name = args.model

    embedding_size = args.embedding_size
    representation_size = args.representation_size

    restore_path = args.restore

    seed = 0

    np.random.seed(seed)
    tf.set_random_seed(seed)

    logger.info('Reading corpus ..')

    snli = SNLI()
    data_is = snli.parse(path=data_path)

    bos_idx, eos_idx, unk_idx = 1, 2, 3

    assert restore_path is not None

    vocab_path = '{}_index_to_token.p'.format(restore_path)

    logger.info('Restoring vocabulary from {} ..'.format(vocab_path))
    with open(vocab_path, 'rb') as f:
        index_to_token = pickle.load(f)

    token_to_index = {token: index for index, token in index_to_token.items()}

    entailment_idx, neutral_idx, contradiction_idx = 0, 1, 2

    label_to_index = {
        'entailment': entailment_idx,
        'neutral': neutral_idx,
        'contradiction': contradiction_idx,
    }

    name_to_model = {
        'cbilstm': ConditionalBiLSTM,
        'ff-dam': FeedForwardDAM,
        'ff-damp': FeedForwardDAMP,
        'ff-dams': FeedForwardDAMS,
        'esim': ESIM
    }

    model_class = name_to_model[model_name]

    token_kwargs = dict(bos_idx=bos_idx, eos_idx=eos_idx, unk_idx=unk_idx)

    data_tensors = util.to_tensors(data_is, token_to_index, label_to_index, **token_kwargs)

    sequence1_ph = tf.placeholder(dtype=tf.int32, shape=[None, None], name='sequence1')
    sequence1_len_ph = tf.placeholder(dtype=tf.int32, shape=[None], name='sequence1_length')

    sequence2_ph = tf.placeholder(dtype=tf.int32, shape=[None, None], name='sequence2')
    sequence2_len_ph = tf.placeholder(dtype=tf.int32, shape=[None], name='sequence2_length')

    label_ph = tf.placeholder(dtype=tf.int32, shape=[None], name='label')
    dropout_keep_prob_ph = tf.placeholder(tf.float32, name='dropout_keep_prob')

    placeholders = {
        'sequence1': sequence1_ph,
        'sequence1_length': sequence1_len_ph,
        'sequence2': sequence2_ph,
        'sequence2_length': sequence2_len_ph,
        'label': label_ph,
        'dropout': dropout_keep_prob_ph
    }

    # Disable Dropout at evaluation time
    data_tensors['dropout'] = 1.0

    clipped_sequence1 = tfutil.clip_sentence(sequence1_ph, sequence1_len_ph)
    clipped_sequence2 = tfutil.clip_sentence(sequence2_ph, sequence2_len_ph)

    logger.info('Initializing the Model')

    discriminator_scope_name = 'discriminator'
    with tf.variable_scope(discriminator_scope_name):
        vocab_size = max(token_to_index.values()) + 1
        embedding_layer = tf.get_variable('embeddings', shape=[vocab_size, embedding_size])

        sequence1_embedding = tf.nn.embedding_lookup(embedding_layer, clipped_sequence1)
        sequence2_embedding = tf.nn.embedding_lookup(embedding_layer, clipped_sequence2)

        model_kwargs = {
            'sequence1': sequence1_embedding,
            'sequence1_length': sequence1_len_ph,
            'sequence2': sequence2_embedding,
            'sequence2_length': sequence2_len_ph,
            'representation_size': representation_size,
            'dropout_keep_prob': dropout_keep_prob_ph
        }

        model = model_class(**model_kwargs)

        logits = model()

        predictions = tf.argmax(logits, axis=1, name='predictions')

    discriminator_vars = tfutil.get_variables_in_scope(discriminator_scope_name)

    trainable_discriminator_vars = list(discriminator_vars)
    trainable_discriminator_vars.remove(embedding_layer)

    predictions_int = tf.cast(predictions, tf.int32)
    labels_int = tf.cast(label_ph, tf.int32)
    accuracy = tf.cast(tf.equal(x=predictions_int, y=labels_int), tf.float32)

    saver = tf.train.Saver(discriminator_vars, max_to_keep=1)

    session_config = tf.ConfigProto()
    session_config.gpu_options.allow_growth = True
    # session_config.log_device_placement = True

    with tf.Session(config=session_config) as session:
        logger.info('Total Parameters: {}'
                    .format(tfutil.count_trainable_parameters()))
        logger.info('Total Discriminator Parameters: {}'
                    .format(tfutil.count_trainable_parameters(var_list=discriminator_vars)))
        logger.info('Total Trainable Discriminator Parameters: {}'
                    .format(tfutil.count_trainable_parameters(var_list=trainable_discriminator_vars)))

        saver.restore(session, restore_path)

        accuracies = evaluation.evaluate(session, data_tensors, placeholders, accuracy, batch_size=256)

        logger.info("Accuracy: {}".format(util.stats(accuracies)))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])
