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

REPORT_LOSS_INTERVAL = 100


def save_model(save_path, saver, session, index_to_token):
    if save_path:
        with open('{}_index_to_token.p'.format(save_path), 'wb') as fs:
            pickle.dump(index_to_token, fs)
        saved_path = saver.save(session, save_path)
        logger.info('Model saved in {}'.format(saved_path))
    return


def main(argv):
    logger.info('Command line: {}'.format(' '.join(arg for arg in argv)))

    def fmt(prog):
        return argparse.HelpFormatter(prog, max_help_position=100, width=200)

    argparser = argparse.ArgumentParser('Regularising RTE/NLI models via Adversarial Training', formatter_class=fmt)

    argparser.add_argument('--train', '-t', action='store', type=str, default='data/snli/snli_1.0_train.jsonl.gz')
    argparser.add_argument('--valid', '-v', action='store', type=str, default='data/snli/snli_1.0_dev.jsonl.gz')
    argparser.add_argument('--test', '-T', action='store', type=str, default='data/snli/snli_1.0_test.jsonl.gz')

    argparser.add_argument('--test2', action='store', type=str, default=None)

    argparser.add_argument('--model', '-m', action='store', type=str, default='ff-dam',
                           choices=['cbilstm', 'ff-dam', 'esim'])
    argparser.add_argument('--optimizer', '-o', action='store', type=str, default='adagrad',
                           choices=['adagrad', 'adam'])

    argparser.add_argument('--embedding-size', action='store', type=int, default=300)
    argparser.add_argument('--representation-size', '-r', action='store', type=int, default=200)
    argparser.add_argument('--batch-size', '-b', action='store', type=int, default=32)
    argparser.add_argument('--epochs', '-e', action='store', type=int, default=1)

    argparser.add_argument('--dropout-keep-prob', '-d', action='store', type=float, default=1.0)
    argparser.add_argument('--learning-rate', '--lr', action='store', type=float, default=0.1)
    argparser.add_argument('--clip', '-c', action='store', type=float, default=None)
    argparser.add_argument('--seed', action='store', type=int, default=0)
    argparser.add_argument('--glove', action='store', type=str, default=None)

    argparser.add_argument('--restore', action='store', type=str, default=None)
    argparser.add_argument('--save', action='store', type=str, default=None)

    argparser.add_argument('--check-interval', '--check-every', '-C',
                           action='store', type=int, default=None)

    # The following parameters are devoted to regularization
    for rule_index in range(1, 5 + 1):
        argparser.add_argument('--regularizer{}-weight'.format(rule_index),
                               '-{}'.format(rule_index),
                               action='store', type=float, default=None)

    argparser.add_argument('--regularizer-inputs', '--ri', '-R', nargs='+', type=str)
    argparser.add_argument('--regularizer-nb-samples', '--rns', '-S', type=int, default=0)
    argparser.add_argument('--regularizer-nb-flips', '--rnf', '-F', type=int, default=0)

    args = argparser.parse_args(argv)

    # Command line arguments
    train_path = args.train
    valid_path = args.valid
    test_path = args.test

    test2_path = args.test2

    model_name = args.model
    optimizer_name = args.optimizer

    embedding_size = args.embedding_size
    representation_size = args.representation_size
    batch_size = args.batch_size
    nb_epochs = args.epochs

    dropout_keep_prob = args.dropout_keep_prob
    learning_rate = args.learning_rate
    clip_value = args.clip
    seed = args.seed
    glove_path = args.glove

    restore_path = args.restore
    save_path = args.save

    check_interval = args.check_interval

    # The following parameters are devoted to regularization
    r1_weight = args.regularizer1_weight
    r2_weight = args.regularizer2_weight
    r3_weight = args.regularizer3_weight
    r4_weight = args.regularizer4_weight
    r5_weight = args.regularizer5_weight

    r_input_paths = args.regularizer_inputs or []
    nb_r_samples = args.regularizer_nb_samples
    nb_r_flips = args.regularizer_nb_flips

    r_weights = [r1_weight, r2_weight, r3_weight, r4_weight, r5_weight]
    is_regularized = not all(r_weight is None for r_weight in r_weights)

    np.random.seed(seed)
    rs = np.random.RandomState(seed)
    tf.set_random_seed(seed)

    logger.info('Reading corpus ..')

    snli = SNLI()
    train_is = snli.parse(path=train_path)
    valid_is = snli.parse(path=valid_path)
    test_is = snli.parse(path=test_path)

    test2_is = snli.parse(path=test2_path) if test2_path else None

    # Discrete/symbolic inputs used by the regularizers
    regularizer_is = [i for path in r_input_paths for i in snli.parse(path=path)]

    # Filtering out unuseful information
    regularizer_is = [
        {k: v for k, v in instance.items() if k in {'sentence1_parse_tokens', 'sentence2_parse_tokens'}}
        for instance in regularizer_is]

    all_is = train_is + valid_is + test_is

    if test2_is is not None:
        all_is += test2_is

    # Enumeration of tokens start at index=3:
    #   index=0 PADDING
    #   index=1 START_OF_SENTENCE
    #   index=2 END_OF_SENTENCE
    #   index=3 UNKNOWN_WORD
    bos_idx, eos_idx, unk_idx = 1, 2, 3

    # Words start at index 4
    start_idx = 1 + 3

    if restore_path is None:
        token_lst = [tkn for inst in all_is for tkn in inst['sentence1_parse_tokens'] + inst['sentence2_parse_tokens']]

        from collections import Counter
        token_cnt = Counter(token_lst)

        # Sort the tokens according to their frequency and lexicographic ordering
        sorted_vocabulary = sorted(token_cnt.keys(), key=lambda t: (- token_cnt[t], t))

        index_to_token = {idx: tkn for idx, tkn in enumerate(sorted_vocabulary, start=start_idx)}
    else:
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

    name_to_optimizer = {
        'adagrad': tf.train.AdagradOptimizer,
        'adam': tf.train.AdamOptimizer
    }

    name_to_model = {
        'cbilstm': ConditionalBiLSTM,
        'ff-dam': FeedForwardDAM,
        'ff-damp': FeedForwardDAMP,
        'ff-dams': FeedForwardDAMS,
        'esim': ESIM
    }

    optimizer_class = name_to_optimizer[optimizer_name]
    optimizer = optimizer_class(learning_rate=learning_rate)

    model_class = name_to_model[model_name]

    token_kwargs = dict(bos_idx=bos_idx, eos_idx=eos_idx, unk_idx=unk_idx)

    train_tensors = util.to_tensors(train_is, token_to_index, label_to_index, **token_kwargs)
    valid_tensors = util.to_tensors(valid_is, token_to_index, label_to_index, **token_kwargs)
    test_tensors = util.to_tensors(test_is, token_to_index, label_to_index, **token_kwargs)

    test2_tensors = None
    if test2_is is not None:
        test2_tensors = util.to_tensors(test2_is, token_to_index, label_to_index, **token_kwargs)

    train_sequence1 = train_tensors['sequence1']
    train_sequence1_len = train_tensors['sequence1_length']

    train_sequence2 = train_tensors['sequence2']
    train_sequence2_len = train_tensors['sequence2_length']

    train_label = train_tensors['label']

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
    valid_tensors['dropout'] = 1.0
    test_tensors['dropout'] = 1.0

    if test2_tensors is not None:
        test2_tensors['dropout'] = 1.0

    clipped_sequence1 = tfutil.clip_sentence(sequence1_ph, sequence1_len_ph)
    clipped_sequence2 = tfutil.clip_sentence(sequence2_ph, sequence2_len_ph)

    vocab_size = max(token_to_index.values()) + 1

    logger.info('Initializing the Model')

    discriminator_scope_name = 'discriminator'
    with tf.variable_scope(discriminator_scope_name):

        embedding_matrix_value = E.embedding_matrix(nb_tokens=vocab_size,
                                                    embedding_size=embedding_size,
                                                    token_to_index=token_to_index,
                                                    glove_path=glove_path,
                                                    unit_norm=True,
                                                    rs=rs, dtype=np.float32)

        embedding_layer = tf.get_variable('embeddings',
                                          initializer=tf.constant(embedding_matrix_value),
                                          trainable=False)

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

        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label_ph)
        loss = tf.reduce_mean(losses)

    if is_regularized:
        logger.info('Initializing the Regularizers')

        regularizer_placeholders = R.get_placeholders('regularizer')

        r_sequence1_ph = regularizer_placeholders['sequence1']
        r_sequence1_len_ph = regularizer_placeholders['sequence1_length']

        r_sequence2_ph = regularizer_placeholders['sequence2']
        r_sequence2_len_ph = regularizer_placeholders['sequence2_length']

        r_clipped_sequence1 = tfutil.clip_sentence(r_sequence1_ph, r_sequence1_len_ph)
        r_clipped_sequence2 = tfutil.clip_sentence(r_sequence2_ph, r_sequence2_len_ph)

        r_sequence1_embedding = tf.nn.embedding_lookup(embedding_layer, r_clipped_sequence1)
        r_sequence2_embedding = tf.nn.embedding_lookup(embedding_layer, r_clipped_sequence2)

        r_model_kwargs = model_kwargs.copy()
        r_model_kwargs.update({
            'sequence1': r_sequence1_embedding,
            'sequence1_length': r_sequence1_len_ph,

            'sequence2': r_sequence2_embedding,
            'sequence2_length': r_sequence2_len_ph
        })

        r_kwargs = {
            'model_class': model_class,
            'model_kwargs': r_model_kwargs,
            'debug': True
        }

        with tf.variable_scope(discriminator_scope_name):
            if r1_weight:
                r_loss, _ = R.contradiction_acl(is_bi=True, **r_kwargs)
                loss += r1_weight * r_loss
            if r2_weight:
                r_loss, _ = R.entailment_acl(is_bi=True, **r_kwargs)
                loss += r2_weight * r_loss
            if r3_weight:
                r_loss, _ = R.neutral_acl(is_bi=True, **r_kwargs)
                loss += r3_weight * r_loss
            if r4_weight:
                r_loss, _ = R.entailment_reflexive_acl(**r_kwargs)
                loss += r4_weight * r_loss
            if r5_weight:
                r_loss, _ = R.entailment_neutral_acl(is_bi=True, **r_kwargs)
                loss += r5_weight * r_loss

    discriminator_vars = tfutil.get_variables_in_scope(discriminator_scope_name)
    discriminator_init_op = tf.variables_initializer(discriminator_vars)

    trainable_discriminator_vars = list(discriminator_vars)
    trainable_discriminator_vars.remove(embedding_layer)

    discriminator_optimizer_scope_name = 'discriminator_optimizer'
    with tf.variable_scope(discriminator_optimizer_scope_name):
        gradients, v = zip(*optimizer.compute_gradients(loss, var_list=trainable_discriminator_vars))
        if clip_value:
            gradients, _ = tf.clip_by_global_norm(gradients, clip_value)
        training_step = optimizer.apply_gradients(zip(gradients, v))

    discriminator_optimizer_vars = tfutil.get_variables_in_scope(discriminator_optimizer_scope_name)
    discriminator_optimizer_init_op = tf.variables_initializer(discriminator_optimizer_vars)

    predictions_int = tf.cast(predictions, tf.int32)
    labels_int = tf.cast(label_ph, tf.int32)
    accuracy = tf.cast(tf.equal(x=predictions_int, y=labels_int), tf.float32)

    saver = tf.train.Saver(discriminator_vars + discriminator_optimizer_vars, max_to_keep=1)

    session_config = tf.ConfigProto()
    session_config.gpu_options.allow_growth = True
    # session_config.log_device_placement = True

    nb_r_instances = len(regularizer_is)

    r_sampler = WithoutReplacementSampler(nb_instances=nb_r_instances) if is_regularized else None
    r_generator = InstanceGenerator(token_to_index=token_to_index)

    with tf.Session(config=session_config) as session:
        logger.info('Total Parameters: {}'
                    .format(tfutil.count_trainable_parameters()))
        logger.info('Total Discriminator Parameters: {}'
                    .format(tfutil.count_trainable_parameters(var_list=discriminator_vars)))
        logger.info('Total Trainable Discriminator Parameters: {}'
                    .format(tfutil.count_trainable_parameters(var_list=trainable_discriminator_vars)))

        if restore_path is not None:
            saver.restore(session, restore_path)
        else:
            session.run([discriminator_init_op, discriminator_optimizer_init_op])

        nb_instances = train_sequence1.shape[0]
        batches = util.make_batches(size=nb_instances, batch_size=batch_size)

        loss_values = []
        best_valid_accuracy = None

        iteration_index = 0
        for epoch in range(1, nb_epochs + 1):
            order = rs.permutation(nb_instances)
            shuf_sequence1 = train_sequence1[order]
            shuf_sequence2 = train_sequence2[order]
            shuf_sequence1_len = train_sequence1_len[order]
            shuf_sequence2_len = train_sequence2_len[order]
            shuf_label = train_label[order]

            # Semi-sorting
            order = util.semi_sort(shuf_sequence1_len, shuf_sequence2_len)
            shuf_sequence1 = shuf_sequence1[order]
            shuf_sequence2 = shuf_sequence2[order]
            shuf_sequence1_len = shuf_sequence1_len[order]
            shuf_sequence2_len = shuf_sequence2_len[order]
            shuf_label = shuf_label[order]

            for batch_idx, (batch_start, batch_end) in enumerate(batches, start=1):
                iteration_index += 1

                batch_sequence1 = shuf_sequence1[batch_start:batch_end]
                batch_sequence2 = shuf_sequence2[batch_start:batch_end]
                batch_sequence1_len = shuf_sequence1_len[batch_start:batch_end]
                batch_sequence2_len = shuf_sequence2_len[batch_start:batch_end]
                batch_label = shuf_label[batch_start:batch_end]

                batch_max_size1 = np.max(batch_sequence1_len)
                batch_max_size2 = np.max(batch_sequence2_len)

                batch_sequence1 = batch_sequence1[:, :batch_max_size1]
                batch_sequence2 = batch_sequence2[:, :batch_max_size2]

                current_batch_size = batch_sequence1.shape[0]

                batch_feed_dict = {
                    sequence1_ph: batch_sequence1,
                    sequence1_len_ph: batch_sequence1_len,

                    sequence2_ph: batch_sequence2,
                    sequence2_len_ph: batch_sequence2_len,

                    label_ph: batch_label,
                    dropout_keep_prob_ph: dropout_keep_prob
                }

                if is_regularized:
                    r_instances = [regularizer_is[index] for index in r_sampler.sample(nb_r_samples)]

                    c_instances = []
                    for r_instance in r_instances:
                        r_sentence1 = r_instance['sentence1_parse_tokens']
                        r_sentence2 = r_instance['sentence2_parse_tokens']

                        f_sentence1_lst, f_sentence2_lst = r_generator.flip(r_sentence1, r_sentence2, nb_r_flips)

                        for f_sentence1, f_sentence2 in zip(f_sentence1_lst, f_sentence2_lst):
                            c_instance = {
                                'sentence1_parse_tokens': f_sentence1,
                                'sentence2_parse_tokens': f_sentence2
                            }
                            c_instances += [c_instance]

                    r_instances += c_instances
                    r_tensors = util.to_tensors(r_instances, token_to_index, label_to_index, **token_kwargs)

                    assert len(r_instances) == r_tensors['sequence1'].shape[0]
                    # logging.info('Regularising on {} samples ..'.format(len(r_instances)))

                    batch_feed_dict.update({
                        r_sequence1_ph: r_tensors['sequence1'],
                        r_sequence1_len_ph: r_tensors['sequence1_length'],

                        r_sequence2_ph: r_tensors['sequence2'],
                        r_sequence2_len_ph: r_tensors['sequence2_length'],
                    })

                _, loss_value = session.run([training_step, loss], feed_dict=batch_feed_dict)
                loss_values += [loss_value / current_batch_size]

                if len(loss_values) >= REPORT_LOSS_INTERVAL:
                    logger.info("Epoch {0}, Batch {1}\tLoss: {2}".format(epoch, batch_idx, util.stats(loss_values)))
                    loss_values = []

                # every k iterations, check whether accuracy improves
                if check_interval is not None and iteration_index % check_interval == 0:
                    accuracies_valid = evaluation.evaluate(session, valid_tensors, placeholders, accuracy, batch_size=256)
                    accuracies_test = evaluation.evaluate(session, test_tensors, placeholders, accuracy, batch_size=256)

                    accuracies_test2 = None
                    if test2_tensors is not None:
                        accuracies_test2 = evaluation.evaluate(session, test2_tensors, placeholders, accuracy,
                                                               batch_size=256)

                    logger.info("Epoch {0}\tBatch {1}\tValidation Accuracy: {2}, Test Accuracy: {3}"
                                .format(epoch, batch_idx, util.stats(accuracies_valid), util.stats(accuracies_test)))

                    if accuracies_test2 is not None:
                        logger.info("Epoch {0}\tBatch {1}\tValidation Accuracy: {2}, Test2 Accuracy: {3}"
                                    .format(epoch, batch_idx, util.stats(accuracies_valid),
                                            util.stats(accuracies_test2)))

                    if best_valid_accuracy is None or best_valid_accuracy < np.mean(accuracies_valid):
                        best_valid_accuracy = np.mean(accuracies_valid)
                        logger.info("Epoch {0}\tBatch {1}\tBest Validation Accuracy: {2}, Test Accuracy: {3}"
                                    .format(epoch, batch_idx, util.stats(accuracies_valid), util.stats(accuracies_test)))

                        if accuracies_test2 is not None:
                            logger.info("Epoch {0}\tBatch {1}\tBest Validation Accuracy: {2}, Test2 Accuracy: {3}"
                                        .format(epoch, batch_idx, util.stats(accuracies_valid),
                                                util.stats(accuracies_test2)))

                        save_model(save_path, saver, session, index_to_token)

            # End of epoch statistics
            accuracies_valid = evaluation.evaluate(session, valid_tensors, placeholders, accuracy, batch_size=256)
            accuracies_test = evaluation.evaluate(session, test_tensors, placeholders, accuracy, batch_size=256)

            accuracies_test2 = None
            if test2_tensors is not None:
                accuracies_test2 = evaluation.evaluate(session, test2_tensors, placeholders, accuracy, batch_size=256)

            logger.info("Epoch {0}\tValidation Accuracy: {1}, Test Accuracy: {2}"
                        .format(epoch, util.stats(accuracies_valid), util.stats(accuracies_test)))

            if accuracies_test2 is not None:
                logger.info("Epoch {0}\tValidation Accuracy: {1}, Test2 Accuracy: {2}"
                            .format(epoch, util.stats(accuracies_valid), util.stats(accuracies_test2)))

            if best_valid_accuracy is None or best_valid_accuracy < np.mean(accuracies_valid):
                best_valid_accuracy = np.mean(accuracies_valid)
                logger.info("Epoch {0}\tBest Validation Accuracy: {1}, Test Accuracy: {2}"
                            .format(epoch, util.stats(accuracies_valid), util.stats(accuracies_test)))

                if accuracies_test2 is not None:
                    logger.info("Epoch {0}\tBest Validation Accuracy: {1}, Test2 Accuracy: {2}"
                                .format(epoch, util.stats(accuracies_valid), util.stats(accuracies_test2)))

                save_model(save_path, saver, session, index_to_token)

    logger.info('Training finished.')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])
