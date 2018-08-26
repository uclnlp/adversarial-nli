#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse

import os
import sys
import json

import pickle
import copy

import numpy as np
import tensorflow as tf

from nnli.parser import SNLI
from nnli.util import make_batches

from nnli import util, tfutil
from nnli.models import ConditionalBiLSTM
from nnli.models import FeedForwardDAM
from nnli.models import ESIM

import nnli.regularizers.base as R

from nnli.generators.base import InstanceGenerator
from nnli.generators.scorer import LMScorer
from nnli.generators.scorer import InstanceScorer

import logging

logger = logging.getLogger(os.path.basename(sys.argv[0]))

# PYTHONPATH=. python3 ./bin/incubator/nli-generate-cli.py --top-k 5 -1 1.0 --epsilon 10.0 --flip
# --nb-examples-per-batch 32


def main(argv):
    logger.info('Command line: {}'.format(' '.join(arg for arg in argv)))

    def fmt(prog):
        return argparse.HelpFormatter(prog, max_help_position=100, width=200)

    argparser = argparse.ArgumentParser('Generating adversarial samples for NLI models', formatter_class=fmt)

    argparser.add_argument('--data', '-d', action='store', type=str, default='data/snli/snli_1.0_train.jsonl.gz')

    argparser.add_argument('--model', '-m', action='store', type=str, default='ff-dam',
                           choices=['cbilstm', 'ff-dam', 'esim1'])

    argparser.add_argument('--embedding-size', action='store', type=int, default=300)
    argparser.add_argument('--representation-size', action='store', type=int, default=200)

    argparser.add_argument('--batch-size', action='store', type=int, default=32)

    argparser.add_argument('--seed', action='store', type=int, default=0)

    argparser.add_argument('--restore', action='store', type=str, default='saved/snli/dam/2/dam')

    # The following parameters are devoted to regularization
    for rule_index in range(1, 5 + 1):
        argparser.add_argument('--regularizer{}-weight'.format(rule_index), '-{}'.format(rule_index),
                               action='store', type=float, default=None)

    # Parameters for adversarial training
    argparser.add_argument('--lm', action='store', type=str, default='saved/snli/lm/1/', help='Language Model')

    # XXX: default to None (disable) - 0.01
    argparser.add_argument('--epsilon', '--eps', action='store', type=float, default=None)
    argparser.add_argument('--nb-corruptions', '--nc', action='store', type=int, default=32)
    argparser.add_argument('--nb-examples-per-batch', '--nepb', action='store', type=int, default=4)

    # XXX: default to -1 (disable) - 4
    argparser.add_argument('--top-k', action='store', type=int, default=-1)

    argparser.add_argument('--flip', '-f', action='store_true', default=False)
    argparser.add_argument('--combine', '-c', action='store_true', default=False)
    argparser.add_argument('--remove', '-r', action='store_true', default=False)
    argparser.add_argument('--scramble', '-s', action='store', type=int, default=None)

    argparser.add_argument('--json', action='store', type=str, default=None)

    args = argparser.parse_args(argv)

    # Command line arguments
    data_path = args.data

    model_name = args.model

    embedding_size = args.embedding_size
    representation_size = args.representation_size

    batch_size = args.batch_size

    seed = args.seed

    restore_path = args.restore

    # The following parameters are devoted to regularization
    r1_weight = args.regularizer1_weight
    r2_weight = args.regularizer2_weight
    r3_weight = args.regularizer3_weight
    r4_weight = args.regularizer4_weight
    r5_weight = args.regularizer5_weight

    lm_path = args.lm
    epsilon = args.epsilon
    nb_corruptions = args.nb_corruptions
    nb_examples_per_batch = args.nb_examples_per_batch
    top_k = args.top_k
    is_flip = args.flip
    is_combine = args.combine
    is_remove = args.remove
    scramble = args.scramble

    json_path = args.json

    np.random.seed(seed)
    tf.set_random_seed(seed)

    logger.debug('Reading corpus ..')

    snli = SNLI()
    data_is = snli.parse(path=data_path)

    logger.info('Data size: {}'.format(len(data_is)))

    # Enumeration of tokens start at index=3:
    #   index=0 PADDING
    #   index=1 START_OF_SENTENCE
    #   index=2 END_OF_SENTENCE
    #   index=3 UNKNOWN_WORD
    bos_idx, eos_idx, unk_idx = 1, 2, 3

    # Words start at index 4
    start_idx = 1 + 3

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
    index_to_label = {k: v for v, k in label_to_index.items()}

    token_kwargs = dict(bos_idx=bos_idx, eos_idx=eos_idx, unk_idx=unk_idx)

    data_tensors = util.to_tensors(data_is, token_to_index, label_to_index, **token_kwargs)

    sequence1 = data_tensors['sequence1']
    sequence1_len = data_tensors['sequence1_length']

    sequence2 = data_tensors['sequence2']
    sequence2_len = data_tensors['sequence2_length']

    label = data_tensors['label']

    sequence1_ph = tf.placeholder(dtype=tf.int32, shape=[None, None], name='sequence1')
    sequence1_len_ph = tf.placeholder(dtype=tf.int32, shape=[None], name='sequence1_length')

    sequence2_ph = tf.placeholder(dtype=tf.int32, shape=[None, None], name='sequence2')
    sequence2_len_ph = tf.placeholder(dtype=tf.int32, shape=[None], name='sequence2_length')

    dropout_keep_prob_ph = tf.placeholder(tf.float32, name='dropout_keep_prob')

    data_tensors['dropout'] = 1.0

    clipped_sequence1 = tfutil.clip_sentence(sequence1_ph, sequence1_len_ph)
    clipped_sequence2 = tfutil.clip_sentence(sequence2_ph, sequence2_len_ph)

    nb_instances = sequence1.shape[0]

    vocab_size = max(token_to_index.values()) + 1

    discriminator_scope_name = 'discriminator'
    with tf.variable_scope(discriminator_scope_name):

        embedding_layer = tf.get_variable('embeddings',
                                          shape=[vocab_size, embedding_size],
                                          initializer=None)

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

        mode_name_to_class = {
            'cbilstm': ConditionalBiLSTM,
            'ff-dam': FeedForwardDAM,
            'esim1': ESIM
        }

        model_class = mode_name_to_class[model_name]

        assert model_class is not None
        model = model_class(**model_kwargs)
        logits = model()
        probabilities = tf.nn.softmax(logits)

        a_pooling_function = tf.reduce_max

        a_model_kwargs = copy.copy(model_kwargs)

        a_sentence1_ph = tf.placeholder(dtype=tf.int32, shape=[None, None], name='a_sentence1')
        a_sentence2_ph = tf.placeholder(dtype=tf.int32, shape=[None, None], name='a_sentence2')

        a_sentence1_len_ph = tf.placeholder(dtype=tf.int32, shape=[None], name='a_sentence1_length')
        a_sentence2_len_ph = tf.placeholder(dtype=tf.int32, shape=[None], name='a_sentence2_length')

        a_clipped_sentence1 = tfutil.clip_sentence(a_sentence1_ph, a_sentence1_len_ph)
        a_clipped_sentence2 = tfutil.clip_sentence(a_sentence2_ph, a_sentence2_len_ph)

        a_sentence1_embedding = tf.nn.embedding_lookup(embedding_layer, a_clipped_sentence1)
        a_sentence2_embedding = tf.nn.embedding_lookup(embedding_layer, a_clipped_sentence2)

        a_model_kwargs.update({
            'sequence1': a_sentence1_embedding, 'sequence1_length': a_sentence1_len_ph,
            'sequence2': a_sentence2_embedding, 'sequence2_length': a_sentence2_len_ph
        })

        a_kwargs = dict(model_class=model_class, model_kwargs=a_model_kwargs,
                        entailment_idx=entailment_idx, contradiction_idx=contradiction_idx, neutral_idx=neutral_idx,
                        pooling_function=a_pooling_function, debug=True)

        a_function_weight_bi_tuple_lst = []

        loss = tf.constant(0.0)

        if r1_weight:
            r_loss, _ = R.contradiction_acl(is_bi=True, **a_kwargs)
            a_function_weight_bi_tuple_lst += [(R.contradiction_acl, r1_weight, True)]
            loss += r1_weight * r_loss
        if r2_weight:
            r_loss, _ = R.entailment_acl(is_bi=True, **a_kwargs)
            a_function_weight_bi_tuple_lst += [(R.entailment_acl, r2_weight, True)]
            loss += r2_weight * r_loss
        if r3_weight:
            r_loss, _ = R.neutral_acl(is_bi=True, **a_kwargs)
            a_function_weight_bi_tuple_lst += [(R.neutral_acl, r3_weight, True)]
            loss += r3_weight * r_loss
        if r4_weight:
            r_loss, _ = R.entailment_reflexive_acl(**a_kwargs)
            a_function_weight_bi_tuple_lst += [(R.entailment_reflexive_acl, r4_weight, False)]
            loss += r4_weight * r_loss
        if r5_weight:
            r_loss, _ = R.entailment_neutral_acl(is_bi=True, **a_kwargs)
            a_function_weight_bi_tuple_lst += [(R.entailment_neutral_acl, r5_weight, True)]
            loss += r5_weight * r_loss

    discriminator_vars = tfutil.get_variables_in_scope(discriminator_scope_name)

    trainable_discriminator_vars = list(discriminator_vars)
    trainable_discriminator_vars.remove(embedding_layer)

    saver = tf.train.Saver(discriminator_vars, max_to_keep=1)

    session_config = tf.ConfigProto()
    session_config.gpu_options.allow_growth = True

    instance_generator = InstanceGenerator(token_to_index=token_to_index)

    instance_scorer = None
    if top_k is not None:
        with tf.variable_scope(discriminator_scope_name):
            instance_scorer = InstanceScorer(embedding_layer=embedding_layer,
                                             token_to_index=token_to_index,
                                             model_class=model_class,
                                             model_kwargs=model_kwargs,
                                             i_pooling_function=tf.reduce_sum,
                                             a_function_weight_bi_tuple_lst=a_function_weight_bi_tuple_lst)

    a_batch_size = (nb_corruptions * is_flip) + \
                   (nb_corruptions * is_remove) + \
                   (nb_corruptions * is_combine) + \
                   (nb_corruptions * (scramble is not None))

    lm_scorer_adversarial_batch = lm_scorer_batch = None
    if epsilon is not None:
        lm_scorer_adversarial_batch = LMScorer(embedding_layer=embedding_layer,
                                               token_to_index=token_to_index,
                                               batch_size=a_batch_size)

        lm_scorer_batch = LMScorer(embedding_layer=embedding_layer,
                                   token_to_index=token_to_index,
                                   batch_size=batch_size, reuse=True)

        lm_vars = lm_scorer_adversarial_batch.get_vars()
        lm_saver = tf.train.Saver(lm_vars, max_to_keep=1)

    A_rs = np.random.RandomState(0)

    sentence_pair_to_original_pair = {}
    original_pair_to_label = {}
    sentence_pair_to_corruption_type = {}

    rs = np.random.RandomState(seed)

    with tf.Session(config=session_config) as session:

        if lm_scorer_adversarial_batch is not None:
            lm_ckpt = tf.train.get_checkpoint_state(lm_path)
            lm_saver.restore(session, lm_ckpt.model_checkpoint_path)

        saver.restore(session, restore_path)

        batches = make_batches(size=nb_instances, batch_size=batch_size)

        for batch_idx, (batch_start, batch_end) in enumerate(batches):
            # order = np.arange(nb_instances)
            order = rs.permutation(nb_instances)

            sentences1, sentences2 = sequence1[order], sequence2[order]
            sizes1, sizes2 = sequence1_len[order], sequence2_len[order]
            labels = label[order]

            batch_sentences1 = sentences1[batch_start:batch_end]
            batch_sentences2 = sentences2[batch_start:batch_end]

            batch_sizes1 = sizes1[batch_start:batch_end]
            batch_sizes2 = sizes2[batch_start:batch_end]

            batch_labels = labels[batch_start:batch_end]

            batch_max_size1 = np.max(batch_sizes1)
            batch_max_size2 = np.max(batch_sizes2)

            batch_sentences1 = batch_sentences1[:, :batch_max_size1]
            batch_sentences2 = batch_sentences2[:, :batch_max_size2]

            # Remove the BOS token from sentences
            o_batch_size = batch_sentences1.shape[0]
            o_sentences1, o_sentences2 = [], []

            for i in range(o_batch_size):
                o_sentences1 += [[idx for idx in batch_sentences1[i, 1:].tolist() if idx != 0]]
                o_sentences2 += [[idx for idx in batch_sentences2[i, 1:].tolist() if idx != 0]]

            # Parameters for adversarial training:
            # a_epsilon, a_nb_corruptions, a_nb_examples_per_batch, a_is_flip, a_is_combine, a_is_remove, a_is_scramble
            selected_sentence1, selected_sentence2 = [], []

            # First, add all training sentences
            selected_sentence1 += o_sentences1
            selected_sentence2 += o_sentences2

            op2idx = {}
            for idx, (o1, o2) in enumerate(zip(o_sentences1, o_sentences2)):
                op2idx[(tuple(o1), tuple(o2))] = idx

            for a, b, c in zip(selected_sentence1, selected_sentence2, batch_labels):
                sentence_pair_to_original_pair[(tuple(a), tuple(b))] = (a, b)
                original_pair_to_label[(tuple(a), tuple(b))] = c
                sentence_pair_to_corruption_type[(tuple(a), tuple(b))] = 'none'

            c_idxs = A_rs.choice(o_batch_size, nb_examples_per_batch, replace=False)
            for c_idx in c_idxs:
                o_sentence1 = o_sentences1[c_idx]
                o_sentence2 = o_sentences2[c_idx]

                sentence_pair_to_original_pair[(tuple(o_sentence1), tuple(o_sentence2))] = (o_sentence1, o_sentence2)
                sentence_pair_to_corruption_type[(tuple(o_sentence1), tuple(o_sentence2))] = 'none'

                # Generating Corruptions
                c_sentence1_lst, c_sentence2_lst = [], []
                if is_flip:
                    corruptions_1, corruptions_2 = instance_generator.flip(sentence1=o_sentence1, sentence2=o_sentence2,
                                                                           nb_corruptions=nb_corruptions)
                    c_sentence1_lst += corruptions_1
                    c_sentence2_lst += corruptions_2

                    for _corruption_1, _corruption_2 in zip(corruptions_1, corruptions_2):
                        sentence_pair_to_original_pair[(tuple(_corruption_1), tuple(_corruption_2))] = (o_sentence1, o_sentence2)
                        if (tuple(_corruption_1), tuple(_corruption_2)) not in sentence_pair_to_corruption_type:
                            sentence_pair_to_corruption_type[(tuple(_corruption_1), tuple(_corruption_2))] = 'flip'

                if is_remove:
                    corruptions_1, corruptions_2 = instance_generator.remove(sentence1=o_sentence1, sentence2=o_sentence2,
                                                                             nb_corruptions=nb_corruptions)
                    c_sentence1_lst += corruptions_1
                    c_sentence2_lst += corruptions_2

                    for _corruption_1, _corruption_2 in zip(corruptions_1, corruptions_2):
                        sentence_pair_to_original_pair[(tuple(_corruption_1), tuple(_corruption_2))] = (o_sentence1, o_sentence2)
                        if (tuple(_corruption_1), tuple(_corruption_2)) not in sentence_pair_to_corruption_type:
                            sentence_pair_to_corruption_type[(tuple(_corruption_1), tuple(_corruption_2))] = 'remove'

                if is_combine:
                    corruptions_1, corruptions_2 = instance_generator.combine(sentence1=o_sentence1, sentence2=o_sentence2,
                                                                              nb_corruptions=nb_corruptions)
                    c_sentence1_lst += corruptions_1
                    c_sentence2_lst += corruptions_2

                    for _corruption_1, _corruption_2 in zip(corruptions_1, corruptions_2):
                        sentence_pair_to_original_pair[(tuple(_corruption_1), tuple(_corruption_2))] = (o_sentence1, o_sentence2)
                        if (tuple(_corruption_1), tuple(_corruption_2)) not in sentence_pair_to_corruption_type:
                            sentence_pair_to_corruption_type[(tuple(_corruption_1), tuple(_corruption_2))] = 'combine'

                if scramble is not None:
                    corruptions_1, corruptions_2 = instance_generator.scramble(sentence1=o_sentence1, sentence2=o_sentence2,
                                                                               nb_corruptions=nb_corruptions,
                                                                               nb_pooled_sentences=scramble,
                                                                               sentence_pool=o_sentences1 + o_sentences2)
                    c_sentence1_lst += corruptions_1
                    c_sentence2_lst += corruptions_2

                    for _corruption_1, _corruption_2 in zip(corruptions_1, corruptions_2):
                        sentence_pair_to_original_pair[(tuple(_corruption_1), tuple(_corruption_2))] = (o_sentence1, o_sentence2)
                        if (tuple(_corruption_1), tuple(_corruption_2)) not in sentence_pair_to_corruption_type:
                            sentence_pair_to_corruption_type[(tuple(_corruption_1), tuple(_corruption_2))] = 'scramble'

                if epsilon is not None and lm_scorer_adversarial_batch is not None:
                    # Scoring them against a Language Model
                    log_perp1 = lm_scorer_adversarial_batch.score(session, c_sentence1_lst)
                    log_perp2 = lm_scorer_adversarial_batch.score(session, c_sentence2_lst)

                    log_perp_o1 = lm_scorer_batch.score(session, o_sentences1)
                    log_perp_o2 = lm_scorer_batch.score(session, o_sentences2)

                    low_lperp_idxs = []
                    for i, (c1, c2, _lp1, _lp2) in enumerate(zip(c_sentence1_lst, c_sentence2_lst, log_perp1, log_perp2)):
                        o1, o2 = sentence_pair_to_original_pair[(tuple(c1), tuple(c2))]
                        idx = op2idx[(tuple(o1), tuple(o2))]
                        _log_perp_o1 = log_perp_o1[idx]
                        _log_perp_o2 = log_perp_o2[idx]

                        if (_lp1 + _lp2) <= (_log_perp_o1 + _log_perp_o2 + epsilon):
                            low_lperp_idxs += [i]
                else:
                    low_lperp_idxs = range(len(c_sentence1_lst))

                selected_sentence1 += [c_sentence1_lst[i] for i in low_lperp_idxs]
                selected_sentence2 += [c_sentence2_lst[i] for i in low_lperp_idxs]

            selected_scores = None

            sentence_pair_to_score = {}

            # Now in selected_sentence1 and selected_sentence2 we have the most offending examples
            if top_k >= 0 and instance_scorer is not None:
                iscore_values = instance_scorer.iscore(session, selected_sentence1, selected_sentence2)

                for a, b, c in zip(selected_sentence1, selected_sentence2, iscore_values):
                    sentence_pair_to_score[(tuple(a), tuple(b))] = c

                top_k_idxs = np.argsort(iscore_values)[::-1][:top_k]

                selected_sentence1 = [selected_sentence1[i] for i in top_k_idxs]
                selected_sentence2 = [selected_sentence2[i] for i in top_k_idxs]

                selected_scores = [iscore_values[i] for i in top_k_idxs]

            def decode(sentence_ids):
                return ' '.join([index_to_token[idx] for idx in sentence_ids])

            def infer(s1_ids, s2_ids):
                a = np.array([[bos_idx] + s1_ids])
                b = np.array([[bos_idx] + s2_ids])

                c = np.array([1 + len(s1_ids)])
                d = np.array([1 + len(s2_ids)])

                inf_feed = {
                    sequence1_ph: a, sequence2_ph: b,
                    sequence1_len_ph: c, sequence2_len_ph: d,
                    dropout_keep_prob_ph: 1.0
                }
                pv = session.run(probabilities, feed_dict=inf_feed)
                return {
                    'ent': str(pv[0, entailment_idx]),
                    'neu': str(pv[0, neutral_idx]),
                    'con': str(pv[0, contradiction_idx])
                }

            logger.info("No. of generated pairs: {}".format(len(selected_sentence1)))

            for i, (s1, s2, score) in enumerate(zip(selected_sentence1, selected_sentence2, selected_scores)):
                o1, o2 = sentence_pair_to_original_pair[(tuple(s1), tuple(s2))]
                lbl = original_pair_to_label[(tuple(o1), tuple(o2))]
                corr = sentence_pair_to_corruption_type[(tuple(s1), tuple(s2))]

                oiscore = sentence_pair_to_score.get((tuple(o1), tuple(o2)), 1.0)

                print('[{}] Original 1: {}'.format(i, decode(o1)))
                print('[{}] Original 2: {}'.format(i, decode(o2)))
                print('[{}] Original Label: {}'.format(i, index_to_label[lbl]))
                print('[{}] Original Inconsistency Loss: {}'.format(i, oiscore))

                print('[{}] Sentence 1: {}'.format(i, decode(s1)))
                print('[{}] Sentence 2: {}'.format(i, decode(s2)))

                print('[{}] Inconsistency Loss: {}'.format(i, score))

                print('[{}] Corruption type: {}'.format(i, corr))

                print('[{}] (before) s1 -> s2: {}'.format(i, str(infer(o1, o2))))
                print('[{}] (before) s2 -> s1: {}'.format(i, str(infer(o2, o1))))

                print('[{}] (after) s1 -> s2: {}'.format(i, str(infer(s1, s2))))
                print('[{}] (after) s2 -> s1: {}'.format(i, str(infer(s2, s1))))

                jdata = {
                    'original_sentence1': decode(o1),
                    'original_sentence2': decode(o2),
                    'original_inconsistency_loss': str(oiscore),

                    'original_label': index_to_label[lbl],

                    'sentence1': decode(s1),
                    'sentence2': decode(s2),

                    'inconsistency_loss': str(score),
                    'inconsistency_loss_increase': str(score - oiscore),

                    'corruption': str(corr),

                    'probabilities_before_s1_s2': infer(o1, o2),
                    'probabilities_before_s2_s1': infer(o2, o1),

                    'probabilities_after_s1_s2': infer(s1, s2),
                    'probabilities_after_s2_s1': infer(s2, s1)
                }

                if json_path is not None:
                    with open(json_path, 'a') as f:
                        json.dump(jdata, f)
                        f.write('\n')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])
