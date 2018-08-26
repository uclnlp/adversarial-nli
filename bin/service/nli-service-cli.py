#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse

import os
import sys

import pickle

import nltk

import tensorflow as tf

from nnli import tfutil

from nnli.models import ConditionalBiLSTM
from nnli.models import FeedForwardDAM
from nnli.models import FeedForwardDAMP
from nnli.models import FeedForwardDAMS
from nnli.models import ESIM

from flask import Flask, request, jsonify
from flask.views import View

from werkzeug.serving import WSGIRequestHandler, BaseWSGIServer

import logging

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger(os.path.basename(sys.argv[0]))

WSGIRequestHandler.protocol_version = "HTTP/1.1"
BaseWSGIServer.protocol_version = "HTTP/1.1"

app = Flask('nli-service')

# Run with:
# $ ./bin/nli-service-cli.py -R saved/snli/dam/2/dam
# $ ./bin/nli-service-cli.py -R saved/snli/esim/2/esim -r 300 -m esim -p 9001
# $ ./bin/nli-service-cli.py -R saved/snli/cbilstm/2/cbilstm -r 300 -m cbilstm -p 9002


class InvalidAPIUsage(Exception):
    """
    Class used for handling error messages.
    """
    DEFAULT_STATUS_CODE = 400

    def __init__(self, message, status_code=None, payload=None):
        Exception.__init__(self)
        self.message = message
        self.status_code = self.DEFAULT_STATUS_CODE

        if status_code is not None:
            self.status_code = status_code

        self.payload = payload

    def to_dict(self):
        rv = dict(self.payload or ())
        rv['message'] = self.message
        return rv


@app.errorhandler(InvalidAPIUsage)
def handle_invalid_usage(error):
    response = jsonify(error.to_dict())
    response.status_code = error.status_code
    return response


def main(argv):
    def formatter(prog):
        return argparse.HelpFormatter(prog, max_help_position=100, width=200)

    argparser = argparse.ArgumentParser('NLI Service', formatter_class=formatter)

    argparser.add_argument('--model', '-m', action='store', type=str, default='ff-dam',
                           choices=['cbilstm', 'ff-dam', 'ff-damp', 'ff-dams', 'esim'])

    argparser.add_argument('--embedding-size', '-e', action='store', type=int, default=300)
    argparser.add_argument('--representation-size', '-r', action='store', type=int, default=200)

    argparser.add_argument('--restore', '-R', action='store', type=str, default=None, required=True)

    argparser.add_argument('--port', '-p', action='store', type=int, default=8889, required=False)

    args = argparser.parse_args(argv)

    model_name = args.model

    embedding_size = args.embedding_size
    representation_size = args.representation_size

    restore_path = args.restore

    port = args.port

    with open('{}_index_to_token.p'.format(restore_path), 'rb') as f:
        index_to_token = pickle.load(f)

    token_to_index = {token: index for index, token in index_to_token.items()}

    entailment_idx, neutral_idx, contradiction_idx = 0, 1, 2

    name_to_model = {
        'cbilstm': ConditionalBiLSTM,
        'ff-dam': FeedForwardDAM,
        'ff-damp': FeedForwardDAMP,
        'ff-dams': FeedForwardDAMS,
        'esim': ESIM
    }

    model_class = name_to_model[model_name]

    sequence1_ph = tf.placeholder(dtype=tf.int32, shape=[None, None], name='sequence1')
    sequence1_len_ph = tf.placeholder(dtype=tf.int32, shape=[None], name='sequence1_length')

    sequence2_ph = tf.placeholder(dtype=tf.int32, shape=[None, None], name='sequence2')
    sequence2_len_ph = tf.placeholder(dtype=tf.int32, shape=[None], name='sequence2_length')

    dropout_keep_prob_ph = tf.placeholder(tf.float32, name='dropout_keep_prob')

    clipped_sequence1 = tfutil.clip_sentence(sequence1_ph, sequence1_len_ph)
    clipped_sequence2 = tfutil.clip_sentence(sequence2_ph, sequence2_len_ph)

    vocab_size = max(token_to_index.values()) + 1

    logger.info('Initializing the Model')

    discriminator_scope_name = 'discriminator'
    with tf.variable_scope(discriminator_scope_name):
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

        probabilities = tf.nn.softmax(logits)

    discriminator_vars = tfutil.get_variables_in_scope(discriminator_scope_name)
    tokenizer = nltk.tokenize.TreebankWordTokenizer()

    # Enumeration of tokens start at index=3:
    #   index=0 PADDING
    #   index=1 START_OF_SENTENCE
    #   index=2 END_OF_SENTENCE
    #   index=3 UNKNOWN_WORD
    bos_idx, eos_idx, unk_idx = 1, 2, 3

    with tf.Session() as session:
        saver = tf.train.Saver(discriminator_vars, max_to_keep=1)
        saver.restore(session, restore_path)

        class Service(View):
            methods = ['GET', 'POST']

            def dispatch_request(self):
                sentence1 = request.form['sentence1'] if 'sentence1' in request.form else request.args.get('sentence1')
                sentence2 = request.form['sentence2'] if 'sentence2' in request.form else request.args.get('sentence2')

                sentence1_tokens = tokenizer.tokenize(sentence1)
                sentence2_tokens = tokenizer.tokenize(sentence2)

                sentence1_ids = [bos_idx] + [token_to_index.get(t, unk_idx) for t in sentence1_tokens]
                sentence2_ids = [bos_idx] + [token_to_index.get(t, unk_idx) for t in sentence2_tokens]

                sentence1_len = len(sentence1_ids)
                sentence2_len = len(sentence2_ids)

                # Compute the answer
                feed_dict = {
                    sequence1_ph: [sentence1_ids],
                    sequence2_ph: [sentence2_ids],

                    sequence1_len_ph: [sentence1_len],
                    sequence2_len_ph: [sentence2_len],

                    dropout_keep_prob_ph: 1.0
                }

                probabilities_value = session.run(probabilities, feed_dict=feed_dict)[0]

                answer = {
                    'neutral': str(probabilities_value[neutral_idx]),
                    'contradiction': str(probabilities_value[contradiction_idx]),
                    'entailment': str(probabilities_value[entailment_idx])
                }

                return jsonify(answer)

        app.add_url_rule('/nnli', view_func=Service.as_view('request'))
        app.run(host='0.0.0.0', port=port, debug=False)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    main(sys.argv[1:])
