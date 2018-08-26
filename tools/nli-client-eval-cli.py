#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse

import gzip
import json

import os
import sys

import nltk
import requests

import logging

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(os.path.basename(sys.argv[0]))


def call_service(url, sentence1, sentence2):
    data = {
        'sentence1': sentence1,
        'sentence2': sentence2
    }
    res = requests.post(url, data=data)
    res_json = res.json()
    return res_json


def main(argv):
    def formatter(prog):
        return argparse.HelpFormatter(prog, max_help_position=100, width=200)

    argparser = argparse.ArgumentParser('NLI Client', formatter_class=formatter)

    argparser.add_argument('path', action='store', type=str)
    argparser.add_argument('--url', '-u', action='store', type=str, default='http://127.0.0.1:8889/nnli')

    args = argparser.parse_args(argv)

    path = args.path
    url = args.url

    nb_predictions = 0.0
    nb_matching_predictions = 0.0

    with gzip.open(path, 'rb') as f:
        for line in f:
            decoded_line = line.decode('utf-8')
            obj = json.loads(decoded_line)

            gold_label = obj['gold_label']

            # sentence1 = obj['sentence1']
            # sentence2 = obj['sentence2']

            sentence1_parse = obj['sentence1_parse']
            sentence2_parse = obj['sentence2_parse']

            sentence1_tree = nltk.Tree.fromstring(sentence1_parse)
            sentence2_tree = nltk.Tree.fromstring(sentence2_parse)

            sentence1_tokens = sentence1_tree.leaves()
            sentence2_tokens = sentence2_tree.leaves()

            if gold_label in {'contradiction', 'entailment', 'neutral'}:
                prediction = call_service(url=url,
                                          sentence1=' '.join(sentence1_tokens),
                                          sentence2=' '.join(sentence2_tokens))

                predicted_label = max(prediction, key=prediction.get)

                nb_predictions += 1.0
                nb_matching_predictions += 1.0 if gold_label == predicted_label else 0.0

                print(gold_label, predicted_label, 1.0 if gold_label == predicted_label else 0.0)
                print(nb_matching_predictions / nb_predictions)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    main(sys.argv[1:])
