# -*- coding: utf-8 -*-

import json

import nltk
from nnli.io import iopen

import logging

logger = logging.getLogger(__name__)


class SNLI:
    def __init__(self):
        pass

    def to_instance(self, d):
        sentence1 = d['sentence1']
        sentence1_parse = d['sentence1_parse']
        sentence1_tree = nltk.Tree.fromstring(sentence1_parse)
        sentence1_parse_tokens = sentence1_tree.leaves()

        sentence2 = d['sentence2']
        sentence2_parse = d['sentence2_parse']
        sentence2_tree = nltk.Tree.fromstring(sentence2_parse)
        sentence2_parse_tokens = sentence2_tree.leaves()

        gold_label = d['gold_label']

        instance = {
            'sentence1': sentence1,
            'sentence1_parse': sentence1_parse,
            'sentence1_parse_tokens': sentence1_parse_tokens,
            'sentence2': sentence2,
            'sentence2_parse': sentence2_parse,
            'sentence2_parse_tokens': sentence2_parse_tokens,
            'gold_label': gold_label
        }

        return instance

    def parse(self, path):
        res = []
        with iopen(path, 'rb') as f:
            for line in f:
                decoded_line = line.decode('utf-8')
                obj = json.loads(decoded_line)
                instance = self.to_instance(obj)
                if instance['gold_label'] in {'entailment', 'neutral', 'contradiction'}:
                    res += [instance]
        return res
