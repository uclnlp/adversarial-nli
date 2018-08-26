# -*- coding: utf-8 -*-

import nltk
from nltk.tree import Tree

import redis

import numpy as np

from nnli.generators.parser import Parser
from nnli.generators import operators


class InstanceGenerator:
    def __init__(self, token_to_index,
                 corenlp_url='http://127.0.0.1:9000',
                 is_remove=True,
                 is_merge=True,
                 nb_words=1024,
                 bos_idx=1, eos_idx=2, unk_idx=3,
                 seed=0):

        self.token_to_index = token_to_index
        self.index_to_token = {k: v for v, k in self.token_to_index.items()}
        self.unk_token = self.index_to_token.get(unk_idx, '<UNK>')

        self.corenlp_url = corenlp_url
        self.is_remove = is_remove
        self.is_merge = is_merge
        self.nb_words = nb_words

        self.bos_idx = bos_idx
        self.eos_idx = eos_idx
        self.unk_idx = unk_idx

        self.rs = np.random.RandomState(seed)

        self.parser = None
        self.tokenizer = nltk.tokenize.TreebankWordTokenizer()

        self.cache = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
        try:
            self.cache.client_list()
        except redis.ConnectionError:
            self.cache = dict()

        # self.str_to_tree_cache = dict()
        self.str_to_tree_cache = None

    def _get_parser(self):
        if self.parser is None:
            self.parser = Parser(url=self.corenlp_url)
        return self.parser

    def _str_to_tree(self, tree_str):
        if self.str_to_tree_cache is not None:
            if tree_str not in self.str_to_tree_cache:
                self.str_to_tree_cache[tree_str] = Tree.fromstring(tree_str)
            res = self.str_to_tree_cache[tree_str]
        else:
            res = Tree.fromstring(tree_str)
        return res.copy(deep=True)

    def scramble(self, sentence1, sentence2, nb_corruptions,
                 sentence_pool=None,
                 nb_pooled_sentences=10):
        sentence1_str, sentence2_str = sentence1, sentence2

        assert sentence_pool is not None
        sentence_pool_size = len(sentence_pool)

        # Select some sentences from sentence_pool
        sentence_lst = [sentence_pool[i] for i in self.rs.choice(sentence_pool_size, nb_pooled_sentences, replace=False)]
        sentence_lst_str = sentence_lst

        # If the input is arrays of indices, turn each array into a string first
        if isinstance(sentence1, list):
            sentence1_str = self._to_string(sentence1)
            sentence2_str = self._to_string(sentence2)
            sentence_lst_str = [self._to_string(sentence) for sentence in sentence_lst]

        # Parse the sentences
        tree1 = self._parse(sentence1_str)
        tree2 = self._parse(sentence2_str)
        tree_lst = [self._parse(sentence_str) for sentence_str in sentence_lst_str]

        tree1_variations = operators.combine_trees(tree1, tree2)
        tree2_variations = operators.combine_trees(tree1, tree2)

        for tree in tree_lst:
            tree1_variations += operators.combine_trees(tree1, tree)
            tree2_variations += operators.combine_trees(tree, tree2)

        # Remove duplicate trees, working around the issue that Tree is not hashable
        tree1_variations = [self._str_to_tree(s) for s in list(set([str(t) for t in tree1_variations]))]
        tree2_variations = [self._str_to_tree(s) for s in list(set([str(t) for t in tree2_variations]))]

        res1, res2 = [], []
        for idx in range(nb_corruptions):
            tree1_c, tree2_c = tree1, tree2
            if self.rs.randint(0, 2):
                tree1_c = tree1_variations[self.rs.randint(low=0, high=len(tree1_variations))]
            else:
                tree2_c = tree2_variations[self.rs.randint(low=0, high=len(tree2_variations))]

            res1 += [[self.token_to_index.get(tkn, self.unk_idx) for tkn in tree1_c.leaves()]]
            res2 += [[self.token_to_index.get(tkn, self.unk_idx) for tkn in tree2_c.leaves()]]

        # If the input was strings, the output should also be strings
        if isinstance(sentence1, str):
            new_res1 = [self._to_string(e) for e in res1]
            new_res2 = [self._to_string(e) for e in res2]
            res1, res2 = new_res1, new_res2

        return res1, res2

    def combine(self, sentence1, sentence2, nb_corruptions):
        sentence1_str, sentence2_str = sentence1, sentence2

        # If the input is arrays of indices, turn each array into a string first
        if isinstance(sentence1, list):
            sentence1_str = self._to_string(sentence1)
            sentence2_str = self._to_string(sentence2)

        # Parse the sentences
        tree1 = self._parse(sentence1_str)
        tree2 = self._parse(sentence2_str)

        tree1_variations = operators.combine_trees(tree1, tree2)
        tree2_variations = operators.combine_trees(tree1, tree2)

        # Remove duplicate trees, working around the issue that Tree is not hashable
        tree1_variations = [self._str_to_tree(s) for s in list(set([str(t) for t in tree1_variations]))]
        tree2_variations = [self._str_to_tree(s) for s in list(set([str(t) for t in tree2_variations]))]

        res1, res2 = [], []
        for idx in range(nb_corruptions):
            tree1_c, tree2_c = tree1, tree2
            if self.rs.randint(0, 2):
                tree1_c = tree1_variations[self.rs.randint(low=0, high=len(tree1_variations))]
            else:
                tree2_c = tree2_variations[self.rs.randint(low=0, high=len(tree2_variations))]

            res1 += [[self.token_to_index.get(tkn, self.unk_idx) for tkn in tree1_c.leaves()]]
            res2 += [[self.token_to_index.get(tkn, self.unk_idx) for tkn in tree2_c.leaves()]]

        # If the input was strings, the output should also be strings
        if isinstance(sentence1, str):
            new_res1 = [self._to_string(e) for e in res1]
            new_res2 = [self._to_string(e) for e in res2]
            res1, res2 = new_res1, new_res2

        return res1, res2

    def remove(self, sentence1, sentence2, nb_corruptions):
        sentence1_str, sentence2_str = sentence1, sentence2

        # If the input is arrays of indices, turn each array into a string first
        if isinstance(sentence1, list):
            sentence1_str = self._to_string(sentence1)
            sentence2_str = self._to_string(sentence2)

        # Parse the sentences
        tree1 = self._parse(sentence1_str)
        tree2 = self._parse(sentence2_str)

        tree1_variations = operators.remove_subtree(tree1)
        tree2_variations = operators.remove_subtree(tree2)

        # Remove duplicate trees, working around the issue that Tree is not hashable
        tree1_variations = [self._str_to_tree(s) for s in list(set([str(t) for t in tree1_variations]))]
        tree2_variations = [self._str_to_tree(s) for s in list(set([str(t) for t in tree2_variations]))]

        res1, res2 = [], []
        for idx in range(nb_corruptions):
            tree1_c, tree2_c = tree1, tree2
            if self.rs.randint(0, 2):
                tree1_c = tree1_variations[self.rs.randint(low=0, high=len(tree1_variations))]
            else:
                tree2_c = tree2_variations[self.rs.randint(low=0, high=len(tree2_variations))]

            res1 += [[self.token_to_index.get(tkn, self.unk_idx) for tkn in tree1_c.leaves()]]
            res2 += [[self.token_to_index.get(tkn, self.unk_idx) for tkn in tree2_c.leaves()]]

        # If the input was strings, the output should also be strings
        if isinstance(sentence1, str):
            new_res1 = [self._to_string(e) for e in res1]
            new_res2 = [self._to_string(e) for e in res2]
            res1, res2 = new_res1, new_res2

        return res1, res2

    def flip(self, sentence1, sentence2, nb_corruptions):
        # If the input is strings, turn each string in a list of token indices.
        if isinstance(sentence1, str):
            assert isinstance(sentence2, str)
            sentence1_tokens = self._tokenize(sentence1)
            sentence2_tokens = self._tokenize(sentence2)

            sentence1_indexes = [self.token_to_index.get(tkn, self.unk_idx) for tkn in sentence1_tokens]
            sentence2_indexes = [self.token_to_index.get(tkn, self.unk_idx) for tkn in sentence2_tokens]
        else:
            assert isinstance(sentence1, list) and isinstance(sentence2, list)
            sentence1_indexes = sentence1
            sentence2_indexes = sentence2

        corruptions1 = np.repeat(a=[sentence1_indexes], repeats=nb_corruptions, axis=0)
        corruptions2 = np.repeat(a=[sentence2_indexes], repeats=nb_corruptions, axis=0)

        sentence1_len, sentence2_len = len(sentence1_indexes), len(sentence2_indexes)

        for idx in range(nb_corruptions):
            if 1 < sentence1_len - 1 and 1 < sentence2_len - 1:
                new_word = self.rs.randint(low=4, high=self.nb_words)

                if self.rs.randint(0, 2):
                    where_to_corrupt1 = self.rs.randint(low=1, high=sentence1_len - 1)
                    corruptions1[idx, where_to_corrupt1] = new_word
                else:
                    where_to_corrupt2 = self.rs.randint(low=1, high=sentence2_len - 1)
                    corruptions2[idx, where_to_corrupt2] = new_word

        res1, res2 = corruptions1.tolist(), corruptions2.tolist()

        # If the input is strings, the outputs needs to be strings as well.
        if isinstance(sentence1, str):
            new_res1 = [self._to_string(e) for e in res1]
            new_res2 = [self._to_string(e) for e in res2]
            res1, res2 = new_res1, new_res2

        return res1, res2

    def _parse(self, sentence):
        s = self._parse_str(sentence)
        return self._str_to_tree(s)

    def _parse_str(self, sentence):
        if ('Generator', 'parse', sentence) not in self.cache:
            parser = self._get_parser()
            tree = parser.parse(sentence)
            s_tree = str(tree)
            self.cache[('Generator', 'parse', sentence)] = s_tree
        return self.cache[('Generator', 'parse', sentence)]

    def _to_string(self, sentence_indexes):
        return ' '.join([self.index_to_token.get(idx, self.unk_token) for idx in sentence_indexes])

    def _tokenize(self, sentence):
        return self.tokenizer.tokenize(sentence)
