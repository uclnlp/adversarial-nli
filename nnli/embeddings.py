# -*- coding: utf-8 -*-

import gensim

import numpy as np
from numpy import linalg as LA

import logging

logger = logging.getLogger(__name__)


def embedding_matrix(nb_tokens, embedding_size, token_to_index,
                     glove_path=None,
                     unit_norm=True,
                     rs=None,
                     dtype=None):
    rs = rs if rs else np.random.RandomState(0)
    embedding_matrix_np = rs.uniform(low=0.0, high=1.0,
                                     size=(nb_tokens, embedding_size))
    if glove_path is not None:
        word_to_vector = load_glove(path=glove_path, words={token for token in token_to_index})
        for word, vector in word_to_vector.items():
            word_id = token_to_index[word]
            vector_np = np.array(vector)
            embedding_matrix_np[word_id, :] = vector_np
    if unit_norm:
        norms = LA.norm(embedding_matrix_np, axis=1, keepdims=True)
        embedding_matrix_np /= norms
    if dtype is not None:
        embedding_matrix_np = embedding_matrix_np.astype(dtype)
    return embedding_matrix_np


def load_glove(path, words=None):
    word_to_embedding = {}
    with open(path, 'r') as stream:
        for n, line in enumerate(stream):
            if not isinstance(line, str):
                line = line.decode('utf-8')
            split_line = line.split(' ')
            word = split_line[0]
            if words is None or word in words:
                try:
                    word_to_embedding[word] = [float(f) for f in split_line[1:]]
                except ValueError:
                    logger.error('{}\t{}\t{}'.format(n, word, str(split_line)))
    return word_to_embedding


def load_word2vec(path, words=None, binary=True):
    word_to_embedding = {}
    model = gensim.models.KeyedVectors.load_word2vec_format(path, binary=binary)
    for word in words:
        if word in model:
            word_to_embedding[word] = model[word].tolist()
    return word_to_embedding
