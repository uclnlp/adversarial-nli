# -*- coding: utf-8 -*-

import nltk


def extract_nouns(sentence, index_to_token=None, unk_idx=3):
    token_to_index = None
    if index_to_token is not None:
        token_to_index = {token: index for index, token in index_to_token.items()}

    if len(sentence) < 1:
        return []

    is_int = False
    if isinstance(sentence[0], int):
        is_int = True
        new_sentence = [token_to_index.get(index, unk_idx) for index in sentence]
        sentence = new_sentence

    tokens = nltk.word_tokenize(sentence)
    _token_pos_lst = nltk.pos_tag(tokens)
    token_pos_lst = []

    nnp_token = nnps_token = None
    nouns = []

    for token_pos in _token_pos_lst:
        token, pos = token_pos
        # If it's a NNP ..
        if pos in {'NNP'}:
            # Aggregate it with previous NNPs in the sequence
            nnp_token = '{} {}'.format(nnp_token, token) if nnp_token else token
        elif pos in {'NNPS'}:
            nnps_token = '{} {}'.format(nnps_token, token) if nnps_token else token
        # Otherwise ..
        else:
            # If there was a sequence of NNP tokens before ..
            if nnp_token:
                # Add such tokens to the list as a single token
                token_pos_lst += [(nnp_token, 'NNP')]
                nouns += [nnp_token]

                nnp_token = None

            if nnps_token:
                token_pos_lst += [(nnps_token, 'NNPS')]
                nouns += [nnps_token]

                nnps_token = None
            # And then add the current token as well
            token_pos_lst += [token_pos]

    if nnp_token:
        token_pos_lst += [(nnp_token, 'NNP')]
        nouns += [nnp_token]

    if nnps_token:
        token_pos_lst += [(nnps_token, 'NNPS')]
        nouns += [nnps_token]

    res = nouns
    if is_int:
        res = []
        for noun in nouns:
            noun_tokens = nltk.word_tokenize(noun)
            noun_indexes = [token_to_index.get(token, unk_idx) for token in noun_tokens]
            res += [noun_indexes]

    return res
