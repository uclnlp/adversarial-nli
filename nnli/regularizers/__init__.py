# -*- coding: utf-8 -*-

from nnli.regularizers.base import contradiction_acl
from nnli.regularizers.base import entailment_acl
from nnli.regularizers.base import neutral_acl

from nnli.regularizers.base import entailment_reflexive_acl
from nnli.regularizers.base import entailment_neutral_acl

from nnli.regularizers.placeholders import get_placeholders

__all__ = [
    'contradiction_acl',
    'entailment_acl',
    'neutral_acl',

    'entailment_reflexive_acl',
    'entailment_neutral_acl',

    'get_placeholders'
]
