# -*- coding: utf-8 -*-

import gzip
import bz2

import logging

logger = logging.getLogger(__name__)


def iopen(path, *args, **kwargs):
    _open = open
    if path.endswith('.gz'):
        _open = gzip.open
    elif path.endswith('.bz2'):
        _open = bz2.open
    return _open(path, *args, **kwargs)
