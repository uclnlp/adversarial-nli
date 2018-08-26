# -*- coding: utf-8 -*-

import numpy as np


class WithoutReplacementSampler:
    def __init__(self, nb_instances, rs=None):
        self.nb_instances = nb_instances
        self.rs = rs

        self.order = self.rs.permutation(self.nb_instances) if rs else np.arange(self.nb_instances)
        self.position = 0

    @staticmethod
    def parse(path, snli):
        instances = snli.parse(path=path)
        return instances

    def sample(self, nb_samples):
        for _ in range(nb_samples):
            yield self.order[self.position % self.nb_instances]
            self.position += 1
