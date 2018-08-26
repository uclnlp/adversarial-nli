# -*- coding: utf-8 -*-

import tensorflow as tf

from nnli.models import tfutil
from nnli.models import BaseRTEModel

import logging

logger = logging.getLogger(__name__)


class ConditionalBiLSTM(BaseRTEModel):
    def __init__(self, representation_size=300, dropout_keep_prob=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.representation_size = representation_size
        self.dropout_keep_prob = dropout_keep_prob

        with tf.variable_scope('lstm', reuse=self.reuse) as _:
            fused_rnn = tf.contrib.rnn.LSTMBlockFusedCell(self.representation_size)
            # [batch, 2*output_dim] -> [batch, num_classes]
            _, q_states = tfutil.fused_birnn(fused_rnn, self.sequence1, sequence_length=self.sequence1_length,
                                             dtype=tf.float32, time_major=False, scope="sequence1_rnn")

            outputs, _ = tfutil.fused_birnn(fused_rnn, self.sequence2, sequence_length=self.sequence2_length,
                                            dtype=tf.float32, initial_state=q_states, time_major=False, scope="sequence2_rnn")

            outputs = tf.concat([outputs[0], outputs[1]], axis=2)
            hidden = tf.layers.dense(outputs, self.representation_size, tf.nn.relu, name="hidden") * tf.expand_dims(
                tf.sequence_mask(self.sequence2_length, maxlen=tf.shape(outputs)[1], dtype=tf.float32), 2)
            hidden = tf.reduce_max(hidden, axis=1)
            # [batch, dim] -> [batch, num_classes]
            outputs = tf.layers.dense(hidden, self.nb_classes, name="classification")
            self.logits = outputs

    def __call__(self):
            return self.logits
