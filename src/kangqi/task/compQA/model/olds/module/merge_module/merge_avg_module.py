# -*- coding: utf-8 -*-

import tensorflow as tf
from .merge_base_module import MergeBaseModule


class MergeAvgModule(MergeBaseModule):

    def __init__(self, q_max_len, sc_max_len, dim_q_hidden, dim_sk_hidden,
                 dim_merge_hidden):
        super(MergeAvgModule, self).__init__(q_max_len = q_max_len,
                                             sc_max_len = sc_max_len,
                                             dim_q_hidden = dim_q_hidden,
                                             dim_sk_hidden= dim_sk_hidden)
        self.dim_merge_hidden = dim_merge_hidden

    def forward(self, q_hidden, q_len, sc_hidden, sc_len, reuse=None):
        q_mask = tf.sequence_mask(
            lengths=tf.to_int32(q_len),
            maxlen=tf.to_int32(self.q_max_len),
            dtype=tf.float32
        )
        q_mask = tf.expand_dims(q_mask, -1)
        q_sum = tf.reduce_sum(q_hidden * q_mask, axis=1)
        mask_sum = tf.reduce_sum(q_mask, axis=1)
        q_avg = tf.div(q_sum, mask_sum)

        sc_mask = tf.sequence_mask(
            lengths=tf.to_int32(sc_len),
            maxlen=tf.to_int32(self.sc_max_len),
            dtype=tf.float32
        )
        sc_mask = tf.expand_dims(sc_mask, -1)
        sc_sum = tf.reduce_sum(sc_hidden * sc_mask, axis=1)
        mask_sum = tf.reduce_sum(sc_mask, axis=1)
        sc_avg = tf.div(sc_sum, mask_sum)

        with tf.name_scope('MergeModule'):     # , reuse=tf.AUTO_REUSE):
            w_merge = tf.get_variable(
                name='w_merge',
                shape=(self.dim_q_hidden + self.dim_sk_hidden, self.dim_merge_hidden),
                initializer=tf.contrib.layers.xavier_initializer()
            )
            b_merge = tf.get_variable(
                name='b_merge',
                shape=(self.dim_merge_hidden,)
            )
            merge_input = tf.concat([q_avg, sc_avg], axis=-1, name='merge_input')     # (batch, n_input_hidden)
            merge_hidden = tf.nn.relu(tf.matmul(merge_input, w_merge) + b_merge,
                                      name='merge_hidden')                                  # (-1, n_merge_hidden)

            w_final = tf.get_variable(
                name='w_final',
                shape=(self.dim_merge_hidden, 1),
                initializer=tf.contrib.layers.xavier_initializer()
            )
            b_final = tf.get_variable(
                name='b_final',
                shape=(1,)
            )
            logits = tf.reshape(tf.matmul(merge_hidden, w_final) + b_final,
                                shape=[-1], name='logits')  # (-1,)
        return logits
