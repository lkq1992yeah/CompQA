# -*- coding: utf-8 -*-

import tensorflow as tf

from .merge_base_module import MergeBaseModule
from ...u import show_tensor

from kangqi.util.LogUtil import LogInfo


class MergeFCModule(MergeBaseModule):

    def __init__(self, q_max_len, sc_max_len, dim_q_hidden, dim_sk_hidden,
                 dim_sc_hidden, dim_merge_hidden, final_merge='concat'):
        super(MergeFCModule, self).__init__(q_max_len=q_max_len,
                                            sc_max_len=sc_max_len,
                                            dim_q_hidden=dim_q_hidden,
                                            dim_sk_hidden=dim_sk_hidden)
        self.dim_sc_hidden = dim_sc_hidden
        self.dim_merge_hidden = dim_merge_hidden
        self.final_merge = final_merge

        self.dim_flat = sc_max_len * dim_sk_hidden
        self.dim_input_hidden = dim_q_hidden + dim_sc_hidden

    # Input:
    #   q_hidden: (batch, q_max_len, dim_q_hidden)
    #   q_len: (batch, ) as int32
    #   sc_hidden: (batch, sc_max_len, dim_sk_hidden)
    #   sc_len: (batch, ) as int32
    # Output:
    #   score: (batch, )
    def forward(self, q_hidden, q_len, sc_hidden, sc_len, reuse=None):
        LogInfo.begin_track('MergeFCModule forward: ')

        with tf.variable_scope('MergeFCModule', reuse=reuse):
            dyn_batch = tf.shape(q_hidden)[0]
            dummy_att_tensor = tf.get_variable(name='dummy_att_mat',
                                               shape=(dyn_batch, self.q_max_len, self.sc_max_len))
            dummy_q_weight = tf.get_variable(name='dummy_q_weight',
                                             shape=(dyn_batch, self.q_max_len))
            dummy_sc_weight = tf.get_variable(name='dummy_sc_weight',
                                              shape=(dyn_batch, self.sc_max_len))

            sum_q_hidden = tf.reduce_sum(
                q_hidden,
                axis=1,
                name='sum_q_hidden'
            )  # (batch, dim_q_hidden)
            q_len_mat = tf.cast(
                tf.expand_dims(q_len, axis=1),
                dtype=tf.float32,
                name='q_len_mat'
            )   # (batch, 1) as float32
            avg_q_hidden = tf.div(sum_q_hidden, q_len_mat, name='avg_q_hidden')     # (batch, dim_q_hidden)
            show_tensor(avg_q_hidden)

            flat_sc_hidden = tf.reshape(
                sc_hidden,
                shape=[-1, self.dim_flat],
                name='flat_sc_hidden'
            )       # (batch, sc_max_len * dim_sk_hidden)
            w_sc_hidden = tf.get_variable(name='w_sc_hidden',
                                          shape=(self.dim_flat, self.dim_sc_hidden),
                                          initializer=tf.contrib.layers.xavier_initializer())
            b_sc_hidden = tf.get_variable(name='b_sc_hidden',
                                          shape=(self.dim_sc_hidden,))
            fc_sc_hidden = tf.nn.relu(
                tf.matmul(flat_sc_hidden, w_sc_hidden) + b_sc_hidden,
                name='fc_sc_hidden'
            )  # (ds, n_sc_hidden)
            show_tensor(fc_sc_hidden)

            logits = None
            if self.final_merge == 'concat':
                w_merge = tf.get_variable(name='w_merge',
                                          shape=(self.dim_input_hidden, self.dim_merge_hidden),
                                          initializer=tf.contrib.layers.xavier_initializer())
                b_merge = tf.get_variable(name='b_merge',
                                          shape=(self.dim_merge_hidden,))
                merge_input = tf.concat(
                    [avg_q_hidden, fc_sc_hidden],
                    axis=-1,
                    name='merge_input'
                )  # (batch, dim_input_hidden)
                merge_hidden = tf.nn.relu(
                    tf.matmul(merge_input, w_merge) + b_merge,
                    name='merge_hidden'
                )  # (batch, dim_merge_hidden)

                w_final = tf.get_variable(name='w_final',
                                          shape=(self.dim_merge_hidden, 1),
                                          initializer=tf.contrib.layers.xavier_initializer())
                b_final = tf.get_variable(name='b_final',
                                          shape=(1,))
                logits = tf.reshape(
                    tf.matmul(merge_hidden, w_final) + b_final,
                    shape=[-1],
                    name='logits'
                )  # (-1,)
            elif self.final_merge == 'cos':
                assert self.dim_q_hidden == self.dim_sc_hidden
                lf_norm = tf.sqrt(
                    tf.reduce_sum(avg_q_hidden ** 2, axis=1, keep_dims=True) + 1e-6,
                    name='lf_norm'
                )
                q_norm_hidden = tf.div(avg_q_hidden, lf_norm,
                                       name='q_norm_hidden')    # (batch, dim_q_hidden)
                rt_norm = tf.sqrt(
                    tf.reduce_sum(fc_sc_hidden ** 2, axis=1, keep_dims=True) + 1e-6,
                    name='rt_norm'
                )
                sc_norm_hidden = tf.div(fc_sc_hidden, rt_norm,
                                        name='sc_norm_hidden')  # (batch, dim_sc_hidden)
                logits = tf.reduce_sum(
                    q_norm_hidden * sc_norm_hidden,
                    axis=1,
                    name='logits'
                )   # (batch, )

        LogInfo.end_track()
        return logits, fc_sc_hidden, dummy_att_tensor, dummy_q_weight, dummy_sc_weight
        # (batch, ), (batch, dim_sc_hidden), (batch, q_max_len, sc_max_len),
        # (batch, q_max_len), (batch, sc_max_len)
