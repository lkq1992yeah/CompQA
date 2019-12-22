# -*- coding: utf-8 -*-

import tensorflow as tf

from merge_base_module import MergeBaseModule
from kangqi.util.LogUtil import LogInfo


class MergeAttModule(MergeBaseModule):

    def __init__(self, q_max_len, sc_max_len, dim_q_hidden, dim_sk_hidden,
                 dim_merge_hidden, final_merge='concat'):
        super(MergeAttModule, self).__init__(q_max_len=q_max_len,
                                             sc_max_len=sc_max_len,
                                             dim_q_hidden=dim_q_hidden,
                                             dim_sk_hidden=dim_sk_hidden)
        self.dim_merge_hidden = dim_merge_hidden
        self.final_merge = final_merge

    def forward(self, q_hidden, q_len, sc_hidden, sc_len, reuse=None):
        """
        :param q_hidden: [B, T1, dim1] 
        :param q_len: [B, ]
        :param sc_hidden: [B, T2, dim2]
        :param sc_len: [B, ]
        :param reuse: 
        :return: score [B, ], att_matrix [B, T1, T2]
        """
        LogInfo.begin_track('MergeAttModule forward: ')

        with tf.variable_scope('MergeFCModule', reuse=reuse):

            # Fully connected layers to transform both left and right tensor
            # into a tensor with `dim_merge_hidden` units
            # [B, T1, dim]
            att_left = tf.contrib.layers.fully_connected(
                inputs=q_hidden,
                num_outputs=self.dim_merge_hidden,
                activation_fn=None,
                scope="att_keys",
                reuse=reuse
            )
            # [B, T2, dim]
            att_right = tf.contrib.layers.fully_connected(
                inputs=sc_hidden,
                num_outputs=self.dim_merge_hidden,
                activation_fn=None,
                scope="att_query",
                reuse=reuse
            )
            # [B, T1, 1, dim]
            att_left = tf.expand_dims(att_left, axis=2)
            # [B, T1, T2, dim]
            att_left = tf.tile(att_left, multiples=[1, 1, self.sc_max_len, 1])
            # [B, T2, 1, dim]
            att_right = tf.expand_dims(att_right, axis=2)
            # [B, T2, T1, dim]
            att_right = tf.tile(att_right, multiples=[1, 1, self.q_max_len, 1])
            # [B, T1, T2, dim]
            att_right = tf.transpose(att_right, perm=[0, 2, 1, 3])

            v_att = tf.get_variable(
                name="v_att",
                shape=[self.dim_merge_hidden],
                dtype=tf.float32
            )

            # [B, T1, T2] Bahdanau Attention: v * tanh(W1x1 + W2x2)
            att_matrix = tf.reduce_sum(v_att * tf.tanh(att_left + att_right), axis=3)

            # [B, T1]
            left_mask = tf.sequence_mask(
                lengths=tf.to_int32(q_len),
                maxlen=tf.to_int32(self.q_max_len),
                dtype=tf.float32)
            # [B, T2]
            right_mask = tf.sequence_mask(
                lengths=tf.to_int32(sc_len),
                maxlen=tf.to_int32(self.sc_max_len),
                dtype=tf.float32)

            # [B, 1, T1]
            left_sum_mask = tf.expand_dims(left_mask, axis=1)
            # [B, T2, T1]
            left_sum_mask = tf.tile(left_sum_mask, multiples=[1, self.sc_max_len, 1])
            # [B, T1, T2]
            left_sum_mask = tf.transpose(left_sum_mask, perm=[0, 2, 1])
            # [B, 1, T2]
            right_sum_mask = tf.expand_dims(right_mask, axis=1)
            # [B, T1, T2]
            right_sum_mask = tf.tile(right_sum_mask, multiples=[1, self.q_max_len, 1])

            # [B, T1]
            att_val_left = tf.reduce_sum(att_matrix * right_sum_mask, axis=2)

            # [B, T2]
            att_val_right = tf.reduce_sum(att_matrix * left_sum_mask, axis=1)

            # Replace all scores for padded inputs with tf.float32.min
            left_val = att_val_left * left_mask + ((1.0 - left_mask) * tf.float32.min)
            right_val = att_val_right * right_mask + ((1.0 - right_mask) * tf.float32.min)

            # Normalize the scores
            left_normalized = tf.nn.softmax(left_val, name="left_normalized")
            right_normalized = tf.nn.softmax(right_val, name="right_normalized")

            # Calculate the weighted average of the attention inputs
            # according to the attention values
            # [B, dim]
            left_weighted = tf.expand_dims(left_normalized, axis=2) * q_hidden
            left_weighted = tf.reduce_sum(left_weighted, axis=1)

            # [B, dim]
            right_weighted = tf.expand_dims(right_normalized, axis=2) * sc_hidden
            right_weighted = tf.reduce_sum(right_weighted, axis=1)

            logits = None
            if self.final_merge == 'concat':
                logits = tf.contrib.layers.fully_connected(
                    inputs=tf.concat([left_weighted, right_weighted], axis=1),
                    num_outputs=1,
                    activation_fn=None,
                    scope="output",
                    reuse=reuse
                )
            elif self.final_merge == 'cos':
                assert self.dim_q_hidden == self.dim_sk_hidden
                lf_norm = tf.sqrt(
                    tf.reduce_sum(left_weighted ** 2, axis=1, keep_dims=True) + 1e-6,
                    name='lf_norm'
                )
                q_norm_hidden = tf.div(left_weighted, lf_norm,
                                       name='q_norm_hidden')  # (batch, dim_q_hidden)
                rt_norm = tf.sqrt(
                    tf.reduce_sum(right_weighted ** 2, axis=1, keep_dims=True) + 1e-6,
                    name='rt_norm'
                )
                sc_norm_hidden = tf.div(right_weighted, rt_norm,
                                        name='sc_norm_hidden')  # (batch, dim_sc_hidden)
                logits = tf.reduce_sum(
                    q_norm_hidden * sc_norm_hidden,
                    axis=1,
                    name='logits'
                )  # (batch, )
        LogInfo.end_track()

        return logits, right_weighted, att_matrix, left_normalized, right_normalized
        # (batch, ), (batch, dim_sc_hidden), (batch, q_max_len, sc_max_len),
        # (batch, q_max_len), (batch, sc_max_len)

if __name__ == '__main__':
    test = MergeAttModule(q_max_len=8,
                          sc_max_len=4,
                          dim_q_hidden=5,
                          dim_sk_hidden=4,
                          dim_merge_hidden=3)

    score, sc_vec, mat, lf_val, rt_val = test.forward(
        q_hidden=tf.random_normal(shape=[3, 8, 5]),
        q_len=[4, 5, 4],
        sc_hidden=tf.random_normal(shape=[3, 4, 4]),
        sc_len=[2, 3, 3]
    )

    print(score.get_shape().as_list())
    print(mat.get_shape().as_list())
