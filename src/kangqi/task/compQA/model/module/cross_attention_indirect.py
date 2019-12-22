"""
Author: Kangqi Luo
Goal: Combine the structure of ABCNN-1 and AF-attention
(A Decomposable Attention Model for Natural Language Inference)
We are using the module in compQA scenario, where the rhs (path) is represented by both pwords and preds.
Therefore, we send'em together into the module, making it a little bit more complex than a normal CrossAtt layer.
"""

import tensorflow as tf
from . import att_layer

from kangqi.util.LogUtil import LogInfo


class IndirectCrossAttention:

    def __init__(self, lf_max_len, rt_max_len, dim_att_hidden, att_func):
        self.lf_max_len = lf_max_len
        self.rt_max_len = rt_max_len
        self.dim_att_hidden = dim_att_hidden
        LogInfo.logs('IndirectCrossAttention: lf_max_len = %d, rt_max_len = %d, dim_att_hidden = %d, att_func = %s.',
                     lf_max_len, rt_max_len, dim_att_hidden, att_func)

        assert att_func in ('dot', 'bilinear', 'bahdanau', 'bdot')
        self.att_func = getattr(att_layer, 'cross_att_' + att_func)

    def forward(self, lf_input, lf_mask, rt_input, rt_mask):
        """
        :param lf_input:    (ds, lf_max_len, dim_hidden)
        :param lf_mask:     (ds, lf_max_len) as float32
        :param rt_input:    (ds, rt_max_len, dim_hidden)
        :param rt_mask:     (ds, rt_max_len) as float32
        """
        with tf.variable_scope('cross_att_indirect', reuse=tf.AUTO_REUSE):
            lf_cube_mask = tf.stack([lf_mask] * self.rt_max_len,
                                    axis=-1, name='lf_cube_mask')  # (ds, lf_max_len, rt_max_len)
            rt_cube_mask = tf.stack([rt_mask] * self.lf_max_len,
                                    axis=1, name='rt_cube_mask')  # (ds, lf_max_len, rt_max_len)
            cube_mask = tf.multiply(lf_cube_mask, rt_cube_mask, name='cube_mask')

            """ Calculate cross attention matrix """
            raw_att_mat = self.att_func(lf_input=lf_input, rt_input=rt_input,
                                        lf_max_len=self.lf_max_len,
                                        rt_max_len=self.rt_max_len,
                                        dim_att_hidden=self.dim_att_hidden)
            masked_att_mat = raw_att_mat * cube_mask + tf.float32.min * (1. - cube_mask)
            # padding: -inf

            """ Attention normalize & produce att_repr """
            att_norm_for_lf = tf.nn.softmax(masked_att_mat, dim=2, name='att_norm_for_lf')
            att_norm_for_rt = tf.nn.softmax(masked_att_mat, dim=1, name='att_norm_for_rt')
            # for_lf: sum_j A[:,j] = 1.
            # for_rt: sum_i A[i,:] = 1.
            
            lf_att_repr = tf.matmul(att_norm_for_lf, rt_input, name='lf_att_repr')  # (ds, lf_max_len, dim_emb)
            rt_att_repr = tf.matmul(tf.transpose(att_norm_for_rt, perm=[0, 2, 1]),  # (ds, rt_max_len, lf_max_len)
                                    lf_input, name='rt_att_repr')  # (ds, rt_max_len, dim_emb)

        return lf_att_repr, rt_att_repr, raw_att_mat

    # @staticmethod
    # def att_norm_col_wise(att_mat):
    #     sum_of_cols = 1e-4 + tf.reduce_mean(att_mat, axis=1, name='sum_of_cols')    # (ds, rt_max_len)
    #     sum_of_cols = tf.expand_dims(sum_of_cols, axis=1)   # (ds, 1, rt_max_len)
    #     att_norm = tf.div(att_mat, sum_of_cols, name='att_norm_col_wise')
    #     # (ds, lf_max_len, rt_max_len), sum(att_norm[:, j]) = 1
    #     # att_norm[:, j]: the distribution over left words for each word-j at right side
    #     return att_norm
    #
    # @staticmethod
    # def att_norm_row_wise(att_mat):
    #     sum_of_rows = 1e-4 + tf.reduce_sum(att_mat, axis=2, name='sum_of_rows')     # (ds, lf_max_len)
    #     sum_of_rows = tf.expand_dims(sum_of_rows, axis=2)   # (ds, lf_max_len, 1)
    #     att_norm = tf.div(att_mat, sum_of_rows, name='att_norm_row_wise')
    #     # (ds, lf_max_len, rt_max_len), sum(att_norm[i, :]) = 1
    #     # att_norm[i, :]: the distribution over right words for each word-i at left side
    #     return att_norm
    #
    # def construct_att_weights(self, att_mat):
    #     """
    #     Parikh: Go through formula (2) in AF-attention paper
    #     :param att_mat: (ds, q_max_len, p_max_len + pw_max_len)
    #     :return: 3 attention weights (q, p, pw) and the split attention matrices
    #     """
    #     """ Naive v.s. Parikh: just different from the normalizing direction!! """
    #     p_att_mat, pw_att_mat = tf.split(value=att_mat,
    #                                      num_or_size_splits=[self.p_max_len, self.pw_max_len],
    #                                      axis=2)    # (ds, q_max_len, p_max_len | pw_max_len)
    #     if self.att_norm_mode == 'parikh':
    #         att_wt_q = self.att_norm_col_wise(att_mat=att_mat)      # (ds, q_max_len, p_max_len+pw_max_len)
    #         att_wt_p = self.att_norm_row_wise(att_mat=p_att_mat)    # (ds, q_max_len, p_max_len)
    #         att_wt_pw = self.att_norm_row_wise(att_mat=pw_att_mat)  # (ds, q_max_len, pw_max_len)
    #     else:       # naive
    #         att_wt_q = self.att_norm_row_wise(att_mat=att_mat)
    #         att_wt_p = self.att_norm_col_wise(att_mat=p_att_mat)
    #         att_wt_pw = self.att_norm_col_wise(att_mat=pw_att_mat)
    #     return p_att_mat, pw_att_mat, att_wt_q, att_wt_p, att_wt_pw
