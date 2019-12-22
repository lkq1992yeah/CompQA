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

    def __init__(self, q_max_len, p_max_len, pw_max_len, dim_input,
                 dim_att_hidden, att_calc_mode, att_norm_mode, att_merge_mode):
        self.q_max_len = q_max_len
        self.p_max_len = p_max_len
        self.pw_max_len = pw_max_len
        self.dim_input = dim_input
        self.dim_att_hidden = dim_att_hidden  # hidden dim size used in attention (Bahdanau or bilinear)
        self.att_calc_mode = att_calc_mode
        self.att_norm_mode = att_norm_mode
        self.att_merge_mode = att_merge_mode

        self.lf_max_len = self.q_max_len
        self.rt_max_len = self.p_max_len + self.pw_max_len
        self.dim_output = self.dim_input    # Default: output_size == input_size
        self.reuse = tf.AUTO_REUSE

        assert isinstance(self.lf_max_len, int)
        assert isinstance(self.rt_max_len, int)

        """
        bahdanau: a = u . relu(W[x_1 : x_2] + b)
        bilinear: a = x_1 . W . x_2
        bahdanau-dot: t_i = relu(Wx_i + b), a = t_1 . t_2  
        """
        assert self.att_calc_mode in ('bahdanau', 'bilinear', 'bahdanau_dot')
        assert self.att_norm_mode in ('parikh', 'naive')
        assert self.att_merge_mode in ('sum', 'concat')
        # concat: actually concat+FC

        cross_att_func_name = 'cross_att_' + self.att_calc_mode
        self.cross_att_func = getattr(att_layer, cross_att_func_name)

    def forward(self, q_input, p_input, pw_input, q_len, p_len, pw_len):
        """
        Produce the attention matrix, and the rich representation of questions, predicates and pwords
            x_input: (ds, x_max_len, dim_emb == dim_input)
            x_len: (ds, )
        :return: q_output, (p_output, p_att_mat), (pw_output, pw_att_mat)
            x_output: (ds, x_max_len, dim_output)
            x_att_mat: (ds, q_max_len, x_max_len)
        """
        assert q_len.dtype == p_len.dtype == pw_len.dtype == tf.int32
        with tf.variable_scope('cross_att_indirect', reuse=self.reuse):

            """ Get 3-dim mask """
            q_mask, p_mask, pw_mask = [
                tf.sequence_mask(lengths=x_len, maxlen=x_max_len, dtype=tf.float32)
                for x_len, x_max_len in zip(
                    (q_len, p_len, pw_len),
                    (self.q_max_len, self.p_max_len, self.pw_max_len)
                )
            ]       # (ds, x_max_len) as mask

            lf_mask = q_mask                                    # (ds, lf_max-len)
            rt_mask = tf.concat([p_mask, pw_mask],
                                axis=-1, name='rt_mask')        # (ds, rt_max_len)
            lf_cube_mask = tf.stack([lf_mask] * self.rt_max_len, axis=-1)   # (ds, lf_max_len, rt_max_len)
            rt_cube_mask = tf.stack([rt_mask] * self.lf_max_len, axis=1)    # (ds, lf_max_len, rt_max_len)
            cube_mask = lf_cube_mask * rt_cube_mask

            """ Calculate cross attention matrix """
            lf_input = q_input  # (ds, lf_max_len, dim_emb)
            rt_input = tf.concat([p_input, pw_input], axis=1, name='rt_input')  # (ds, rt_max_len, dim_emb)
            args = {'lf_input': lf_input, 'lf_max_len': self.lf_max_len,
                    'rt_input': rt_input, 'rt_max_len': self.rt_max_len,
                    'dim_att_hidden': self.dim_att_hidden}
            raw_att_mat = self.cross_att_func(**args)
            masked_att_mat = raw_att_mat * cube_mask + tf.float32.min * (1. - cube_mask)
            att_mat = tf.exp(masked_att_mat, name='att_mat')        # (ds, lf_max_len, rt_max_len)
            # after exp(): valid grid > 0, padding grid = 0.

            """ Calcuate attention repr of q, given p and pw information """
            [p_att_mat, pw_att_mat,
             att_wt_q, att_wt_p, att_wt_pw] = self.construct_att_weights(att_mat=att_mat)
            q_att_repr = tf.matmul(att_wt_q, rt_input, name='q_att_repr')   # (ds, lf_max_len, dim_emb)
            p_att_repr = tf.matmul(tf.transpose(att_wt_p, perm=[0, 2, 1]),      # (ds, p_max_len, q_max_len)
                                   q_input, name='p_att_repr')              # (ds, p_max_len, dim_emb)
            pw_att_repr = tf.matmul(tf.transpose(att_wt_pw, perm=[0, 2, 1]),    # (ds, pw_max_len, q_max_len)
                                    q_input, name='pw_att_repr')            # (ds, pw_max_len, dim_emb)
            # * Note: batch matrix multiplication through calling tf.matmul

            """ Finally: construct outputs """
            output_tensors = []
            for x_input, x_att_repr, x_name in zip((q_input, p_input, pw_input),
                                                   (q_att_repr, p_att_repr, pw_att_repr),
                                                   ('q', 'p', 'pw')):
                if self.att_merge_mode == 'sum':
                    # Something like shortcut
                    x_output = tf.add(x_input, x_att_repr,
                                      name=x_name+'_output')    # (ds, x_max_len, dim_emb)
                else:       # concat & FC, following formula (3) in AF-attention
                    x_concat = tf.concat([x_input, x_att_repr], axis=-1,
                                         name=x_name+'_concat')     # (ds, lf_max_len, dim_emb * 2)
                    x_output = tf.contrib.layers.fully_connected(inputs=x_concat,
                                                                 num_outputs=self.dim_output,
                                                                 activation_fn=tf.nn.relu,
                                                                 scope='merge')
                output_tensors.append(x_output)
            q_output, p_output, pw_output = output_tensors

        return q_output, (p_output, p_att_mat), (pw_output, pw_att_mat)

    @staticmethod
    def att_norm_col_wise(att_mat):
        sum_of_cols = 1e-4 + tf.reduce_mean(att_mat, axis=1, name='sum_of_cols')    # (ds, rt_max_len)
        sum_of_cols = tf.expand_dims(sum_of_cols, axis=1)   # (ds, 1, rt_max_len)
        att_norm = tf.div(att_mat, sum_of_cols, name='att_norm_col_wise')
        # (ds, lf_max_len, rt_max_len), sum(att_norm[:, j]) = 1
        # att_norm[:, j]: the distribution over left words for each word-j at right side
        return att_norm

    @staticmethod
    def att_norm_row_wise(att_mat):
        sum_of_rows = 1e-4 + tf.reduce_sum(att_mat, axis=2, name='sum_of_rows')     # (ds, lf_max_len)
        sum_of_rows = tf.expand_dims(sum_of_rows, axis=2)   # (ds, lf_max_len, 1)
        att_norm = tf.div(att_mat, sum_of_rows, name='att_norm_row_wise')
        # (ds, lf_max_len, rt_max_len), sum(att_norm[i, :]) = 1
        # att_norm[i, :]: the distribution over right words for each word-i at left side
        return att_norm

    def construct_att_weights(self, att_mat):
        """
        Parikh: Go through formula (2) in AF-attention paper
        :param att_mat: (ds, q_max_len, p_max_len + pw_max_len)
        :return: 3 attention weights (q, p, pw) and the split attention matrices
        """
        """ Naive v.s. Parikh: just different from the normalizing direction!! """
        p_att_mat, pw_att_mat = tf.split(value=att_mat,
                                         num_or_size_splits=[self.p_max_len, self.pw_max_len],
                                         axis=2)    # (ds, q_max_len, p_max_len | pw_max_len)
        if self.att_norm_mode == 'parikh':
            att_wt_q = self.att_norm_col_wise(att_mat=att_mat)      # (ds, q_max_len, p_max_len+pw_max_len)
            att_wt_p = self.att_norm_row_wise(att_mat=p_att_mat)    # (ds, q_max_len, p_max_len)
            att_wt_pw = self.att_norm_row_wise(att_mat=pw_att_mat)  # (ds, q_max_len, pw_max_len)
        else:       # naive
            att_wt_q = self.att_norm_row_wise(att_mat=att_mat)
            att_wt_p = self.att_norm_col_wise(att_mat=p_att_mat)
            att_wt_pw = self.att_norm_col_wise(att_mat=pw_att_mat)
        return p_att_mat, pw_att_mat, att_wt_q, att_wt_p, att_wt_pw
