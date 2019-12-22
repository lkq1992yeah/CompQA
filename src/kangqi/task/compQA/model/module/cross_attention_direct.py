"""
Author: Kangqi Luo
Goal: Follow ABCNN-2 and xusheng/model/attention.py
"""

import tensorflow as tf
from . import att_layer
from kangqi.util.LogUtil import LogInfo


class DirectCrossAttention:

    def __init__(self, lf_max_len, rt_max_len, dim_att_hidden, att_func):
        self.lf_max_len = lf_max_len
        self.rt_max_len = rt_max_len
        self.dim_att_hidden = dim_att_hidden
        LogInfo.logs('DirectCrossAttention: lf_max_len = %d, rt_max_len = %d, dim_att_hidden = %d, att_func = %s.',
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
        with tf.variable_scope('cross_att_direct', reuse=tf.AUTO_REUSE):
            lf_cube_mask = tf.stack([lf_mask] * self.rt_max_len,
                                    axis=-1, name='lf_cube_mask')   # (ds, lf_max_len, rt_max_len)
            rt_cube_mask = tf.stack([rt_mask] * self.lf_max_len,
                                    axis=1, name='rt_cube_mask')    # (ds, lf_max_len, rt_max_len)
            cube_mask = tf.multiply(lf_cube_mask, rt_cube_mask, name='cube_mask')
            # show_tensor(lf_cube_mask)
            # show_tensor(rt_cube_mask)
            # show_tensor(cube_mask)

            """ Calculate cross attention matrix """
            raw_att_mat = self.att_func(lf_input=lf_input, rt_input=rt_input,
                                        lf_max_len=self.lf_max_len,
                                        rt_max_len=self.rt_max_len,
                                        dim_att_hidden=self.dim_att_hidden)
            # (ds, lf_max_len, rt_max_len)

            """
            Kangqi on 180211:
            In ABCNN-2, there doesn't exist the exp(.) step,
            but due to their definition of matrix A, all elements in A are non-negative.
            Therefore we add this exp(.) layer for trial.
            """
            # masked_att_mat = raw_att_mat * cube_mask + tf.float32.min * (1. - cube_mask)
            # att_mat = tf.exp(masked_att_mat, name='att_mat')  # (ds, lf_max_len, rt_max_len)
            # # after exp(): valid grid > 0, padding grid = 0.
            #
            # att_val_lf = tf.reduce_sum(att_mat, axis=2)     # (ds, lf_max_len)
            # att_val_rt = tf.reduce_sum(att_mat, axis=1)     # (ds, rt_max_len)

            """
            Kangqi on 180217:
            we try to use averaged attention score at each row / column.
            att_val_lf: sum over right words, therefore divide by "rt_words", verse visa.
            The trick just acts as a dynamic cool down before softmax,
            which reduces the affect of severe weight gathering.
            """
            # masked_att_mat = raw_att_mat * cube_mask        # padding: 0
            #
            # lf_words = tf.maximum(1.0, tf.reduce_sum(lf_mask, axis=-1, keep_dims=True))
            # rt_words = tf.maximum(1.0, tf.reduce_sum(rt_mask, axis=-1, keep_dims=True))   # (ds, 1) as float32
            # # make sure we are not dividing zero
            #
            # att_val_lf = tf.reduce_sum(masked_att_mat, axis=2) / rt_words  # (ds, lf_max_len)
            # att_val_rt = tf.reduce_sum(masked_att_mat, axis=1) / lf_words  # (ds, rt_max_len)
            #
            # # before softmax: set padding row / columns to be -inf
            # att_val_lf = att_val_lf * lf_mask + tf.float32.min * (1. - lf_mask)
            # att_val_rt = att_val_rt * rt_mask + tf.float32.min * (1. - rt_mask)
            #
            # # Normalize the scores
            # lf_norm = tf.nn.softmax(att_val_lf, name='lf_norm')
            # rt_norm = tf.nn.softmax(att_val_rt, name='rt_norm')

            """
            Kangqi on 180220:
            After discussing with Kenny, I realized that I should first perform exp and then averaging.
            a big negative logit (like -40) means "no contribution", rather than "negative contribution".
            If I perform averaging over logits, then other positive cells are affected by othe big negative logits.
            The only issue is, I can't remember why I edited the code of 180211 (40 lines above). 
            I must made other mistakes then.
            """
            masked_att_mat = raw_att_mat * cube_mask + tf.float32.min * (1. - cube_mask)
            # padding by -inf, because we are going to apply exp(.)
            # max_mask = tf.reduce_max(masked_att_mat)
            # masked_att_mat = tf.subtract(masked_att_mat, max_mask, name='masked_att_mat')
            # # avoid we apply exp(.) on large numbers, resulting in inf or nan
            exp_att_mat = tf.exp(masked_att_mat, name='exp_att_mat')

            att_val_lf = tf.reduce_sum(exp_att_mat, axis=2)      # (ds, lf_max_len)
            att_val_rt = tf.reduce_sum(exp_att_mat, axis=1)      # (ds, rt_max_len)
            # For padding row / column, their corresponding sum must be 0.
            # Therefore we don't need to apply masking here.

            # perform averaging, rather than softmax. (remeber to avoid nan)
            lf_norm = tf.div(att_val_lf, 1e-6 + tf.reduce_sum(att_val_lf, axis=-1, keep_dims=True), name='lf_norm')
            rt_norm = tf.div(att_val_rt, 1e-6 + tf.reduce_sum(att_val_rt, axis=-1, keep_dims=True), name='rt_norm')

            """ Finally: Calculate weighted sum of inputs according to the attention values """
            lf_weighted = tf.expand_dims(lf_norm, axis=2) * lf_input    # (ds, lf_max_len, dim_hidden)
            lf_weighted = tf.reduce_sum(lf_weighted, axis=1)            # (ds, dim_hidden)
            rt_weighted = tf.expand_dims(rt_norm, axis=2) * rt_input
            rt_weighted = tf.reduce_sum(rt_weighted, axis=1)

            return lf_weighted, rt_weighted, raw_att_mat, lf_norm, rt_norm
