import tensorflow as tf

from . import att_layer
from kangqi.util.LogUtil import LogInfo


class SimpleAttention:

    def __init__(self, lf_max_len, dim_att_hidden, att_func):
        self.lf_max_len = lf_max_len
        self.dim_att_hidden = dim_att_hidden
        LogInfo.logs('SimpleAttention: lf_max_len = %d, dim_att_hidden = %d, att_func = %s.',
                     lf_max_len, dim_att_hidden, att_func)

        assert att_func in ('dot', 'bilinear', 'bahdanau', 'bdot')
        self.att_func = getattr(att_layer, 'cross_att_' + att_func)

    def forward(self, lf_input, lf_mask, fix_rt_input):
        """
        :param lf_input:        (ds, lf_max_len, dim_hidden)
        :param lf_mask:         (ds, lf_max_len) as float32
        :param fix_rt_input:    (ds, dim_hidden), no timestamps
        """
        rt_input = tf.expand_dims(fix_rt_input, axis=1, name='rt_input')    # (ds, 1, dim_hidden)
        with tf.variable_scope('simple_att', reuse=tf.AUTO_REUSE):
            raw_att_mat = self.att_func(lf_input=lf_input, rt_input=rt_input,
                                        lf_max_len=self.lf_max_len, rt_max_len=1,
                                        dim_att_hidden=self.dim_att_hidden)     # (ds, lf_max_len, 1)
            raw_att_mat = tf.reshape(raw_att_mat, shape=[-1, self.lf_max_len], name='raw_att_mat')
            # (ds, lf_max_len)

            masked_att_mat = raw_att_mat * lf_mask + tf.float32.min * (1. - lf_mask)
            lf_norm = tf.nn.softmax(masked_att_mat, name='lf_norm')     # (ds, lf_max_len)
            lf_weighted = tf.expand_dims(lf_norm, axis=2) * lf_input    # (ds, lf_max_len, dim_hidden)
            lf_weighted = tf.reduce_sum(lf_weighted, axis=1,
                                        name='lf_weighted')             # (ds, dim_hidden)

        return lf_weighted, raw_att_mat, lf_norm
