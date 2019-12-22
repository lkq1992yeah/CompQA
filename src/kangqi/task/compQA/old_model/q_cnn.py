# -*- coding: utf-8 -*-

import tensorflow as tf

from kangqi.util.tf.tf_basics import weight_variable, bias_variable, conv2d, maxpool2d
from kangqi.util.LogUtil import LogInfo

# from kangqi.util.LogUtil import LogInfo

class QuestionCNN(object):

    def __init__(self, n_maxlen, n_input, n_hidden, n_window, activation=tf.nn.relu):
        self.n_maxlen = n_maxlen
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_window = n_window
        self.activation = activation
        pass

    # Note: data_size = ds = batch * PN
    # q_tf: tensor for question input (ds, n_maxlen, n_input)
    def build(self, q_tf):
        x = tf.expand_dims(q_tf, 1)     # (ds, 1, n_maxlen, n_input)
        W_conv = weight_variable([1, self.n_window, self.n_input, self.n_hidden])
        b_conv = bias_variable([self.n_hidden])
        h_conv = self.activation(conv2d(x, W_conv) + b_conv)
        # h_conv: (ds, 1, n_maxlen - n_window + 1, n_hidden)
        h_pool = maxpool2d(h_conv, [1, self.n_maxlen - self.n_window + 1])
        # h_pool: (ds, 1, 1, n_hidden)

        cnn_output_tf = tf.reshape(h_pool, [-1, self.n_hidden])
        LogInfo.logs('cnn_output_tf compiled. %s', cnn_output_tf.get_shape().as_list())
        LogInfo.logs('* QuestionCNN built.')
        return cnn_output_tf