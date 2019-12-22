# -*- coding:utf-8 -*-

import tensorflow as tf

from kangqi.util.tf.tf_basics import conv2d, maxpool2d, xavier_weight_init, conv2d_xavier_weight_init

class LetterTriGramCNN(object):

    def __init__(self, n_max_len, n_hash, n_window, n_conv, n_hidden, scope_name='letter_trigram_cnn'):
        self.n_conv = n_conv
        self.n_max_len = n_max_len
        self.n_window = n_window
        self.scope_name = scope_name

        with tf.variable_scope(scope_name, initializer=xavier_weight_init()):
            self.W_conv = tf.get_variable(
                            name='W_conv',
                            shape=(1, n_window, n_hash, n_conv),
                            initializer=conv2d_xavier_weight_init())
            self.b_conv = tf.get_variable(
                            name='b_conv',
                            shape=(n_conv,))
            self.W_hidden = tf.get_variable(
                            name='W_hidden',
                            shape=(n_conv, n_hidden))
            self.b_hidden = tf.get_variable(
                            name='b_hidden',
                            shape=(n_hidden,))


    # input: (data, n_max_len, n_hash)
    # output: (data, n_hidden)
    def build(self, input):
        with tf.variable_scope(self.scope_name):
            input_conv = tf.expand_dims(input, 1)       # (data, 1, n_max_len, n_hash)
            h_conv = tf.nn.relu(conv2d(input_conv, self.W_conv) + self.b_conv)
            h_pool = maxpool2d(
                        h_conv,
                        [1, self.n_max_len-self.n_window+1]) # (data, 1, 1, n_conv)
            h_pool = tf.reshape(h_pool, [-1, self.n_conv])  # (data, n_conv)
            h_output = tf.nn.relu(tf.matmul(h_pool, self.W_hidden) + self.b_hidden)
            return h_output     # (data, n_hidden)
