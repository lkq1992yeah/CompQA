"""
Author: Kangqi Luo
Goal: Neural Tensor Network layer
NTN can be a replacement of cosine similarity
Regarded as a combination of multiple bi-linear modules
"""

import tensorflow as tf


class NeuralTensorNetwork(object):

    def __init__(self, dim_hidden, blocks):
        self.dim_hidden = dim_hidden
        self.blocks = blocks
        # main param in NTN: (dim_hidden, blocks * dim_hidden)
        # several blocks of 2-dim matrices

    def forward(self, x_input, y_input, reuse=None):
        """
        For each data, provided with two vectors, calculate the final score through NTN
        :param x_input: (data_size, dim_hidden)
        :param y_input: (data_size, dim_hidden)
        :param reuse: reuse flag in variable_scope
        :return: (data_size, ) returning the final score of each <x, y> data
        """
        block_detail = tf.contrib.layers.fully_connected(
            inputs=x_input,
            num_outputs=self.blocks*self.dim_hidden,
            activation_fn=None,
            scope='NTN',
            reuse=reuse
        )       # (data_size, blocks * dim_hidden)
        linear_rep = tf.reshape(block_detail,
                                shape=[-1, self.blocks, self.dim_hidden],
                                name='linear_rep')  # (data_size, blocks, dim_hidden)
        y_rep = tf.stack([y_input] * self.blocks,
                         axis=1, name='y_rep')      # (data_size, blocks, dim_hidden)
        bilinear_rep = tf.reduce_sum(linear_rep * y_rep,
                                     axis=-1, name='bilinear_rep')      # (data_size, blocks)
        # now got the bilinear score (xWy) of each block

        ntn_score = tf.contrib.layers.fully_connected(
            inputs=bilinear_rep,
            num_outputs=1,
            activation_fn=None,
            scope='NTN_final',
            reuse=reuse
        )       # (data_size, 1)
        ntn_score = tf.squeeze(ntn_score, axis=-1, name='ntn_score')
        return ntn_score
