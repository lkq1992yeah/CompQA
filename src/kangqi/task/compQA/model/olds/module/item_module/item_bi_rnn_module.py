# -*- coding:utf-8 -*-

import tensorflow as tf
from xusheng.model.rnn_encoder import BidirectionalRNNEncoder

from .item_base_module import ItemBaseModule
from ...u import show_tensor

from kangqi.util.LogUtil import LogInfo


class ItemBiRNNModule(ItemBaseModule):

    # Below: parameters in BidirectionalRNNEncoder
    # key_list = ["cell_class", "num_units", "dropout_input_keep_prob",
    #             "dropout_output_keep_prob", "num_layers", "reuse"]
    # Without any attention mechanism
    def __init__(self, item_max_len, dim_wd_emb, dim_item_hidden, rnn_config):
        super(ItemBiRNNModule, self).__init__(item_max_len=item_max_len,
                                              dim_wd_emb=dim_wd_emb,
                                              dim_item_hidden=dim_item_hidden)
        rnn_config['num_units'] = dim_item_hidden / 2  # bidirectional
        self.rnn_encoder = BidirectionalRNNEncoder(rnn_config, mode=tf.contrib.learn.ModeKeys.TRAIN)

    # Input:
    #   item_wd_embedding: (batch, item_max_len, dim_wd_emb)
    #   item_len: (batch, ) as int32
    # Output:
    #   item_wd_hidden: (batch, dim_item_hidden)
    def forward(self, item_wd_embedding, item_len, reuse=None):
        LogInfo.begin_track('ItemBiRNNModule forward: ')

        with tf.variable_scope('ItemBiRNNModule', reuse=reuse):
            # stamps = item_wd_embedding.get_shape().as_list()[1]
            stamps = self.item_max_len
            show_tensor(item_wd_embedding)
            birnn_inputs = tf.unstack(item_wd_embedding, num=stamps, axis=1, name='birnn_inputs')
            # rnn_input: a list of stamps elements: (batch, n_emb)
            encoder_output = self.rnn_encoder.encode(inputs=birnn_inputs,
                                                     sequence_length=item_len,
                                                     reuse=reuse)
            birnn_outputs = tf.stack(encoder_output.outputs, axis=1,
                                     name='birnn_outputs')  # (data_size, q_len, n_hidden_emb)
            LogInfo.logs('birnn_output = %s', birnn_outputs.get_shape().as_list())

            sum_wd_hidden = tf.reduce_sum(birnn_outputs, axis=1)  # (data_size, n_hidden_emb)
            item_len_mat = tf.cast(tf.expand_dims(item_len, axis=1),
                                   dtype=tf.float32)  # (data_size, 1) as float
            item_wd_hidden = tf.div(sum_wd_hidden,
                                    tf.maximum(item_len_mat, 1),        # avoid dividing by 0
                                    name='item_wd_hidden')  # (data_size, n_hidden_emb)
            LogInfo.logs('item_wd_hidden = %s', item_wd_hidden.get_shape().as_list())

        LogInfo.end_track()
        return item_wd_hidden
