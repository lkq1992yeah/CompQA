# -*- coding:utf-8 -*-

import tensorflow as tf

from .q_base_module import QBaseModule
from xusheng.model.rnn_encoder import BidirectionalRNNEncoder

from kangqi.util.LogUtil import LogInfo


class QBiRNNModule(QBaseModule):

    # Below: parameters in BidirectionalRNNEncoder
    # key_list = ["cell_class", "num_units", "dropout_input_keep_prob",
    #             "dropout_output_keep_prob", "num_layers", "reuse"]
    # Without any attention mechanism
    def __init__(self, dim_q_hidden, rnn_config, q_max_len=None, dim_wd_emb=None):
        super(QBiRNNModule, self).__init__(q_max_len=q_max_len,
                                           dim_wd_emb=dim_wd_emb,
                                           dim_q_hidden=dim_q_hidden)

        rnn_config['num_units'] = dim_q_hidden / 2      # bidirectional
        self.rnn_encoder = BidirectionalRNNEncoder(rnn_config, mode=tf.contrib.learn.ModeKeys.TRAIN)

    def forward(self, q_embedding, q_len, reuse=None):
        LogInfo.begin_track('QBiRNNModule forward: ')

        with tf.variable_scope('QBiRNNModule', reuse=reuse):
            # stamps = q_embedding.get_shape().as_list()[1]
            stamps = self.q_max_len
            birnn_inputs = tf.unstack(q_embedding, num=stamps, axis=1, name='birnn_inputs')
            # rnn_input: a list of stamps elements: (batch, n_emb)
            encoder_output = self.rnn_encoder.encode(inputs=birnn_inputs,
                                                     sequence_length=q_len,
                                                     reuse=reuse)
            q_hidden = tf.stack(
                encoder_output.outputs,
                axis=1,
                name='q_hidden'
            )  # (batch, q_max_len, dim_q_hidden)

        LogInfo.end_track()
        return q_hidden
