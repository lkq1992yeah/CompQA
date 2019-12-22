# -*- coding:utf-8 -*-

# Goal: schema --> list
import tensorflow as tf
from xusheng.model.rnn_encoder import BidirectionalRNNEncoder

from .sk_base_module import SkBaseModule
from ...u import show_tensor

from kangqi.util.LogUtil import LogInfo


class SkBiRNNModule(SkBaseModule):

    def __init__(self, path_max_len, dim_item_hidden, dim_kb_emb, dim_sk_hidden,
                 data_source, rnn_config):
        super(SkBiRNNModule, self).__init__(path_max_len=path_max_len,
                                            dim_item_hidden=dim_item_hidden,
                                            dim_kb_emb=dim_kb_emb,
                                            dim_sk_hidden=dim_sk_hidden)
        self.data_source = data_source
        assert self.data_source in ('kb', 'word', 'both')

        rnn_config['num_units'] = dim_sk_hidden / 2
        self.rnn_encoder = BidirectionalRNNEncoder(rnn_config, mode=tf.contrib.learn.ModeKeys.TRAIN)

    # Input:
    #   path_wd_hidden: (batch, path_max_len, dim_item_hidden)
    #   path_kb_hidden: (batch, path_max_len, dim_kb_emb)
    #   path_len: (batch, ) as int32
    #   focus_wd_hidden: (batch, dim_item_hidden)
    #   focus_kb_hidden: (batch, dim_kb_emb)
    # Output:
    #   sk_hidden: (batch, dim_sk_hidden)
    def forward(self, path_wd_hidden, path_kb_hidden, path_len, focus_wd_hidden, focus_kb_hidden, reuse=None):
        LogInfo.begin_track('SkBiRNNModule forward: ')

        with tf.variable_scope('SkBiRNNModule', reuse=reuse):
            if self.data_source == 'kb':
                use_path_hidden = path_kb_hidden
                use_focus_hidden = focus_kb_hidden
            elif self.data_source == 'word':
                use_path_hidden = path_wd_hidden
                use_focus_hidden = focus_wd_hidden
            else:
                use_path_hidden = tf.concat([path_kb_hidden, path_wd_hidden],
                                            axis=-1, name='use_path_hidden')
                # (batch, path_max_len, dim_item_hidden + dim_kb_hidden)
                use_focus_hidden = tf.concat([focus_kb_hidden, focus_wd_hidden],
                                             axis=-1, name='use_focus_hidden')
                # (batch, dim_item_hidden + dim_kb_hidden)

            use_path_emb_input = tf.concat([tf.expand_dims(use_focus_hidden, axis=1), use_path_hidden],
                                           axis=1,
                                           name='use_path_emb_input')       # (batch, path_max_len + 1, dim_use)
            show_tensor(use_path_emb_input)
            use_path_len = path_len + 1
            stamps = self.path_max_len + 1
            birnn_inputs = tf.unstack(use_path_emb_input, num=stamps, axis=1, name='birnn_inputs')
            encoder_output = self.rnn_encoder.encode(inputs=birnn_inputs,
                                                     sequence_length=use_path_len,
                                                     reuse=reuse)
            rnn_outputs = tf.stack(encoder_output.outputs,
                                   axis=1, name='rnn_outputs')  # (batch, path_max_len + 1, dim_sk_hidden)

            # Since we are in the BiRNN mode, we are simply taking average.

            sum_sk_hidden = tf.reduce_sum(rnn_outputs, axis=1,
                                          name='sum_sk_hidden')  # (batch, dim_sk_hidden)
            use_path_len_mat = tf.cast(
                tf.expand_dims(use_path_len, axis=1),
                dtype=tf.float32,
                name='use_path_len_mat'
            )  # (batch, 1) as float32
            sk_hidden = tf.div(sum_sk_hidden, use_path_len_mat, name='sk_hidden')  # (batch, dim_sk_hidden)

        LogInfo.end_track()
        return sk_hidden
