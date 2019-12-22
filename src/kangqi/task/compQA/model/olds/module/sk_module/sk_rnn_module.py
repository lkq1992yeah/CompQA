# -*- coding:utf-8 -*-

# Goal: schema --> list
import tensorflow as tf
from xusheng.model.rnn_encoder import UnidirectionalRNNEncoder

from .sk_base_module import SkBaseModule
from ...u import show_tensor

from kangqi.util.LogUtil import LogInfo


class SkRNNModule(SkBaseModule):

    def __init__(self, path_max_len, dim_item_hidden, dim_kb_emb, dim_sk_hidden,
                 data_source, rnn_config):
        super(SkRNNModule, self).__init__(path_max_len=path_max_len,
                                          dim_item_hidden=dim_item_hidden,
                                          dim_kb_emb=dim_kb_emb,
                                          dim_sk_hidden=dim_sk_hidden)
        self.data_source = data_source
        assert self.data_source in ('kb', 'word', 'both')

        rnn_config['num_units'] = dim_sk_hidden
        self.rnn_encoder = UnidirectionalRNNEncoder(rnn_config, mode=tf.contrib.learn.ModeKeys.TRAIN)

    # Input:
    #   path_wd_hidden: (batch, path_max_len, dim_item_hidden)
    #   path_kb_hidden: (batch, path_max_len, dim_kb_emb)
    #   path_len: (batch, ) as int32
    #   focus_wd_hidden: (batch, dim_item_hidden)
    #   focus_kb_hidden: (batch, dim_kb_emb)
    # Output:
    #   sk_hidden: (batch, dim_sk_hidden)
    def forward(self, path_wd_hidden, path_kb_hidden, path_len, focus_wd_hidden, focus_kb_hidden, reuse=None):
        LogInfo.begin_track('SkRNNModule forward: ')

        with tf.variable_scope('SkRNNModule', reuse=reuse):
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

            # Strategy 1: focus as initial state, but the RNN hidden dim must be equal to dim_kb_emb
            # # dim_use: the last dimension used in use_path_hidden and use_focus_hidden
            # initial_state = use_focus_hidden        # (batch, dim_use)
            # path_emb_input = use_path_hidden        # (batch, path_max_len, dim_use)
            # stamps = self.path_max_len
            # rnn_inputs = tf.unstack(path_emb_input, num=stamps, axis=1, name='rnn_inputs')
            # encoder_output = self.rnn_encoder.encode(inputs=rnn_inputs,
            #                                          sequence_length=path_len,
            #                                          initial_state=initial_state,
            #                                          reuse=reuse)
            # rnn_outputs = tf.stack(
            #     encoder_output.outputs,
            #     axis=1,
            #     name='rnn_outputs'
            # )   # (batch, path_max_len, dim_sk_hidden)
            # LogInfo.logs('outputs = %s', rnn_outputs.get_shape().as_list())
            #
            # # Now pick the representative hidden vector for each skeleton
            # # Determined by the length of each one
            # row_tf = tf.cast(tf.range(tf.shape(path_emb_input)[0]), tf.int32, name='row_tf')
            # col_tf = tf.maximum(path_len - 1, 0, name='col_tf')

            # Strategy 2: default initial state, focus+path as RNN input.
            use_path_emb_input = tf.concat([tf.expand_dims(use_focus_hidden,axis=1), use_path_hidden],
                                            axis=1,
                                            name='use_path_emb_input')       # (batch, path_max_len + 1, dim_use)
            show_tensor(use_path_emb_input)
            use_path_len = path_len + 1
            stamps = self.path_max_len + 1
            rnn_inputs = tf.unstack(use_path_emb_input, num=stamps, axis=1, name='rnn_inputs')
            encoder_output = self.rnn_encoder.encode(inputs=rnn_inputs,
                                                     sequence_length=use_path_len,
                                                     reuse=reuse)
            rnn_outputs = tf.stack(encoder_output.outputs,
                                   axis=1, name='rnn_outputs')  # (batch, path_max_len + 1, dim_sk_hidden)
            row_tf = tf.cast(tf.range(tf.shape(use_path_emb_input)[0]), tf.int32, name='row_tf')
            col_tf = path_len           # use_path_len - 1

            # choose the position of the output vector representation of a path (starting from 0)
            # For padding paths (path_len = 0), we make sure that col_tf won't become -1.
            # Since encoder_output takes path_len_mat into consideration,
            # the padding path will always return zero vector as its vector representation.
            coo_tf = tf.stack(values=[row_tf, col_tf], axis=-1, name='coo_tf')
            sk_hidden = tf.gather_nd(
                params=rnn_outputs,
                indices=coo_tf,
                name='sk_hidden'
            )  # (batch, dim_sk_hidden)

        LogInfo.end_track()
        return sk_hidden
