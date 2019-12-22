"""
Author: Kangqi Luo
Goal: Multi-task model: segment part
"""

import tensorflow as tf

from ..module.seq_helper import seq_encoding
from xusheng.model.rnn_encoder import BidirectionalRNNEncoder

from kangqi.util.LogUtil import LogInfo


class SegmentKernel:

    def __init__(self, rnn_config, q_max_len, num_classes):
        self.rnn_config = rnn_config
        self.rnn_config['reuse'] = tf.AUTO_REUSE
        rnn_units = rnn_config['num_units']

        self.dim_seg_hidden = 2 * rnn_units
        self.q_max_len = q_max_len
        self.num_classes = num_classes

    def forward(self, v_emb, v_len, tag_indices, mode):
        """
        :param v_emb:          (ds, q_max_len, dim_emb)
        :param v_len:          (ds,) as int
        :param tag_indices:    (ds, q_max_len) as int
        :param mode: TRAIN / INFER
        """
        LogInfo.begin_track('Build kernel: [segment_kernel]')
        assert mode in (tf.contrib.learn.ModeKeys.INFER, tf.contrib.learn.ModeKeys.TRAIN)
        encoder_args = {'config': self.rnn_config, 'mode': mode}
        seg_encoder = BidirectionalRNNEncoder(**encoder_args)

        with tf.variable_scope('segment_kernel', reuse=tf.AUTO_REUSE):
            transition = tf.get_variable(
                name='transition', dtype=tf.float32,
                shape=[self.num_classes, self.num_classes]
            )  # (num_classes, num_classes) as transition matrix
            v_hidden = seq_encoding(
                emb_input=v_emb,
                len_input=v_len,
                encoder=seg_encoder
            )  # (ds, q_max_len, dim_seg_hidden)
            v_hidden_flat = tf.reshape(v_hidden, [-1, self.dim_seg_hidden])     # (ds * q_max_len, dim_seg_hidden)
            seg_logits = tf.reshape(
                tf.contrib.layers.fully_connected(
                    inputs=v_hidden_flat,
                    num_outputs=self.num_classes,
                    activation_fn=None,
                    scope='fc'),
                shape=[-1, self.q_max_len, self.num_classes],
                name='seg_logits'
            )   # (ds, q_max_len, num_classes)
            log_lik, _ = tf.contrib.crf.crf_log_likelihood(
                inputs=seg_logits,
                tag_indices=tag_indices,
                sequence_lengths=v_len,
                transition_params=transition
            )
            best_seg, viterbi_score = tf.contrib.crf.crf_decode(
                potentials=seg_logits,
                transition_params=transition,
                sequence_length=v_len
            )
            # output_seq: (ds, q_max_len) as int
        LogInfo.end_track()

        return v_hidden, seg_logits, log_lik, best_seg
