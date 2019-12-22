"""
Author: Kangqi Luo
Date: 180315
Goal: Relation matching without using attention
"""

import tensorflow as tf

from .base_relation_matching_kernel import BaseRelationMatchingKernel
from ..module.seq_helper import seq_encoding, seq_hidden_max_pooling, seq_hidden_averaging

from xusheng.model.rnn_encoder import BidirectionalRNNEncoder
from kangqi.util.LogUtil import LogInfo


class NoAttRelationMatchingKernel(BaseRelationMatchingKernel):

    def __init__(self, qw_max_len, sc_max_len, p_max_len, pw_max_len, dim_emb,
                 path_usage, dim_att_hidden, att_func,
                 att_merge_mode, seq_merge_mode, scoring_mode,
                 rnn_config=None, cnn_config=None, residual=False):

        assert path_usage == 'pwOnly'
        assert cnn_config is None
        assert seq_merge_mode in ('max', 'avg', 'fwbw')

        BaseRelationMatchingKernel.__init__(
            self, qw_max_len, sc_max_len, p_max_len, pw_max_len, dim_emb,
            path_usage, dim_att_hidden, att_func, att_merge_mode, seq_merge_mode,
            scoring_mode, rnn_config, cnn_config, residual
        )

    """ No attention at all """

    def forward(self, qw_emb, qw_len, sc_len, p_emb, pw_emb, p_len, pw_len, mode):
        """
        :param qw_emb:  (ds, qw_max_len, dim_qw_emb)
        :param qw_len:  (ds, )
        :param sc_len:  (ds, )
        :param p_emb:   (ds, sc_max_len, p_max_len, dim_p_emb)
        :param pw_emb:  (ds, sc_max_len, pw_max_len, dim_pw_emb)
        :param p_len:   (ds, sc_max_len)
        :param pw_len:  (ds, sc_max_len)
        :param mode:    tf.contrib.learn.ModeKeys. TRAIN / INFER
        :return:        (ds, ) as the overall relation matching score
        """
        LogInfo.begin_track('Build kernel: [noatt_rm_kernel]')
        assert mode in (tf.contrib.learn.ModeKeys.INFER, tf.contrib.learn.ModeKeys.TRAIN)
        LogInfo.logs('repr_mode = %s, scoring_mode = %s', self.repr_mode, self.scoring_mode)
        encoder_args = {'config': self.rnn_config, 'mode': mode}
        rnn_encoder = BidirectionalRNNEncoder(**encoder_args)

        comb_tensor_list = []
        for tensor_input in (p_emb, pw_emb, p_len, pw_len):
            ori_shape = tensor_input.get_shape().as_list()
            comb_shape = [-1] + ori_shape[2:]  # keep the dimensions after (ds, sc_max_len)
            comb_tensor_list.append(tf.reshape(tensor_input, shape=comb_shape))
        p_emb, pw_emb, p_len, pw_len = comb_tensor_list
        # p/pw_emb: (ds * sc_max_len, x_max_len, dim_x_emb)
        # p/pw_len: (ds * sc_max_len,)

        with tf.variable_scope('noatt_rm_kernel', reuse=tf.AUTO_REUSE):
            with tf.variable_scope('qw_repr', reuse=tf.AUTO_REUSE):
                if self.seq_merge_mode == 'fwbw':
                    q_rep = seq_encoding(emb_input=qw_emb, len_input=qw_len, encoder=rnn_encoder, fwbw=True)
                    # (ds, dim_hidden)
                else:
                    q_hidden = seq_encoding(emb_input=qw_emb, len_input=qw_len, encoder=rnn_encoder)
                    if self.seq_merge_mode == 'avg':
                        q_rep = seq_hidden_averaging(seq_hidden_input=q_hidden, len_input=qw_len)
                    else:
                        q_rep = seq_hidden_max_pooling(seq_hidden_input=q_hidden, len_input=qw_len)
                    # (ds, dim_hidden)
            q_rep = tf.reshape(
                tf.stack([q_rep] * self.sc_max_len, axis=1),
                shape=[-1, self.dim_hidden],
                name='q_rep'
            )  # (ds * sc_max_len, dim_hidden)

            with tf.variable_scope('pw_repr', reuse=tf.AUTO_REUSE):
                if self.seq_merge_mode == 'fwbw':
                    pw_rep = seq_encoding(emb_input=pw_emb, len_input=pw_len, encoder=rnn_encoder, fwbw=True)
                    # (ds, dim_hidden)
                else:
                    pw_hidden = seq_encoding(emb_input=pw_emb, len_input=pw_len, encoder=rnn_encoder)
                    if self.seq_merge_mode == 'avg':
                        pw_rep = seq_hidden_averaging(seq_hidden_input=pw_hidden, len_input=pw_len)
                    else:
                        pw_rep = seq_hidden_max_pooling(seq_hidden_input=pw_hidden, len_input=pw_len)
                    # (ds * sc_max_len, dim_hidden)

            final_ret_dict = self.final_merge(
                q_rep=q_rep, path_rep=pw_rep, sc_len=sc_len, sc_max_len=self.sc_max_len,
                dim_hidden=self.dim_hidden, scoring_mode=self.scoring_mode
            )

        LogInfo.end_track()
        return final_ret_dict       # rm_score, rm_path_score (optional)
