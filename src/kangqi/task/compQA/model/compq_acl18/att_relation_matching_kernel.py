"""
Author: Kangqi Luo
Date: 180323
Goal: Relation matching using traditional attention
"""

import tensorflow as tf

from .base_relation_matching_kernel import BaseRelationMatchingKernel
from ..module.seq_helper import seq_encoding, seq_hidden_max_pooling, seq_hidden_averaging
from ..module.simple_attention import SimpleAttention

from xusheng.model.rnn_encoder import BidirectionalRNNEncoder
from kangqi.util.LogUtil import LogInfo


class AttRelationMatchingKernel(BaseRelationMatchingKernel):

    def __init__(self, qw_max_len, sc_max_len, p_max_len, pw_max_len, dim_emb,
                 path_usage, dim_att_hidden, att_func,
                 att_merge_mode, seq_merge_mode, scoring_mode,
                 rnn_config=None, cnn_config=None, residual=False):

        assert path_usage == 'pwOnly'
        assert att_func != 'None'
        assert cnn_config is None
        assert seq_merge_mode in ('max', 'avg', 'fwbw', 'nfwbw')

        BaseRelationMatchingKernel.__init__(
            self, qw_max_len, sc_max_len, p_max_len, pw_max_len, dim_emb,
            path_usage, dim_att_hidden, att_func, att_merge_mode, seq_merge_mode,
            scoring_mode, rnn_config, cnn_config, residual
        )

    """ using simple attention: attention over q-words, given the fixed path representation """

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
        LogInfo.begin_track('Build kernel: [att_rm_kernel]')
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

        with tf.variable_scope('att_rm_kernel', reuse=tf.AUTO_REUSE):
            with tf.variable_scope('qw_repr', reuse=tf.AUTO_REUSE):
                qw_hidden = self.apply_seq_repr(input_emb=qw_emb, input_len=qw_len, mode=mode)
                # (ds, qw_max_len, dim_hidden)
                if self.residual:           # RNN hidden + RNN input
                    LogInfo.logs('Applying residual at qw_repr.')
                    assert self.dim_hidden == self.dim_emb
                    qw_hidden = tf.add(qw_hidden, qw_emb, name='qw_hidden_residual')
                    # (ds, qw_max_len, dim_hidden)
                qw_mask = tf.sequence_mask(lengths=qw_len,
                                           maxlen=self.qw_max_len,
                                           dtype=tf.float32,
                                           name='qw_mask')      # (ds, qw_max_len)
            qw_hidden = tf.reshape(
                tf.stack([qw_hidden] * self.sc_max_len, axis=1),
                shape=[-1, self.qw_max_len, self.dim_hidden],
                name='qw_hidden'
            )       # (ds * sc_max_len, qw_max_len, dim_hidden)
            qw_mask = tf.reshape(
                tf.stack([qw_mask] * self.sc_max_len, axis=1),
                shape=[-1, self.qw_max_len], name='qw_mask'
            )       # (ds * sc_max_len, qw_max_len)

            with tf.variable_scope('pw_repr', reuse=tf.AUTO_REUSE):
                if self.seq_merge_mode in ('fwbw', 'nfwbw'):
                    pw_rep = seq_encoding(emb_input=pw_emb, len_input=pw_len, encoder=rnn_encoder, fwbw=True)
                    # (ds * sc_max_len, dim_hidden)
                else:
                    pw_hidden = seq_encoding(emb_input=pw_emb, len_input=pw_len, encoder=rnn_encoder)
                    if self.seq_merge_mode == 'avg':
                        pw_rep = seq_hidden_averaging(seq_hidden_input=pw_hidden, len_input=pw_len)
                    else:
                        pw_rep = seq_hidden_max_pooling(seq_hidden_input=pw_hidden, len_input=pw_len)
                    # (ds * sc_max_len, dim_hidden)

            # Ready for attention calculation
            LogInfo.logs('Sequence merge mode: %s', self.seq_merge_mode)
            if self.seq_merge_mode != 'nfwbw':
                simple_att = SimpleAttention(lf_max_len=self.qw_max_len,
                                             dim_att_hidden=self.dim_att_hidden,
                                             att_func=self.att_func)
                q_att_rep, att_mat, q_weight = simple_att.forward(lf_input=qw_hidden,
                                                                  lf_mask=qw_mask,
                                                                  fix_rt_input=pw_rep)
                # q_att_rep: (ds * sc_max_len, dim_hidden)
                # att_mat:   (ds * sc_max_len, qw_max_len)
                # q_weight:  (ds * sc_max_len, qw_max_len)
                att_mat = tf.reshape(att_mat, shape=[-1, self.sc_max_len, self.qw_max_len],
                                     name='att_mat')        # (ds, sc_max_len, qw_max_len)
                q_weight = tf.reshape(q_weight, shape=[-1, self.sc_max_len, self.qw_max_len],
                                      name='q_weight')      # (ds, sc_max_len, qw_max_len)
                final_ret_dict = self.final_merge(
                    q_rep=q_att_rep, path_rep=pw_rep, sc_len=sc_len, sc_max_len=self.sc_max_len,
                    dim_hidden=self.dim_hidden, scoring_mode=self.scoring_mode
                )
                final_ret_dict['rm_att_mat'] = att_mat
                final_ret_dict['rm_q_weight'] = q_weight
                # rm_score, rm_path_score (optional), rm_att_mat, rm_q_weight
            else:
                """ Working in nfwbw mode, the fw/bw information are separated & calculating attention """
                fw_qw_hidden, bw_qw_hidden = tf.split(qw_hidden, num_or_size_splits=2, axis=-1)
                # both (ds * sc_max_len, qw_max_len, dim_hidden / 2)
                fw_pw_rep, bw_pw_rep = tf.split(pw_rep, num_or_size_splits=2, axis=-1)
                # both (ds * sc_max_len, dim_hidden / 2)
                simple_att = SimpleAttention(lf_max_len=self.qw_max_len,
                                             dim_att_hidden=self.dim_att_hidden,
                                             att_func=self.att_func)
                fw_q_att_rep, fw_att_mat, fw_q_weight = simple_att.forward(lf_input=fw_qw_hidden,
                                                                           lf_mask=qw_mask,
                                                                           fix_rt_input=fw_pw_rep)
                bw_q_att_rep, bw_att_mat, bw_q_weight = simple_att.forward(lf_input=bw_qw_hidden,
                                                                           lf_mask=qw_mask,
                                                                           fix_rt_input=bw_pw_rep)
                # fw/bw_q_att_rep: (ds * sc_max_len, dim_hidden / 2)
                # fw/bw_att_mat:   (ds * sc_max_len, qw_max_len)
                # fw/bw_q_weight:  (ds * sc_max_len, qw_max_len)
                fw_att_mat = tf.reshape(fw_att_mat, shape=[-1, self.sc_max_len, self.qw_max_len],
                                        name='fw_att_mat')  # (ds, sc_max_len, qw_max_len)
                bw_att_mat = tf.reshape(bw_att_mat, shape=[-1, self.sc_max_len, self.qw_max_len],
                                        name='bw_att_mat')  # (ds, sc_max_len, qw_max_len)
                fw_q_weight = tf.reshape(fw_q_weight, shape=[-1, self.sc_max_len, self.qw_max_len],
                                         name='fw_q_weight')  # (ds, sc_max_len, qw_max_len)
                bw_q_weight = tf.reshape(bw_q_weight, shape=[-1, self.sc_max_len, self.qw_max_len],
                                         name='bw_q_weight')  # (ds, sc_max_len, qw_max_len)
                q_att_rep = tf.concat([fw_q_att_rep, bw_q_att_rep], axis=-1, name='q_att_rep')
                # (ds * sc_max_len, dim_hidden)
                final_ret_dict = self.final_merge(
                    q_rep=q_att_rep, path_rep=pw_rep, sc_len=sc_len, sc_max_len=self.sc_max_len,
                    dim_hidden=self.dim_hidden, scoring_mode=self.scoring_mode
                )
                final_ret_dict['rm_fw_att_mat'] = fw_att_mat
                final_ret_dict['rm_bw_att_mat'] = bw_att_mat
                final_ret_dict['rm_fw_q_weight'] = fw_q_weight
                final_ret_dict['rm_bw_q_weight'] = bw_q_weight
                # rm_score, rm_path_score (optional), rm_fw/bw_att_mat, rm_fw/bw_q_weight
        LogInfo.end_track()
        return final_ret_dict

    def apply_seq_repr(self, input_emb, input_len, mode):
        assert self.repr_mode in ('raw', 'cnn', 'rnn')
        LogInfo.logs('apply_seq_repr: %s', self.repr_mode)
        if self.repr_mode == 'raw':
            return input_emb
        elif self.repr_mode == 'cnn':
            return tf.layers.conv1d(inputs=input_emb,
                                    padding='same',
                                    activation=tf.nn.relu,
                                    reuse=tf.AUTO_REUSE,
                                    **self.cnn_config)  # (ds, x_max_len, num_filters == dim_hidden)
        else:
            encoder_args = {'config': self.rnn_config, 'mode': mode}
            rnn_encoder = BidirectionalRNNEncoder(**encoder_args)
            return seq_encoding(emb_input=input_emb, len_input=input_len, encoder=rnn_encoder)
        # (ds, x_max_len, dim_hidden)
