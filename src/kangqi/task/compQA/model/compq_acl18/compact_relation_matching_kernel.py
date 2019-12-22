"""
Author: Kangqi Luo
Date: 180211
Copied many codes from compq_overall/compact_relation_matching_kernel.py
"""

import tensorflow as tf

from .base_relation_matching_kernel import BaseRelationMatchingKernel

from ..module.seq_helper import seq_encoding
from ..module.cross_attention_direct import DirectCrossAttention
# from ..u import show_tensor

from xusheng.model.rnn_encoder import BidirectionalRNNEncoder

from kangqi.util.LogUtil import LogInfo


class CompactRelationMatchingKernel(BaseRelationMatchingKernel):

    def __init__(self, qw_max_len, sc_max_len, p_max_len, pw_max_len, dim_emb,
                 path_usage, dim_att_hidden, att_func,
                 att_merge_mode, seq_merge_mode, scoring_mode,
                 rnn_config=None, cnn_config=None, residual=False):

        assert att_func != 'None'

        BaseRelationMatchingKernel.__init__(
            self, qw_max_len, sc_max_len, p_max_len, pw_max_len, dim_emb,
            path_usage, dim_att_hidden, att_func, att_merge_mode, seq_merge_mode,
            scoring_mode, rnn_config, cnn_config, residual
        )

    """ The ABCNN2 """

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
        LogInfo.begin_track('Build kernel: [abcnn2_rm_kernel]')
        assert mode in (tf.contrib.learn.ModeKeys.INFER, tf.contrib.learn.ModeKeys.TRAIN)
        LogInfo.logs('repr_mode = %s, scoring_mode = %s', self.repr_mode, self.scoring_mode)

        comb_tensor_list = []
        for tensor_input in (p_emb, pw_emb, p_len, pw_len):
            ori_shape = tensor_input.get_shape().as_list()
            comb_shape = [-1] + ori_shape[2:]  # keep the dimensions after (ds, sc_max_len)
            comb_tensor_list.append(tf.reshape(tensor_input, shape=comb_shape))
        p_emb, pw_emb, p_len, pw_len = comb_tensor_list
        # p/pw_emb: (ds * sc_max_len, x_max_len, dim_x_emb)
        # p/pw_len: (ds * sc_max_len,)

        with tf.variable_scope('abcnn2_rm_kernel', reuse=tf.AUTO_REUSE):
            with tf.variable_scope('qw_repr', reuse=tf.AUTO_REUSE):
                qw_hidden = self.apply_seq_repr(input_emb=qw_emb, input_len=qw_len, mode=mode)
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
                pw_hidden = self.apply_seq_repr(input_emb=pw_emb, input_len=pw_len, mode=mode)
                # (ds * sc_max_len, pw_max_len, dim_hidden)
                pw_mask = tf.sequence_mask(lengths=pw_len,
                                           maxlen=self.pw_max_len,
                                           dtype=tf.float32,
                                           name='pw_mask')
                # (ds * sc_max_len, pw_max_len)

            with tf.variable_scope('p_repr', reuse=tf.AUTO_REUSE):
                # p_hidden = self.apply_seq_repr(input_emb=p_emb, input_len=p_len, mode=mode)
                """ Always just use predicate raw embedding as the hidden state """
                # TODO: assert dim_emb == dim_hidden
                p_hidden = p_emb
                # (ds * sc_max_len, p_max_len, dim_emb == dim_hidden)
                p_mask = tf.sequence_mask(lengths=p_len,
                                          maxlen=self.p_max_len,
                                          dtype=tf.float32,
                                          name='p_mask')
                # (ds * sc_max_len, p_max_len)

            """ pw & p join together """
            LogInfo.logs('Path Usage: %s', self.path_usage)
            if self.path_usage == 'pwp':
                path_hidden = tf.concat([pw_hidden, p_hidden], axis=1, name='path_hidden')
                path_mask = tf.concat([pw_mask, p_mask], axis=1, name='path_mask')
                merge_max_len = self.pw_max_len + self.p_max_len
            else:           # pwOnly
                path_hidden = pw_hidden
                path_mask = pw_mask
                merge_max_len = self.pw_max_len

            """ cross attention & calculate final score """
            cross_att = DirectCrossAttention(lf_max_len=self.qw_max_len,
                                             rt_max_len=merge_max_len,
                                             dim_att_hidden=self.dim_att_hidden,
                                             att_func=self.att_func)
            q_att_rep, path_att_rep, att_mat, q_weight, path_weight = cross_att.forward(
                lf_input=qw_hidden, lf_mask=qw_mask,
                rt_input=path_hidden, rt_mask=path_mask
            )
            # q_att_rep / path_att_rep: (ds * sc_max_len, dim_hidden)
            # att_mat: (ds * sc_max_len, qw_max_len, merge_max_len)
            # q_weight: (ds * sc_max_len, qw_max_len)
            # path_weight: (ds * sc_max_len, merge_max_len)
            att_mat = tf.reshape(att_mat, shape=[-1, self.sc_max_len, self.qw_max_len, merge_max_len],
                                 name='att_mat')  # (ds, sc_max_len, qw_max_len, merge_max_len)
            q_weight = tf.reshape(q_weight, shape=[-1, self.sc_max_len, self.qw_max_len],
                                  name='q_weight')  # (ds, sc_max_len, qw_max_len)
            path_weight = tf.reshape(path_weight, shape=[-1, self.sc_max_len, merge_max_len],
                                     name='path_weight')  # (ds, sc_max_len, merge_max_len)

            final_ret_dict = self.final_merge(
                q_rep=q_att_rep, path_rep=path_att_rep,
                sc_len=sc_len, sc_max_len=self.sc_max_len,
                dim_hidden=self.dim_hidden, scoring_mode=self.scoring_mode
            )
            final_ret_dict['rm_att_mat'] = att_mat
            final_ret_dict['rm_q_weight'] = q_weight
            final_ret_dict['rm_path_weight'] = path_weight

        LogInfo.end_track()
        return final_ret_dict       # rm_score, rm_path_score (optional), rm_att_mat, rm_q_weight, rm_path_weight

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
