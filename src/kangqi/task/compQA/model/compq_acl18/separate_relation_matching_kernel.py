"""
Author: Kangqi Luo
Date: 180211
Goal: Following Xusheng's idea, try another framework of Relation Matching
"""

import tensorflow as tf

from .base_relation_matching_kernel import BaseRelationMatchingKernel

from ..module.cross_attention_indirect import IndirectCrossAttention
from ..module.seq_helper import seq_encoding, seq_hidden_max_pooling, seq_hidden_averaging

from xusheng.model.rnn_encoder import BidirectionalRNNEncoder
from kangqi.util.LogUtil import LogInfo


class SeparatedRelationMatchingKernel(BaseRelationMatchingKernel):

    def __init__(self, qw_max_len, sc_max_len, p_max_len, pw_max_len, dim_emb,
                 path_usage, dim_att_hidden, att_func,
                 att_merge_mode, seq_merge_mode, scoring_mode,
                 rnn_config, cnn_config, residual=False):

        assert path_usage == 'pwOnly'
        assert att_merge_mode in ('sum', 'concat')
        assert seq_merge_mode in ('fwbw', 'avg', 'max')

        BaseRelationMatchingKernel.__init__(
            self, qw_max_len, sc_max_len, p_max_len, pw_max_len, dim_emb,
            path_usage, dim_att_hidden, att_func, att_merge_mode, seq_merge_mode,
            scoring_mode, rnn_config, cnn_config, residual
        )

    """ The ABCNN1 """

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
        LogInfo.begin_track('Build kernel: [abcnn1_rm_kernel]')
        LogInfo.logs('att_merge_mode = %s, seq_merge_mode = %s, scoring_mode = %s',
                     self.att_merge_mode, self.seq_merge_mode, self.scoring_mode)

        with tf.variable_scope('abcnn1_rm_kernel', reuse=tf.AUTO_REUSE):
            """ Preprocess: reshaping, merge ds and sc_max_len into one dimension """
            qw_emb = tf.reshape(
                tf.stack([qw_emb] * self.sc_max_len, axis=1),
                shape=(-1, self.qw_max_len, self.dim_emb),
                name='qw_emb'
            )       # (ds * sc_max_len, qw_max_len, dim_emb)
            qw_len = tf.reshape(
                tf.stack([qw_len] * self.sc_max_len, axis=1),
                shape=(-1,),
                name='qw_len'
            )       # (ds * sc_max_len, )
            comb_tensor_list = []
            for tensor_input in (p_emb, pw_emb, p_len, pw_len):
                ori_shape = tensor_input.get_shape().as_list()
                comb_shape = [-1] + ori_shape[2:]  # keep the dimensions after (ds, sc_max_len)
                comb_tensor_list.append(tf.reshape(tensor_input, shape=comb_shape))
            p_emb, pw_emb, p_len, pw_len = comb_tensor_list
            # p/pw_emb: (ds * sc_max_len, x_max_len, dim_x_emb)
            # p/pw_len: (ds * sc_max_len,)

            """ Cross attention & calculate attention repr """
            qw_mask = tf.sequence_mask(lengths=qw_len,
                                       maxlen=self.qw_max_len,
                                       dtype=tf.float32,
                                       name='qw_mask')  # (ds * sc_max_len, qw_max_len)
            pw_mask = tf.sequence_mask(lengths=pw_len,
                                       maxlen=self.pw_max_len,
                                       dtype=tf.float32,
                                       name='pw_mask')  # (ds * sc_max_len, pw_max_len)
            cross_att = IndirectCrossAttention(lf_max_len=self.qw_max_len,
                                               rt_max_len=self.pw_max_len,
                                               dim_att_hidden=self.dim_att_hidden,
                                               att_func=self.att_func)
            qw_att_repr, pw_att_repr, att_mat = cross_att.forward(
                lf_input=qw_emb, lf_mask=qw_mask,
                rt_input=pw_emb, rt_mask=pw_mask
            )
            if self.att_merge_mode == 'sum':
                qw_att_merge = tf.add(qw_emb, qw_att_repr, name='qw_att_merge')
                pw_att_merge = tf.add(pw_emb, pw_att_repr, name='pw_att_merge')
                # (ds * sc_max_len, x_max_len, dim_emb)
            else:  # concat
                qw_att_merge = tf.concat([qw_emb, qw_att_repr], axis=-1, name='qw_att_merge')
                pw_att_merge = tf.concat([pw_emb, pw_att_repr], axis=-1, name='pw_att_merge')
                # (ds * sc_max_len, x_max_len, 2*dim_emb)

            """ RNN as following steps """
            """ Want to share RNN parameters? Put'em into one var_scope """
            with tf.variable_scope('qw_repr', reuse=tf.AUTO_REUSE):
                qw_hidden = self.get_seq_hidden(seq_emb=qw_att_merge, seq_len=qw_len, mode=mode)
            with tf.variable_scope('pw_repr', reuse=tf.AUTO_REUSE):
                pw_hidden = self.get_seq_hidden(seq_emb=pw_att_merge, seq_len=pw_len, mode=mode)
            # (ds * sc_max_len, dim_hidden)

            """ Final: separate cosine and return score """
            att_mat = tf.reshape(att_mat, shape=[-1, self.sc_max_len, self.qw_max_len, self.pw_max_len],
                                 name='att_mat')  # (ds, sc_max_len, qw_max_len, pw_max_len)
            final_ret_dict = self.final_merge(
                q_rep=qw_hidden, path_rep=pw_hidden,
                sc_len=sc_len, sc_max_len=self.sc_max_len,
                dim_hidden=self.dim_hidden, scoring_mode=self.scoring_mode
            )
            final_ret_dict['rm_att_mat'] = att_mat

        LogInfo.end_track()
        return final_ret_dict       # score, path_score (optional), att_mat

    def get_seq_hidden(self, seq_emb, seq_len, mode):
        encoder_args = {'config': self.rnn_config, 'mode': mode}
        rnn_encoder = BidirectionalRNNEncoder(**encoder_args)
        if self.seq_merge_mode == 'fwbw':
            return seq_encoding(emb_input=seq_emb, len_input=seq_len,
                                encoder=rnn_encoder, fwbw=True)
        else:
            seq_hidden = seq_encoding(emb_input=seq_emb, len_input=seq_len, encoder=rnn_encoder)
            if self.seq_merge_mode == 'max':
                return seq_hidden_max_pooling(seq_hidden_input=seq_hidden, len_input=seq_len)
            else:       # avg
                return seq_hidden_averaging(seq_hidden_input=seq_hidden, len_input=seq_len)
        # all conditions: (ds, dim_hidden)
