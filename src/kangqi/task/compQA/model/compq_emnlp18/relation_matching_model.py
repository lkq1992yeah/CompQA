"""
Author: Kangqi Luo
Date: 180323
Goal: Relation matching using traditional attention
"""

import tensorflow as tf

from ..module.seq_helper import seq_encoding_with_aggregation, seq_encoding, seq_hidden_max_pooling
from ..module.simple_attention import SimpleAttention

from xusheng.model.rnn_encoder import BidirectionalRNNEncoder
from kangqi.util.tf.cosine_sim import cosine_sim

from kangqi.util.LogUtil import LogInfo


class RelationMatchingKernel:

    def __init__(self, dim_emb, qw_max_len,
                 seq_merge_mode, scoring_mode,
                 rnn_config=None, att_config=None):
        self.dim_emb = dim_emb
        self.qw_max_len = qw_max_len
        self.rnn_config = rnn_config
        self.scoring_mode = scoring_mode
        self.seq_merge_mode = seq_merge_mode
        self.att_config = att_config

        if self.rnn_config is not None:
            self.rnn_config['reuse'] = tf.AUTO_REUSE
            self.dim_hidden = 2 * rnn_config['num_units']
        else:           # no RNN, just directly using embedding
            self.dim_hidden = dim_emb

    def forward(self, path_size, qw_emb, qw_len, pw_emb, pw_len, mode):
        """
        :param path_size: (ds, )
        :param qw_emb:  (ds, path_max_size, qw_max_len, dim_qw_emb)
        :param qw_len:  (ds, path_max_size)
        :param pw_emb:  (ds, path_max_size, pw_max_len, dim_pw_emb)
        :param pw_len:  (ds, path_max_size)
        :param mode:    tf.contrib.learn.ModeKeys. TRAIN / INFER
        """
        rm_ret_dict = {}    # <tensor_name, tensor>
        LogInfo.begin_track('Build kernel: [rm_kernel]')
        assert mode in (tf.contrib.learn.ModeKeys.INFER, tf.contrib.learn.ModeKeys.TRAIN)

        dyn_path_max_size = tf.shape(qw_emb)[1]
        rnn_encoder = None
        if self.rnn_config is not None:
            encoder_args = {'config': self.rnn_config, 'mode': mode}
            rnn_encoder = BidirectionalRNNEncoder(**encoder_args)

        """ Merge first & second dimension: ds * path_max_size = DS """
        comb_tensor_list = []
        for tensor_input in (qw_emb, qw_len, pw_emb, pw_len):
            ori_shape = tensor_input.get_shape().as_list()
            comb_shape = [-1] + ori_shape[2:]  # keep the dimensions after (ds, path_max_size)
            comb_tensor_list.append(tf.reshape(tensor_input, shape=comb_shape))
        qw_emb, qw_len, pw_emb, pw_len = comb_tensor_list

        """ pw side representation """
        pw_repr = seq_encoding_with_aggregation(emb_input=pw_emb, len_input=pw_len,
                                                rnn_encoder=rnn_encoder,
                                                seq_merge_mode=self.seq_merge_mode)
        # (DS, dim_hidden), that is (ds * path_max_size, dim_hidden)

        """ attention with qw repr """
        if self.att_config is not None:
            dim_att_len = self.att_config['dim_att_hidden']
            att_func = self.att_config['att_func']
            qw_hidden = seq_encoding(emb_input=qw_emb, len_input=qw_len, encoder=rnn_encoder)
            # (DS, qw_max_len, dim_hidden)
            qw_mask = tf.sequence_mask(lengths=qw_len,
                                       maxlen=self.qw_max_len,
                                       dtype=tf.float32,
                                       name='qw_mask')  # (DS, qw_max_len)
            simple_att = SimpleAttention(lf_max_len=self.qw_max_len,
                                         dim_att_hidden=dim_att_len,
                                         att_func=att_func)
            q_att_rep, att_mat, q_weight = simple_att.forward(lf_input=qw_hidden,
                                                              lf_mask=qw_mask,
                                                              fix_rt_input=pw_repr)
            # q_att_rep: (DS, dim_hidden)
            # att_mat:   (DS, qw_max_len)
            # q_weight:  (DS, qw_max_len)
            att_mat = tf.reshape(att_mat, shape=[-1, dyn_path_max_size, self.qw_max_len],
                                 name='att_mat')  # (ds, path_max_size, qw_max_len)
            q_weight = tf.reshape(q_weight, shape=[-1, dyn_path_max_size, self.qw_max_len],
                                  name='q_weight')  # (ds, path_max_size, qw_max_len)
            rm_ret_dict['rm_att_mat'] = att_mat
            rm_ret_dict['rm_q_weight'] = q_weight
            qw_repr = q_att_rep
        else:       # no attention, similar with above
            qw_repr = seq_encoding_with_aggregation(emb_input=qw_emb, len_input=qw_len,
                                                    rnn_encoder=rnn_encoder,
                                                    seq_merge_mode=self.seq_merge_mode)

        """ Calculating final score """
        final_ret_dict = self.final_merge(
            qw_repr=qw_repr, pw_repr=pw_repr, path_size=path_size,
            dyn_path_max_size=dyn_path_max_size,
            dim_hidden=self.dim_hidden,
            scoring_mode=self.scoring_mode
        )
        rm_ret_dict.update(final_ret_dict)

        LogInfo.end_track()
        return rm_ret_dict

    @staticmethod
    def final_merge(qw_repr, pw_repr, path_size, dyn_path_max_size, dim_hidden, scoring_mode):
        """
        Copied from compq_acl18.BaseRelationMatchingKernel
        :param qw_repr:             (ds * path_max_len, dim_hidden)
        :param pw_repr:             (ds * path_max_len, dim_hidden)
        :param path_size:           (ds, )
        :param dyn_path_max_size:   A tensor representing the max path_len in this batch
        :param dim_hidden:      dim_hidden
        :param scoring_mode:    compact / separated
        """
        LogInfo.logs('scoring_mode = [%s]', scoring_mode)
        if scoring_mode == 'compact':
            # aggregate by max-pooling, then overall cosine
            qw_repr = tf.reshape(qw_repr, shape=[-1, dyn_path_max_size, dim_hidden],
                                 name='qw_repr')  # (ds, path_max_size, dim_hidden)
            pw_repr = tf.reshape(pw_repr, shape=[-1, dyn_path_max_size, dim_hidden],
                                 name='pw_repr')
            q_final_repr = seq_hidden_max_pooling(seq_hidden_input=qw_repr, len_input=path_size)
            p_final_repr = seq_hidden_max_pooling(seq_hidden_input=pw_repr, len_input=path_size)
            # (ds, dim_hidden)
            score = cosine_sim(lf_input=q_final_repr, rt_input=p_final_repr)  # (ds, )
            return {'rm_score': score}
        else:
            # separately calculate cosine, then sum together (with threshold control)
            raw_score = cosine_sim(lf_input=qw_repr, rt_input=pw_repr)  # (ds * path_max_len, )
            raw_score = tf.reshape(raw_score, shape=[-1, dyn_path_max_size], name='raw_score')  # (ds, path_max_len)
            sim_ths = tf.get_variable(name='sim_ths', dtype=tf.float32, shape=[])
            path_score = tf.subtract(raw_score, sim_ths, name='path_score')  # add penalty to each potential seq.
            sc_mask = tf.sequence_mask(lengths=path_size,
                                       maxlen=dyn_path_max_size,
                                       dtype=tf.float32,
                                       name='sc_mask')  # (ds, sc_max_len) as mask
            score = tf.reduce_sum(path_score * sc_mask, axis=-1, name='score')  # (ds, )
            return {'rm_score': score, 'rm_path_score': path_score}
