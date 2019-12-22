"""
Author: Kangqi Luo
Date: 180416
Goal: Implementing entity linking kernel in a MemNN style.
"""

import tensorflow as tf

from ..module.seq_helper import seq_encoding, seq_encoding_with_aggregation

from xusheng.model.rnn_encoder import BidirectionalRNNEncoder
from kangqi.util.tf.cosine_sim import cosine_sim
from kangqi.util.LogUtil import LogInfo


class MemEntityLinkingKernel:

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
        else:  # no RNN, just directly using embedding
            self.dim_hidden = dim_emb

    def forward(self, el_size, qw_emb, qw_len,
                pw_sup_emb, pw_sup_len, type_trans,
                el_sup_mask, el_type_signa, el_indv_feats, el_comb_feats, mode):
        """
        Note: number of paths in a schema == number of entities in the schema
        local_mem_size: the local number of relevant paths in the current batch.
        :param el_size:         (ds, )
        :param qw_emb:          (ds, el_max_size, qw_max_len, dim_emb)
        :param qw_len:          (ds, el_max_size)
        :param pw_sup_emb:      (local_mem_size, pw_max_len, dim_emb)
        :param pw_sup_len:      (local_mem_size,)
        :param type_trans:      (local_mem_size, dim_type)
        :param el_sup_mask:     (ds, el_max_size, local_mem_size)
        :param el_type_signa:   (ds, el_max_size, dim_type)
        :param el_indv_feats:   (ds, el_max_size, el_feat_size)
        :param el_comb_feats:   (ds, 1)
        :param mode:    TRAIN / INFER
        """
        """
        180416:
        Let's assume ds=16*2=32, el_max_size=3, qw_max_len=20, dim_emb=300, local_mem_size=6K
        Then ds*el_max_size*qw_max_len ~= 2K
        """
        LogInfo.begin_track('Build kernel: [el_kernel]')
        assert mode in (tf.contrib.learn.ModeKeys.INFER, tf.contrib.learn.ModeKeys.TRAIN)

        rnn_encoder = None
        if self.rnn_config is not None:
            encoder_args = {'config': self.rnn_config, 'mode': mode}
            rnn_encoder = BidirectionalRNNEncoder(**encoder_args)
        raw_shape = tf.shape(el_sup_mask)
        el_max_size = raw_shape[1]
        local_mem_size = raw_shape[2]
        dim_type = tf.shape(type_trans)[1]

        """ Possible reshapes """
        qw_emb = tf.reshape(qw_emb, [-1, self.qw_max_len, self.dim_emb])
        # (ds * el_max_size, qw_max_len, dim_emb)
        qw_len = tf.reshape(qw_len, [-1])       # (ds * el_max_size)

        """ Calculate attention / non-attention question representation """
        pw_sup_repr = seq_encoding_with_aggregation(emb_input=pw_sup_emb, len_input=pw_sup_len,
                                                    rnn_encoder=rnn_encoder,
                                                    seq_merge_mode=self.seq_merge_mode)
        # (local_mem_size, dim_hidden)
        if self.att_config is not None:
            att_func = self.att_config['att_func']
            assert att_func == 'dot'        # TODO: Currently only support dot product
            qw_hidden = seq_encoding(emb_input=qw_emb, len_input=qw_len, encoder=rnn_encoder)
            # (ds*el_max_size, qw_max_len, dim_hidden)
            qw_mask = tf.sequence_mask(lengths=qw_len,
                                       maxlen=self.qw_max_len,
                                       dtype=tf.float32,
                                       name='qw_mask')  # (ds*el_max_size, qw_max_len)
            flat_qw_hidden = tf.reshape(qw_hidden, shape=[-1, self.dim_hidden], name='flat_qw_hidden')
            # (ds*el_max_size*qw_max_len, dim_hidden)

            """ Step 1: Very simple & fast way to calculate dot attention """
            raw_mutual_att_mat = tf.matmul(
                flat_qw_hidden,
                tf.transpose(pw_sup_repr),
                name='raw_mutual_att_mat'
            )   # (ds*el_max_size*qw_max_len, local_mem_size)
            mutual_att_mat = tf.reshape(
                raw_mutual_att_mat,
                shape=[-1, self.qw_max_len, local_mem_size],
                name='mutual_att_mat')
            # (ds*el_max_size, qw_max_len, local_mem_size)

            """ Step 2: Prepare masked att_mat and normalized distribution """
            qw_mask_3dim = tf.expand_dims(qw_mask, axis=-1, name='qw_mask_3dim')
            # (ds*el_max_size, qw_max_len, 1)
            masked_att_mat = (
                qw_mask_3dim * mutual_att_mat +
                (1. - qw_mask_3dim) * mutual_att_mat * tf.float32.min
            )   # (ds*el_max_size, qw_max_len, local_mem_size)
            unnorm_weight = tf.transpose(masked_att_mat, [0, 2, 1], name='masked_att_mat')
            # (ds*el_max_size, local_mem_size, qw_max_len)
            norm_weight = tf.nn.softmax(unnorm_weight, name='norm_weight')

            """ Step 3: Got final qw_repr w.r.t different support paths """
            qw_repr = tf.matmul(norm_weight, qw_hidden, name='qw_repr')
            # batch_matmul: (ds*el_max_size, local_mem_size, qw_max_len)

        else:       # noAtt, very simple
            raw_qw_repr = seq_encoding_with_aggregation(emb_input=qw_emb, len_input=qw_len,
                                                        rnn_encoder=rnn_encoder,
                                                        seq_merge_mode=self.seq_merge_mode)
            # (ds*el_max_size, dim_hidden)
            qw_repr = tf.expand_dims(raw_qw_repr, axis=1, name='qw_repr')
            # (ds*el_max_size, 1, dim_hidden)

        with tf.variable_scope('el_kernel', reuse=tf.AUTO_REUSE):
            """ Calculate cosine similarity """
            flat_pw_sup_repr = tf.expand_dims(pw_sup_repr, axis=0, name='flat_pw_sup_repr')
            # (1, local_mem_size, dim_hidden)
            sim_score = cosine_sim(
                lf_input=qw_repr,               # (ds*el_max_size, [1 or local_mem_size], qw_max_len)
                rt_input=flat_pw_sup_repr       # (1, local_mem_size, dim_hidden)
            )
            # (ds*el_max_size, local_mem_size)

            """ Turning into type distribution """
            flat_el_sup_mask = tf.reshape(el_sup_mask, shape=[-1, local_mem_size], name='flat_el_sup_mask')
            # (ds*el_max_size, local_mem_size)
            mask_score = flat_el_sup_mask * sim_score + (1. - flat_el_sup_mask) * tf.float32.min
            pred_prob = tf.nn.softmax(logits=mask_score, name='pred_prob')
            # (ds*el_max_size, local_mem_size)
            raw_type_prob = tf.matmul(pred_prob, type_trans, name='raw_type_prob')
            # (ds*el_max_size, dim_type)
            type_prob = tf.reshape(raw_type_prob, shape=[-1, el_max_size, dim_type], name='type_prob')
            # (ds, el_max_size, dim_type)
            type_match_score = tf.reduce_sum(el_type_signa*type_prob,
                                             axis=-1, keep_dims=True,
                                             name='type_match_score')   # (ds, el_max_size, 1)

            """ Feature concat and produce scores """
            el_indv_concat = tf.concat([type_match_score, el_indv_feats],
                                       axis=-1, name='el_indv_concat')  # (ds, el_max_size, 1+el_feat_size)
            el_mask = tf.sequence_mask(lengths=el_size, maxlen=el_max_size,
                                       dtype=tf.float32, name='el_mask')    # (ds, el_max_size)
            sum_indv_feats = tf.reduce_sum(
                el_indv_concat * tf.expand_dims(el_mask, axis=-1),
                axis=1, name='sum_indv_feats'
            )   # (ds, 1+el_feat_size)
            final_feats = tf.concat([sum_indv_feats, el_comb_feats], axis=-1, name='final_feats')
            # (ds, 1+el_max_size+1) --> type_match + indv_feats + comb_feat
            el_score = tf.contrib.layers.fully_connected(
                inputs=final_feats,
                num_outputs=1,
                activation_fn=None,
                scope='out_fc',
                reuse=tf.AUTO_REUSE
            )  # (ds, 1), representing type matching score

        LogInfo.end_track()
        return el_score, final_feats
