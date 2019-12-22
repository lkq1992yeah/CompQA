"""
Author: Kangqi Luo
Date: 18-01-30
Goal: Following Xusheng's idea, try another framework of Relation Matching
"""

import tensorflow as tf

from ..module.cross_attention_indirect import IndirectCrossAttention
from ..module.seq_helper import seq_encoding, seq_hidden_max_pooling

from xusheng.model.rnn_encoder import BidirectionalRNNEncoder
from kangqi.util.tf.cosine_sim import cosine_sim


class SeparatedRelationMatchingKernel:

    def __init__(self, n_words, n_mids, dim_emb,
                 q_max_len, sc_max_len, path_max_len, pword_max_len,
                 cross_att_config, rnn_config,
                 path_merge_mode, final_score_mode,
                 use_preds=True, use_pwords=True,
                 reuse=tf.AUTO_REUSE, verbose=0):
        self.reuse = reuse
        self.verbose = verbose

        self.q_max_len = q_max_len
        self.sc_max_len = sc_max_len
        self.path_max_len = path_max_len
        self.pword_max_len = pword_max_len
        self.dim_emb = dim_emb

        self.use_preds = use_preds
        self.use_pwords = use_pwords
        self.path_merge_mode = path_merge_mode
        self.final_score_mode = final_score_mode
        assert self.path_merge_mode in ('sum', 'max')
        assert self.final_score_mode in ('cos', 'dot')

        self.rnn_config = rnn_config
        cross_att_config['q_max_len'] = self.q_max_len
        cross_att_config['p_max_len'] = self.path_max_len
        cross_att_config['pw_max_len'] = self.pword_max_len
        cross_att_config['dim_input'] = self.dim_emb
        self.cross_att_config = cross_att_config

        with tf.variable_scope('Embedding_Lookup', reuse=reuse):
            with tf.device('/cpu:0'):
                self.w_embedding_init = tf.placeholder(dtype=tf.float32,
                                                       shape=(n_words, dim_emb),
                                                       name='w_embedding_init')
                self.m_embedding_init = tf.placeholder(dtype=tf.float32,
                                                       shape=(n_mids, dim_emb),
                                                       name='m_embedding_init')
                self.w_embedding = tf.get_variable(name='w_embedding',
                                                   initializer=self.w_embedding_init)
                self.m_embedding = tf.get_variable(name='m_embedding',
                                                   initializer=self.m_embedding_init)

    def get_score(self, mode, qwords_embedding, qwords_len, sc_len,
                  preds_embedding, preds_len, pwords_embedding, pwords_len):
        """
        Produce the final similarity score.
        This function is the most important part in the optm/eval model.
        Just use cosine similarity
        :param mode: tf.contrib.learn.ModeKeys.TRAIN/INFER, which affects the dropout setting
        :param qwords_embedding:    (ds, q_max_len, dim_emb)
        :param qwords_len:          (ds, )
        :param sc_len:              (ds, )
        :param preds_embedding:     (ds, sc_max_len, path_max_len, dim_emb)
        :param preds_len:           (ds, sc_max_len)
        :param pwords_embedding:    (ds, sc_max_len, pword_max_len, dim_emb)
        :param pwords_len:          (ds, sc_max_len)
        :return: score and attention matrices
           pred_att_mat:    (ds, sc_max_len, q_max_len, path_max_len)
           pword_att_mat:   (ds, sc_max_len, q_max_len, pword_max_len)
           score:           (ds,)
        """
        assert mode in (tf.contrib.learn.ModeKeys.TRAIN, tf.contrib.learn.ModeKeys.INFER)
        encoder_args = {'config': self.rnn_config, 'mode': mode}
        # set dropout according to the current mode (TRAIN/INFER)
        q_encoder = BidirectionalRNNEncoder(**encoder_args)
        pred_encoder = BidirectionalRNNEncoder(**encoder_args)
        pword_encoder = BidirectionalRNNEncoder(**encoder_args)
        cross_att = IndirectCrossAttention(**self.cross_att_config)

        with tf.name_scope('separated_relation_matching_kernel'):

            """ Preprocess: reshaping, merge ds and sc_max_len into one dimension """
            qwords_embedding = tf.reshape(
                tf.stack([qwords_embedding] * self.sc_max_len, axis=1),
                shape=(-1, self.q_max_len, self.dim_emb),
                name='qwords_hidden'
            )       # (ds * sc_max_len, q_max_len, dim_hidden)
            qwords_len = tf.reshape(
                tf.stack([qwords_len] * self.sc_max_len, axis=1),
                shape=(-1,),
                name='qwords_len'
            )       # (ds * sc_max_len, )
            comb_tensor_list = []
            for tensor_input in (preds_embedding, preds_len, pwords_embedding, pwords_len):
                ori_shape = tensor_input.get_shape().as_list()
                comb_shape = [-1] + ori_shape[2:]       # keep the dimensions after (ds, sc_max_len)
                # show_tensor(tensor_input)
                # LogInfo.logs('ori_shape: %s, comb_shape: %s', ori_shape, comb_shape)
                comb_tensor_list.append(tf.reshape(tensor_input, shape=comb_shape))
            [preds_embedding, preds_len, pwords_embedding, pwords_len] = comb_tensor_list
            # (ds * sc_max_len, xxxxxxx)
            # for tensor in comb_tensor_list:
            #     show_tensor(tensor)

            """ Step 1: Intra-attention (Optional) """
            # TODO: Question and pred_words

            """ Step 2: Cross-attention, make sure pword and preds treat properly """
            qwords_att_embedding, preds_att_info, pwords_att_info = cross_att.forward(
                q_input=qwords_embedding,
                p_input=preds_embedding,
                pw_input=pwords_embedding,
                q_len=qwords_len, p_len=preds_len, pw_len=pwords_len
            )
            preds_att_embedding, preds_att_mat = preds_att_info
            pwords_att_embedding, pwords_att_mat = pwords_att_info
            # x_embedding: (ds * sc_max_len, x_max_len, dim_emb)
            # x_att_mat: (ds * sc_max_len, q_max_len, x_max_len)

            """ Step 3: Perform RNN over embeddings """
            """ Want to share RNN parameters? Put'em into one var_scope """
            with tf.variable_scope('qwords', reuse=self.reuse):
                qwords_hidden = seq_encoding(
                    emb_input=qwords_att_embedding, len_input=qwords_len,
                    encoder=q_encoder, reuse=self.reuse
                )  # (ds * sc_max_len, q_max_len, dim_hidden)
                qword_final_hidden = seq_hidden_max_pooling(
                    seq_hidden_input=qwords_hidden, len_input=qwords_len)
            with tf.variable_scope('preds', reuse=self.reuse):
                preds_hidden = seq_encoding(
                    emb_input=preds_att_embedding, len_input=preds_len,
                    encoder=pred_encoder, reuse=self.reuse
                )  # (ds * sc_max_len, path_max_len, dim_hidden)
                pred_final_hidden = seq_hidden_max_pooling(
                    seq_hidden_input=preds_hidden, len_input=preds_len)
            with tf.variable_scope('pwords', reuse=self.reuse):
                pwords_hidden = seq_encoding(
                    emb_input=pwords_att_embedding, len_input=pwords_len,
                    encoder=pword_encoder, reuse=self.reuse
                )  # (ds * sc_max_len, pword_max_len, dim_hidden)
                pword_final_hidden = seq_hidden_max_pooling(
                    seq_hidden_input=pwords_hidden, len_input=pwords_len)
            # x_final_hidden: (ds * sc_max_len, dim_hidden)

            """ Step 4: Path merging, calculate final score """
            # TODO: use pword/pred or not
            if self.path_merge_mode == 'sum':
                path_final_hidden = tf.add(pword_final_hidden, pred_final_hidden,
                                           name='path_final_hidden')  # (ds * sc_max_len, dim_hidden)
            else:                   # max
                path_final_hidden = tf.reduce_max(
                    tf.stack([pword_final_hidden,
                              pred_final_hidden], axis=0),  # (2, ds * sc_max_len, dim_hidden)
                    axis=0, name='path_final_hidden')  # (ds * sc_max_len, dim_hidden)

            if self.final_score_mode == 'cos':
                path_score = cosine_sim(lf_input=qword_final_hidden,
                                        rt_input=path_final_hidden)     # (ds * sc_max_len, )
            else:                   # dot
                path_score = tf.reduce_sum(qword_final_hidden * path_final_hidden,
                                           axis=-1)     # (ds * sc_max_len, )
            path_score = tf.reshape(path_score, shape=[-1, self.sc_max_len],
                                    name='path_score')  # (ds, sc_max_len)
            sc_mask = tf.sequence_mask(lengths=sc_len,
                                       maxlen=self.sc_max_len,
                                       dtype=tf.float32,
                                       name='sc_mask')  # (ds, sc_max_len) as mask
            score = tf.reduce_sum(path_score * sc_mask, axis=-1, name='score')  # (ds, )

        pred_att_mat = tf.reshape(preds_att_mat, [-1, self.sc_max_len, self.q_max_len, self.path_max_len],
                                  name='pred_att_mat')          # (ds, sc_max_len, q_max_len, path_max_len)
        pword_att_mat = tf.reshape(pwords_att_mat, [-1, self.sc_max_len, self.q_max_len, self.pword_max_len],
                                   name='pword_att_mat')        # (ds, sc_max_len, q_max_len, pword_max_len)
        return pred_att_mat, pword_att_mat, score
