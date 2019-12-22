"""
Author: Kangqi Luo
Goal: Implement NN in yu2017improved
Relation Detection Only
Each data: <q, r+, r->
"""

import tensorflow as tf

from .u import seq_encoding, schema_encoding, seq_hidden_max_pooling, get_merge_function
from ..u import show_tensor

from ..base_model import BaseModel
from xusheng.model.rnn_encoder import BidirectionalRNNEncoder

from kangqi.util.LogUtil import LogInfo


class SimpqOptmModel(BaseModel):

    def __init__(self, sess, n_words, n_preds, dim_emb,
                 q_max_len, path_max_len, pword_max_len,
                 dim_hidden, rnn_cell, merge_config,
                 margin, learning_rate, optm_name,
                 reuse=tf.AUTO_REUSE, verbose=0):
        LogInfo.begin_track('SimpqOptmModel Building ...')
        super(SimpqOptmModel, self).__init__(sess=sess, ob_batch_num=100, verbose=verbose)

        assert optm_name in ('Adam', 'Adadelta', 'Adagrad', 'GradientDescent')
        optm_name += 'Optimizer'

        # ======== declare sub-modules ======== #

        num_units = dim_hidden / 2          # bidirectional
        rnn_config = {'num_units': num_units, 'cell_class': rnn_cell}
        encoder_args = {'config': rnn_config, 'mode': tf.contrib.learn.ModeKeys.TRAIN}

        q_encoder = BidirectionalRNNEncoder(**encoder_args)
        pred_encoder = BidirectionalRNNEncoder(**encoder_args)
        pword_encoder = BidirectionalRNNEncoder(**encoder_args)
        merge_func = get_merge_function(merge_config=merge_config, dim_hidden=dim_hidden, reuse=reuse)
        LogInfo.logs('Sub-modules declared.')

        # ======== define tensors ======== #

        q_words_input = tf.placeholder(dtype=tf.int32,
                                       shape=[None, q_max_len],
                                       name='q_words_input')            # (data_size, q_max_len)
        q_words_len_input = tf.placeholder(dtype=tf.int32,
                                           shape=[None],
                                           name='q_words_len_input')    # (data_size, )
        self.optm_input_tf_list = [q_words_input, q_words_len_input]

        sc_tensor_groups = []       # [ pos_tensors, neg_tensors ]
        for cate in ('pos', 'neg'):
            preds_input = tf.placeholder(dtype=tf.int32,
                                         shape=[None, path_max_len],
                                         name=cate+'_preds_input')          # (data_size, path_max_len)
            preds_len_input = tf.placeholder(dtype=tf.int32,
                                             shape=[None],
                                             name=cate+'_preds_len_input')  # (data_size, )
            pwords_input = tf.placeholder(dtype=tf.int32,
                                          shape=[None, pword_max_len],
                                          name=cate+'_pwords_input')        # (data_size, pword_max_len)
            pwords_len_input = tf.placeholder(dtype=tf.int32,
                                              shape=[None],
                                              name=cate+'_pwords_len_input')    # (data_size, )
            tensor_group = [preds_input, preds_len_input, pwords_input, pwords_len_input]
            sc_tensor_groups.append(tensor_group)
            self.optm_input_tf_list += tensor_group
        LogInfo.begin_track('Showing %d input tensors:', len(self.optm_input_tf_list))
        for tensor in self.optm_input_tf_list:
            show_tensor(tensor)
        LogInfo.end_track()

        # ======== start building model ======== #

        with tf.variable_scope('Embedding_Lookup', reuse=reuse):
            with tf.device('/cpu:0'):
                self.w_embedding_init = tf.placeholder(dtype=tf.float32,
                                                       shape=(n_words, dim_emb),
                                                       name='w_embedding_init')
                self.p_embedding_init = tf.placeholder(dtype=tf.float32,
                                                       shape=(n_preds, dim_emb),
                                                       name='p_embedding_init')
                w_embedding = tf.get_variable(name='w_embedding',
                                              initializer=self.w_embedding_init)
                p_embedding = tf.get_variable(name='p_embedding',
                                              initializer=self.p_embedding_init)

                q_words_embedding = tf.nn.embedding_lookup(params=w_embedding,
                                                           ids=q_words_input,
                                                           name='q_embedding')      # (batch, q_max_len, dim_emb)

        with tf.variable_scope('Question', reuse=reuse):
            q_words_hidden = seq_encoding(
                emb_input=q_words_embedding,
                len_input=q_words_len_input,
                encoder=q_encoder, reuse=reuse)         # (data_size, q_max_len, dim_emb)
            # q_hidden = tf.reduce_max(q_words_hidden,
            #                          axis=1, name='q_hidden')    # (data_size, dim_hidden)
            q_hidden = seq_hidden_max_pooling(seq_hidden_input=q_words_hidden,
                                              len_input=q_words_len_input)
            # TODO: Currently we just follow yu2017.

        logits_list = []        # store two tensors: positive and negative score
        for cate, sc_tensor_group in zip(('pos', 'neg'), sc_tensor_groups):
            LogInfo.logs('Calculate score at %s side ...', cate)
            preds_input, preds_len_input, pwords_input, pwords_len_input = sc_tensor_group
            with tf.variable_scope('Embedding_Lookup', reuse=reuse):
                with tf.device("/cpu:0"):
                    preds_embedding = tf.nn.embedding_lookup(
                        params=p_embedding, ids=preds_input, name='preds_embedding'
                    )       # (batch, path_max_len, dim_emb)
                    pwords_embedding = tf.nn.embedding_lookup(
                        params=w_embedding, ids=pwords_input, name='pwords_embedding'
                    )       # (batch, pword_max_len, dim_emb)
            with tf.variable_scope('Schema', reuse=reuse):
                with tf.variable_scope('Path', reuse=reuse):
                    preds_hidden = seq_encoding(
                        emb_input=preds_embedding,
                        len_input=preds_len_input,
                        encoder=pred_encoder, reuse=reuse)      # (data_size, path_max_len, dim_hidden)
                with tf.variable_scope('Pword', reuse=reuse):
                    pwords_hidden = seq_encoding(
                        emb_input=pwords_embedding,
                        len_input=pwords_len_input,
                        encoder=pword_encoder, reuse=reuse)     # (data_size, pword_max_len, dim_hidden)
                schema_hidden = schema_encoding(
                    preds_hidden=preds_hidden, preds_len=preds_len_input,
                    pwords_hidden=pwords_hidden, pwords_len=pwords_len_input)
            with tf.variable_scope('Merge', reuse=reuse):
                # logits = cosine_sim(lf_input=q_hidden, rt_input=schema_hidden)    # (data_size, )
                logits = merge_func(q_hidden, schema_hidden)  # (data_size, )
            logits_list.append(logits)

        # ======== define loss and updates ======== #

        pos_logits, neg_logits = logits_list
        margin_loss = tf.nn.relu(neg_logits + margin - pos_logits,
                                 name='margin_loss')
        self.avg_loss = tf.reduce_mean(margin_loss, name='avg_loss')
        tf.summary.scalar('avg_loss', self.avg_loss, collections=['optm'])
        optimizer = getattr(tf.train, optm_name)
        self.optm_step = optimizer(learning_rate).minimize(self.avg_loss)
        self.optm_summary = tf.summary.merge_all(key='optm')
        LogInfo.logs('* avg_loss and optm_step defined.')

        LogInfo.end_track()
