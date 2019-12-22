# -*- coding:utf-8 -*-

import tensorflow as tf

from .q_sc_base_model import QScBaseModel

from kangqi.util.LogUtil import LogInfo


class QScLogRegModel(QScBaseModel):

    def __init__(self, sess, q_max_len, n_words, n_wd_emb,
                 sk_num, sk_max_len, n_kb_emb,
                 q_module_config, sc_module_config, n_merge_hidden,
                 pn_size, keep_prob, learning_rate, verbose=0):
        LogInfo.begin_track('QScLogRegModel compiling ...')
        super(QScLogRegModel, self).__init__(
            sess=sess, q_max_len=q_max_len, n_words=n_words, n_wd_emb=n_wd_emb,
            sk_num=sk_num, sk_max_len=sk_max_len, n_kb_emb=n_kb_emb,
            q_module_config=q_module_config, sc_module_config=sc_module_config,
            n_merge_hidden=n_merge_hidden, pn_size=pn_size, keep_prob=keep_prob, verbose=verbose)

        self.optm_focus_emb_input = tf.placeholder(dtype=tf.float32,
                                                   shape=[None, sk_num, n_kb_emb],
                                                   name='optm_focus_emb_input')
        self.optm_path_emb_input = tf.placeholder(dtype=tf.float32,
                                                  shape=[None, sk_num, sk_max_len, n_kb_emb],
                                                  name='optm_path_emb_input')
        self.optm_path_len_input = tf.placeholder(dtype=tf.int32,
                                                  shape=[None, sk_num],
                                                  name='optm_path_len_input')
        self.optm_label_input = tf.placeholder(dtype=tf.float32,
                                               shape=[None],
                                               name='optm_label_input')
        self.optm_input_tf_list = [self.q_input,
                                   # self.q_emb_input,
                                   self.q_len_input,
                                   self.optm_focus_emb_input,
                                   self.optm_path_emb_input,
                                   self.optm_path_len_input,
                                   self.optm_label_input]

        LogInfo.begin_track('Optm Part Building ...')
        self.optm_sc_hidden = self.sc_module.forward(
            focus_emb_input=self.optm_focus_emb_input,
            path_emb_input=self.optm_path_emb_input,
            path_len_input=self.optm_path_len_input,
            reuse=True          # call forward() for the second time, the first time is called in QScBaseModel
        )       # optm_sc_hidden: (data_size, n_sc_hidden)
        LogInfo.logs('* optm_sc_hidden built: %s', self.optm_sc_hidden.get_shape().as_list())

        optm_logits = self.merge_module.forward(q_hidden=self.q_hidden,
                                                sc_hidden=self.optm_sc_hidden,
                                                dropout_switch=self.dropout_switch,
                                                keep_prob_input=self.keep_prob_input)
        neg_log_lik = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.optm_label_input,
                                                              logits=optm_logits)
        self.avg_loss = tf.reduce_mean(neg_log_lik)
        self.optm_step = tf.train.AdamOptimizer(learning_rate).minimize(self.avg_loss)
        LogInfo.logs('* avg_loss and optm_step defined.')
        LogInfo.end_track()

        self.saver = tf.train.Saver()
        LogInfo.logs('* saver defined.')

        LogInfo.end_track()
