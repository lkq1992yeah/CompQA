# -*- coding:utf-8 -*-
# Author: Kangqi Luo
# Goal: Yih pre train

import tensorflow as tf
import numpy as np

from ..module.letter_trigram_cnn import LetterTriGramCNN

from kangqi.util.LogUtil import LogInfo
from kangqi.util.config import ConfigDict
from kangqi.util.tf.siamese import SiameseNetwork
from kangqi.util.tf.tf_basics import cosine2d, xavier_weight_init

def work(cd):
    with open(cd.data_fp, 'rb') as br:
        q_t3 = np.load(br)          # (data, q_max_len, n_hash)
        skw_t3 = np.load(br)        # (data, skw_max_len, n_hash)
        gold_vec = np.load(br)      # (data, )

def build_model(cd):
    LogInfo.begin_track('Building yih_pre_train Model ... ')
    with tf.variable_scope('yih_pre_train', initializer=xavier_weight_init()):
        q_input = tf.placeholder(tf.float32,
                        [None, cd.q_max_len, cd.n_hash], 'q_input')
        skw_input = tf.placeholder(tf.float32,
                        [None, cd.skw_max_len, cd.n_hash], 'skw_input')
        gold = tf.placeholder(tf.float32, [None, ], 'gold')

        q_cnn_module = LetterTriGramCNN(
                            n_max_len=cd.q_max_len,
                            n_hash=cd.n_hash,
                            n_window=cd.n_window,
                            n_conv=cd.n_conv,
                            n_hidden=cd.n_hidden,
                            scope_name='q_cnn')
        sk_cnn_module = LetterTriGramCNN(
                            n_max_len=cd.skw_max_len,
                            n_hash=cd.n_hash,
                            n_window=cd.n_window,
                            n_conv=cd.n_conv,
                            n_hidden=cd.n_hidden,
                            scope_name='sk_cnn')
        siamese_module = SiameseNetwork(
                lf_module=q_cnn_module,
                rt_module=sk_cnn_module,
                merge_func=cosine2d)            # output: (data, 1)
        sim = siamese_module.build(q_input, skw_input)
        
        w_final = tf.Variable(tf.ones((1,1)), name='w_final')
        b_final = tf.Variable(tf.zeros((1,)), name='b_final')
        score = tf.matmul(sim, w_final) + b_final   # (data, 1)
        
        flat_sim = tf.reshape(sim, [-1])        # (data, )
        flat_score = tf.reshape(score, [-1])    # (data, )

        err_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=gold, logits=flat_score)
        l2_loss_list = []
        for var in tf.global_variables():
            LogInfo.logs('* %s', var.name)
            l2_loss_list.append(tf.nn.l2_loss(var))
        l2_loss = tf.add_n(l2_loss_list) * cd.l2_reg
        final_loss = err_loss + l2_loss
        train_step = tf.train.AdamOptimizer(cd.learning_rate).minimize(final_loss)

    LogInfo.end_track('Compiled.')
    return train_step, final_loss, flat_sim


if __name__ == '__main__':
    work_dir = 'runnings/compQA/yih_pre_train'
    cd = ConfigDict(work_dir + '/param_config')
    train_step, final_loss = build_model(cd)
