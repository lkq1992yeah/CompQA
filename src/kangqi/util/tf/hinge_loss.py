# -*- coding: utf-8 -*-

#==============================================================================
# Author: Kangqi Luo
# Goal: Implement the Hinge-Loss as the final loss function
#==============================================================================

import tensorflow as tf

from kangqi.util.LogUtil import LogInfo

class HingeLoss(object):

    def __init__(self, margin, learning_rate, aggregation=tf.reduce_max):
        self.margin = margin
        self.learning_rate = learning_rate
        self.aggregation = aggregation
        # aggregation: could be max or sum (over each row), and these strategies
        #              stands for different variations of HingeLoss

    # score_tf, best_tf, mask_tf: (batch, PN)
    # best_tf means gold_tf, which is a 0-or-1 matrix
    # Currently, best_tf[:, 0] is always equal to 1 in the training data.
    def build_old(self, score_tf, best_tf, mask_tf):
        first_score_tf = tf.expand_dims(score_tf[:, 0], axis=1) # (batch, 1)
        delta_score_tf = score_tf - first_score_tf  # (batch, PN) get neg_score - pos_score at each position
        L = tf.nn.relu((self.margin + delta_score_tf) * mask_tf * (1.0 - best_tf))
        # (batch, PN), get max(0, margin - pos_score + neg_score) as loss value, but ignore the best cell and padding cells

        loss_tf = tf.reduce_max(L, reduction_indices=1) # (batch, )
        final_loss_tf = tf.reduce_mean(loss_tf) # (, )
        train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(final_loss_tf)
        LogInfo.logs('* Hinge loss compiled.')

        return final_loss_tf, train_step


    # score_tf, best_tf, mask_tf: (batch, PN)
    # best_tf means gold_tf, which is a matrix of multiple 1-hot vectors
    # Now remove the restriction that best_tf[*, 0] must be 1
    # tf_list, l1_reg and l2_reg: used for adding regularizations
    def build(self, score_tf, best_tf, mask_tf, param_tf_list=None, l1_reg=0., l2_reg=0.):
        first_score_tf = tf.reduce_sum(score_tf * best_tf, axis=1, keep_dims=True) # (batch, 1)
        # Only keeps the score of positive data
        delta_score_tf = score_tf - first_score_tf  # (batch, PN) get neg_score - pos_score at each position
        L = tf.nn.relu((self.margin + delta_score_tf) * mask_tf * (1.0 - best_tf))
        # (batch, PN), get max(0, margin - pos_score + neg_score) as loss value,
        # but ignore the best cell and padding cells (L is always 0 at these cells)
        loss_tf = self.aggregation(L, axis=1) # (batch, )
        final_loss_tf = tf.reduce_sum(loss_tf) # (, )
        # this time we use sum loss rather than average loss (refer to TransE)

        if param_tf_list is not None:
            reg_tf_list = []
            for param_tf in param_tf_list:
                reg_tf_list.append(         # each reg_tf is a scalar tensor
                    l2_reg * tf.reduce_sum(param_tf ** 2) +
                    l1_reg * tf.reduce_sum(tf.abs(param_tf)))
            sum_reg_tf = tf.reduce_sum(tf.stack(reg_tf_list))
            final_loss_tf = final_loss_tf + sum_reg_tf
        train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(final_loss_tf)
        LogInfo.logs('* Hinge loss compiled.')

        return final_loss_tf, train_step