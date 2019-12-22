# -*- coding: utf-8 -*-

#==============================================================================
# Author: Kangqi Luo
# Goal: Implement the speedup version of RankNet.
# The main code copies from toy.ranknet.ranknet_improved.py, but here are two differences:
# 1. Each batch consists several lists.
# 2. We have padding candidates (mask_tf will be used)
# I think, this code would be used in multiple tasks (not only in compQA, but in any L2R situations)
#==============================================================================

import numpy as np
import tensorflow as tf
# from kangqi.util.LogUtil import LogInfo

class RankNet(object):

    def __init__(self, batch_size, list_len, learning_rate):
        self.batch_size = batch_size
        self.list_len = list_len
        self.learning_rate = learning_rate

    # score_tf: (batch_size, list_len)
    # Output: several (batch_size, list_len, var_shape) tensor, forming in a list.
    def get_gradient_tf_list(self, score_tf):
        # LogInfo.begin_track('RankNet genearating gradients ... ')
        grad_tf_list = []   # the return value

        scan = 0
        for var in tf.global_variables():
            scan += 1
            # LogInfo.begin_track('Variable %d / %d %s: ',
            #             scan, len(tf.global_variables()), var.get_shape().as_list())
            per_row_grad_tf_list = []
            for row_idx in range(self.batch_size):
                # LogInfo.begin_track('row_idx = %d / %d: ', row_idx + 1, self.batch_size)
                local_grad_tf_list = []
                for item_idx in range(self.list_len):
                    # if (item_idx + 1) % 50 == 0:
                        # LogInfo.logs('item_idx = %d / %d', item_idx + 1, self.list_len)
                    local_grad_tf = tf.gradients(score_tf[row_idx, item_idx], var)[0] # ("var_shape", )
                    local_grad_tf_list.append(local_grad_tf)
                per_row_grad_tf = tf.stack(local_grad_tf_list, axis=0)
                per_row_grad_tf_list.append(per_row_grad_tf)
                # per_row_grad_tf: (list_len, "var_shape")
                # LogInfo.end_track()
            grad_tf = tf.stack(per_row_grad_tf_list, axis=0)
            grad_tf_list.append(grad_tf)
            # LogInfo.logs('grad_tf: %s', grad_tf.get_shape().as_list())
            # grad_tf: (batch_size, list_len, "var_shape")
            # LogInfo.end_track()

#        More slower...
#        flat_score_tf = tf.reshape(score_tf, [-1]) # (batch_size * list_len)
#        for var in tf.global_variables():
#            var_shape = var.get_shape().as_list()
#            grad_shape = [self.batch_size, self.list_len] + var_shape
#            local_grad_tf_list = []
#            for item_idx in range(self.batch_size * self.list_len):
#                local_grad_tf = tf.gradients(flat_score_tf[item_idx], var)[0]
#                local_grad_tf_list.append(local_grad_tf)
#            grad_tf = tf.reshape(
#                tf.stack(local_grad_tf_list, axis=0),   # (batch_size * list_len, "var_shape")
#                grad_shape
#            )                       # (batch_size, list_len, "var_shape")
#            grad_tf_list.append(grad_tf)
#            # LogInfo.logs('grad_tf: %s', grad_tf.get_shape().as_list())

        # LogInfo.end_track('grad_tf_list compiled for %d variables.', len(grad_tf_list))
        return grad_tf_list

    # item_tf: (batch, list_len), could be predicted or labeled score.
    # goal: generate pairwise_tf (batch, list_len, list_len)
    # where pairwise_tf[i, j, k] = operation(item_tf[i, j], item[i, k])
    def get_pairwise_tf(self, item_tf, operation):
        x_tf = tf.reshape(item_tf, [-1, 1]) # (batch * list_len, 1)
        msk_tf = tf.constant(np.ones((1, self.list_len), dtype='float32')) # (1, list_len)
        mat_tf = tf.matmul(x_tf, msk_tf) # (batch * list_len, list_len)
        mat_tf = tf.reshape(mat_tf, [-1, self.list_len, self.list_len])
        # mat_tf: (batch, list_len, list_len), mat_tf[i, j, *] = item_tf[i, j]
        trans_tf = tf.transpose(mat_tf, perm=[0, 2, 1])
        # trans_tf: (batch, list_len, list_len), trans_tf[i, * j] = item_tf[i, j]

        pairwise_tf = operation(mat_tf, trans_tf)
        # pairwise_tf[i, j, k] = operation(item[i, j], item[i, k])
        # LogInfo.logs('pairwise_tf compiled. %s', pairwise_tf.get_shape().as_list())
        return pairwise_tf


    # Given score of each single item, return the pairwise loss
    # score_tf, label_tf, mask_tf: (batch_size, list_len)
    # Return the pairwise differecne: both based on predict score, or on gold label,
    # and also return the average loss of this batch.
    def get_loss_tf(self, score_tf, label_tf, mask_tf):
        score_diff_tf = self.get_pairwise_tf(score_tf, tf.subtract)
        label_diff_tf = self.get_pairwise_tf(label_tf, tf.subtract) # (batch, list_len, list_len)

        pred_tf = tf.nn.sigmoid(score_diff_tf)
        gold_tf = tf.nn.sigmoid(label_diff_tf)  # (batch, list_len, list_len)
        useful_pair_tf = self.get_pairwise_tf(mask_tf, tf.multiply) # (batch, list_len, list_len)

        loss_tf = -1.0 * useful_pair_tf * (
                gold_tf * tf.log(tf.clip_by_value(pred_tf, 1e-6, 1.0)) +
                (1.0 - gold_tf) * tf.log(tf.clip_by_value(1.0 - pred_tf, 1e-6, 1.0))
        ) # (batch, list_len, list_len)  Each cell is the pair loss (padding cells are always 0)

        sum_loss_tf = tf.reduce_sum(loss_tf, axis=[1, 2])             # (batch, ) storing sum loss
        sum_useful_tf = tf.reduce_sum(useful_pair_tf, axis=[1, 2])    # (batch, ) storing number of useful pairs
        avg_loss_tf = sum_loss_tf / sum_useful_tf   # (batch, ) storing avg. loss over each list.
        # TODO: sum_useful_tf maybe 0, be careful with this corner case!

        final_loss_tf = tf.reduce_mean(avg_loss_tf)
        # LogInfo.logs('final_loss_tf compiled. %s', final_loss_tf.get_shape().as_list())
        return pred_tf, gold_tf, useful_pair_tf, final_loss_tf

    # score_tf, label_tf, mask_tf: (batch_size, list_len)
    # return:
    # 1. sum_labmda_tf: (batch_size, list_len)
    # 2. final_loss_tf: average loss over these batches.
    # Note: we shall correctly handle mask.
    def get_lambda_tf(self, score_tf, label_tf, mask_tf):
        pred_tf, gold_tf, useful_pair_tf, final_loss_tf = self.get_loss_tf(score_tf, label_tf, mask_tf)
        lambda_tf = (pred_tf - gold_tf) * useful_pair_tf   # (batch, list_len, list_len)
        sum_lambda_tf = tf.reduce_sum(lambda_tf, axis=2) # (batch, list_len)
        # LogInfo.logs('sum_lambda_tf compiled. %s', sum_lambda_tf.get_shape().as_list())

        return final_loss_tf, sum_lambda_tf

    # grad_tf_list: several (batch_size, list_len, var_shape) tensor forming a list
    # sum_lambda_tf: (batch_size, list_len) as the weight of each gradient.
    def get_update_list(self, grad_tf_list, sum_lambda_tf):
        update_list = []
        use_sum_lambda_tf = tf.reshape(sum_lambda_tf, [1, self.batch_size * self.list_len])
        # use_sum_lambda_tf: (1, batch * list_len)
        for var, grad_tf in zip(tf.global_variables(), grad_tf_list):
            # grad_tf: (batch, list_len, var_shape)
            var_shape = grad_tf.get_shape().as_list()[2: ]
            # get the shape of each var (ignoring batch and list_len)
            use_grad_tf = tf.reshape(grad_tf, [-1, np.prod(var_shape)])
            # use_grad_tf: (batch * list_len, size_of_var_shape)
            use_merged_grad_tf = tf.matmul(use_sum_lambda_tf, use_grad_tf) # (1, size_of_var_shape)
            merged_grad_tf = tf.reshape(use_merged_grad_tf, var_shape)
            upd = tf.assign(var, var - self.learning_rate * merged_grad_tf)
            update_list.append(upd)
        # LogInfo.logs('update_list compiled, len = %d.', len(update_list))
        return update_list


    # The simplest RankNet: building pairwise loss, and perform gradient descent.
    def build(self, score_tf, label_tf, mask_tf):
        pred_tf, gold_tf, useful_pair_tf, final_loss_tf = self.get_loss_tf(score_tf, label_tf, mask_tf)
        train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(final_loss_tf)
        # LogInfo.logs('train_step (normal) built.')
        return final_loss_tf, train_step

    # The version of building lambda-based RankNet.
    # score_tf, label_tf, mask_tf: (batch, list_len)
    def build_improved(self, score_tf, label_tf, mask_tf):
        grad_tf_list = self.get_gradient_tf_list(score_tf)
        final_loss_tf, sum_lambda_tf = self.get_lambda_tf(score_tf, label_tf, mask_tf)
        update_list = self.get_update_list(grad_tf_list, sum_lambda_tf)
        # LogInfo.logs('update_list (lambda-based) built.')
        return final_loss_tf, update_list


