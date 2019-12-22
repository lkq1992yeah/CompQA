# -*- coding: utf-8 -*-

#==============================================================================
# Author: Kangqi Luo
# Goal: Implement the basic version of LambdaRank
# Maybe its slow ...
#==============================================================================

import tensorflow as tf
from kangqi.util.LogUtil import LogInfo

class LambdaRank(object):

    def __init__(self, metric, gold_diff_method, batch_size, list_len, learning_rate):
        self.metric = metric
        self.gold_diff_method = gold_diff_method # use {0, 0.5, 1} or sigmoid(sc_i - sc_j)
        assert self.gold_diff_method in ('Sigmoid', 'Discrete')

        self.batch_size = batch_size
        self.list_len = list_len
        self.learning_rate = learning_rate

    # score_tf: (batch_size, list_len)
    # Output: several (batch_size, list_len, var_shape) tensor, forming in a list.
    # We first borrow the function from ranknet_improved.py
    def get_gradient_tf_list(self, score_tf):
        LogInfo.begin_track('LambdaRank genearating gradients ... ')
        grad_tf_list = []   # the return value

        scan = 0
        for var in tf.global_variables():
            scan += 1
            LogInfo.begin_track('Variable %d / %d %s: ',
                        scan, len(tf.global_variables()), var.get_shape().as_list())
            per_row_grad_tf_list = []
            for row_idx in range(self.batch_size):
                LogInfo.begin_track('row_idx = %d / %d: ', row_idx + 1, self.batch_size)
                local_grad_tf_list = []
                for item_idx in range(self.list_len):
                    if (item_idx + 1) % 50 == 0:
                        LogInfo.logs('item_idx = %d / %d', item_idx + 1, self.list_len)
                    local_grad_tf = tf.gradients(score_tf[row_idx, item_idx], var)[0] # ("var_shape", )
                    local_grad_tf_list.append(local_grad_tf)
                per_row_grad_tf = tf.stack(local_grad_tf_list, axis=0)
                per_row_grad_tf_list.append(per_row_grad_tf)
                # per_row_grad_tf: (list_len, "var_shape")
                LogInfo.end_track()
            grad_tf = tf.stack(per_row_grad_tf_list, axis=0)
            grad_tf_list.append(grad_tf)
            LogInfo.logs('grad_tf: %s', grad_tf.get_shape().as_list())
            # grad_tf: (batch_size, list_len, "var_shape")
            LogInfo.end_track()
        return grad_tf_list

    def build_update(self, merged_grad_tf_list):
        update_list = []
        for var, merged_grad_tf in zip(tf.global_variables(), merged_grad_tf_list):
            upd = tf.assign(var, var - self.learning_rate * merged_grad_tf)
            update_list.append(upd)
        LogInfo.logs('update_list compiled, len = %d.', len(update_list))
        return update_list

