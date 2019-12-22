# -*- coding:utf-8 -*-

import tensorflow as tf
from kangqi.util.LogUtil import LogInfo


# score_tf, metric_tf, mask_tf: (batch, PN)
def get_best_result(score_tf, metric_tf, mask_tf):
    useful_score_tf = score_tf - 100000000.0 * (1.0 - mask_tf)
    # (batch, PN) gives very low score to padding schemas
    predict_row_tf = tf.cast(tf.range(tf.shape(mask_tf)[0]), tf.int64)      # (batch, )
    predict_col_tf = tf.argmax(useful_score_tf, axis=1)   # (batch, ) of a int64 tensor
    # Get the co-ordination of the best candidate
    coo_tf = tf.stack(values=[predict_row_tf, predict_col_tf], axis=-1)
    metric_tf = tf.gather_nd(metric_tf, coo_tf, name='metric_tf')
    # (cases, ) returning whether we correctly link the corresponding cell

    return metric_tf


def show_tensor(tensor, name=None):
    show_name = name or tensor.name
    LogInfo.logs('* %s --> %s | %s', show_name, tensor.get_shape().as_list(), str(tensor))
