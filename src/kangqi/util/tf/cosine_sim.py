# -*- coding: utf-8 -*-

# ==============================================================================
# Author: Kangqi Luo
# Goal: Calculate cosine similarity
# ==============================================================================

import numpy as np
import tensorflow as tf

from kangqi.util.LogUtil import LogInfo


# x_tf: (batch, dim)
# return: (batch, 1) as L2-length of each row
def get_length_tf(x_tf):
    return tf.sqrt(tf.reduce_sum(tf.square(x_tf), axis=1, keep_dims=True))


# x_tf, y_tf: (batch, dim)
# x_len_tf, y_len_tf: (batch, 1)
# return: (batch, 1) as cos(x_tf, y_tf)
def get_cosine_tf(x_tf, y_tf, x_len_tf, y_len_tf):
    dot_prod_tf = tf.reduce_sum(x_tf * y_tf, axis=1, keep_dims=True)    # (batch, 1)
    norm_tf = x_len_tf * y_len_tf   # (batch, 1)
    return dot_prod_tf / norm_tf


# x_tf, y_tf: (batch, dim)
# return: (batch, 1) as cos(x_tf, y_tf)
def get_cosine_tf_simple(x_tf, y_tf):
    x_len_tf = get_length_tf(x_tf)
    y_len_tf = get_length_tf(y_tf)
    return get_cosine_tf(x_tf, y_tf, x_len_tf, y_len_tf)


def cosine_sim(lf_input, rt_input):
    """
    This implementation is better
    :param lf_input: (ANY_SHAPE, dim_hidden)
    :param rt_input: (ANY_SHAPE, dim_hidden)
    :return: (ANY_SHAPE, )
    """
    lf_norm = tf.sqrt(
        tf.reduce_sum(lf_input ** 2, axis=-1, keep_dims=True) + 1e-6,
        name='lf_norm'
    )   # (ANY_SHAPE, 1)
    lf_norm_hidden = tf.div(lf_input, lf_norm,
                            name='lf_norm_hidden')  # (ANY_SHAPE, dim_hidden)
    rt_norm = tf.sqrt(
        tf.reduce_sum(rt_input ** 2, axis=-1, keep_dims=True) + 1e-6,
        name='rt_norm'
    )   # (ANY_SHAPE, 1)
    rt_norm_hidden = tf.div(rt_input, rt_norm,
                            name='rt_norm_hidden')  # (ANY_SHAPE, dim_hidden)

    cosine_score = tf.reduce_sum(lf_norm_hidden * rt_norm_hidden,
                                 axis=-1, name='cosine_score')
    # (ANY_SHAPE, )
    return cosine_score


def main():
    from kangqi.util.tf.tf_basics import cosine2d
    sess = tf.InteractiveSession()
    x_tf = tf.placeholder(tf.float32, [2, 2])
    y_tf = tf.placeholder(tf.float32, [2, 2])
    x = np.array([[1., 2.], [3., 4.]], dtype='float32')
    y = np.array([[2., 3.], [4., 1.]], dtype='float32')

    x_len_tf = get_length_tf(x_tf)
    y_len_tf = get_length_tf(y_tf)
    ret_tf = get_cosine_tf(x_tf, y_tf, x_len_tf, y_len_tf)
    comp_tf = cosine2d(x_tf, y_tf)

    x_len_opt, y_len_opt, cos_opt, comp_opt = \
        sess.run([x_len_tf, y_len_tf, ret_tf, comp_tf], {x_tf: x, y_tf: y})

    LogInfo.logs('x_len.: %s', x_len_opt)
    LogInfo.logs('y_len.: %s', y_len_opt)
    LogInfo.logs('cosine: %s', cos_opt)
    LogInfo.logs('cosine2d: %s', comp_opt)
