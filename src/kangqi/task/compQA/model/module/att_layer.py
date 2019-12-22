"""
Author: Kangqi Luo
Date: 18-01-30
Goal: Implement a bunch of different cross-attention operations
      (Just for outputting different att. matrices)
"""

import tensorflow as tf


def expand_both_dims(lf_input, rt_input, lf_max_len, rt_max_len):
    """
    lf_input:   (ds, lf_max_len, dim_emb)
    rt_input:   (ds, rt_max_len, dim_emb)
    """
    expand_lf_input = tf.stack([lf_input] * rt_max_len, axis=2,
                               name='expand_lf_input')  # (ds, lf_max_len, rt_max_len, dim_emb)
    expand_rt_input = tf.stack([rt_input] * lf_max_len, axis=1,
                               name='expand_rt_input')  # (ds, lf_max_len, rt_max_len, dim_emb)
    return expand_lf_input, expand_rt_input


def cross_att_dot(lf_input, rt_input, lf_max_len, rt_max_len, dim_att_hidden):
    """
    bilinear: a = x_1 . x_2
    dim_att_hidden is never used in the function.
    """
    assert dim_att_hidden is not None
    expand_lf_input, expand_rt_input = expand_both_dims(
        lf_input=lf_input, rt_input=rt_input,
        lf_max_len=lf_max_len, rt_max_len=rt_max_len
    )  # both (ds, lf_max_len, rt_max_len, dim_emb)
    att_mat = tf.reduce_sum(expand_lf_input * expand_rt_input, axis=-1,
                            name='att_mat')  # (ds, lf_max_len, rt_max_len)
    return att_mat


def cross_att_bilinear(lf_input, rt_input, lf_max_len, rt_max_len, dim_att_hidden):
    """
    bilinear: a = x_1 . W . x_2
    dim_att_hidden should be equal with dim_hidden,
    otherwise, matmul couldn't work properly
    """
    expand_lf_input, expand_rt_input = expand_both_dims(
        lf_input=lf_input, rt_input=rt_input,
        lf_max_len=lf_max_len, rt_max_len=rt_max_len
    )  # both (ds, lf_max_len, rt_max_len, dim_emb)
    with tf.variable_scope('cross_att_bilinear', reuse=tf.AUTO_REUSE):
        w = tf.get_variable(name='w', dtype=tf.float32,
                            shape=[dim_att_hidden, dim_att_hidden])
        att_mat = tf.reduce_sum(
            input_tensor=tf.multiply(
                x=tf.matmul(expand_lf_input, w),
                y=expand_rt_input   # both (ds, lf_max_len, rt_max_len, dim_att_hidden==dim_emb)
            ), axis=-1, name='att_mat'
        )   # (ds, lf_max_len, rt_max_len)
    return att_mat


def cross_att_bahdanau(lf_input, rt_input, lf_max_len, rt_max_len, dim_att_hidden):
    """
    A = u . relu(W[x_1 : x_2] + b), or understand as:
    A1 = W_1 . x_1
    A2 = W_2 . x_2
    A = u . relu(A1 + A2 + b)
    :param lf_input: (ds, lf_max_len, dim_hidden)
    :param rt_input: (ds, rt_max_len, dim_hidden)
    :param lf_max_len: int value
    :param rt_max_len: int value
    :param dim_att_hidden: the hidden dimension in the attention operation
    :return: (ds, lf_max_len, rt_max_len)
    """
    with tf.variable_scope('cross_att_bahdanau', reuse=tf.AUTO_REUSE):
        lf_att = tf.contrib.layers.fully_connected(inputs=lf_input,
                                                   num_outputs=dim_att_hidden,
                                                   activation_fn=None,
                                                   scope='fc_lf')  # (ds, lf_max_len, dim_att_hidden)
        rt_att = tf.contrib.layers.fully_connected(inputs=rt_input,
                                                   num_outputs=dim_att_hidden,
                                                   activation_fn=None,
                                                   scope='fc_rt')  # (ds, rt_max_len, dim_att_hidden)
        expand_lf_att, expand_rt_att = expand_both_dims(
            lf_input=lf_att, rt_input=rt_att,
            lf_max_len=lf_max_len, rt_max_len=rt_max_len
        )  # both (ds, lf_max_len, rt_max_len, dim_att_hidden)
        u = tf.get_variable(name='u', shape=[dim_att_hidden], dtype=tf.float32)
        b = tf.get_variable(name='b', shape=[dim_att_hidden], dtype=tf.float32)
        activate = tf.nn.relu(expand_lf_att + expand_rt_att + b)
        att_mat = tf.reduce_sum(activate * u, axis=-1, name='att_mat')      # (ds, lf_max_len, rt_max_len)
    return att_mat


def cross_att_bdot(lf_input, rt_input, lf_max_len, rt_max_len, dim_att_hidden):
    """
    bahdanau-dot: t_i = relu(Wx_i + b), a = t_1 . t_2
    Check AF-attention paper, formula (1)
    """
    with tf.variable_scope('cross_att_bahdanau_dot', reuse=tf.AUTO_REUSE):
        lf_att = tf.contrib.layers.fully_connected(inputs=lf_input,
                                                   num_outputs=dim_att_hidden,
                                                   activation_fn=tf.nn.relu,
                                                   scope='fc_lf')  # (ds, lf_max_len, dim_att_hidden)
        rt_att = tf.contrib.layers.fully_connected(inputs=rt_input,
                                                   num_outputs=dim_att_hidden,
                                                   activation_fn=tf.nn.relu,
                                                   scope='fc_rt')  # (ds, rt_max_len, dim_att_hidden)
    expand_lf_att, expand_rt_att = expand_both_dims(
        lf_input=lf_att, rt_input=rt_att,
        lf_max_len=lf_max_len, rt_max_len=rt_max_len
    )  # both (ds, lf_max_len, rt_max_len, dim_att_hidden)
    att_mat = tf.reduce_sum(expand_lf_att * expand_rt_att, axis=-1,
                            name='att_mat')     # (ds, lf_max_len, rt_max_len)
    return att_mat
