# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
# from tensorflow.python.ops import rnn_cell, rnn   # for 0.12.1

from kangqi.util.LogUtil import LogInfo

class SkeletonRNN:

    def __init__(self, n_steps, n_input, n_hidden,
                 rnn_unit='RNN', combine='FwBk'):
        self.n_steps = n_steps
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.rnn_unit = rnn_unit
        self.combine = combine
        # several combination strategies:
        # 'FwBk': concatenate final state of forward / backward unit
        # 'FwBk_FC': add a FC layer on that
        # 'MaxPool': max pooling on each output
        assert self.combine in ('FwBk', 'FwBk_FC', 'MaxPool')

    # Note: data_size = ds = batch * PN
    # sk_tf: tensor for skeleton input (data_size, n_steps, n_input)
    def build(self, sk_tf):
        x = tf.transpose(sk_tf, [1, 0, 2])  # (n_steps, ds, n_input)
        x = tf.reshape(x, [-1, self.n_input])    # (n_steps * ds, n_input)
        x = tf.split(x, self.n_steps, axis=0) # an array with n_steps tensors: (ds, n_input)
        # x = tf.split(0, n_steps, x) # an array with n_steps tensors: (ds, n_input) # for 0.12.1

        rnn_fw_cell = rnn.BasicRNNCell(self.n_hidden)
        rnn_bw_cell = rnn.BasicRNNCell(self.n_hidden)
        output_tf_list, final_fw_tf, final_bk_tf = \
            rnn.static_bidirectional_rnn(rnn_fw_cell, rnn_bw_cell, x, dtype=tf.float32)

        # rnn_fw_cell = rnn_cell.BasicRNNCell(n_hidden)
        # rnn_bw_cell = rnn_cell.BasicRNNCell(n_hidden)
        # output_tf_list, final_fw_tf, final_bk_tf = \
        #     rnn.bidirectional_rnn(rnn_fw_cell, rnn_bw_cell, x, dtype=tf.float32) # for 0.12.1


        # rnn_output contains 3 elements:
        # 1. a list of n_steps tensors, each one has the shape (ds, 2 * n_hidden)
        # 2. final state of forward cell (ds, n_hidden)
        # 3. final state of backward cell (ds, n_hidden)

        # output_tf_list: a list of n_steps tensors, each one: (ds, 2 * n_hidden)
        output_tf = tf.stack(output_tf_list, axis=0)
        # output_tf: (n_steps, ds, 2 * n_hidden)
        sk_rnn_state_tf = tf.transpose(output_tf, [1, 0, 2])
        # (ds, n_step, 2 * n_hidden): get the hidden vector at each state

        if self.combine == 'FwBk':
            # rnn_output_tf: (ds, 2 * n_hidden)
            rnn_output_tf = tf.concat(values=[final_fw_tf, final_bk_tf], axis=1)
            # rnn_output_tf = tf.concat(concat_dim=1, values=[final_fw_tf, final_bk_tf]) # for 0.12.1

        LogInfo.logs('sk_rnn_state_tf compiled. %s', sk_rnn_state_tf.get_shape().as_list())
        LogInfo.logs('rnn_output_tf compiled. %s', rnn_output_tf.get_shape().as_list())
        LogInfo.logs('* SkeletonRNN built.')
        return sk_rnn_state_tf, rnn_output_tf


if __name__ == '__main__':
    LogInfo.begin_track('[sk_rnn] starts ... ')
    ds = 2
    steps = 3
    input_dim = 20
    hidden_dim = 5
    shape = (ds, steps, input_dim)
    sk_tf = tf.placeholder(tf.float32, shape=shape)
    LogInfo.logs('Input tensor defined.')

    rnn_output_tf = SkeletonRNN(
        n_steps=steps, n_input=input_dim, n_hidden=hidden_dim).build(sk_tf)
    LogInfo.logs('Output compiled.')

    sk = np.random.randn(*shape)
    sess = tf.InteractiveSession()
    sess.run(tf.initialize_all_variables())
    rnn_output = sess.run(rnn_output_tf, {sk_tf: sk})
    LogInfo.begin_track('Ouptut shape %s: %s', rnn_output.shape, rnn_output)

#    for item in rnn_output[0]: LogInfo.logs('%s', item.shape)
#    LogInfo.logs('%s', rnn_output[1].shape)
#    LogInfo.logs('%s', rnn_output[2].shape)

    LogInfo.end_track()