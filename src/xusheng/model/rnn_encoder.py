"""
Static RNN
"""

import tensorflow as tf
import copy
from collections import namedtuple
from xusheng.util.tf_util import get_rnn_cell

from kangqi.util.LogUtil import LogInfo


EncoderOutput = namedtuple(
    "EncoderOutput",
    ["outputs", "final_state", "attention_values", "attention_values_length"])


class UnidirectionalRNNEncoder(object):
    def __init__(self, config, mode):
        self.params = copy.deepcopy(config)
        # only keep dropout during training
        if mode != tf.contrib.learn.ModeKeys.TRAIN:
            self.params["dropout_input_keep_prob"] = 1.0
            self.params["dropout_output_keep_prob"] = 1.0
            self.params["reuse"] = True

    # Kangqi: add the initial_state and reuse paremeter
    # initial state: we are going to set the initial state by looking at the focus entity
    # reuse: we may call encode() several times in the model, so the rnn_cell shall be reused after the first time.
    def encode(self, inputs, sequence_length, initial_state=None, reuse=None):
        # cell = get_rnn_cell(**self.params["rnn_cell"])
        self.params['reuse'] = reuse            # KQ: temporary solution
        cell = get_rnn_cell(**self.params)
        # Kangqi: I'm sure the "rnn_cell" name in the parameter dict is not needed
        # print cell.state_size
        # LSTM: LSTMStateTuple(c=200, h=200)
        # GRU: 200
        outputs, state = tf.contrib.rnn.static_rnn(cell=cell,
                                                   inputs=inputs,
                                                   initial_state=initial_state,
                                                   sequence_length=sequence_length,
                                                   dtype=tf.float32)

        attention_values = tf.stack(outputs, axis=1)

        return EncoderOutput(
            outputs=outputs,
            final_state=state,
            attention_values=attention_values,
            attention_values_length=sequence_length)


class BidirectionalRNNEncoder(object):
    def __init__(self, config, mode):
        self.params = copy.deepcopy(config)
        # only keep dropout during training
        if mode != tf.contrib.learn.ModeKeys.TRAIN:
            self.params['keep_prob'] = 1.0
        LogInfo.logs('Show Bi-RNN param: %s', self.params)

    def encode(self, inputs, sequence_length, reuse=None):
        self.params['reuse'] = reuse            # KQ: temporary solution
        cell_fw = get_rnn_cell(**self.params)
        cell_bw = get_rnn_cell(**self.params)

        outputs, output_state_fw, output_state_bw = \
            tf.contrib.rnn.static_bidirectional_rnn(cell_fw=cell_fw, cell_bw=cell_bw, inputs=inputs,
                                                    sequence_length=sequence_length,
                                                    dtype=tf.float32)

        attention_values = tf.stack(outputs, axis=1)

        return EncoderOutput(
            outputs=outputs,
            final_state=(output_state_fw, output_state_bw),
            attention_values=attention_values,
            attention_values_length=sequence_length)
