import copy
import tensorflow as tf
from collections import namedtuple
from xusheng.util.tf_util import get_rnn_cell
from xusheng.model.attention import AttentionLayerBahdanau, AttentionLayerLuong
from tensorflow.python.layers import core as layers_core



'''
outputs: A list of the same length as decoder_inputs of 2D Tensors with
    shape [batch_size, output_size] containing generated outputs.
state: The state of each cell at the final time-step.
    It is a 2D Tensor of shape [batch_size, cell.state_size].
'''
DecoderOutput = namedtuple(
    "DecoderOutput", ["outputs", "sample_id", "final_state"])


class BasicRNNDecoder(object):
    def __init__(self, cell_params, attention_params,
                 encoder_values, encoder_values_length, mode):
        self.cell_params = copy.deepcopy(cell_params)
        self.attention_params = copy.deepcopy(attention_params)
        # only keep dropout during training
        if mode != tf.contrib.learn.ModeKeys.TRAIN:
            self.cell_params["dropout_input_keep_prob"] = 1.0
            self.cell_params["dropout_output_keep_prob"] = 1.0
            self.cell_params["reuse"] = True

        self.cell = get_rnn_cell(**self.cell_params)

        # num_units: The depth of the query mechanism.
        if self.attention_params["type"] == "Bahdanau":
            att = AttentionLayerBahdanau(encoder_values, encoder_values_length,
                                         self.attention_params["att_num_units"])
        elif self.attention_params["type"] == "Luong":
            att = AttentionLayerLuong(encoder_values, encoder_values_length,
                                      self.attention_params["att_num_units"])
        else:
            att = None

        if att is not None:
            # attention_layer_size
            self.cell = tf.contrib.seq2seq.AttentionWrapper(self.cell, att.attention_mechanism,
                                                            self.attention_params["attention_layer_size"])

    def read_only_decode(self, inputs, sequence_length, embedding, batch_size, mode,
                         initial_state=None, tgt_vocab_size=None):
        """
         cell: core_rnn_cell.RNNCell defining the cell function and size.
         inputs: A list of 2D Tensors [batch_size, input_size].
         initial_state: 2D Tensor with shape [batch_size, cell.state_size].
        """
        if mode != tf.contrib.learn.ModeKeys.INFER:
            helper = tf.contrib.seq2seq.TrainingHelper(inputs=inputs,
                                                       sequence_length=sequence_length,
                                                       time_major=False)
        else:
            # TODO: seq2seq slot filling need sos_tag & eos_tag! previous 7 tags are not enough! => 9
            tgt_sos_id = 0
            tgt_eos_id = 1
            helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding=embedding,
                                                              start_tokens=tf.fill([batch_size], tgt_sos_id),
                                                              end_token=tgt_eos_id)

        output_layer = None
        if tgt_vocab_size is not None:
            output_layer = layers_core.Dense(tgt_vocab_size, use_bias=False)

        dtype = self.cell.dtype
        if initial_state is None:
            decoder_initial_state = self.cell.zero_state(batch_size, dtype)
        else:
            decoder_initial_state = self.cell.zero_state(batch_size, dtype).clone(
                cell_state=initial_state)

        decoder = tf.contrib.seq2seq.BasicDecoder(cell=self.cell, helper=helper,
                                                  initial_state=decoder_initial_state,
                                                  output_layer=output_layer)

        '''
        If time_major == False (default), this will be a `Tensor` shaped:
          `[batch_size, max_time, cell.output_size]`.

        If time_major == True, this will be a `Tensor` shaped:
          `[max_time, batch_size, cell.output_size]`.
        '''

        final_outputs, final_state, _ = tf.contrib.seq2seq.dynamic_decode(decoder=decoder,
                                                                          output_time_major=False,
                                                                          impute_finished=True)

        return DecoderOutput(outputs=final_outputs.rnn_output,
                             sample_id=final_outputs.sample_id,
                             final_state=final_state)

    def feed_forward_decode(self, inputs, sequence_length, embedding, sample_prob, batch_size, mode,
                            initial_state=None, tgt_vocab_size=None):
        """
         cell: core_rnn_cell.RNNCell defining the cell function and size.
         inputs: A list of 2D Tensors [batch_size, input_size].
         initial_state: 2D Tensor with shape [batch_size, cell.state_size].
        """
        if mode != tf.contrib.learn.ModeKeys.INFER:
            helper = tf.contrib.seq2seq.ScheduledEmbeddingTrainingHelper(inputs=inputs,
                                                                         sequence_length=sequence_length,
                                                                         embedding=embedding,
                                                                         sampling_probability=sample_prob,
                                                                         time_major=False)
        else:
            # TODO: seq2seq slot filling need sos_tag & eos_tag! previous 7 tags are not enough! => 9
            tgt_sos_id = 0
            tgt_eos_id = 1
            helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding=embedding,
                                                              start_tokens=tf.fill([batch_size], tgt_sos_id),
                                                              end_token=tgt_eos_id)

        output_layer = None
        if tgt_vocab_size is not None:
            output_layer = layers_core.Dense(tgt_vocab_size, use_bias=False)

        dtype = self.cell.dtype
        if initial_state is None:
            decoder_initial_state = self.cell.zero_state(batch_size, dtype)
        else:
            decoder_initial_state = self.cell.zero_state(batch_size, dtype).clone(
                cell_state=initial_state)

        decoder = tf.contrib.seq2seq.BasicDecoder(cell=self.cell, helper=helper,
                                                  initial_state=decoder_initial_state,
                                                  output_layer=output_layer)

        '''
        If time_major == False (default), this will be a `Tensor` shaped:
          `[batch_size, max_time, cell.output_size]`.

        If time_major == True, this will be a `Tensor` shaped:
          `[max_time, batch_size, cell.output_size]`.
        '''

        final_outputs, final_state, _ = tf.contrib.seq2seq.dynamic_decode(decoder=decoder,
                                                                          output_time_major=False,
                                                                          impute_finished=True)

        return DecoderOutput(outputs=final_outputs.rnn_output,
                             sample_id=final_outputs.sample_id,
                             final_state=final_state)

