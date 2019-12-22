"""
Author: Kangqi Luo
Goal: Define the sequence-related operations.
"""

import tensorflow as tf

from kangqi.util.tf.cosine_sim import cosine_sim
from kangqi.util.tf.ntn import NeuralTensorNetwork
from kangqi.util.LogUtil import LogInfo


def get_merge_function(merge_config, dim_hidden, reuse):
    """
    Judge whether to use cosine similarity, or use a NTN layer
    """
    assert merge_config['name'] in ('cosine', 'NTN', 'FC')
    if merge_config['name'] == 'cosine':
        merge_func = cosine_sim
        LogInfo.logs('Merge function: cosine')
    elif merge_config['name'] == 'NTN':
        ntn = NeuralTensorNetwork(dim_hidden=dim_hidden,
                                  blocks=merge_config['blocks'])
        LogInfo.logs('Merge function: NTN')

        def merge_func(x, y):
            return ntn.forward(x, y, reuse=reuse)
    else:
        def merge_func(x, y):
            concat_input = tf.concat([x, y], axis=-1, name='concat_input')      # (data_size, 2 * dim_hidden)
            fc_hidden = tf.contrib.layers.fully_connected(
                inputs=concat_input,
                num_outputs=merge_config['dim_fc'],
                activation_fn=tf.nn.relu,
                scope='FC',
                reuse=reuse
            )  # (data_size, dim_fc)
            fc_score = tf.contrib.layers.fully_connected(
                inputs=fc_hidden,
                num_outputs=1,
                activation_fn=None,
                scope='FC_final',
                reuse=reuse
            )  # (data_size, 1)
            fc_score = tf.squeeze(fc_score, axis=-1, name='fc_score')
            return fc_score
    return merge_func


def seq_encoding(emb_input, len_input, encoder, fwbw=False, reuse=tf.AUTO_REUSE):
    """
    Just a small wrapper: given the embedding input, length input and encoder, return the encoded result
    :param emb_input: (data_size, stamps, dim_emb)
    :param len_input: (data_size, ) as int32
    :param encoder: the BidirectionalRNNEncoder instance
    :param fwbw: only use the concat of last fw & bw state
    :param reuse: reuse flag (used in generating RNN/GRU/LSTM- Cell)
    :return: (data_size, stamps, dim_hidden)
    """
    if encoder is None:
        return emb_input        # just use word embedding, without other operations
    rnn_input = tf.unstack(emb_input, axis=1, name='emb_input')  # stamp * (data_size, dim_emb)
    encoder_output = encoder.encode(inputs=rnn_input,
                                    sequence_length=len_input,
                                    reuse=reuse)
    if not fwbw:
        out_hidden = tf.stack(encoder_output.outputs, axis=1, name='out_hidden')  # (data_size, stamp, dim_hidden)
        return out_hidden
    else:
        out_hidden = tf.concat(encoder_output.final_state, axis=-1, name='out_hidden')  # (data_size, dim_hidden)
        return out_hidden


def schema_encoding(preds_hidden, preds_len, pwords_hidden, pwords_len):
    """
    Given the pred-/pword- sequence embedding after Bidirectional RNN layer,
    return the final representation of the schema.
    The detail implementation is varied (max pooling, average ...), controlled by the detail config.
    Currently we follow yu2017 and use max-pooling.
    :param preds_hidden:    (data_size, pred_max_len, dim_hidden)
    :param preds_len:       (data_size, )
    :param pwords_hidden:   (data_size, pword_max_len, dim_hidden)
    :param pwords_len:      (data_size, )
    :return: (data_size, dim_hidden), the final representation of the schema
    """
    masked_preds_hidden = seq_hidden_masking_before_pooling(seq_hidden_input=preds_hidden,
                                                            len_input=preds_len)
    masked_pwords_hidden = seq_hidden_masking_before_pooling(seq_hidden_input=pwords_hidden,
                                                             len_input=pwords_len)
    masked_merge_hidden = tf.concat(
        [masked_preds_hidden, masked_pwords_hidden],
        axis=1, name='masked_merge_hidden'
    )       # (data_size, pred_max_len + pword_max_len, dim_hidden)
    schema_hidden = tf.reduce_max(masked_merge_hidden,
                                  axis=1, name='schema_hidden')     # (data_size, dim_hidden)
    return schema_hidden


def seq_hidden_masking(seq_hidden_input, len_input, mask_value):
    """
    According to the length of each data, set the padding hidden vector into some pre-defined mask value
    Then we can perform max_pooling
    :param seq_hidden_input:    (data_size, max_len, dim_hidden)
    :param len_input:           (data_size, ) as int
    :param mask_value: tf.float32.min or 0
    :return: (data_size, stamps, dim_hidden) with masked.
    """
    max_len = tf.shape(seq_hidden_input)[1]     # could be int or int-tensor
    mask = tf.sequence_mask(lengths=len_input, maxlen=max_len,
                            dtype=tf.float32, name='mask')      # (data_size, max_len)
    exp_mask = tf.expand_dims(mask, axis=-1, name='exp_mask')   # (data_size, max_len, 1)
    masked_hidden = exp_mask * seq_hidden_input + (1.0 - exp_mask) * mask_value
    return masked_hidden


def seq_hidden_masking_before_pooling(seq_hidden_input, len_input):
    return seq_hidden_masking(seq_hidden_input, len_input, mask_value=tf.float32.min)


def seq_hidden_masking_before_averaging(seq_hidden_input, len_input):
    return seq_hidden_masking(seq_hidden_input, len_input, mask_value=0)


def seq_hidden_max_pooling(seq_hidden_input, len_input):
    """
    Perform max pooling over the sequences.
    Should perform masking before pooling, due to different length of sequences.
    :param seq_hidden_input:    (data_size, max_len, dim_hidden)
    :param len_input:           (data_size, ) as int
    :return: (data_size, dim_hidden)
    """
    masked_hidden = seq_hidden_masking_before_pooling(seq_hidden_input, len_input)
    final_hidden = tf.reduce_max(masked_hidden, axis=1, name='final_hidden')   # (-1, dim_hidden)
    return final_hidden


def seq_hidden_averaging(seq_hidden_input, len_input):
    """
    Average hidden vectors over all the stamps of the sequence.
    For the padding position of each sequence, their vectors are always 0 (controlled by BidirectionalRNNEncoder)
    For the padding sequences, their length is 0, we shall avoid dividing by 0.
    :param seq_hidden_input:    (data_size, max_len, dim_hidden)
    :param len_input:           (data_size, ) as int
    :return: (data_size, dim_hidden) as the averaged vector repr.
    """
    masked_hidden = seq_hidden_masking_before_averaging(seq_hidden_input, len_input)
    sum_seq_hidden = tf.reduce_sum(
        masked_hidden, axis=1, name='sum_seq_hidden'
    )  # (-1, dim_hidden)
    seq_len_mat = tf.cast(
        tf.expand_dims(
            tf.maximum(len_input, 1),  # for padding sequence, their length=0, we avoid dividing by 0
            axis=1
        ), dtype=tf.float32, name='seq_len_mat'
    )  # (-1, 1) as float32
    seq_avg_hidden = tf.div(sum_seq_hidden, seq_len_mat,
                            name='seq_avg_hidden')  # (-1, dim_hidden)
    return seq_avg_hidden


def seq_encoding_with_aggregation(emb_input, len_input, rnn_encoder, seq_merge_mode):
    """
    Given sequence embedding, return the aggregated representation of the whole sequence.
    Consider using or not using RNN
    :param emb_input: (ds, max_len, dim_emb)
    :param len_input: (ds, )
    :param rnn_encoder: Xusheng's BidirectionalRNNEncoder
    :param seq_merge_mode: fwbw / max / avg
    :return: (ds, dim_hidden)
    """
    is_fwbw = seq_merge_mode == 'fwbw'
    if rnn_encoder is not None:
        hidden_repr = seq_encoding(emb_input=emb_input, len_input=len_input,
                                   encoder=rnn_encoder, fwbw=is_fwbw)     # (ds, dim_hidden)
    else:
        hidden_repr = emb_input  # (ds, max_len, dim_emb)
    final_repr = None
    if seq_merge_mode == 'fwbw':
        final_repr = hidden_repr
    elif seq_merge_mode == 'avg':
        final_repr = seq_hidden_averaging(seq_hidden_input=hidden_repr, len_input=len_input)
    elif seq_merge_mode == 'max':
        final_repr = seq_hidden_max_pooling(seq_hidden_input=hidden_repr, len_input=len_input)
    return final_repr       # (ds, dim_hidden)
