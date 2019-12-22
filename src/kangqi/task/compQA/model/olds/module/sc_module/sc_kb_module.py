# -*- coding:utf-8 -*-

# Goal: schema --> list of skeleton --> Single Direction of LSTM --> Merge

import tensorflow as tf
from xusheng.model.rnn_encoder import UnidirectionalRNNEncoder

from kangqi.util.LogUtil import LogInfo


class SchemaKBOnlyModule(object):

    def __init__(self, rnn_config, sk_num, sk_max_len,
                 n_entities, n_preds, n_kb_emb,
                 n_sc_hidden, mode=tf.contrib.learn.ModeKeys.TRAIN):
        self.sk_num = sk_num
        self.sk_max_len = sk_max_len
        self.n_kb_emb = n_kb_emb
        self.n_sc_hidden = n_sc_hidden
        self.e_embedding_init = tf.placeholder(dtype=tf.float32,
                                               shape=(n_entities, n_kb_emb),
                                               name='e_embedding_init')
        self.p_embedding_init = tf.placeholder(dtype=tf.float32,
                                               shape=(n_preds, n_kb_emb),
                                               name='p_embedding_init')
        self.rnn_encoder = UnidirectionalRNNEncoder(rnn_config, mode)

    # focus_input: (data_size, sk_num) as int
    # path_input: (data_size, sk_num, sk_max_len) as int
    # path_len_input: (data_size, sk_num) as int
    def forward(self, focus_input, path_input, path_len_input, reuse=None):
        LogInfo.begin_track('SchemaKBOnlyModule forward: ')
        with tf.variable_scope('SchemaKBOnlyModule', reuse=reuse):  # , reuse=tf.AUTO_REUSE):
            with tf.device('/cpu:0'):
                e_embedding = tf.get_variable(name='e_embedding', initializer=self.e_embedding_init)
                p_embedding = tf.get_variable(name='p_embedding', initializer=self.p_embedding_init)
                initial_state = tf.nn.embedding_lookup(
                    params=e_embedding,
                    ids=tf.reshape(focus_input, [-1]),
                    name='initial_state')       # (ds*sk_num, n_kb_emb)
                path_emb_input = tf.nn.embedding_lookup(
                    params=p_embedding,
                    ids=tf.reshape(path_input, [-1, self.sk_max_len]),
                    name='path_emb_input')      # (ds*sk_num, sk_max_len, n_kb_emb)
            path_len_vec = tf.reshape(path_len_input, [-1],
                                      name='path_len_vec')  # (ds*sk_num, )
            rnn_inputs = tf.unstack(path_emb_input,
                                    num=self.sk_max_len,
                                    axis=1,
                                    name='rnn_inputs')      # [(ds*sk_num, n_kb_emb) * sk_max_len]
            encoder_output = self.rnn_encoder.encode(inputs=rnn_inputs,
                                                     sequence_length=path_len_vec,
                                                     initial_state=initial_state,
                                                     reuse=reuse)
            rnn_outputs = tf.stack(encoder_output.outputs, axis=1, name='rnn_outputs')
            # (data_size * sk_num, path_len, n_hidden_emb)
            LogInfo.logs('outputs = %s', rnn_outputs.get_shape().as_list())

            # Now pick the representative hidden vector for each skeleton
            # Determined by the length of each one
            row_tf = tf.cast(tf.range(tf.shape(path_emb_input)[0]), tf.int32)
            col_tf = tf.maximum(path_len_vec - 1, 0)
            # choose the position of the output vector representation of a path (starting from 0)
            # For padding paths (path_len = 0), we make sure that col_tf won't beome -1.
            # Since encoder_output takes path_len_mat into consideration,
            # the padding path will always return zero vector as its vector representation.
            coo_tf = tf.stack(values=[row_tf, col_tf], axis=-1)
            sk_hidden_tf = tf.gather_nd(params=rnn_outputs,
                                        indices=coo_tf,
                                        name='sk_hidden_tf')  # (data_size * sk_num, n_hidden_emb)
            sk_merge_hidden_tf = tf.reshape(sk_hidden_tf, [-1, self.sk_num * self.n_kb_emb],
                                            name='sk_merge_hidden_tf')  # (ds, sk_num*n_hidden_emb)

            w_hidden = tf.get_variable(
                name='w_hidden',
                shape=(self.sk_num * self.n_kb_emb, self.n_sc_hidden),
                initializer=tf.contrib.layers.xavier_initializer()
            )
            b_hidden = tf.get_variable(
                name='b_hidden',
                shape=(self.n_sc_hidden,)
            )
            sc_hidden_tf = tf.nn.relu(tf.matmul(sk_merge_hidden_tf, w_hidden) + b_hidden,
                                      name='sc_hidden_tf')  # (ds, n_sc_hidden)
            LogInfo.logs('sc_hidden_tf = %s', sc_hidden_tf.get_shape().as_list())
        LogInfo.end_track()
        return sc_hidden_tf
