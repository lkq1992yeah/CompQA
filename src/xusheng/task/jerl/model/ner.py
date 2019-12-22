"""
Single model for Named Entity Recognition
Written by Xusheng Luo
"""

import tensorflow as tf
import numpy as np

from xusheng.model.attention import AttentionLayerBahdanau
from xusheng.util.log_util import LogInfo


class NER(object):
    def __init__(self, config, mode, init_embedding=None):
        self.config = config
        self.mode = mode
        self.embedding = init_embedding
        self.transfer = config.get("transfer")
        if init_embedding is None:
            self.init_embedding = np.zeros([self.config.get('vocab_size'),
                                            self.config.get('word_dim')],
                                           dtype=np.float32)
        else:
            self.init_embedding = init_embedding
        self.__build_graph()
        self.saver = tf.train.Saver()

    @staticmethod
    def forward(config, lstm_output, ner_label, seq_len):
        with tf.variable_scope("ner_parameter"):
            W = tf.get_variable(
                shape=[config.get('lstm_dim')*2, config.get('num_classes')],
                initializer=tf.truncated_normal_initializer(stddev=0.01),
                dtype=tf.float32,
                name="weights",
                regularizer=tf.contrib.layers.l2_regularizer(config.get('l2_reg_lambda'))
            )
            b = tf.Variable(tf.zeros([config.get('num_classes')],
                                     dtype=tf.float32,
                                     name="bias"))

        with tf.variable_scope("ner_logits"):
            size = tf.shape(lstm_output)[0]
            output = tf.reshape(lstm_output, [-1, 2 * config.get('lstm_dim')])
            matricized_unary_scores = tf.matmul(output, W) + b
            logits = tf.reshape(matricized_unary_scores, [size, -1, config.get('num_classes')])

        with tf.variable_scope("ner_loss"):
            # CRF log likelihood
            log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(
                logits, ner_label, seq_len)

            loss = tf.reduce_mean(-log_likelihood)

        return logits, transition_params, loss

    def __build_graph(self):
        self.seq_idx = tf.placeholder(dtype=tf.int32,
                                      shape=[None, self.config.get("max_seq_len")])
        self.seq_len = tf.placeholder(dtype=tf.int32,
                                      shape=[None, ])
        self.ner_label = tf.placeholder(dtype=tf.int32,
                                        shape=[None, self.config.get("max_seq_len")])
        self.dropout_keep_prob = tf.placeholder(dtype=tf.float32,)

        with tf.variable_scope("embedding"):
            self.embedding = tf.Variable(self.init_embedding,
                                         dtype=tf.float32,
                                         name="embedding")

        with tf.variable_scope("lstm"):
            seq_len = tf.cast(self.seq_len, tf.int64)
            seq_embedding = tf.nn.embedding_lookup(self.embedding, self.seq_idx)
            for num in range(self.config.get('layer_size')):
                self.fw_cell = tf.nn.rnn_cell.BasicLSTMCell(self.config.get('lstm_dim'))
                self.bw_cell = tf.nn.rnn_cell.BasicLSTMCell(self.config.get('lstm_dim'))

                (forward_output, backward_output), _ = tf.nn.bidirectional_dynamic_rnn(
                    self.fw_cell,
                    self.bw_cell,
                    seq_embedding,
                    time_major=False,
                    dtype=tf.float32,
                    sequence_length=seq_len,
                    scope='layer_' + str(num)
                )
                output = tf.concat(values=[forward_output, backward_output],
                                   axis=2)
                lstm_output = tf.nn.dropout(output, keep_prob=self.dropout_keep_prob)

        self.logits, self.transition_params, self.loss = self.forward(
            self.config, lstm_output, self.ner_label, self.seq_len
        )

        with tf.variable_scope("train_ops"):
            self.optimizer = tf.train.AdamOptimizer(self.config.get('lr'))
            self.global_step = tf.Variable(0, name="global_step", trainable=False)

            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars),
                                              self.config.get('gradient_clip'))
            self.train_op = self.optimizer.apply_gradients(zip(grads, tvars),
                                                           global_step=self.global_step)

    def train_step(self, sess, x_batch, y_batch, seq_len_batch):
        feed_dict = {
            self.seq_idx: x_batch,
            self.ner_label: y_batch,
            self.seq_len: seq_len_batch,

        }
        _, step, loss = sess.run(
            [self.train_op, self.global_step, self.loss],
            feed_dict)

        return step, loss

    def decode(self, sess, x, seq_len):
        feed_dict = {
            self.seq_idx: x,
            self.seq_len: seq_len,
            self.dropout_keep_prob: 1.0
        }

        logits, transition_params = sess.run(
            [self.logits, self.transition_params], feed_dict)

        y_pred = []
        for logits_, seq_len_ in zip(logits, seq_len):
            logits_ = logits_[:seq_len_]
            viterbi_sequence, _ = tf.contrib.crf.viterbi_decode(
                logits_, transition_params)
            y_pred.append(viterbi_sequence)

        return y_pred

    def save(self, session, dir_path):
        import os
        if not (os.path.isdir(dir_path)):
            os.mkdir(dir_path)
        fp = dir_path + "/best_model"
        self.saver.save(session, fp)
        LogInfo.logs("Model saved into %s.", fp)

    def load(self, session, fp):
        LogInfo.logs("Loading Model from %s", fp)
        self.saver.restore(session, fp)
        LogInfo.logs("Model loaded from %s", fp)


if __name__ == "__main__":
    print()
