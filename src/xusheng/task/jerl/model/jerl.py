"""
Main model for Joint Entity Recognition and Linking (JERL)
Written by Xusheng Luo
"""

import tensorflow as tf
import numpy as np

from xusheng.task.jerl.model.ner import NER
from xusheng.task.jerl.model.el import EntityLinker
from xusheng.model.attention import AttentionLayerBahdanau
from xusheng.util.log_util import LogInfo


class JERL(object):

    def __init__(self, config, mode, init_embedding):
        self.config = config
        self.mode = mode
        self.init_embedding = init_embedding
        self.transfer = config.get("transfer")
        self.__build_graph()
        self.saver = tf.train.Saver()

    def __build_graph(self):
        # Common initial part for NER and EL
        self.seq_idx = tf.placeholder(dtype=tf.int32,
                                      shape=[None, self.config.get("max_seq_len")])
        self.seq_len = tf.placeholder(dtype=tf.int32,
                                      shape=[None, ])
        self.ner_label = tf.placeholder(dtype=tf.int32,
                                        shape=[None, self.config.get("max_seq_len")])
        self.dropout_keep_prob = tf.placeholder(dtype=tf.float32, )

        with tf.variable_scope("embedding"):
            self.embedding = tf.Variable(self.init_embedding,
                                         dtype=tf.float32,
                                         name="embedding")
        with tf.variable_scope("parameter"):
            self.W = tf.get_variable(
                shape=[self.config.get('lstm_dim') * 2, self.config.get('num_classes')],
                initializer=tf.truncated_normal_initializer(stddev=0.01),
                dtype=tf.float32,
                name="weights",
                regularizer=tf.contrib.layers.l2_regularizer(self.config.get('l2_reg_lambda'))
            )
            self.b = tf.Variable(tf.zeros([self.config.get('num_classes')],
                                          dtype=tf.float32,
                                          name="bias"))

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

        # ============================  NER ============================ #
        self.ner_logits, self.transition_params, self.ner_loss = NER.forward(
            self.config, lstm_output, self.ner_label, self.seq_len
        )
        # ======================== Entity Linking ======================= #

        self.el_loss = EntityLinker.forward()

        # ============================= Joint =========================== #

        # one option: co-train two losses
        self.loss = self.ner_loss + self.el_loss

        # another option may be train one of loss for each batch step

        with tf.variable_scope("train_ops"):
            self.optimizer = tf.train.AdamOptimizer(self.config.get('lr'))
            self.global_step = tf.Variable(0, name="global_step", trainable=False)

            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars),
                                              self.config.get('gradient_clip'))
            self.train_op = self.optimizer.apply_gradients(zip(grads, tvars),
                                                           global_step=self.global_step)

    def save(self, session, dir_path):
        import os
        if not(os.path.isdir(dir_path)):
            os.mkdir(dir_path)
        fp = dir_path + "/best_model"
        self.saver.save(session, fp)
        LogInfo.logs("Model saved into %s.", fp)

    def load(self, session, fp):
        LogInfo.logs("Loading Model from %s", fp)
        self.saver.restore(session, fp)
        LogInfo.logs("Model loaded from %s", fp)

