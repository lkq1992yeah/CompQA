"""
Main model to identify user intention for multi-pinlei query
"""

import tensorflow as tf

from xusheng.model.rnn_encoder import BidirectionalRNNEncoder
from xusheng.model.attention import AttentionLayerBahdanau_old, AttentionLayerAvg_old
from xusheng.model.loss import hinge_loss
from xusheng.util.log_util import LogInfo
from xusheng.util.tf_util import get_optimizer


class IntentionIdentifier(object):

    def __init__(self, config, mode, embedding_vocab):
        self.config = config
        self.mode = mode
        self.embedding = embedding_vocab
        self.transfer = config.get("transfer")
        self._build_graph()
        self.saver = tf.train.Saver()

    def _build_graph(self):
        self.context_idx = tf.placeholder(dtype=tf.int32,
                                          shape=[None, self.config.get("max_seq_len")])
        self.context_seq = tf.placeholder(dtype=tf.int32,
                                          shape=[None, ])
        self.pinlei_idx = tf.placeholder(dtype=tf.int32,
                                         shape=[None, ])

        with tf.device('/cpu:0'), tf.name_scope("embedding_layer"):
            # LogInfo.logs("Embedding shape: %s (%d*%d).", self.embedding.shape,
            #              self.config.get("vocab_size"), self.config.get("embedding_dim"))
            term_embedding = tf.get_variable(
                name="embedding",
                shape=[self.config.get("vocab_size"), self.config.get("embedding_dim")],
                dtype=tf.float32,
                initializer=tf.constant_initializer(self.embedding)
            )
            self.context_embedding = tf.nn.embedding_lookup(term_embedding, self.context_idx)
            self.pinlei_embedding = tf.nn.embedding_lookup(term_embedding, self.pinlei_idx)
            # shape = [max_seq_len, batch_size, embedding_dim], feed to rnn_encoder
            self.context_slice = [
                tf.squeeze(_input, [1])
                for _input in tf.split(self.context_embedding,
                                       self.config.get("max_seq_len"),
                                       axis=1)
            ]

        # bi-LSTM
        with tf.name_scope("rnn_encoder"):
            rnn_config = dict()
            key_list = ["cell_class", "num_units", "dropout_input_keep_prob",
                        "dropout_output_keep_prob", "num_layers", "reuse"]
            for key in key_list:
                rnn_config[key] = self.config.get(key)
            rnn_encoder = BidirectionalRNNEncoder(rnn_config, self.mode)
            self.encoder_output = rnn_encoder.encode(self.context_slice, self.context_seq)

        # attention mechanism
        with tf.name_scope("attention"):
            att_config = dict()
            key_list = ["num_units"]
            for key in key_list:
                att_config[key] = self.config.get(key)

            if self.config.get("attention") == "bah":
                att = AttentionLayerBahdanau_old(att_config)
                self.query_hidden = att.build(self.pinlei_embedding,
                                              self.encoder_output.attention_values,
                                              self.encoder_output.attention_values_length)
            elif self.config.get("attention") == "avg":
                att = AttentionLayerAvg_old()
                self.query_hidden = att.build(self.encoder_output.attention_values,
                                              self.encoder_output.attention_values_length)

        self.hidden_dim = self.query_hidden.get_shape().as_list()[-1]

        # training parameters
        with tf.name_scope("parameters"):
            self.W_p = tf.get_variable(name="W_p",
                                       shape=[self.config.get("embedding_dim"), self.hidden_dim],
                                       dtype=tf.float32,
                                       initializer
                                       =tf.contrib.layers.xavier_initializer(uniform=True))
            self.b_p = tf.get_variable(name="b_p",
                                       shape=[self.hidden_dim],
                                       dtype=tf.float32,
                                       initializer=tf.constant_initializer(0.0))
            self.W_f = tf.get_variable(name="W_f",
                                       shape=[self.hidden_dim * 2, self.hidden_dim],
                                       dtype=tf.float32,
                                       initializer
                                       =tf.contrib.layers.xavier_initializer(uniform=True))
            self.b_f = tf.get_variable(name="b_f",
                                       shape=[self.hidden_dim],
                                       dtype=tf.float32,
                                       initializer=tf.constant_initializer(0.0))
            self.W_o = tf.get_variable(name="W_o",
                                       shape=[self.hidden_dim, 1],
                                       dtype=tf.float32,
                                       initializer
                                       =tf.contrib.layers.xavier_initializer(uniform=True))
            self.b_o = tf.get_variable(name="b_o",
                                       shape=[1],
                                       dtype=tf.float32,
                                       initializer=tf.constant_initializer(0.0))
        # above bi-LSTM + attention
        with tf.name_scope("score"):
            self.pinlei_hidden = self.transfer(tf.add(tf.matmul(self.pinlei_embedding, self.W_p), self.b_p))
            self.final = self.transfer(tf.add(tf.matmul(tf.concat([self.query_hidden, self.pinlei_hidden], 1),
                                                        self.W_f),
                                              self.b_f))
            # self.score = tf.add(tf.matmul(self.final, self.W_o), self.b_o)  # tensorflow 1.0.0
            self.score = tf.nn.xw_plus_b(self.final, self.W_o, self.b_o)

        # hinge loss
        if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
            self.loss = hinge_loss(self.score,
                                   int(self.config.get("batch_size") / self.config.get("PN")),
                                   self.config.get("PN"),
                                   self.config.get("margin"))
            self.train_op = get_optimizer(self.config.get("optimizer"),
                                          self.config.get("lr")).minimize(self.loss)

    def train(self, session, batch_data):
        context_idx, context_seq, pinlei_idx = batch_data
        feed_dict = {
            self.context_idx: context_idx,
            self.context_seq: context_seq,
            self.pinlei_idx: pinlei_idx
        }

        return_list = session.run([self.train_op,
                                   self.loss,
                                   self.score,
                                   self.context_embedding,
                                   self.pinlei_embedding,
                                   self.context_slice,
                                   self.encoder_output,
                                   self.query_hidden,
                                   self.pinlei_hidden], feed_dict)

        return return_list

    def eval(self, session, eval_data):
        context_idx, context_seq, pinlei_idx = eval_data
        feed_dict = {
            self.context_idx: context_idx,
            self.context_seq: context_seq,
            self.pinlei_idx: pinlei_idx
        }

        score_list = session.run(self.score, feed_dict)

        return score_list

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


if __name__ == "__main__":
    import sys
    import numpy as np
    from xusheng.util.config import ConfigDict
    config = ConfigDict("runnings/%s/%s/param_config"
                        % (sys.argv[1], sys.argv[2]))
    model = IntentionIdentifier(config=config,
                                mode=tf.contrib.learn.ModeKeys.TRAIN,
                                embedding_vocab=np.array([[1, 2]]))