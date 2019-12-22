"""
Multi-task model for intent detection, sequence labeling and entity linking
Intent detection: 3 classes including Pinlei, Property-key, Property-value
Sequence labeling: 3 tags including Pinlei, Property-key, Property-value
Entity Linking: link all tags to KG entities
"""

import tensorflow as tf

from xusheng.model.rnn_encoder import BidirectionalRNNEncoder
from xusheng.model.attention import AttentionLayerBahdanau, AttentionLayerAvg
from xusheng.model.loss import softmax_sequence_loss, hinge_loss, cosine_sim
from xusheng.util.tf_util import get_optimizer
from xusheng.util.log_util import LogInfo


class MultiTaskModel(object):

    def __init__(self, config, mode, embedding_vocab):
        self.config = config
        self.mode = mode
        self.embedding_vocab = embedding_vocab
        self.transfer = config.get("transfer")
        self._build_graph()
        self.saver = tf.train.Saver()

    def _build_graph(self):
        self.query_idx = tf.placeholder(dtype=tf.int32,
                                        shape=[None, self.config.get("max_seq_len")])
        self.query_len = tf.placeholder(dtype=tf.int32,
                                        shape=[None, ])
        self.label = tf.placeholder(dtype=tf.int32,
                                    shape=[None, self.config.get("max_seq_len")])
        self.intent = tf.placeholder(dtype=tf.int32,
                                     shape=[None, ])
        self.link_mask = tf.placeholder(dtype=tf.int32,
                                        shape=[None, self.config.get("max_seq_len")])
        self.entity_idx = tf.placeholder(dtype=tf.int32,
                                         shape=[None, self.config.get("PN")])

        with tf.device('/cpu:0'), tf.name_scope("embedding_layer"):
            term_embedding = tf.get_variable(
                name="embedding",
                shape=[self.config.get("vocab_size"), self.config.get("embedding_dim")],
                dtype=tf.float32,
                initializer=tf.constant_initializer(self.embedding_vocab)
            )
            self.query_embedding = tf.nn.embedding_lookup(term_embedding, self.query_idx)
            self.entity_embedding = tf.nn.embedding_lookup(term_embedding, self.entity_idx)
            # tf.split:    Tensor -> list tensors
            # tf.stack:    list of tensors -> list of tensors
            self.query_slice = [
                tf.squeeze(_input, [1])
                for _input in tf.split(self.query_embedding,
                                       self.config.get("max_seq_len"),
                                       axis=1)
            ]

        # bi-LSTM
        with tf.name_scope("rnn_encoder"):
            rnn_config = dict()
            key_list = ["cell_class", "num_units", "dropout_input_keep_prob",
                        "dropout_output_keep_prob", "num_layers"]
            for key in key_list:
                rnn_config[key] = self.config.get(key)
            rnn_encoder = BidirectionalRNNEncoder(rnn_config, self.mode)
            self.encoder_output = rnn_encoder.encode(self.query_slice, self.query_len)

        # hidden representation for intent detection
        with tf.name_scope("intent_hidden"):
            # average attention
            att_config = dict()
            key_list = ["num_units"]
            for key in key_list:
                att_config[key] = self.config.get(key)

            att = AttentionLayerAvg()
            self.query_hidden_avg = att.build(self.encoder_output.attention_values,
                                              self.encoder_output.attention_values_length)

        self.hidden_dim = self.query_hidden_avg.get_shape().as_list()[-1]

        # training parameters
        with tf.name_scope("parameters"):
            self.W_i = tf.get_variable(name="W_i",
                                       shape=[self.hidden_dim,
                                              self.config.get("intent_num")],
                                       dtype=tf.float32,
                                       initializer
                                       =tf.contrib.layers.xavier_initializer(uniform=True))
            self.b_i = tf.get_variable(name="b_i",
                                       shape=[self.config.get("intent_num")],
                                       dtype=tf.float32,
                                       initializer=tf.constant_initializer(0.0))
            self.W_l = tf.get_variable(name="W_l",
                                       shape=[self.hidden_dim,
                                              self.config.get("label_num")],
                                       dtype=tf.float32,
                                       initializer
                                       =tf.contrib.layers.xavier_initializer(uniform=True))
            self.b_l = tf.get_variable(name="b_l",
                                       shape=[self.config.get("label_num")],
                                       dtype=tf.float32,
                                       initializer=tf.constant_initializer(0.0))
            self.W_e = tf.get_variable(name="W_e",
                                       shape=[self.hidden_dim*2,
                                              self.config.get("embedding_dim")],
                                       dtype=tf.float32,
                                       initializer
                                       =tf.contrib.layers.xavier_initializer(uniform=True))
            self.b_e = tf.get_variable(name="b_e",
                                       shape=[self.config.get("embedding_dim")],
                                       dtype=tf.float32,
                                       initializer=tf.constant_initializer(0.0))

        # above bi-LSTM

        # ---------------------------------- Intent Detection --------------------------- #
        self.intent_layer = tf.nn.xw_plus_b(self.query_hidden_avg, self.W_i, self.b_i)

        # ---------------------------------- Sequence Labeling -------------------------- #
        self.outputs = tf.reshape(tensor=self.encoder_output.outputs,
                                  shape=[-1, self.hidden_dim])
        self.label_layer = tf.nn.xw_plus_b(self.outputs, self.W_l, self.b_l)
        # [B, T, class_num]
        self.label_layer = tf.reshape(tensor=self.label_layer,
                                      shape=[-1, self.config.get("max_seq_len"),
                                             self.config.get("label_num")])

        # ---------------------------------- Entity Linking--- -------------------------- #
        """
        notice that entity linking in evaluation step is based on the result of sequence nlu
        so we do two-step evaluation
        """

        # [B, h_dim]
        self.mention = add_mask_then_avg(self.encoder_output.attention_values, self.link_mask)
        # [B, h_dim]
        self.context = add_mask_then_avg(self.encoder_output.attention_values, 1-self.link_mask)
        # [B, w2v_dim]
        self.left = tf.nn.xw_plus_b(tf.concat([self.mention, self.context], axis=1),
                                    self.W_e,
                                    self.b_e)
        # [B, 1, w2v_dim]
        self.left = tf.expand_dims(self.left, axis=1)
        # [B, PN, w2v_dim]
        self.left = tf.tile(self.left, multiples=[1, self.config.get("PN"), 1])
        # [B*PN, w2v_dim]
        self.left = tf.reshape(self.left, shape=[-1, self.config.get("embedding_dim")])
        # [B*PN, w2v_dim]
        self.right = tf.reshape(self.entity_embedding,
                                shape=[-1, self.config.get("embedding_dim")])

        # [B*PN, ]
        self.link_score = cosine_sim(self.left, self.right)

        # ===================================== Loss ====================================== #
        if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
            # loss for intent detection
            self.intent_loss = \
                tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.intent_layer,
                                                               labels=self.intent,
                                                               name="intent_loss")
            self.intent_loss = tf.reduce_mean(self.intent_loss)

            # loss for sequence nlu
            self.label_loss = softmax_sequence_loss(logits=self.label_layer,
                                                    targets=self.label,
                                                    sequence_length=self.query_len)
            self.label_loss = tf.reduce_mean(self.label_loss)

            # loss for entity linking
            self.link_loss = hinge_loss(scores=self.link_score,
                                        row=self.config.get("batch_size"),
                                        col=self.config.get("PN"),
                                        margin=self.config.get("margin"))

            # train op, currently three losses have equal weights
            self.train_op = get_optimizer(self.config.get("optimizer"),
                                          self.config.get("lr")).minimize(self.intent_loss +
                                                                          self.label_loss +
                                                                          self.link_loss)

    def train(self, session, batch_data):
        query_idx, query_len, label, intent, link_mask, entity_idx = batch_data
        feed_dict = {
            self.query_idx: query_idx,
            self.query_len: query_len,
            self.label: label,
            self.intent: intent,
            self.link_mask: link_mask,
            self.entity_idx: entity_idx
        }

        return_list = session.run([self.train_op,
                                   self.intent_loss,
                                   self.label_loss,
                                   self.link_loss,
                                   self.intent_layer,
                                   self.label_layer,
                                   self.link_score], feed_dict)

        return return_list

    def eval_intent_and_label(self, session, eval_data):
        query_idx, query_len, label, intent = eval_data
        feed_dict = {
            self.query_idx: query_idx,
            self.query_len: query_len,
        }

        return_list = session.run([self.intent_layer,
                                   self.label_layer], feed_dict)

        return return_list

    def eval_link(self, session, eval_data):
        """
        notice "link_mask" in eval_data is based on previous
        sequence nlu result
        "entity idx" is the output of candidate generation
        """
        query_idx, query_len, link_mask, entity_idx = eval_data
        feed_dict = {
            self.query_idx: query_idx,
            self.query_len: query_len,
            self.link_mask: link_mask,
            self.entity_idx: entity_idx
        }

        return_list = session.run([self.link_score], feed_dict)

        return return_list

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
    model = MultiTaskModel(config=config,
                           mode=tf.contrib.learn.ModeKeys.TRAIN,
                           embedding_vocab=np.array([[1, 2]]))
    LogInfo.logs("Model compiled successfully!")


