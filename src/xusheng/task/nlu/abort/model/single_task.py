"""
Single-task model for named entity recognition
char+word embedding + bi-LSTM + CRF
"""

import tensorflow as tf

from xusheng.model.rnn_encoder import BidirectionalRNNEncoder
from xusheng.model.attention import AttentionLayerBahdanau, AttentionLayerAvg
from xusheng.model.loss import softmax_sequence_loss, hinge_loss, cosine_sim
from xusheng.util.tf_util import get_optimizer
from xusheng.util.log_util import LogInfo

class NER(object):

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

        self.batch_size = self.config.get("batch_size")

        with tf.device('/cpu:0'), tf.name_scope("embedding_layer"):
            term_embedding = tf.get_variable(
                name="embedding",
                shape=[self.config.get("vocab_size"), self.config.get("embedding_dim")],
                dtype=tf.float32,
                initializer=tf.constant_initializer(self.embedding_vocab)
            )
            self.query_embedding = tf.nn.embedding_lookup(term_embedding, self.query_idx)
            # tf.split:    Tensor -> list tensors
            # tf.stack:    list of tensors -> one tensor
            self.query_slice = [
                tf.squeeze(_input, [1])
                for _input in tf.split(self.query_embedding,
                                       self.config.get("max_seq_len"),
                                       axis=1)
            ]
            # better style: use unstack!  one tensor -> list of tensors
            # equal to the above one
            # self.query_slice = tf.unstack(self.query_embedding, axis=1)

        # bi-LSTM
        with tf.name_scope("rnn_encoder"):
            rnn_config = dict()
            key_list = ["cell_class", "num_units", "dropout_input_keep_prob",
                        "dropout_output_keep_prob", "num_layers"]
            for key in key_list:
                rnn_config[key] = self.config.get(key)
            rnn_encoder = BidirectionalRNNEncoder(rnn_config, self.mode)
            self.biLstm = rnn_encoder.encode(self.query_slice, self.query_len)

        # output dim = 2 * rnn cell dim (fw + bw)
        self.hidden_dim = self.config.get("num_units") * 2
        self.biLstm_clip = tf.clip_by_value(self.biLstm.attention_values,
                                            -self.config.get("grad_clip"),
                                            self.config.get("grad_clip"))
        # training parameters
        with tf.name_scope("parameters"):
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

        # above bi-LSTM
        self.outputs = tf.reshape(tensor=self.biLstm_clip,
                                  shape=[-1, self.hidden_dim])
        self.label_matrix = tf.nn.xw_plus_b(self.outputs, self.W_l, self.b_l)
        # [B, T, label_num]
        self.logits = tf.reshape(tensor=self.label_matrix,
                                 shape=[-1, self.config.get("max_seq_len"),
                                        self.config.get("label_num")])
        # [label_num, label_num]
        self.transition_mat = tf.get_variable(
            "transitions",
            shape=[self.config.get("label_num")+1, self.config.get("label_num")+1],
            initializer=tf.contrib.layers.xavier_initializer(uniform=True))

        # ===================================== Loss ====================================== #
        if self.mode == tf.contrib.learn.ModeKeys.TRAIN:

            # # softmax sequence loss for sequence nlu
            # self.loss = softmax_sequence_loss(logits=self.logits,
            #                                   targets=self.label,
            #                                   sequence_length=self.query_len)
            # self.loss = tf.reduce_mean(self.loss)

            # padding logits for crf loss, length += 1
            small = -1000.0
            start_logits = tf.concat(
                [small * tf.ones(shape=[self.batch_size, 1, self.config.get("label_num")]),
                 tf.zeros(shape=[self.batch_size, 1, 1])],
                axis=-1
            )
            LogInfo.logs(start_logits.get_shape().as_list())
            pad_logits = tf.cast(small * tf.ones([self.batch_size,
                                                  self.config.get("max_seq_len"), 1]), tf.float32)
            LogInfo.logs(pad_logits.get_shape().as_list())
            self.logits = tf.concat([self.logits, pad_logits], axis=-1)
            self.logits = tf.concat([start_logits, self.logits], axis=1)
            LogInfo.logs(self.logits.get_shape().as_list())
            targets = tf.concat(
                [tf.cast(self.config.get("label_num")*tf.ones([self.batch_size, 1]),
                         tf.int32),
                 self.label], axis=-1
            )
            LogInfo.logs(targets.get_shape().as_list())

            # CRF layer
            self.log_likelihood, self.transition_mat = \
                tf.contrib.crf.crf_log_likelihood(
                    inputs=self.logits,
                    tag_indices=targets,
                    transition_params=self.transition_mat,
                    sequence_lengths=self.query_len+1)
            self.loss = tf.reduce_mean(-self.log_likelihood)

            # train op
            self.global_step = tf.Variable(0, name="global_step",  trainable=False)
            optimizer = get_optimizer(self.config.get("optimizer"), self.config.get("lr"))
            grads_and_vars = optimizer.compute_gradients(self.loss)
            self.train_op = optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)

    def train(self, session, batch_data):
        query_idx, query_len, label, _, _, _ = batch_data
        feed_dict = {
            self.query_idx: query_idx,
            self.query_len: query_len,
            self.label: label,
        }

        return_list = session.run([self.train_op,
                                   self.loss,
                                   self.logits,
                                   self.transition_mat], feed_dict)

        return return_list

    def eval(self, session, eval_data):
        query_idx, query_len = eval_data
        feed_dict = {
            self.query_idx: query_idx,
            self.query_len: query_len,
        }

        return_list = session.run([self.logits], feed_dict)

        return return_list

    def decode(self, logits, lengths):
        """
        :param logits: [B, T, num_tags]float32, logits
        :param lengths: [B]int32, real length of each sequence
        :return: sequence of tags of real lengths
        """
        # inference final labels usa Viterbi Algorithm
        paths = []
        small = -1000.0
        start = np.asarray([[small]*self.config.get("label_num") + [0]])
        matrix = self.transition_mat.eval()
        for score, length in zip(logits, lengths):
            score = score[:length]
            pad = small * np.ones([length, 1])
            logits = np.concatenate([score, pad], axis=1)
            logits = np.concatenate([start, logits], axis=0)
            path, _ = tf.contrib.crf.viterbi_decode(logits, matrix)

            paths.append(path[1:])
        return paths

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
    model = NER(config=config,
                mode=tf.contrib.learn.ModeKeys.TRAIN,
                embedding_vocab=np.array([[1, 2]]))
    LogInfo.logs("Model compiled successfully!")


