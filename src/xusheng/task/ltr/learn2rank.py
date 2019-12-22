"""
Learn to rank by ranknet
"""

import tensorflow as tf

from kangqi.util.tf.ranknet.ranknet_improved import RankNet
from xusheng.util.log_util import LogInfo

class Learn2Ranker(object):

    def __init__(self, config, mode):
        self.config = config
        self.mode = mode
        self.hidden_dim = config.get("hidden_dim")
        self._build_graph()
        self.saver = tf.train.Saver()

    def _build_graph(self):
        self.X = tf.placeholder(dtype=tf.float32,
                                shape=[None, self.config.get("list_len"), self.config.get("feat_size")])
        self.Y_true = tf.placeholder(dtype=tf.float32,
                                     shape=[None, self.config.get("list_len")])

        # training parameters
        with tf.name_scope("parameters"):
            self.W_1 = tf.get_variable(name="W_1",
                                       shape=[self.config.get("feat_size"), self.hidden_dim],
                                       dtype=tf.float32,
                                       initializer
                                       =tf.contrib.layers.xavier_initializer(uniform=True))
            self.b_1 = tf.get_variable(name="b_1",
                                       shape=[self.hidden_dim],
                                       dtype=tf.float32,
                                       initializer=tf.constant_initializer(0.0))
            self.W_2 = tf.get_variable(name="W_2",
                                       shape=[self.hidden_dim, self.hidden_dim],
                                       dtype=tf.float32,
                                       initializer
                                       =tf.contrib.layers.xavier_initializer(uniform=True))
            self.b_2 = tf.get_variable(name="b_2",
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
        # 3-layer fully connected
        with tf.name_scope("score"):
            self.X = tf.reshape(self.X, shape=[-1, self.config.get("feat_size")])
            self.layer_1 = tf.nn.relu(tf.nn.xw_plus_b(self.X, self.W_1, self.b_1))
            self.layer_2 = tf.nn.relu(tf.nn.xw_plus_b(self.layer_1, self.W_2, self.b_2))
            self.output_layer = tf.nn.xw_plus_b(self.layer_2, self.W_o, self.b_o)

        # ranknet loss
        if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
            self.scores = tf.reshape(self.output_layer, shape=[-1,
                                                               self.config.get("list_len"),
                                                               self.config.get("feat_size")])
            ranknet = RankNet(self.config.get("batch_size"), self.config.get("list_len"),
                              self.config.get("lr"), gold_method='binary')
            mask = tf.ones(shape=[self.config.get("batch_size"),
                                  self.config.get("list_len")],
                           dtype='float32')
            # ranknet code uses Adam optimizer by default
            self.loss, self.opt_op = ranknet.build(self.scores, self.Y_true, mask)

    def train(self, session, batch_data):
        X, Y_true = batch_data
        feed_dict = {
            self.X: X,
            self.Y_true: Y_true,
        }

        _, loss = session.run([self.opt_op,
                               self.loss], feed_dict)

        return loss

    def eval(self, session, eval_data):
        X, = eval_data
        feed_dict = {
            self.X: X
        }

        score_matrix = session.run(self.scores, feed_dict)

        return score_matrix

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
    model = Learn2Ranker(config=config,
                         mode=tf.contrib.learn.ModeKeys.TRAIN)
