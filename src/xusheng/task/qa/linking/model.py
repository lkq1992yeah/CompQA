"""
Entity Linking for Web Q.
train on <mention, entity> pairs, give a score
"""

import tensorflow as tf

class EntityLinker(object):

    def __init__(self, config):
        self.config = config
        self._forward()
        self.saver = tf.train.Saver()

    def _forward(self):
        self.feats = tf.placeholder(dtype=tf.float32,
                                    shape=[None, self.config.get("feat_num")])
        self.labels = tf.placeholder(dtype=tf.int32,
                                     shape=[None, 1])

        with tf.variable_scope("parameter"):
            self.W = tf.get_variable(name="el_W",
                                     shape=[self.config.get("feat_num"), 1],
                                     dtype=tf.float32,
                                     initializer=tf.truncated_normal_initializer)
            self.b = tf.get_variable(name="el_b",
                                     shape=[1],
                                     dtype=tf.float32,
                                     initializer=tf.truncated_normal_initializer)

        self.logits = tf.nn.xw_plus_b(self.feats, self.W, self.b)
        self.labels = tf.cast(self.labels, tf.float32)

        with tf.variable_scope("loss"):
            self.loss = tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.logits,
                labels=self.labels
            )
            self.loss = tf.reduce_mean(self.loss)

        with tf.variable_scope("train_ops"):
            self.optimizer = tf.train.AdamOptimizer(self.config.get("lr"))

            self.global_step = tf.Variable(0, name="global_step", trainable=False)

            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars),
                                              self.config.get("gradient_clip"))
            self.train_op = self.optimizer.apply_gradients(zip(grads, tvars),
                                                           global_step=self.global_step)

    def train(self, sess, x_batch, y_batch):
        feed_dict = {
            self.feats: x_batch,
            self.labels: y_batch
        }

        _, step, loss = sess.run(
            [self.train_op, self.global_step, self.loss],
            feed_dict
        )

        return step, loss

    def test(self, sess, x):
        feed_dict = {
            self.feats: x
        }

        logits = sess.run([self.logits], feed_dict)
        rets = tf.nn.sigmoid(logits)

        return rets

    def save(self, sess, dir_path):
        import os
        if not(os.path.isdir(dir_path)):
            os.mkdir(dir_path)
        fp = dir_path + "/best_model"
        return self.saver.save(sess, fp)

    def load(self, sess, fp):
        self.saver.restore(sess, fp)

