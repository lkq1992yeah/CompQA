import numpy as np
import tensorflow as tf

from kangqi.util.LogUtil import LogInfo
from kangqi.util.tf.cosine_sim import get_cosine_tf, get_length_tf
from xusheng.util.tf_util import noise_weight_variable, noise_bias_variable


# unfinished, used Kangqi's non-joint model in CIKM submission

class NonJointModel(object):

    def __init__(self, pre_train, batch_size, margin, rows, columns, PN,
                 d_w2v, d_vec, d_hidden, d_final, lr):

        # initialize parameters
        self.batch_size = batch_size
        self.rows = rows
        self.columns = columns
        self.PN = PN
        self.margin = margin
        self.d_w2v = d_w2v
        self.d_vec = d_vec
        self.d_hidden = d_hidden
        self.d_final = d_final
        self.transfer = tf.nn.tanh
        self.optimizer = tf.train.AdamOptimizer(lr)
        self.running_ret = dict()

        # input: batch size * rows * columns * w2v_dim
        # note that rows & columns are fixed!!! that's a problem of jointly modeling_2017 of table data!
        self.cells = tf.placeholder(tf.float32, [None, self.d_w2v])
        self.entities = tf.placeholder(tf.float32, [None, self.d_w2v])
        self.length = self.rows + self.columns - 2
        self.contexts = tf.placeholder(tf.float32, [None, self.d_w2v])
        # OLD self.contexts = tf.placeholder(tf.float32, [None, 1, self.length, self.d_w2v])

        self.sess = tf.Session()

        # build the model graph
        self._build_graph()

        self.saver = tf.train.Saver()
        # initialize global variables & start a session
        init = tf.global_variables_initializer()
        self.sess.run(init)

        if pre_train != "random":
            LogInfo.logs("Loading pre-trained translation weights...")
            self._get_translation_weights(pre_train)
        else:
            LogInfo.logs("W/o pre-trained translation weights.")

    def _initialize_weights(self):
        weights = dict()

        # weight matrix for translation
        weights['w_trans'] = noise_weight_variable([self.d_w2v, self.d_w2v], 'w_trans')
        weights['b_trans'] = noise_bias_variable([self.d_w2v], 'b_trans')

        # NOT USED! weight matrix from cell_w2v -> cell_vec
        weights['wc'] = noise_weight_variable([self.d_w2v, self.d_vec], 'wc')
        weights['bc'] = noise_bias_variable([self.d_vec], 'bc')

        # NOT USED! weight matrix from entity_w2v -> ent_vec
        weights['we'] = noise_weight_variable([self.d_w2v, self.d_vec], 'we')
        weights['be'] = noise_bias_variable([self.d_vec], 'be')

        # weight matrix from concat(cell_vec, context_vec) -> left mention feature
        weights['w1'] = noise_weight_variable([self.d_vec * 2, self.d_hidden], 'w1')
        weights['b1'] = noise_bias_variable([self.d_hidden], 'b1')

        # weight matrix from ent_vec -> right entity feature
        weights['w2'] = noise_weight_variable([self.d_vec, self.d_hidden], 'w2')
        weights['b2'] = noise_bias_variable([self.d_hidden], 'b2')

        return weights

    def _get_translation_weights(self, source):

        # weight_fp = "/home/kangqi/workspace/PythonProject/runnings/tabel/translation" \
        #             "/try_0/weights.Mlinear-cosine_D100_lr0.0050_reg0.0000"
        if source == "common":
            weight_fp = "/home/xusheng/TabelProject/data/weight/weights.from_common_words"
        else:
            weight_fp = "/home/xusheng/TabelProject/data/weight/weights.from_wiki_inter_link"
        with open(weight_fp, 'rb') as fin:
            W_value = np.load(fin)
            b_value = np.load(fin)
        self.sess.run([self.weights['w_trans'].assign(W_value),
                       self.weights['b_trans'].assign(b_value)])

    def _build_mention_vec(self):

        # shape: (batch size, d_vec)
        # self.cell_vec = self.transfer(tf.add(tf.matmul(self.cells, self.weights['wc']),
        #                                      self.weights['bc']))
        self.cell_vec = tf.add(tf.matmul(self.cells, self.weights['w_trans']),
                               self.weights['b_trans'])

        self.context_vec = tf.add(tf.matmul(self.contexts, self.weights['w_trans']),
                                  self.weights['b_trans'])

        self.mention_vec = self.transfer(tf.add(tf.matmul(tf.concat([self.cell_vec,
                                                                     self.context_vec], 1),
                                                          self.weights['w1']),
                                                self.weights['b1']))

    def _build_entity_vec(self):
        # shape: (batch size * rows * columns, d_vec)

        self.entity_vec = self.transfer(tf.add(tf.matmul(self.entities, self.weights['w2']),
                                               self.weights['b2']))

    def _build_score_layer(self):
        # cosine similarity
        self.score_layer = get_cosine_tf(self.mention_vec, self.entity_vec,
                                         get_length_tf(self.mention_vec), get_length_tf(self.entity_vec))
        # ------------------------- LOSS -------------------------------- #
        self.scores = tf.reshape(self.score_layer, [-1, self.PN])
        self.scores_pos = tf.slice(self.scores, [0, 0], [self.batch_size / self.PN, 1])
        self.scores_neg = tf.slice(self.scores, [0, 1], [self.batch_size / self.PN, self.PN - 1])

        self.loss_matrix = tf.maximum(0., self.margin - self.scores_pos + self.scores_neg)
        # we could also use tf.nn.relu here
        self.loss = tf.reduce_mean(self.loss_matrix)
        self.opt_op = self.optimizer.minimize(self.loss)
        # ------------------------- LOSS -------------------------------- #

    def _build_graph(self): pass

    def save(self, directory, idx):
        import os
        if not(os.path.isdir(directory)):
            os.mkdir(directory)
        fp =  directory + "/" + str(idx)
        LogInfo.logs("Saving Model into %s...", fp)
        self.saver.save(self.sess, fp)
        LogInfo.logs("Model saved into %s.", fp)

    def load(self, fp):
        LogInfo.logs("Loading Model from %s", fp)
        self.saver.restore(self.sess, fp)
        LogInfo.logs("Model loaded from %s", fp)

    def check_weight(self):
        run_list = [self.weights['w_trans']]
        ret = self.sess.run(run_list)
        self.running_ret['w_trans'] = ret[0]
        return self.running_ret

    def train(self, input_data):
        # get feed_dict from input_data & run the model
        run_list = [self.opt_op, self.loss, self.weights['w_trans']]
        ret = self.sess.run(run_list, feed_dict={
            self.cells: input_data['cell'],
            self.entities: input_data['entity'],
            self.contexts: input_data['context'],
        })
        self.running_ret['loss'] = ret[1]
        self.running_ret['w_trans'] = ret[2]

        return self.running_ret

    def eval(self, input_data):
        ret = self.sess.run([self.score_layer, self.weights], feed_dict={
            self.cells: input_data['cell'],
            self.entities: input_data['entity'],
            self.contexts: input_data['context']
        })
        self.running_ret['eval_score'] = ret[0]
        self.running_ret['weights_eval'] = ret[1]
        return self.running_ret


