"""
Main model for CIKM submission "Cross-Lingual Entity Linking for Web Tables"
Joint means link all cells in one table at the same time
"""

import numpy as np
import tensorflow as tf

from kangqi.util.LogUtil import LogInfo
from kangqi.util.tf.ranknet.ranknet_improved import RankNet
from xusheng.util.tf_util import noise_weight_variable, noise_bias_variable, conv2d, maxpool2d


class JointModel(object):

    def __init__(self, trans_pre_train, nonj_pre_train,
                 batch_size, margin, rows, columns, PN,
                 d_w2v, d_vec, d_hidden, d_final, lr,
                 istrain=True, keep_prob=0.5):

        # initialize parameters
        self.batch_size = batch_size
        self.rows = rows
        self.columns = columns
        self.PN = PN
        self.table_num = self.batch_size / self.PN

        self.d_w2v = d_w2v
        self.d_vec = d_vec
        self.d_hidden = d_hidden
        self.d_final = d_final

        self.transfer = tf.nn.relu

        self.margin = margin
        self.learning_rate = lr
        self.optimizer = tf.train.AdamOptimizer(lr)
        self.running_ret = dict()
        if istrain:
            self.istrain = True
        else:
            self.istrain = False
        self.keep_prob = keep_prob

        # input: batch size * rows * columns * w2v_dim
        # note that rows & columns are fixed!!! that's a problem of jointly modeling_2017 of table data!
        self.cells = tf.placeholder(tf.float32, [None, self.d_w2v])
        self.entities = tf.placeholder(tf.float32, [None, self.d_w2v])
        self.coherence = tf.placeholder(tf.float32, [None, self.d_w2v])
        self.length = self.rows + self.columns - 2
        self.contexts = tf.placeholder(tf.float32, [None, self.d_w2v])
        # corrupt ratio for each table, shape=(table_num, PN)
        self.corrupts = tf.placeholder(tf.float32, [None, self.PN])
        # OLD self.contexts = tf.placeholder(tf.float32, [None, 1, self.length, self.d_w2v])

        # get the mask to remove all zero-zero pairs
        # !!! attention the mask is based on entity table!
        x = tf.reduce_sum(self.entities, 1)
        zero = tf.constant(0, dtype=tf.float32)
        # shape = (batch size * rows * cols)
        self.where_list = tf.cast(tf.not_equal(x, zero), dtype=tf.float32)
        # shape = (batch size, rows, cols)
        self.where_table = tf.reshape(self.where_list,
                                      [-1, self.rows*self.columns])
        self.where_cell = tf.reshape(self.where_list,
                                     [-1, self.rows, self.columns])
        # shape = (batch size, 1
        self.where_sum = tf.reciprocal(tf.reduce_sum(self.where_table, 1))

        # build the model graph
        self._build_graph()

        self.saver = tf.train.Saver()

    def _initialize_weights_old(self):
        weights = dict()

        # weight matrix for translation
        weights['w_trans'] = noise_weight_variable([self.d_w2v, self.d_w2v], 'w_trans')
        weights['b_trans'] = noise_bias_variable([self.d_w2v], 'b_trans')

        # weight matrix from cell_w2v -> cell_vec
        weights['wc'] = noise_weight_variable([self.d_w2v, self.d_vec], 'wc')
        weights['bc'] = noise_bias_variable([self.d_vec], 'bc')

        # weight matrix from entity_w2v -> ent_vec
        # todo: consider gloss by a CNN layer
        weights['we'] = noise_weight_variable([self.d_w2v, self.d_vec], 'we')
        weights['be'] = noise_bias_variable([self.d_vec], 'be')

        # weight matrix from row cells & column cells -> context vec
        # todo: consider specialize header vector
        # weights['wcx'] = noise_weight_variable([1, 3, self.d_w2v, self.d_vec], 'wcx')
        # weights['bcx'] = noise_bias_variable([self.d_vec], 'bcx')

        # weight matrix from concat(cell, ent) -> feature NO.1
        weights['w1'] = noise_weight_variable([self.d_vec * 2, self.d_hidden], 'w1')
        weights['b1'] = noise_bias_variable([self.d_hidden], 'b1')

        # weight matrix from concat(context, ent) -> feature NO.2
        weights['w2'] = noise_weight_variable([self.d_vec * 2, self.d_hidden], 'w2')
        weights['b2'] = noise_bias_variable([self.d_hidden], 'b2')

        # define weight matrix from concatenation of 1~3 features -> final_vec in derived class
        weights['bf'] = noise_bias_variable([self.d_final], 'bf')

        # weight matrix from final_vec -> score_layer
        weights['wo'] = noise_weight_variable([self.d_final, 1], 'wo')
        weights['bo'] = noise_bias_variable([1], 'bo')

        return weights

    def _initialize_weights(self):
        weights = dict()

        # weight matrix for translation
        weights['w_trans'] = tf.get_variable(
            shape=[self.d_w2v, self.d_w2v],
            name='w_trans',
            dtype=tf.float32,
            initializer=tf.contrib.layers.xavier_initializer(uniform=True)
        )
        weights['b_trans'] = tf.get_variable(
            shape=[self.d_w2v],
            name='b_trans',
            dtype = tf.float32,
            initializer = tf.constant_initializer(0.0)
        )

        # weight matrix from concat(cell, ent) -> feature NO.1
        weights['w1'] = tf.get_variable(
            shape=[self.d_w2v*2, self.d_hidden],
            name='w1',
            dtype=tf.float32,
            initializer=tf.contrib.layers.xavier_initializer(uniform=True)
        )
        weights['b1'] = tf.get_variable(
            shape=[self.d_hidden],
            name='b1',
            dtype = tf.float32,
            initializer = tf.constant_initializer(0.0)
        )

        # weight matrix from concat(context, ent) -> feature NO.2
        weights['w2'] = tf.get_variable(
            shape=[self.d_w2v*2, self.d_hidden],
            name='w2',
            dtype=tf.float32,
            initializer=tf.contrib.layers.xavier_initializer(uniform=True)
        )
        weights['b2'] = tf.get_variable(
            shape=[self.d_hidden],
            name='b2',
            dtype = tf.float32,
            initializer = tf.constant_initializer(0.0)
        )

        # define weight matrix from concatenation of 1~3 features -> final_vec in derived class
        weights['bf'] = tf.get_variable(
            shape=[self.d_final],
            name='bf',
            dtype = tf.float32,
            initializer = tf.constant_initializer(0.0)
        )

        # weight matrix from final_vec -> score_layer
        weights['wo'] = tf.get_variable(
            shape=[self.d_final, 1],
            name='wo',
            dtype=tf.float32,
            initializer=tf.contrib.layers.xavier_initializer(uniform=True)
        )
        weights['bo'] = tf.get_variable(
            shape=[1],
            name='bo',
            dtype = tf.float32,
            initializer = tf.constant_initializer(0.0)
        )

        return weights

    # translation pre-train
    def _get_translation_weights_old(self, setting):
        # weight_fp = "/home/kangqi/workspace/PythonProject/runnings/tabel/translation" \
        #             "/try_0/weights.Mlinear-cosine_D100_lr0.0050_reg0.0000"
        if setting == "common":
            weight_fp = "/home/xusheng/TabelProject/data/weight/weights.from_common_words"
        elif setting == "500":
            weight_fp = "/home/kangqi/workspace/PythonProject/runnings/tabel/translation/try_5/Mlinear-cosine_D100_keep0500_lr0.0050_reg0.0000/weights"
        elif setting == "1000":
            weight_fp = "/home/kangqi/workspace/PythonProject/runnings/tabel/translation/try_5/Mlinear-cosine_D100_keep1000_lr0.0050_reg0.0000/weights"
        elif setting == "1500":
            weight_fp = "/home/kangqi/workspace/PythonProject/runnings/tabel/translation/try_5/Mlinear-cosine_D100_keep1500_lr0.0050_reg0.0000/weights"
        elif setting == "2000":
            weight_fp = "/home/kangqi/workspace/PythonProject/runnings/tabel/translation/try_5/Mlinear-cosine_D100_keep2000_lr0.0050_reg0.0000/weights"
        elif setting == "2500":
            weight_fp = "/home/kangqi/workspace/PythonProject/runnings/tabel/translation/try_5/Mlinear-cosine_D100_keep2500_lr0.0050_reg0.0000/weights"
        elif setting == "3000":
            weight_fp = "/home/kangqi/workspace/PythonProject/runnings/tabel/translation/try_5/Mlinear-cosine_D100_keep3000_lr0.0050_reg0.0000/weights"
        with open(weight_fp, 'rb') as fin:
            W_value = np.load(fin)
            b_value = np.load(fin)
        self.sess.run([self.weights['w_trans'].assign(W_value),
                       self.weights['b_trans'].assign(b_value)])
        LogInfo.logs("[model] pre-trained translation loaded from %s.", weight_fp)

    def get_translation_weights(self, session, setting):
        # weight_fp = "/home/kangqi/workspace/PythonProject/runnings/tabel/translation" \
        #             "/try_0/weights.Mlinear-cosine_D100_lr0.0050_reg0.0000"
        if setting == "common":
            weight_fp = "/home/xusheng/TabelProject/data/weight/weights.from_common_words"
        elif setting == "500":
            weight_fp = "/home/kangqi/workspace/PythonProject/runnings/tabel/translation/try_5/Mlinear-cosine_D100_keep0500_lr0.0050_reg0.0000/weights"
        elif setting == "1000":
            weight_fp = "/home/kangqi/workspace/PythonProject/runnings/tabel/translation/try_5/Mlinear-cosine_D100_keep1000_lr0.0050_reg0.0000/weights"
        elif setting == "1500":
            weight_fp = "/home/kangqi/workspace/PythonProject/runnings/tabel/translation/try_5/Mlinear-cosine_D100_keep1500_lr0.0050_reg0.0000/weights"
        elif setting == "2000":
            weight_fp = "/home/kangqi/workspace/PythonProject/runnings/tabel/translation/try_5/Mlinear-cosine_D100_keep2000_lr0.0050_reg0.0000/weights"
        elif setting == "2500":
            weight_fp = "/home/kangqi/workspace/PythonProject/runnings/tabel/translation/try_5/Mlinear-cosine_D100_keep2500_lr0.0050_reg0.0000/weights"
        elif setting == "3000":
            weight_fp = "/home/kangqi/workspace/PythonProject/runnings/tabel/translation/try_5/Mlinear-cosine_D100_keep3000_lr0.0050_reg0.0000/weights"
        with open(weight_fp, 'rb') as fin:
            W_value = np.load(fin)
            b_value = np.load(fin)
        session.run([self.weights['w_trans'].assign(W_value),
                     self.weights['b_trans'].assign(b_value)])
        LogInfo.logs("[model] pre-trained translation loaded from %s.", weight_fp)

    # non-joint weights pre-train
    def _get_nonjoint_weights_old(self, setting):
        if setting == "trans":
            weight_fp = "/home/kangqi/workspace/PythonProject/runnings/tabel/e2e/noj_model_try_6/" \
                        "Mfc_Arelu_dh100_df200_m3.00_reg0.0005_Reg0.0005_keep1.00_b20_lr0.0002/" \
                        "weights"
        elif setting == "no_trans":
            weight_fp = "/home/kangqi/workspace/PythonProject/runnings/tabel/e2e/noj_model_try_8/" \
                        "Mfc_Arelu_dh100_df200_m2.00_reg0.0005_Reg0.0010_keep1.00_b20_lr0.0002/" \
                        "weights"

        with open(weight_fp, 'rb') as fin:
            fin.readline()
            w_trans = np.load(fin)
            b_trans = np.load(fin)
            w_mention = np.load(fin)
            b_mention = np.load(fin)
            w_context = np.load(fin)
            b_context = np.load(fin)
            w_final = np.load(fin)
            len_of_pre_wf = np.shape(w_final)[0]
            len_of_model_wf = self.sess.run(tf.shape(self.weights['wf'])[0])
            LogInfo.logs("[model] len of pre_wf: %d.", len_of_pre_wf)
            LogInfo.logs("[model] len of model_wf: %d.", len_of_model_wf)
            if len_of_pre_wf < len_of_model_wf:
                LogInfo.logs("[model] pre-trained w_final shape < weights[wf]: %d.", len_of_pre_wf)
                padding_len = len_of_model_wf - len_of_pre_wf
                padding = np.random.normal(0.01, 0.01, (padding_len, self.d_final))
                w_final = np.concatenate((w_final, padding), axis=0)
                LogInfo.logs("[model] padding added: %d.", np.shape(w_final)[0])

            b_final = np.load(fin)
            w_output = np.load(fin)
            b_output = np.load(fin)
        self.sess.run([self.weights['w_trans'].assign(w_trans),
                       self.weights['b_trans'].assign(b_trans),
                       self.weights['w1'].assign(w_mention),
                       self.weights['b1'].assign(b_mention),
                       self.weights['w2'].assign(w_context),
                       self.weights['b2'].assign(b_context),
                       self.weights['wf'].assign(w_final),
                       self.weights['bf'].assign(b_final),
                       self.weights['wo'].assign(w_output),
                       self.weights['bo'].assign(b_output),
                       ])
        LogInfo.logs("[model] pre-trained weights from non-joint model loaded.")

    def get_nonjoint_weights(self, session, setting):
        if setting == "trans":
            weight_fp = "/home/kangqi/workspace/PythonProject/runnings/tabel/e2e/noj_model_try_6/" \
                        "Mfc_Arelu_dh100_df200_m3.00_reg0.0005_Reg0.0005_keep1.00_b20_lr0.0002/" \
                        "weights"
        elif setting == "no_trans":
            weight_fp = "/home/kangqi/workspace/PythonProject/runnings/tabel/e2e/noj_model_try_8/" \
                        "Mfc_Arelu_dh100_df200_m2.00_reg0.0005_Reg0.0010_keep1.00_b20_lr0.0002/" \
                        "weights"

        with open(weight_fp, 'rb') as fin:
            fin.readline()
            w_trans = np.load(fin)
            b_trans = np.load(fin)
            w_mention = np.load(fin)
            b_mention = np.load(fin)
            w_context = np.load(fin)
            b_context = np.load(fin)
            w_final = np.load(fin)
            len_of_pre_wf = np.shape(w_final)[0]
            len_of_model_wf = session.run(tf.shape(self.weights['wf'])[0])
            LogInfo.logs("[model] len of pre_wf: %d.", len_of_pre_wf)
            LogInfo.logs("[model] len of model_wf: %d.", len_of_model_wf)
            if len_of_pre_wf < len_of_model_wf:
                LogInfo.logs("[model] pre-trained w_final shape < weights[wf]: %d.", len_of_pre_wf)
                padding_len = len_of_model_wf - len_of_pre_wf
                padding = np.random.normal(0.01, 0.01, (padding_len, self.d_final))
                w_final = np.concatenate((w_final, padding), axis=0)
                LogInfo.logs("[model] padding added: %d.", np.shape(w_final)[0])

            b_final = np.load(fin)
            w_output = np.load(fin)
            b_output = np.load(fin)
        session.run([self.weights['w_trans'].assign(w_trans),
                     self.weights['b_trans'].assign(b_trans),
                     self.weights['w1'].assign(w_mention),
                     self.weights['b1'].assign(b_mention),
                     self.weights['w2'].assign(w_context),
                     self.weights['b2'].assign(b_context),
                     self.weights['wf'].assign(w_final),
                     self.weights['bf'].assign(b_final),
                     self.weights['wo'].assign(w_output),
                     self.weights['bo'].assign(b_output),
                     ])
        LogInfo.logs("[model] pre-trained weights from non-joint model loaded.")

    def _build_cell_feature(self, cells=None, entities=None):
        if cells is None:
            cells = self.cells
        if entities is None:
            entities = self.entities

        # shape: (batch size * rows * columns, d_vec)
        # self.cell_vec = self.transfer(tf.add(tf.matmul(self.cells, self.weights['wc']),
        #                                      self.weights['bc']))
        # notice that there is no transfer function, same as training the translation matrix
        # it's just a linear transformation
        self.cell_vec = tf.add(tf.matmul(cells, self.weights['w_trans']),
                               self.weights['b_trans'])

        # shape: (batch size * rows * columns, d_vec)
        # self.ent_vec = self.transfer(tf.add(tf.matmul(self.entities, self.weights['we']),
        #                                     self.weights['be']))

        # self.ent_vec = tf.add(tf.matmul(self.entities, self.weights['w_trans']),
        #                       self.weights['b_trans'])

        # concatenate one cell with corresponding entity
        # shape: (batch size*rows*columns, d_hidden)

        self.cell_concat = tf.concat([self.cell_vec, entities], 1)

        self.vec_ONE_single = self.transfer(tf.add(tf.matmul(self.cell_concat,  # change: self.ent_vec
                                                             self.weights['w1']),
                                                   self.weights['b1']))
        if self.istrain:
            self.vec_ONE_single = tf.nn.dropout(
                x=self.vec_ONE_single,
                keep_prob=self.keep_prob
            )

        # set all previous zero-vec cell into zero-vec
        self.vec_ONE_single = tf.transpose(tf.multiply(tf.transpose(self.vec_ONE_single),
                                                       self.where_list))

        self.vec_ONE_single = tf.reshape(self.vec_ONE_single,
                                         [-1, self.rows*self.columns, self.d_hidden])

        # vec_ONE has the shape of (batch_size, d_hidden)
        self.vec_ONE = tf.reduce_sum(self.vec_ONE_single, 1)

        self.vec_ONE = tf.transpose(tf.multiply(tf.transpose(self.vec_ONE), self.where_sum))
        return self.vec_ONE

    def _build_context_feature(self, contexts=None, entities=None):
        if contexts is None:
            contexts = self.contexts
        if entities is None:
            entities = self.entities

        # shape: (batch size * rows * columns, d_vec)
        # self.context_vec = self.transfer(tf.add(tf.matmul(self.cells, self.weights['wc']),
        #                                      self.weights['bc']))
        self.context_vec = tf.add(tf.matmul(contexts, self.weights['w_trans']),
                                  self.weights['b_trans'])

        # shape: (batch size * rows * columns, d_vec)
        # self.ent_vec = self.transfer(tf.add(tf.matmul(self.entities, self.weights['we']),
        #                                     self.weights['be']))

        # self.ent_vec = tf.add(tf.matmul(self.entities, self.weights['w_trans']),
        #                       self.weights['b_trans'])

        # concatenate one cell with corresponding entity
        # shape: (batch size*rows*columns, d_hidden)

        self.context_concat = tf.concat([self.context_vec, entities], 1)

        self.vec_TWO_single = self.transfer(tf.add(tf.matmul(self.context_concat,
                                                             self.weights['w2']),
                                                   self.weights['b2']))

        if self.istrain:
            self.vec_TWO_single = tf.nn.dropout(
                x=self.vec_TWO_single,
                keep_prob=self.keep_prob
            )

        # set all previous zero-vec cell into zero-vec
        self.vec_TWO_single = tf.transpose(tf.multiply(tf.transpose(self.vec_TWO_single),
                                                       self.where_list))

        self.vec_TWO_single = tf.reshape(self.vec_TWO_single,
                                         [-1, self.rows*self.columns, self.d_hidden])

        # vec_TWO has the shape of (batch_size, d_hidden)
        self.vec_TWO = tf.reduce_sum(self.vec_TWO_single, 1)

        self.vec_TWO = tf.transpose(tf.multiply(tf.transpose(self.vec_TWO), self.where_sum))
        return self.vec_TWO

    # hinge loss
    def _build_score_layer(self):
        self.score_layer = tf.add(tf.matmul(self.final_layer, self.weights['wo']), self.weights['bo'])
        # ------------------------- LOSS -------------------------------- #
        self.scores = tf.reshape(self.score_layer, [-1, self.PN])
        self.scores_pos = tf.slice(self.scores, [0, 0], [self.table_num, 1])
        self.scores_neg = tf.slice(self.scores, [0, 1], [self.table_num, self.PN - 1])
        # we could also use tf.nn.relu here
        self.loss_matrix = tf.maximum(0., self.margin - self.scores_pos + self.scores_neg)
        # find the negative with the highest score
        self.loss = tf.reduce_max(self.loss_matrix, axis=1)
        self.loss = tf.reduce_mean(self.loss)
        self.opt_op = self.optimizer.minimize(self.loss)
        # ------------------------- LOSS -------------------------------- #

    # pairwise ranking loss
    def _build_score_layer_rank(self):
        self.score_layer = tf.add(tf.matmul(self.final_layer, self.weights['wo']), self.weights['bo'])
        if self.istrain:
            # ------------------------- LOSS -------------------------------- #
            self.scores = tf.reshape(self.score_layer, [-1, self.PN])
            ranknet = RankNet(self.table_num, self.PN, self.learning_rate, gold_method="binary")
            mask = tf.ones((self.table_num, self.PN), dtype='float32')
            self.loss, self.opt_op = ranknet.build(self.scores, self.corrupts, mask)
            # ------------------------- LOSS -------------------------------- #

    def _build_graph(self): pass

    def save(self, session, directory):
        import os
        if not(os.path.isdir(directory)):
            os.mkdir(directory)
        fp =  directory + "/best_model"
        self.saver.save(session, fp)
        LogInfo.logs("Model saved into %s.", fp)

    def load(self, session, fp):
        LogInfo.logs("Loading Model from %s", fp)
        self.saver.restore(session, fp)
        LogInfo.logs("Model loaded from %s", fp)

    def check_weight(self):
        run_list = [self.weights['w_trans']]
        ret = self.sess.run(run_list)
        self.running_ret['w_trans'] = ret[0]
        return self.running_ret

    def train(self, session, input_data):
        # get feed_dict from input_data & run the model
        run_list = [self.opt_op, self.loss]
        ret = session.run(run_list, feed_dict={
            self.cells: input_data['cell'],
            self.entities: input_data['entity'],
            self.coherence: input_data['coherence'],
            self.contexts: input_data['context'],
            self.corrupts: input_data['corrupt']
        })
        self.running_ret['loss'] = ret[1]
        return self.running_ret

    def eval(self, session, input_data):
        ret = session.run([self.score_layer, self.weights], feed_dict={
            self.cells: input_data['cell'],
            self.entities: input_data['entity'],
            self.coherence: input_data['coherence'],
            self.contexts: input_data['context']
        })
        self.running_ret['eval_score'] = ret[0]
        self.running_ret['weights_eval'] = ret[1]
        return self.running_ret

class MonolingualModel(object):

    def __init__(self, trans_pre_train, nonj_pre_train,
                 batch_size, margin, rows, columns, PN,
                 d_w2v, d_vec, d_hidden, d_final, lr,
                 istrain=True, keep_prob=0.5):

        # initialize parameters
        self.batch_size = batch_size
        self.rows = rows
        self.columns = columns
        self.PN = PN
        self.table_num = self.batch_size / self.PN

        self.d_w2v = d_w2v
        self.d_vec = d_vec
        self.d_hidden = d_hidden
        self.d_final = d_final

        self.transfer = tf.nn.relu

        self.margin = margin
        self.learning_rate = lr
        self.optimizer = tf.train.AdamOptimizer(lr)
        self.running_ret = dict()
        if istrain:
            self.istrain = True
        else:
            self.istrain = False
        self.keep_prob=keep_prob

        # input: batch size * rows * columns * w2v_dim
        # note that rows & columns are fixed!!! that's a problem of jointly modeling_2017 of table data!
        self.cells = tf.placeholder(tf.float32, [None, self.d_w2v])
        self.entities = tf.placeholder(tf.float32, [None, self.d_w2v])
        self.coherence = tf.placeholder(tf.float32, [None, self.d_w2v])
        self.length = self.rows + self.columns - 2
        self.contexts = tf.placeholder(tf.float32, [None, self.d_w2v])
        # corrupt ratio for each table, shape=(table_num, PN)
        self.corrupts = tf.placeholder(tf.float32, [None, self.PN])
        # OLD self.contexts = tf.placeholder(tf.float32, [None, 1, self.length, self.d_w2v])

        # get the mask to remove all zero-zero pairs
        # !!! attention the mask is based on entity table!
        x = tf.reduce_sum(self.entities, 1)
        zero = tf.constant(0, dtype=tf.float32)
        # shape = (batch size * rows * cols)
        self.where_list = tf.cast(tf.not_equal(x, zero), dtype=tf.float32)
        # shape = (batch size, rows, cols)
        self.where_table = tf.reshape(self.where_list,
                                      [-1, self.rows*self.columns])
        self.where_cell = tf.reshape(self.where_list,
                                     [-1, self.rows, self.columns])
        # shape = (batch size, 1
        self.where_sum = tf.reciprocal(tf.reduce_sum(self.where_table, 1))

        # build the model graph
        self._build_graph()

        self.saver = tf.train.Saver()

    def _initialize_weights(self):
        weights = dict()

        # weight matrix for translation
        weights['w_trans'] = tf.get_variable(
            shape=[self.d_w2v, self.d_w2v],
            name='w_trans',
            dtype=tf.float32,
            initializer=tf.contrib.layers.xavier_initializer(uniform=True)
        )
        weights['b_trans'] = tf.get_variable(
            shape=[self.d_w2v],
            name='b_trans',
            dtype = tf.float32,
            initializer = tf.constant_initializer(0.0)
        )

        # weight matrix from concat(cell, ent) -> feature NO.1
        weights['w1'] = tf.get_variable(
            shape=[self.d_w2v*2, self.d_hidden],
            name='w1',
            dtype=tf.float32,
            initializer=tf.contrib.layers.xavier_initializer(uniform=True)
        )
        weights['b1'] = tf.get_variable(
            shape=[self.d_hidden],
            name='b1',
            dtype = tf.float32,
            initializer = tf.constant_initializer(0.0)
        )

        # weight matrix from concat(context, ent) -> feature NO.2
        weights['w2'] = tf.get_variable(
            shape=[self.d_w2v*2, self.d_hidden],
            name='w2',
            dtype=tf.float32,
            initializer=tf.contrib.layers.xavier_initializer(uniform=True)
        )
        weights['b2'] = tf.get_variable(
            shape=[self.d_hidden],
            name='b2',
            dtype = tf.float32,
            initializer = tf.constant_initializer(0.0)
        )

        # define weight matrix from concatenation of 1~3 features -> final_vec in derived class
        weights['bf'] = tf.get_variable(
            shape=[self.d_final],
            name='bf',
            dtype = tf.float32,
            initializer = tf.constant_initializer(0.0)
        )

        # weight matrix from final_vec -> score_layer
        weights['wo'] = tf.get_variable(
            shape=[self.d_final, 1],
            name='wo',
            dtype=tf.float32,
            initializer=tf.contrib.layers.xavier_initializer(uniform=True)
        )
        weights['bo'] = tf.get_variable(
            shape=[1],
            name='bo',
            dtype = tf.float32,
            initializer = tf.constant_initializer(0.0)
        )

        return weights

    def _build_cell_feature(self, cells=None, entities=None):
        if cells is None:
            cells = self.cells
        if entities is None:
            entities = self.entities

        # shape: (batch size * rows * columns, d_vec)
        self.cell_vec = tf.add(tf.matmul(cells, self.weights['w_trans']),
                               self.weights['b_trans'])

        # shape: (batch size * rows * columns, d_vec)
        # self.ent_vec = self.transfer(tf.add(tf.matmul(self.entities, self.weights['we']),
        #                                     self.weights['be']))

        # self.ent_vec = tf.add(tf.matmul(self.entities, self.weights['w_trans']),
        #                       self.weights['b_trans'])

        # concatenate one cell with corresponding entity
        # shape: (batch size*rows*columns, d_hidden)
        self.cell_concat = tf.concat([self.cell_vec, entities], 1)

        self.vec_ONE_single = self.transfer(tf.add(tf.matmul(self.cell_concat,  # change: self.ent_vec
                                                             self.weights['w1']),
                                                   self.weights['b1']))

        if self.istrain:
            self.vec_ONE_single = tf.nn.dropout(
                x=self.vec_ONE_single,
                keep_prob=self.keep_prob
            )

        # set all previous zero-vec cell into zero-vec
        self.vec_ONE_single = tf.transpose(tf.multiply(tf.transpose(self.vec_ONE_single),
                                                       self.where_list))

        self.vec_ONE_single = tf.reshape(self.vec_ONE_single,
                                         [-1, self.rows*self.columns, self.d_hidden])

        # vec_ONE has the shape of (batch_size, d_hidden)
        self.vec_ONE = tf.reduce_sum(self.vec_ONE_single, 1)

        self.vec_ONE = tf.transpose(tf.multiply(tf.transpose(self.vec_ONE), self.where_sum))
        return self.vec_ONE

    def _build_context_feature(self, contexts=None, entities=None):
        if contexts is None:
            contexts = self.contexts
        if entities is None:
            entities = self.entities

        # shape: (batch size * rows * columns, d_vec)
        self.context_vec = tf.add(tf.matmul(contexts, self.weights['w_trans']),
                                  self.weights['b_trans'])

        # shape: (batch size * rows * columns, d_vec)
        # self.ent_vec = self.transfer(tf.add(tf.matmul(self.entities, self.weights['we']),
        #                                     self.weights['be']))

        # self.ent_vec = tf.add(tf.matmul(self.entities, self.weights['w_trans']),
        #                       self.weights['b_trans'])

        # concatenate one cell with corresponding entity
        # shape: (batch size*rows*columns, d_hidden)
        self.context_concat = tf.concat([self.context_vec, entities], 1)

        self.vec_TWO_single = self.transfer(tf.add(tf.matmul(self.context_concat,
                                                             self.weights['w2']),
                                                   self.weights['b2']))

        if self.istrain:
            self.vec_TWO_single = tf.nn.dropout(
                x=self.vec_TWO_single,
                keep_prob=self.keep_prob
            )

        # set all previous zero-vec cell into zero-vec
        self.vec_TWO_single = tf.transpose(tf.multiply(tf.transpose(self.vec_TWO_single),
                                                       self.where_list))

        self.vec_TWO_single = tf.reshape(self.vec_TWO_single,
                                         [-1, self.rows*self.columns, self.d_hidden])

        # vec_TWO has the shape of (batch_size, d_hidden)
        self.vec_TWO = tf.reduce_sum(self.vec_TWO_single, 1)

        self.vec_TWO = tf.transpose(tf.multiply(tf.transpose(self.vec_TWO), self.where_sum))
        return self.vec_TWO

    # hinge loss
    def _build_score_layer(self):
        self.score_layer = tf.add(tf.matmul(self.final_layer, self.weights['wo']), self.weights['bo'])
        # ------------------------- LOSS -------------------------------- #
        self.scores = tf.reshape(self.score_layer, [-1, self.PN])
        self.scores_pos = tf.slice(self.scores, [0, 0], [self.table_num, 1])
        self.scores_neg = tf.slice(self.scores, [0, 1], [self.table_num, self.PN - 1])
        # we could also use tf.nn.relu here
        self.loss_matrix = tf.maximum(0., self.margin - self.scores_pos + self.scores_neg)
        # find the negative with the highest score
        self.loss = tf.reduce_max(self.loss_matrix, axis=1)
        self.loss = tf.reduce_mean(self.loss)
        self.opt_op = self.optimizer.minimize(self.loss)
        # ------------------------- LOSS -------------------------------- #

    # pairwise ranking loss
    def _build_score_layer_rank(self):
        self.score_layer = tf.add(tf.matmul(self.final_layer, self.weights['wo']), self.weights['bo'])
        if self.istrain:
            # ------------------------- LOSS -------------------------------- #
            self.scores = tf.reshape(self.score_layer, [-1, self.PN])
            ranknet = RankNet(self.table_num, self.PN, self.learning_rate, gold_method="binary")
            mask = tf.ones((self.table_num, self.PN), dtype='float32')
            self.loss, self.opt_op = ranknet.build(self.scores, self.corrupts, mask)
            # ------------------------- LOSS -------------------------------- #

    def _build_graph(self): pass

    def save(self, session, directory):
        import os
        if not(os.path.isdir(directory)):
            os.mkdir(directory)
        fp =  directory + "/best_model"
        self.saver.save(session, fp)
        LogInfo.logs("Model saved into %s.", fp)

    def load(self, session, fp):
        LogInfo.logs("Loading Model from %s", fp)
        self.saver.restore(session, fp)
        LogInfo.logs("Model loaded from %s", fp)

    def check_weight(self):
        run_list = [self.weights['w_trans']]
        ret = self.sess.run(run_list)
        self.running_ret['w_trans'] = ret[0]
        return self.running_ret

    def train(self, session, input_data):
        # get feed_dict from input_data & run the model
        run_list = [self.opt_op, self.loss]
        ret = session.run(run_list, feed_dict={
            self.cells: input_data['cell'],
            self.entities: input_data['entity'],
            self.coherence: input_data['coherence'],
            self.contexts: input_data['context'],
            self.corrupts: input_data['corrupt']
        })
        self.running_ret['loss'] = ret[1]
        return self.running_ret

    def eval(self, session, input_data):
        ret = session.run([self.score_layer, self.weights], feed_dict={
            self.cells: input_data['cell'],
            self.entities: input_data['entity'],
            self.coherence: input_data['coherence'],
            self.contexts: input_data['context']
        })
        self.running_ret['eval_score'] = ret[0]
        self.running_ret['weights_eval'] = ret[1]
        return self.running_ret
