import tensorflow as tf

from joint_model import JointModel, MonolingualModel
from xusheng.util.tf_util import noise_weight_variable

class Model1(JointModel):

    # This model uses surface form similarity feature only

    def __init__(self, trans_pre_train="common", nonj_pre_train="trans",
                 batch_size=3*20, margin=0.0005,
                 rows=16, columns=6, PN=20, d_w2v=100, d_vec=100, d_hidden=100,
                 d_final=200, lr=1e-4):
        super(Model1, self).__init__(trans_pre_train=trans_pre_train,
                                     nonj_pre_train=nonj_pre_train,
                                     batch_size=batch_size,
                                     margin=margin,
                                     rows=rows,
                                     columns=columns,
                                     PN=PN,
                                     d_w2v=d_w2v,
                                     d_vec=d_vec,
                                     d_hidden=d_hidden,
                                     d_final=d_final,
                                     lr=lr)

    def _build_graph(self):

        self.weights = self._initialize_weights()
        # weight matrix from concat(vec_1, vec_2) -> final_vec
        self.weights['wf'] = noise_weight_variable([self.d_hidden, self.d_final], 'wf')

        # model 1 only use cell vs. entity surface similarity features
        self._build_cell_feature()

        # concatenate 2 vectors together, shape = (batch_size, d_hidden * 2)
        # final fully connected layer, shape = (batch_size, d_final)
        self.final_layer = self.transfer(tf.add(tf.matmul(self.vec_ONE,
                                                          self.weights['wf']),
                                                self.weights['bf']))

        # output layer has only one number, shape = (batch_size, 1)
        self._build_score_layer()

class Model1_rank(JointModel):

    # This model uses surface form similarity feature only & ranknet loss

    def __init__(self, trans_pre_train="common", nonj_pre_train="trans",
                 batch_size=3*20, margin=0.0005,
                 rows=16, columns=6, PN=20, d_w2v=100, d_vec=100, d_hidden=100,
                 d_final=200, lr=1e-4):
        super(Model1_rank, self).__init__(trans_pre_train=trans_pre_train,
                                          nonj_pre_train=nonj_pre_train,
                                          batch_size=batch_size,
                                          margin=margin,
                                          rows=rows,
                                          columns=columns,
                                          PN=PN,
                                          d_w2v=d_w2v,
                                          d_vec=d_vec,
                                          d_hidden=d_hidden,
                                          d_final=d_final,
                                          lr=lr)

    def _build_graph(self):

        self.weights = self._initialize_weights()
        # weight matrix from concat(vec_1, vec_2) -> final_vec
        self.weights['wf'] = noise_weight_variable([self.d_hidden, self.d_final], 'wf')

        # model 1 only use cell vs. entity surface similarity features
        self._build_cell_feature()

        # concatenate 2 vectors together, shape = (batch_size, d_hidden * 2)
        # final fully connected layer, shape = (batch_size, d_final)
        self.final_layer = self.transfer(tf.add(tf.matmul(self.vec_ONE,
                                                          self.weights['wf']),
                                                self.weights['bf']))

        # output layer has only one number, shape = (batch_size, 1)
        self._build_score_layer_rank()

class Model2(JointModel):

    # This model uses surface form & context similarity features

    def __init__(self, trans_pre_train="common", nonj_pre_train="trans",
                 batch_size=3*20, margin=0.0005,
                 rows=16, columns=6, PN=20, d_w2v=100, d_vec=100, d_hidden=100,
                 d_final=200, lr=1e-4):
        super(Model2, self).__init__(trans_pre_train=trans_pre_train,
                                     nonj_pre_train=nonj_pre_train,
                                     batch_size=batch_size,
                                     margin=margin,
                                     rows=rows,
                                     columns=columns,
                                     PN=PN,
                                     d_w2v=d_w2v,
                                     d_vec=d_vec,
                                     d_hidden=d_hidden,
                                     d_final=d_final,
                                     lr=lr)

    def _build_graph(self):

        self.weights = self._initialize_weights()
        # weight matrix from concat(vec_1, vec_2) -> final_vec
        self.weights['wf'] = noise_weight_variable([self.d_hidden * 2, self.d_final], 'wf')

        # model 2 uses cell vs. entity surface + context similarity features
        self._build_cell_feature()
        self._build_context_feature()

        # concatenate 2 vectors together, shape = (batch_size, d_hidden * 2)
        # final fully connected layer, shape = (batch_size, d_final)
        self.final_layer = self.transfer(tf.add(tf.matmul(tf.concat([self.vec_ONE,
                                                                     self.vec_TWO], 1),
                                                          self.weights['wf']),
                                                self.weights['bf']))

        # output layer has only one number, shape = (batch_size, 1)
        self._build_score_layer()

class Model2_rank(JointModel):

    # This model uses surface form & context similarity features
    # use RankNet as loss function

    def __init__(self, trans_pre_train="common", nonj_pre_train="trans",
                 batch_size=3*20, margin=0.0005,
                 rows=16, columns=6, PN=20, d_w2v=100, d_vec=100, d_hidden=100,
                 d_final=200, lr=1e-4):
        super(Model2_rank, self).__init__(trans_pre_train=trans_pre_train,
                                          nonj_pre_train=nonj_pre_train,
                                          batch_size=batch_size,
                                          margin=margin,
                                          rows=rows,
                                          columns=columns,
                                          PN=PN,
                                          d_w2v=d_w2v,
                                          d_vec=d_vec,
                                          d_hidden=d_hidden,
                                          d_final=d_final,
                                          lr=lr)

    def _build_graph(self):

        self.weights = self._initialize_weights()
        # weight matrix from concat(vec_1, vec_2) -> final_vec
        self.weights['wf'] = noise_weight_variable([self.d_hidden * 2, self.d_final], 'wf')

        # model 2 uses cell vs. entity surface + context similarity features
        self._build_cell_feature()
        self._build_context_feature()

        # concatenate 2 vectors together, shape = (batch_size, d_hidden * 2)
        # final fully connected layer, shape = (batch_size, d_final)
        self.final_layer = self.transfer(tf.add(tf.matmul(tf.concat([self.vec_ONE,
                                                                     self.vec_TWO], 1),
                                                          self.weights['wf']),
                                                self.weights['bf']))

        # output layer has only one number, shape = (batch_size, 1)
        self._build_score_layer_rank()

class Model3(JointModel):

    # This model uses surface form & context similarity & coherence features
    # max-margin loss
    def __init__(self, trans_pre_train="common", nonj_pre_train="trans",
                 batch_size=3*20, margin=0.0005,
                 rows=16, columns=6, PN=20, d_w2v=100, d_vec=100, d_hidden=100,
                 d_final=200, lr=1e-4):
        super(Model3, self).__init__(trans_pre_train=trans_pre_train,
                                     nonj_pre_train=nonj_pre_train,
                                     batch_size=batch_size,
                                     margin=margin,
                                     rows=rows,
                                     columns=columns,
                                     PN=PN,
                                     d_w2v=d_w2v,
                                     d_vec=d_vec,
                                     d_hidden=d_hidden,
                                     d_final=d_final,
                                     lr=lr)

    def _build_graph(self):

        self.weights = self._initialize_weights()
        # weight matrix from concat(vec_1, vec_2, vec_3) -> final_vec
        self.weights['wf'] = noise_weight_variable([self.d_hidden * 3, self.d_final], 'wf')

        # build two similarity features
        self._build_cell_feature()
        self._build_context_feature()

        # build coherence features
        self.vec_Three = self.coherence

        # OLD
        # self.vec_Three, intermediate_dict = \
        #     build_colwise_coherence_feature(self.coherence, self.d_vec, self.d_vec)

        # concatenate all vectors together, shape = (batch_size, d_hidden * n)
        # final fully connected layer, shape = (batch_size, d_final)
        self.final_layer = self.transfer(tf.add(tf.matmul(tf.concat([tf.concat([self.vec_ONE,
                                                                                self.vec_TWO], 1),
                                                                     self.vec_Three], 1),
                                                          self.weights['wf']),
                                                self.weights['bf']))

        # output layer has only one number, shape = (batch_size, 1)
        self._build_score_layer()

class Model3_rank(JointModel):

    # This model uses surface form & context similarity & coherence features
    # and RankNet loss

    def __init__(self, trans_pre_train="common", nonj_pre_train="trans",
                 batch_size=3*20, margin=0.0005,
                 rows=16, columns=6, PN=20, d_w2v=100, d_vec=100, d_hidden=100,
                 d_final=200, lr=1e-4, istrain=True, keep_prob=0.5):
        super(Model3_rank, self).__init__(trans_pre_train=trans_pre_train,
                                          nonj_pre_train=nonj_pre_train,
                                          batch_size=batch_size,
                                          margin=margin,
                                          rows=rows,
                                          columns=columns,
                                          PN=PN,
                                          d_w2v=d_w2v,
                                          d_vec=d_vec,
                                          d_hidden=d_hidden,
                                          d_final=d_final,
                                          lr=lr,
                                          istrain=istrain,
                                          keep_prob=keep_prob)

    def _build_graph(self):

        self.weights = self._initialize_weights()
        # weight matrix from concat(vec_1, vec_2, vec_3) -> final_vec
        self.weights['wf'] = tf.get_variable(
            shape=[self.d_hidden*3, self.d_final],
            name='wf',
            dtype=tf.float32,
            initializer=tf.contrib.layers.xavier_initializer(uniform=True)
        )

        # build two similarity features
        self._build_cell_feature()
        self._build_context_feature()

        # build coherence features
        self.vec_Three = self.coherence

        # OLD
        # self.vec_Three, intermediate_dict = \
        #     build_colwise_coherence_feature(self.coherence, self.d_vec, self.d_vec)

        # concatenate all vectors together, shape = (batch_size, d_hidden * n)
        # final fully connected layer, shape = (batch_size, d_final)
        self.final_layer = self.transfer(tf.add(tf.matmul(tf.concat([tf.concat([self.vec_ONE,
                                                                                self.vec_TWO], 1),
                                                                     self.vec_Three], 1),
                                                          self.weights['wf']),
                                                self.weights['bf']))

        # output layer has only one number, shape = (batch_size, 1)
        self._build_score_layer_rank()

class Model3_rank_avg(JointModel):

    # This model uses surface form & context similarity & coherence features
    # and RankNet loss

    def __init__(self, trans_pre_train="common", nonj_pre_train="trans",
                 batch_size=3*20, margin=0.0005,
                 rows=16, columns=6, PN=20, d_w2v=100, d_vec=100, d_hidden=100,
                 d_final=200, lr=1e-4):
        super(Model3_rank_avg, self).__init__(trans_pre_train=trans_pre_train,
                                              nonj_pre_train=nonj_pre_train,
                                              batch_size=batch_size,
                                              margin=margin,
                                              rows=rows,
                                              columns=columns,
                                              PN=PN,
                                              d_w2v=d_w2v,
                                              d_vec=d_vec,
                                              d_hidden=d_hidden,
                                              d_final=d_final,
                                              lr=lr)

    def _build_graph(self):

        self.weights = self._initialize_weights()
        # weight matrix from concat(vec_1, vec_2, vec_3) -> final_vec
        self.coherence = tf.placeholder(tf.float32, [None, 1])
        self.weights['wf'] = noise_weight_variable([self.d_hidden * 2 + 1, self.d_final], 'wf')

        # build two similarity features
        self._build_cell_feature()
        self._build_context_feature()

        # build coherence features
        self.vec_Three = self.coherence

        # OLD
        # self.vec_Three, intermediate_dict = \
        #     build_colwise_coherence_feature(self.coherence, self.d_vec, self.d_vec)

        # concatenate all vectors together, shape = (batch_size, d_hidden * n)
        # final fully connected layer, shape = (batch_size, d_final)
        self.final_layer = self.transfer(tf.add(tf.matmul(tf.concat([tf.concat([self.vec_ONE,
                                                                                self.vec_TWO], 1),
                                                                     self.vec_Three], 1),
                                                          self.weights['wf']),
                                                self.weights['bf']))

        # output layer has only one number, shape = (batch_size, 1)
        self._build_score_layer_rank()

class Model4(JointModel):

    # This model uses context similarity feature only

    def __init__(self, trans_pre_train="common", nonj_pre_train="trans",
                 batch_size=3*20, margin=0.0005,
                 rows=16, columns=6, PN=20, d_w2v=100, d_vec=100, d_hidden=100,
                 d_final=200, lr=1e-4):
        super(Model4, self).__init__(trans_pre_train=trans_pre_train,
                                     nonj_pre_train=nonj_pre_train,
                                     batch_size=batch_size,
                                     margin=margin,
                                     rows=rows,
                                     columns=columns,
                                     PN=PN,
                                     d_w2v=d_w2v,
                                     d_vec=d_vec,
                                     d_hidden=d_hidden,
                                     d_final=d_final,
                                     lr=lr)

    def _build_graph(self):
        self.weights = self._initialize_weights()
        # weight matrix from concat(vec_1, vec_2) -> final_vec
        self.weights['wf'] = noise_weight_variable([self.d_hidden, self.d_final], 'wf')

        # model 4 only use cell vs. entity context similarity features
        self._build_context_feature()

        # final fully connected layer, shape = (batch_size, d_final)
        self.final_layer = self.transfer(tf.add(tf.matmul(self.vec_TWO,
                                                          self.weights['wf']),
                                                self.weights['bf']))

        # output layer has only one number, shape = (batch_size, 1)
        self._build_score_layer()

class Model4_rank(JointModel):

    # This model uses context similarity feature only & ranknet loss

    def __init__(self, trans_pre_train="common", nonj_pre_train="trans",
                 batch_size=3*20, margin=0.0005,
                 rows=16, columns=6, PN=20, d_w2v=100, d_vec=100, d_hidden=100,
                 d_final=200, lr=1e-4):
        super(Model4_rank, self).__init__(trans_pre_train=trans_pre_train,
                                          nonj_pre_train=nonj_pre_train,
                                          batch_size=batch_size,
                                          margin=margin,
                                          rows=rows,
                                          columns=columns,
                                          PN=PN,
                                          d_w2v=d_w2v,
                                          d_vec=d_vec,
                                          d_hidden=d_hidden,
                                          d_final=d_final,
                                          lr=lr)

    def _build_graph(self):
        self.weights = self._initialize_weights()
        # weight matrix from concat(vec_1, vec_2) -> final_vec
        self.weights['wf'] = noise_weight_variable([self.d_hidden, self.d_final], 'wf')

        # model 4 only use cell vs. entity context similarity features
        self._build_context_feature()

        # final fully connected layer, shape = (batch_size, d_final)
        self.final_layer = self.transfer(tf.add(tf.matmul(self.vec_TWO,
                                                          self.weights['wf']),
                                                self.weights['bf']))

        # output layer has only one number, shape = (batch_size, 1)
        self._build_score_layer_rank()

class Model5(JointModel):
    # This model uses Kenny's coherence feature only

    def __init__(self, trans_pre_train="common", nonj_pre_train="trans",
                 batch_size=3*20, margin=0.0005,
                 rows=16, columns=6, PN=20, d_w2v=100, d_vec=100, d_hidden=100,
                 d_final=200, lr=1e-4):
        super(Model5, self).__init__(trans_pre_train=trans_pre_train,
                                     nonj_pre_train=nonj_pre_train,
                                     batch_size=batch_size,
                                     margin=margin,
                                     rows=rows,
                                     columns=columns,
                                     PN=PN,
                                     d_w2v=d_w2v,
                                     d_vec=d_vec,
                                     d_hidden=d_hidden,
                                     d_final=d_final,
                                     lr=lr)

    def _build_graph(self):
        self.weights = self._initialize_weights()
        # weight matrix from concat(vec_1, vec_2) -> final_vec
        self.weights['wf'] = noise_weight_variable([self.d_hidden, self.d_final], 'wf')

        # model 5 only use coherence features
        self.vec_Three = self.coherence

        # final fully connected layer, shape = (batch_size, d_final)
        self.final_layer = self.transfer(tf.add(tf.matmul(self.vec_Three,
                                                          self.weights['wf']),
                                                self.weights['bf']))

        # output layer has only one number, shape = (batch_size, 1)
        self._build_score_layer()

class Model5_rank(JointModel):
    # This model uses Kenny's coherence feature only

    def __init__(self, trans_pre_train="common", nonj_pre_train="trans",
                 batch_size=3*20, margin=0.0005,
                 rows=16, columns=6, PN=20, d_w2v=100, d_vec=100, d_hidden=100,
                 d_final=200, lr=1e-4):
        super(Model5_rank, self).__init__(trans_pre_train=trans_pre_train,
                                          nonj_pre_train=nonj_pre_train,
                                          batch_size=batch_size,
                                          margin=margin,
                                          rows=rows,
                                          columns=columns,
                                          PN=PN,
                                          d_w2v=d_w2v,
                                          d_vec=d_vec,
                                          d_hidden=d_hidden,
                                          d_final=d_final,
                                          lr=lr)

    def _build_graph(self):
        self.weights = self._initialize_weights()
        # weight matrix from concat(vec_1, vec_2) -> final_vec
        self.weights['wf'] = noise_weight_variable([self.d_hidden, self.d_final], 'wf')

        # model 5 only use coherence features
        self.vec_Three = self.coherence

        # final fully connected layer, shape = (batch_size, d_final)
        self.final_layer = self.transfer(tf.add(tf.matmul(self.vec_Three,
                                                          self.weights['wf']),
                                                self.weights['bf']))

        # output layer has only one number, shape = (batch_size, 1)
        self._build_score_layer_rank()

class Model6_rank(JointModel):

    # This model uses surface form similarity & coherence feature & ranknet loss
    def __init__(self, trans_pre_train="common", nonj_pre_train="trans",
                 batch_size=3*20, margin=0.0005,
                 rows=16, columns=6, PN=20, d_w2v=100, d_vec=100, d_hidden=100,
                 d_final=200, lr=1e-4):
        super(Model6_rank, self).__init__(trans_pre_train=trans_pre_train,
                                          nonj_pre_train=nonj_pre_train,
                                          batch_size=batch_size,
                                          margin=margin,
                                          rows=rows,
                                          columns=columns,
                                          PN=PN,
                                          d_w2v=d_w2v,
                                          d_vec=d_vec,
                                          d_hidden=d_hidden,
                                          d_final=d_final,
                                          lr=lr)

    def _build_graph(self):

        self.weights = self._initialize_weights()
        # weight matrix from concat(vec_1, vec_2) -> final_vec
        self.weights['wf'] = noise_weight_variable([self.d_hidden * 2, self.d_final], 'wf')

        # model 1 only use cell vs. entity surface similarity features
        self._build_cell_feature()
        self.vec_Three = self.coherence

        # concatenate 2 vectors together, shape = (batch_size, d_hidden * 2)
        # final fully connected layer, shape = (batch_size, d_final)
        self.final_layer = self.transfer(tf.add(tf.matmul(tf.concat([self.vec_ONE,
                                                                     self.vec_Three], 1),
                                                          self.weights['wf']),
                                                self.weights['bf']))

        # output layer has only one number, shape = (batch_size, 1)
        self._build_score_layer_rank()

class Model7_rank(JointModel):

    # This model uses surface form similarity & coherence feature & ranknet loss
    def __init__(self, trans_pre_train="common", nonj_pre_train="trans",
                 batch_size=3*20, margin=0.0005,
                 rows=16, columns=6, PN=20, d_w2v=100, d_vec=100, d_hidden=100,
                 d_final=200, lr=1e-4):
        super(Model7_rank, self).__init__(trans_pre_train=trans_pre_train,
                                          nonj_pre_train=nonj_pre_train,
                                          batch_size=batch_size,
                                          margin=margin,
                                          rows=rows,
                                          columns=columns,
                                          PN=PN,
                                          d_w2v=d_w2v,
                                          d_vec=d_vec,
                                          d_hidden=d_hidden,
                                          d_final=d_final,
                                          lr=lr)

    def _build_graph(self):

        self.weights = self._initialize_weights()
        # weight matrix from concat(vec_1, vec_2) -> final_vec
        self.weights['wf'] = noise_weight_variable([self.d_hidden * 2, self.d_final], 'wf')

        # model 1 only use cell vs. entity surface similarity features
        self._build_context_feature()
        self.vec_Three = self.coherence

        # concatenate 2 vectors together, shape = (batch_size, d_hidden * 2)
        # final fully connected layer, shape = (batch_size, d_final)
        self.final_layer = self.transfer(tf.add(tf.matmul(tf.concat([self.vec_TWO,
                                                                     self.vec_Three], 1),
                                                          self.weights['wf']),
                                                self.weights['bf']))

        # output layer has only one number, shape = (batch_size, 1)
        self._build_score_layer_rank()

class Model8_rank(MonolingualModel):

    # monolingual model
    # This model uses surface form & context similarity & coherence features
    # and RankNet loss

    def __init__(self, trans_pre_train="common", nonj_pre_train="trans",
                 batch_size=3*20, margin=0.0005,
                 rows=16, columns=6, PN=20, d_w2v=100, d_vec=100, d_hidden=100,
                 d_final=200, lr=1e-4, istrain=True, keep_prob=0.5):
        super(Model8_rank, self).__init__(trans_pre_train=trans_pre_train,
                                          nonj_pre_train=nonj_pre_train,
                                          batch_size=batch_size,
                                          margin=margin,
                                          rows=rows,
                                          columns=columns,
                                          PN=PN,
                                          d_w2v=d_w2v,
                                          d_vec=d_vec,
                                          d_hidden=d_hidden,
                                          d_final=d_final,
                                          lr=lr,
                                          istrain=istrain,
                                          keep_prob=keep_prob)

    def _build_graph(self):

        self.weights = self._initialize_weights()
        # weight matrix from concat(vec_1, vec_2, vec_3) -> final_vec
        self.weights['wf'] = tf.get_variable(
            shape=[self.d_hidden * 3, self.d_final],
            name='wf',
            dtype=tf.float32,
            initializer=tf.contrib.layers.xavier_initializer(uniform=True)
        )

        # build two similarity features
        self._build_cell_feature()
        self._build_context_feature()

        # build coherence features
        self.vec_Three = self.coherence

        # OLD
        # self.vec_Three, intermediate_dict = \
        #     build_colwise_coherence_feature(self.coherence, self.d_vec, self.d_vec)

        # concatenate all vectors together, shape = (batch_size, d_hidden * n)
        # final fully connected layer, shape = (batch_size, d_final)
        self.final_layer = self.transfer(tf.add(tf.matmul(tf.concat([tf.concat([self.vec_ONE,
                                                                                self.vec_TWO], 1),
                                                                     self.vec_Three], 1),
                                                          self.weights['wf']),
                                                self.weights['bf']))

        # output layer has only one number, shape = (batch_size, 1)
        self._build_score_layer_rank()
