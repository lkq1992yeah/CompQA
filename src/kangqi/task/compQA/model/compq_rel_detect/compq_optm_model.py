"""
Author: Kangqi Luo
Goal: We remove the entities in the question by placeholders
Then we implement our own relation matching method.
Strategy 1: Calculate score between q and each path, then sum it together.
Strategy 2: Max-Pooling among path representations.
Related procedures/modules:
    * Simple Attention
    * Cross Attention
    * Combine vector rep. from both relation words and relation ids
"""

import tensorflow as tf

from ..compq_overall.relation_matching_kernel import RelationMatchingKernel as CompqKernel

from ..u import show_tensor
from ..base_model import BaseModel

from kangqi.util.LogUtil import LogInfo


class CompqOptmModel(BaseModel):

    def __init__(self, sess, compq_kernel,
                 loss_func, learning_rate, optm_name):
        LogInfo.begin_track('CompqOptmModel Building ...')
        assert isinstance(compq_kernel, CompqKernel)
        verbose = compq_kernel.verbose
        assert optm_name in ('Adam', 'Adadelta', 'Adagrad', 'GradientDescent')
        optm_name += 'Optimizer'
        assert loss_func.startswith('H')
        margin = float(loss_func[1:])   # "H0.5" --> 0.5
        # TODO: RankNet support
        super(CompqOptmModel, self).__init__(sess=sess, verbose=verbose)        # TODO: ob_batch_num = 20

        # ======== define tensors ======== #

        q_max_len = compq_kernel.q_max_len
        path_max_len = compq_kernel.path_max_len
        pword_max_len = compq_kernel.pword_max_len

        self.optm_input_tf_list = []
        sc_tensor_groups = []  # [ pos_tensors, neg_tensors ]
        for cate in ('pos', 'neg'):
            qwords_input = tf.placeholder(dtype=tf.int32,
                                          shape=[None, q_max_len],
                                          name=cate + '_qwords_input')  # (data_size, q_max_len)
            qwords_len_input = tf.placeholder(dtype=tf.int32,
                                              shape=[None],
                                              name=cate + '_qwords_len_input')  # (data_size, )
            sc_len_input = tf.placeholder(dtype=tf.int32,
                                          shape=[None],
                                          name=cate + '_sc_len_input')  # (data_size, )
            preds_input = tf.placeholder(dtype=tf.int32,
                                         shape=[None, None, path_max_len],
                                         name=cate + '_preds_input')  # (data_size, sc_max_len, path_max_len)
            preds_len_input = tf.placeholder(dtype=tf.int32,
                                             shape=[None, None],
                                             name=cate + '_preds_len_input')  # (data_size, sc_max_len)
            pwords_input = tf.placeholder(dtype=tf.int32,
                                          shape=[None, None, pword_max_len],
                                          name=cate + '_pwords_input')  # (data_size, sc_max_len, pword_max_len)
            pwords_len_input = tf.placeholder(dtype=tf.int32,
                                              shape=[None, None],
                                              name=cate + '_pwords_len_input')  # (data_size, sc_max_len)
            tensor_group = [qwords_input, qwords_len_input, sc_len_input,
                            preds_input, preds_len_input,
                            pwords_input, pwords_len_input]
            sc_tensor_groups.append(tensor_group)
            self.optm_input_tf_list += tensor_group
        data_weights = tf.placeholder(dtype=tf.float32, shape=[None],
                                      name='data_weights')      # (data_size, )
        self.optm_input_tf_list.append(data_weights)
        LogInfo.begin_track('Showing %d input tensors:', len(self.optm_input_tf_list))
        for tensor in self.optm_input_tf_list:
            show_tensor(tensor)
        LogInfo.end_track()

        # ======== start building model ======== #

        with tf.name_scope('Optm'):
            score_list = []     # store two tensors: positive and negative score
            for cate, sc_tensor_group in zip(('pos', 'neg'), sc_tensor_groups):
                LogInfo.begin_track('Calculate score at %s side ...', cate)
                [qwords_input, qwords_len_input, sc_len_input,
                 preds_input, preds_len_input,
                 pwords_input, pwords_len_input] = sc_tensor_group
                with tf.device("/cpu:0"):
                    qwords_embedding = tf.nn.embedding_lookup(
                        params=compq_kernel.w_embedding,
                        ids=qwords_input, name='q_embedding'
                    )  # (data_size, q_max_len, dim_emb)
                    preds_embedding = tf.nn.embedding_lookup(
                        params=compq_kernel.m_embedding,
                        ids=preds_input, name='preds_embedding'
                    )  # (data_size, sc_max_len, path_max_len, dim_emb)
                    pwords_embedding = tf.nn.embedding_lookup(
                        params=compq_kernel.w_embedding,
                        ids=pwords_input, name='pwords_embedding'
                    )  # (data_size, sc_max_len, pword_max_len, dim_emb)
                # Kernel Function: Calculate scores, given all the information we need
                _, _, score = compq_kernel.get_score(
                    mode=tf.contrib.learn.ModeKeys.TRAIN,
                    qwords_embedding=qwords_embedding, qwords_len=qwords_len_input, sc_len=sc_len_input,
                    preds_embedding=preds_embedding, preds_len=preds_len_input,
                    pwords_embedding=pwords_embedding, pwords_len=pwords_len_input
                )       # (data_size, ), currently we ignore the attention matrix information
                score_list.append(score)
                LogInfo.end_track()

            # ======== define loss and updates ======== #

            pos_score, neg_score = score_list
            margin_loss = tf.nn.relu(neg_score + margin - pos_score,
                                     name='margin_loss')    # (data_size, )
            weighted_margin_loss = tf.multiply(margin_loss,
                                               data_weights,
                                               name='weighted_margin_loss')
            self.avg_loss = tf.reduce_mean(weighted_margin_loss,
                                           name='avg_loss')

        tf.summary.scalar('avg_loss', self.avg_loss, collections=['optm'])
        optimizer = getattr(tf.train, optm_name)
        self.optm_step = optimizer(learning_rate).minimize(self.avg_loss)
        self.optm_summary = tf.summary.merge_all(key='optm')
        LogInfo.logs('* avg_loss and optm_step defined.')

        LogInfo.end_track()
