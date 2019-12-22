"""
Author: Kangqi Luo
Date: 18-01-17
Goal: The evaluation code of the overall model
      (including relation matching, entity linking, and other structural features)
The coding style is very similar with compq_rel_detect/compq_eval_model.py
"""

import tensorflow as tf

from ..u import show_tensor
from ..base_model import BaseModel
from ..general_evaluator import general_evaluate

from kangqi.util.LogUtil import LogInfo


class CompqModel(BaseModel):

    def __init__(self, sess, relation_kernel, entity_kernel, extra_len,
                 objective, loss_func, learning_rate, optm_name, verbose):
        BaseModel.__init__(self, sess=sess, verbose=verbose)

        self.reuse = relation_kernel.reuse
        self.objective = objective
        assert self.objective in ('relation_only', 'normal')

        assert optm_name in ('Adam', 'Adadelta', 'Adagrad', 'GradientDescent')
        optm_name += 'Optimizer'
        assert loss_func.startswith('H')
        margin = float(loss_func[1:])  # "H0.5" --> 0.5
        # TODO: RankNet support

        LogInfo.begin_track('compq_overall/CompqEvalModel Building ...')

        """ Define Tensors """
        q_max_len = relation_kernel.q_max_len
        path_max_len = relation_kernel.path_max_len
        pword_max_len = relation_kernel.pword_max_len

        eval_tf_list, optm_pos_tf_list, optm_neg_tf_list = [
            self.create_tensors(
                q_max_len=q_max_len, path_max_len=path_max_len,
                pword_max_len=pword_max_len, name_prefix=name_prefix,
                extra_len=extra_len
            ) for name_prefix in ('', 'pos_', 'neg_')
        ]       # (EVAL, OPTM-POS, OPTM-NEG)
        self.eval_input_tf_list = eval_tf_list
        self.optm_input_tf_list = optm_pos_tf_list + optm_neg_tf_list
        data_weights = tf.placeholder(dtype=tf.float32, shape=[None],
                                      name='data_weights')  # (data_size, )
        self.optm_input_tf_list.append(data_weights)

        for tf_list, tf_name in ((self.optm_input_tf_list, 'OPTM'),
                                 (self.eval_input_tf_list, 'EVAL')):
            LogInfo.begin_track('Showing %d input tensors for [%s]:', len(tf_list), tf_name)
            for tensor in tf_list:
                show_tensor(tensor)
            LogInfo.end_track()

        """ Build Score Function for Both Optm and Eval """
        config_tups = [(eval_tf_list, tf.contrib.learn.ModeKeys.INFER, 'Eval'),
                       (optm_pos_tf_list, tf.contrib.learn.ModeKeys.TRAIN, 'Optm_Pos'),
                       (optm_neg_tf_list, tf.contrib.learn.ModeKeys.TRAIN, 'Optm_Neg')]
        if self.objective == 'normal':
            eval_score_tup, optm_pos_score_tup, optm_neg_score_tup = [
                self.produce_overall_score(
                    tf_list=tf_list, mode=mode, name_scope=name_scope,
                    relation_kernel=relation_kernel, entity_kernel=entity_kernel
                ) for tf_list, mode, name_scope in config_tups
            ]
            eval_relation_score, eval_linking_score, eval_score = eval_score_tup
            _, _, optm_pos_score = optm_pos_score_tup
            _, _, optm_neg_score = optm_neg_score_tup
            show_tensor(eval_score)
            self.eval_output_tf_tup_list.append(('relation_score', eval_relation_score))
            self.eval_output_tf_tup_list.append(('linking_score', eval_linking_score))
        else:       # relation only
            eval_score_tup, optm_pos_score_tup, optm_neg_score_tup = [
                self.produce_relation_only_score(
                    tf_list=tf_list, mode=mode, name_scope=name_scope,
                    relation_kernel=relation_kernel,
                ) for tf_list, mode, name_scope in config_tups
            ]
            eval_pred_att_mat, eval_pword_att_mat, eval_score = eval_score_tup
            _, _, optm_pos_score = optm_pos_score_tup
            _, _, optm_neg_score = optm_neg_score_tup
            show_tensor(eval_score)
            if eval_pred_att_mat is not None:
                self.eval_output_tf_tup_list.append(('pred_att_mat', eval_pred_att_mat))
            if eval_pword_att_mat is not None:
                self.eval_output_tf_tup_list.append(('pword_att_mat', eval_pword_att_mat))

        # Note: Must ensure 'score' tensor is added, otherwise the evaluator fails to rank all the candidates.
        self.eval_output_tf_tup_list.append(('score', eval_score))
        LogInfo.begin_track('Showing outputs: ')
        for name, tensor in self.eval_output_tf_tup_list:
            LogInfo.logs('%s --> %s', name, tensor.get_shape().as_list())
        LogInfo.end_track()

        """ Build Loss & Update """
        margin_loss = tf.nn.relu(optm_neg_score + margin - optm_pos_score,
                                 name='margin_loss')  # (data_size, )
        weighted_margin_loss = tf.multiply(margin_loss,
                                           data_weights,
                                           name='weighted_margin_loss')
        self.avg_loss = tf.reduce_mean(weighted_margin_loss,
                                       name='avg_loss')
        tf.summary.scalar('avg_loss', self.avg_loss, collections=['optm'])
        optimizer = getattr(tf.train, optm_name)
        self.optm_step = optimizer(learning_rate).minimize(self.avg_loss)
        self.optm_summary = tf.summary.merge_all(key='optm')
        # self.eval_summary = tf.summary.merge_all(key='eval')
        LogInfo.logs('* Avg_loss and optm_step defined.')

        """ Add Concern Parameters """
        param_name_list = []
        if self.objective == 'normal':
            param_name_list = ['linking_fc/weights', 'overall_fc/weights']
        with tf.variable_scope('', reuse=True):
            for param_name in param_name_list:
                var = tf.get_variable(name=param_name)
                self.show_param_tf_tup_list.append((param_name, var))
        LogInfo.begin_track('Showing concern parameters: ')
        for name, tensor in self.show_param_tf_tup_list:
            LogInfo.logs('%s --> %s', name, tensor.get_shape().as_list())
        LogInfo.end_track()

        LogInfo.end_track()

    @staticmethod
    def create_tensors(q_max_len, path_max_len, pword_max_len, extra_len, name_prefix=''):
        """
        Given the necessary length information, create a group of tensors describing a bunch of <Q, sc> pairs
        :param q_max_len: maximum words of a question
        :param path_max_len: maximum length of a path
        :param pword_max_len: maximum length of words in a path
        :param extra_len: length of structural feature
        :param name_prefix: the prefix of tensor names
        :return: a list of tensors
        """
        # Following: copied from compq_rel_detect
        qwords_input = tf.placeholder(dtype=tf.int32,
                                      shape=[None, q_max_len],
                                      name=name_prefix+'q_words_input')  # (data_size, q_max_len)
        qwords_len_input = tf.placeholder(dtype=tf.int32,
                                          shape=[None],
                                          name=name_prefix+'q_words_len_input')  # (data_size, )
        sc_len_input = tf.placeholder(dtype=tf.int32,
                                      shape=[None],
                                      name=name_prefix+'sc_len_input')  # (data_size, )
        preds_input = tf.placeholder(dtype=tf.int32,
                                     shape=[None, None, path_max_len],
                                     name=name_prefix+'preds_input')  # (data_size, sc_max_len, path_max_len)
        preds_len_input = tf.placeholder(dtype=tf.int32,
                                         shape=[None, None],
                                         name=name_prefix+'preds_len_input')  # (data_size, sc_max_len)
        pwords_input = tf.placeholder(dtype=tf.int32,
                                      shape=[None, None, pword_max_len],
                                      name=name_prefix+'pwords_input')  # (data_size, sc_max_len, pword_max_len)
        pwords_len_input = tf.placeholder(dtype=tf.int32,
                                          shape=[None, None],
                                          name=name_prefix+'pwords_len_input')  # (data_size, sc_max_len)
        efeats_input = tf.placeholder(dtype=tf.float32,
                                      shape=[None, None, None],
                                      name=name_prefix+'efeats_input')  # (data_size, e_max_size, e_feat_size)
        etypes_input = tf.placeholder(dtype=tf.int32,
                                      shape=[None, None],
                                      name=name_prefix+'etypes_input')  # (data_size, e_max_size)
        emask_input = tf.placeholder(dtype=tf.float32,
                                     shape=[None, None],
                                     name=name_prefix+'emask_input')    # (data_size, e_max_size)
        extra_input = tf.placeholder(dtype=tf.float32,
                                     shape=[None, extra_len],
                                     name=name_prefix+'extra_input')  # (data_size, extra_len), for extra features
        return [qwords_input, qwords_len_input, sc_len_input,
                preds_input, preds_len_input,
                pwords_input, pwords_len_input,
                efeats_input, etypes_input, emask_input,
                extra_input]

    def produce_overall_score(self, tf_list, mode, name_scope, relation_kernel, entity_kernel):
        """
        Given all necessary tensors and kernel module, return the final score of <Q, sc> pairs
        :param tf_list: a input tensors
        :param mode: TRAIN / INFER
        :param name_scope: the corresponding name_scope (optm_pos, optm_neg, eval)
        :param relation_kernel: for relation matching
        :param entity_kernel: for entity linking
        """
        LogInfo.begin_track('Produce OVERALL score in [%s] ...', name_scope)
        assert mode in (tf.contrib.learn.ModeKeys.TRAIN, tf.contrib.learn.ModeKeys.INFER)
        [qwords_input, qwords_len_input, sc_len_input,
         preds_input, preds_len_input,
         pwords_input, pwords_len_input,
         efeats_input, etypes_input, emask_input, extra_input] = tf_list     # decomposition

        with tf.name_scope(name=name_scope):

            """ Step 1: Basic Embedding Lookup """
            with tf.device("/cpu:0"):
                qwords_embedding = tf.nn.embedding_lookup(
                    params=relation_kernel.w_embedding,
                    ids=qwords_input, name='q_embedding'
                )  # (data_size, q_max_len, dim_emb)
                preds_embedding = tf.nn.embedding_lookup(
                    params=relation_kernel.m_embedding,
                    ids=preds_input, name='preds_embedding'
                )  # (data_size, sc_max_len, path_max_len, dim_emb)
                pwords_embedding = tf.nn.embedding_lookup(
                    params=relation_kernel.w_embedding,
                    ids=pwords_input, name='pwords_embedding'
                )  # (data_size, sc_max_len, pword_max_len, dim_emb)
                etypes_embedding = tf.nn.embedding_lookup(
                    params=relation_kernel.m_embedding,
                    ids=etypes_input, name='etypes_embedding'
                )  # (data_size, e_max_size, dim_emb)

            """ Step 2: Relation Matching & Entity Linking """
            _, _, relation_score = relation_kernel.get_score(
                mode=mode, qwords_embedding=qwords_embedding,
                qwords_len=qwords_len_input, sc_len=sc_len_input,
                preds_embedding=preds_embedding, preds_len=preds_len_input,
                pwords_embedding=pwords_embedding, pwords_len=pwords_len_input
            )  # (data_size, ), also ignore the attention matrix information

            linking_score = entity_kernel.get_score(
                mode=mode, qwords_embedding=qwords_embedding, qwords_len=qwords_len_input,
                efeats=efeats_input, etypes_embedding=etypes_embedding, e_mask=emask_input
            )  # (data_size, )
            out_relation_score = relation_score
            out_linking_score = linking_score

            """ Step 3: Merge Features & Producing Final Score """
            with tf.name_scope('Overall_Merge'):
                relation_score = tf.expand_dims(relation_score, axis=-1,
                                                name='relation_score')  # (data_size, 1)
                linking_score = tf.expand_dims(linking_score, axis=-1,
                                               name='linking_score')    # (data_size, 1)
                hidden_feats = tf.concat([relation_score, linking_score, extra_input],
                                         axis=-1, name='final_feats')  # (data_size, 2 + extra_len)
                hidden_output = tf.contrib.layers.fully_connected(
                    inputs=hidden_feats,
                    num_outputs=1,
                    activation_fn=None,
                    scope='overall_fc',
                    reuse=self.reuse
                )  # (data_size, 1)
                overall_score = tf.reshape(hidden_output, shape=[-1],
                                           name='overall_score')  # (data_size, )

        LogInfo.end_track()
        return out_relation_score, out_linking_score, overall_score     # all (data_size, )

    @staticmethod
    def produce_relation_only_score(tf_list, mode, name_scope, relation_kernel):
        """
        Different from the function above, we only output the relation similarity score
        :param tf_list: a input tensors
        :param mode: TRAIN / INFER
        :param name_scope: the corresponding name_scope (optm_pos, optm_neg, eval)
        :param relation_kernel: for relation matching
        """
        LogInfo.begin_track('Produce RELATION MATCHING score in [%s] ...', name_scope)
        assert mode in (tf.contrib.learn.ModeKeys.TRAIN, tf.contrib.learn.ModeKeys.INFER)
        [qwords_input, qwords_len_input, sc_len_input,
         preds_input, preds_len_input,
         pwords_input, pwords_len_input, _, _, _, _] = tf_list     # decomposition

        with tf.name_scope(name=name_scope):

            """ Step 1: Basic Embedding Lookup """
            with tf.device("/cpu:0"):
                qwords_embedding = tf.nn.embedding_lookup(
                    params=relation_kernel.w_embedding,
                    ids=qwords_input, name='q_embedding'
                )  # (data_size, q_max_len, dim_emb)
                preds_embedding = tf.nn.embedding_lookup(
                    params=relation_kernel.m_embedding,
                    ids=preds_input, name='preds_embedding'
                )  # (data_size, sc_max_len, path_max_len, dim_emb)
                pwords_embedding = tf.nn.embedding_lookup(
                    params=relation_kernel.w_embedding,
                    ids=pwords_input, name='pwords_embedding'
                )  # (data_size, sc_max_len, pword_max_len, dim_emb)

            """ Step 2: Relation Matching """
            pred_att_mat, pword_att_mat, relation_score = relation_kernel.get_score(
                mode=mode, qwords_embedding=qwords_embedding,
                qwords_len=qwords_len_input, sc_len=sc_len_input,
                preds_embedding=preds_embedding, preds_len=preds_len_input,
                pwords_embedding=pwords_embedding, pwords_len=pwords_len_input
            )
            # pred_att_mat: (data_size, sc_max_len, q_max_len, path_max_len), could be None
            # pword_att_mat: (data_size, sc_max_len, q_max_len, pword_max_len), could be None
            # relation_score: (data_size, )
        LogInfo.end_track()
        return pred_att_mat, pword_att_mat, relation_score

    def evaluate(self, data_loader, epoch_idx, ob_batch_num=10, detail_fp=None, result_fp=None, summary_writer=None):
        return general_evaluate(eval_model=self,
                                data_loader=data_loader,
                                epoch_idx=epoch_idx,
                                ob_batch_num=ob_batch_num,
                                detail_fp=detail_fp,
                                result_fp=result_fp,
                                summary_writer=summary_writer)
