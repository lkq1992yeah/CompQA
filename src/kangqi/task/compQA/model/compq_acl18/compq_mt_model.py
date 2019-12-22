"""
Control the multi task module.
"""

import tensorflow as tf
import numpy as np

from .segment_kernel import SegmentKernel
from .compact_relation_matching_kernel import CompactRelationMatchingKernel
from .separate_relation_matching_kernel import SeparatedRelationMatchingKernel
from .noatt_relation_matching_kernel import NoAttRelationMatchingKernel
from .att_relation_matching_kernel import AttRelationMatchingKernel
from .entity_linking_kernel import EntityLinkingKernel
# from ..u import show_tensor

from kangqi.util.LogUtil import LogInfo


class CompqMultiTaskModel:

    def __init__(self, q_max_len, sc_max_len, path_max_len, pword_max_len,
                 type_dist_len, el_feat_size, extra_feat_size,
                 n_words, n_mids, dim_emb, seg_kernel_conf, el_kernel_conf, rm_kernel_conf,
                 use_clause, use_ans_type_dist, conn_name, loss_func, optm_name, learning_rate, full_back_prop=True):
        LogInfo.begin_track('Build [compq_acl18.CompqMultiTaskModel]:')

        self.dim_emb = dim_emb
        self.q_max_len = q_max_len
        self.sc_max_len = sc_max_len
        self.path_max_len = path_max_len
        self.pword_max_len = pword_max_len
        self.type_dist_len = type_dist_len
        self.extra_feat_size = extra_feat_size

        # one-hot auxiliary mask array, used for appending type_dist information to pw_emb and p_emb
        self.p_dir_mask = np.zeros((self.path_max_len,), dtype='int32')
        self.pw_dir_mask = np.zeros((self.pword_max_len,), dtype='int32')
        self.p_dir_mask[0] = 1
        self.pw_dir_mask[0] = 1

        self.n_clauses = 3
        self.dim_clause_emb = 50
        self.use_clause = use_clause
        self.use_ans_type_dist = use_ans_type_dist      # control whether to add the additional type distribution
        self.conn_name = conn_name
        assert optm_name in ('Adam', 'Adadelta', 'Adagrad', 'GradientDescent')
        self.learning_rate = learning_rate
        self.optimizer = getattr(tf.train, optm_name + 'Optimizer')
        self.margin = 0.

        assert loss_func.startswith('H')
        self.margin = float(loss_func[1:])  # "H0.5" --> 0.5

        # self.use_loss_func = ''
        # if loss_func.startswith('H'):
        #     self.use_loss_func = 'hinge'
        #     self.margin = float(loss_func[1:])  # "H0.5" --> 0.5
        # else:
        #     self.use_loss_func = 'ranknet'

        seg_kernel_conf['q_max_len'] = q_max_len
        self.seg_kernel = SegmentKernel(**seg_kernel_conf)
        LogInfo.logs('SegmentKernel defined.')

        el_kernel_conf['qw_max_len'] = q_max_len
        el_kernel_conf['el_max_len'] = sc_max_len
        self.el_kernel = EntityLinkingKernel(**el_kernel_conf)
        LogInfo.logs('EntityLinkingKernel defined.')

        self.rm_kernel_name = rm_kernel_conf['name']
        del rm_kernel_conf['name']
        rm_kernel_conf['qw_max_len'] = q_max_len
        rm_kernel_conf['sc_max_len'] = sc_max_len
        rm_kernel_conf['p_max_len'] = path_max_len
        rm_kernel_conf['pw_max_len'] = pword_max_len
        rm_kernel_conf['dim_emb'] = dim_emb
        if use_clause:      # enrich the embedding dimension
            rm_kernel_conf['dim_emb'] += self.dim_clause_emb
        if self.rm_kernel_name == 'ab2':
            self.rm_kernel = CompactRelationMatchingKernel(**rm_kernel_conf)
            LogInfo.logs('ABCNN-2 RelationMatchingKernel defined.')
        elif self.rm_kernel_name == 'ab1':
            self.rm_kernel = SeparatedRelationMatchingKernel(**rm_kernel_conf)
            LogInfo.logs('ABCNN-1 RelationMatchingKernel defined.')
        elif self.rm_kernel_name == 'noAtt':
            self.rm_kernel = NoAttRelationMatchingKernel(**rm_kernel_conf)
            LogInfo.logs('NoAtt RelationMatchingKernel defined.')
        elif self.rm_kernel_name == 'att':
            self.rm_kernel = AttRelationMatchingKernel(**rm_kernel_conf)
            LogInfo.logs('SimpleAtt RelationMatchingKernel defined.')
        else:
            self.rm_kernel = None
            LogInfo.logs('Error: Unknown relation matching config [%s].', self.rm_kernel_name)

        self.input_tensor_dict = self.input_tensor_definition(
            q_max_len=q_max_len, sc_max_len=sc_max_len,
            path_max_len=path_max_len, pword_max_len=pword_max_len,
            type_dist_len=type_dist_len, el_max_len=sc_max_len,
            el_feat_size=el_feat_size, extra_feat_size=extra_feat_size
        )
        LogInfo.logs('Global input tensors defined.')

        with tf.variable_scope('embedding_lookup', reuse=tf.AUTO_REUSE):
            with tf.device('/cpu:0'):
                self.w_embedding_init = tf.placeholder(
                    dtype=tf.float32, shape=(n_words, dim_emb), name='w_embedding_init')
                self.m_embedding_init = tf.placeholder(
                    dtype=tf.float32, shape=(n_mids, dim_emb), name='m_embedding_init')
                self.w_embedding = tf.get_variable(name='w_embedding', initializer=self.w_embedding_init)
                self.m_embedding = tf.get_variable(name='m_embedding', initializer=self.m_embedding_init)
                self.c_embedding = tf.get_variable(name='c_embedding', dtype=tf.float32,
                                                   shape=[self.n_clauses, self.dim_clause_emb])
                # TODO: clause embedding, current is: no-clause, out-clause, in-clause. works only for 'when'
        LogInfo.logs('Embedding & emb_lookup defined.')

        """ Define the input tensors for different tasks """
        v_only_names = ['v_input', 'v_len', 'clause_input']
        el_only_names = ['el_len', 'el_type_input', 'el_feats']
        rm_only_names = ['e_mask', 'tm_mask', 'gather_pos', 'qw_len',
                         'sc_len', 'p_input', 'pw_input', 'p_len', 'pw_len',
                         'type_dist', 'type_dist_weight']
        self.seg_input_names = v_only_names + ['tag_indices']
        self.el_eval_input_names = v_only_names + el_only_names
        self.rm_eval_input_names = v_only_names + rm_only_names
        self.full_eval_input_names = v_only_names + el_only_names + rm_only_names + ['extra_feats']
        self.el_optm_input_names = self.el_eval_input_names + ['data_weights']
        self.rm_optm_input_names = self.rm_eval_input_names + ['data_weights']
        self.full_optm_input_names = self.full_eval_input_names + ['data_weights']

        """ Build the main graph for both optm and eval """
        self.eval_tensor_dict = self.build_graph(mode_str='eval')
        self.optm_tensor_dict = self.build_graph(mode_str='optm')

        """ Build loss & update for different tasks """
        data_weight = tf.reshape(self.input_tensor_dict['data_weights'], shape=[-1, 2])  # (n_pair, 2)
        data_weight = tf.reduce_mean(data_weight, axis=-1, name='data_weight')  # (n_pair, )
        LogInfo.logs('[rm & el] loss function: Hinge-%.1f', self.margin)

        # Segment task
        # optm_seg_loglik = self.optm_tensor_dict['seg_loglik']
        # self.seg_loss = tf.reduce_mean(-1. * optm_seg_loglik, name='seg_loss')
        # self.seg_update, self.optm_seg_summary = self.build_update_summary(task='seg')
        # LogInfo.logs('Loss & update defined: [Segment]')

        # Entity Linking task
        el_weighted_loss = self.get_pair_loss(optm_score=self.optm_tensor_dict['el_score'],
                                              data_weight=data_weight)
        self.el_loss = tf.reduce_mean(el_weighted_loss, name='el_loss')
        self.el_update, self.optm_el_summary = self.build_update_summary(task='el')
        LogInfo.logs('Loss & update defined: [Entity Linking]')

        # Relation Matching task
        rm_weighted_loss = self.get_pair_loss(optm_score=self.optm_tensor_dict['rm_score'],
                                              data_weight=data_weight)
        self.rm_loss = tf.reduce_mean(rm_weighted_loss, name='rm_loss')
        self.rm_update, self.optm_rm_summary = self.build_update_summary(task='rm')
        LogInfo.logs('Loss & update defined: [Relation Matching]')

        # """ For debug in batch = 1"""
        # self.optm_tensor_dict['pos_rm_score'] = tf.reduce_sum(pos_rm_score)
        # self.optm_tensor_dict['neg_rm_score'] = tf.reduce_sum(neg_rm_score)
        # """"""

        # Full task
        full_weighted_loss = self.get_pair_loss(optm_score=self.optm_tensor_dict['full_score'],
                                                data_weight=data_weight)
        self.full_loss = tf.reduce_mean(full_weighted_loss, name='full_loss')
        LogInfo.logs('Full Back Propagate: %s', full_back_prop)
        if not full_back_prop:      # full task: only update the last FC layer
            with tf.variable_scope('', reuse=tf.AUTO_REUSE):
                spec_var_list = [
                    tf.get_variable(name=param_name)
                    for param_name in ['full_task/final_fc/weights',
                                       'full_task/final_fc/biases']
                ]
            self.full_update, self.optm_full_summary = self.build_update_summary(
                task='full', spec_var_list=spec_var_list
            )
        else:
            self.full_update, self.optm_full_summary = self.build_update_summary(task='full')
        LogInfo.logs('Loss & update defined: [Full Task]')

        LogInfo.end_track('End of model.')

    @staticmethod
    def input_tensor_definition(q_max_len, sc_max_len, path_max_len, pword_max_len,
                                type_dist_len, el_max_len, el_feat_size, extra_feat_size):
        """ Define input tensors of the whole model """
        input_tensor_dict = {
            'v_input': tf.placeholder(dtype=tf.int32, shape=[None, q_max_len], name='v_input'),
            'v_len': tf.placeholder(dtype=tf.int32, shape=[None], name='v_len'),
            'tag_indices': tf.placeholder(dtype=tf.int32, shape=[None, q_max_len], name='tag_indices'),

            'clause_input': tf.placeholder(dtype=tf.int32, shape=[None, q_max_len], name='clause_input'),
            # used for adding clause-based embedding

            # Now mask series
            # 'chunk_mat': tf.placeholder(dtype=tf.float32, shape=[None, q_max_len, q_max_len], name='chunk_mat')
            #  used for generating merge embedding of chunks
            'e_mask': tf.placeholder(dtype=tf.float32, shape=[None, q_max_len], name='e_mask'),
            # used for adding <E> placeholder
            'tm_mask': tf.placeholder(dtype=tf.float32, shape=[None, q_max_len], name='tm_mask'),
            # used for adding <Tm> placeholder
            'gather_pos': tf.placeholder(dtype=tf.int32, shape=[None, q_max_len], name='gather_pos'),
            # used to maintain the sequence of the chunk-ed question
            'qw_len': tf.placeholder(dtype=tf.int32, shape=[None], name='qw_len'),
            # length of chunk-ed question

            # Now entity information
            'el_len': tf.placeholder(dtype=tf.int32, shape=[None], name='el_len'),
            'el_type_input': tf.placeholder(dtype=tf.int32, shape=[None, el_max_len], name='el_type_input'),
            'el_feats': tf.placeholder(dtype=tf.float32, shape=[None, el_max_len, el_feat_size], name='el_feats'),

            # Now predicate id & name sequence detail
            'sc_len': tf.placeholder(dtype=tf.int32, shape=[None], name='sc_len'),
            'p_input': tf.placeholder(dtype=tf.int32, shape=[None, sc_max_len, path_max_len], name='p_input'),
            'pw_input': tf.placeholder(dtype=tf.int32, shape=[None, sc_max_len, pword_max_len], name='pw_input'),
            'p_len': tf.placeholder(dtype=tf.int32, shape=[None, sc_max_len], name='p_len'),
            'pw_len': tf.placeholder(dtype=tf.int32, shape=[None, sc_max_len], name='pw_len'),
            # controlling whether to use the additional explicit type distribution information
            'type_dist': tf.placeholder(dtype=tf.int32, shape=[None, type_dist_len], name='type_dist'),
            'type_dist_weight': tf.placeholder(dtype=tf.float32, shape=[None, type_dist_len], name='type_dist_weight'),

            # structural features of a schema
            'extra_feats': tf.placeholder(dtype=tf.float32, shape=[None, extra_feat_size], name='extra_feats'),
    
            'data_weights': tf.placeholder(dtype=tf.float32, shape=[None], name='data_weights')
            # used for EL / RE, optimize only
            # Note: When optimizing, it takes two rows to store the whole input of <pos, neg> pairs.
            #       In order to make the first dimension be equal across all input tensors,
            #       we actually store the weight of each <pos, neg> TWICE in 'data_weight' tensor.
        }
        return input_tensor_dict

    def build_graph(self, mode_str):
        """
        Build the whole multi-task computational graph for both optm and eval.
        Won't implement loss define & update in this function.
        :param mode_str: TRAIN / INFER
        """
        LogInfo.begin_track('Build graph: [MT-%s]', mode_str)
        mode = tf.contrib.learn.ModeKeys.INFER if mode_str == 'eval' else tf.contrib.learn.ModeKeys.TRAIN
        with tf.device('/cpu:0'):
            v_emb = tf.nn.embedding_lookup(params=self.w_embedding,
                                           ids=self.input_tensor_dict['v_input'],
                                           name='v_emb')
            pw_emb = tf.nn.embedding_lookup(params=self.w_embedding,
                                            ids=self.input_tensor_dict['pw_input'],
                                            name='pw_emb')  # (ds, sc_max_len, pword_max_len, dim_emb)
            p_emb = tf.nn.embedding_lookup(params=self.m_embedding,
                                           ids=self.input_tensor_dict['p_input'],
                                           name='p_emb')    # (ds, sc_max_len, path_max_len, dim_emb)
            el_type_emb = tf.nn.embedding_lookup(params=self.m_embedding,
                                                 ids=self.input_tensor_dict['el_type_input'],
                                                 name='el_type_emb')
            clause_emb = tf.nn.embedding_lookup(params=self.c_embedding,
                                                ids=self.input_tensor_dict['clause_input'],
                                                name='clause_emb')  # (ds, q_max_len, dim_clause_emb)
            type_dist_emb = tf.nn.embedding_lookup(params=self.m_embedding,
                                                   ids=self.input_tensor_dict['type_dist'],
                                                   name='type_dist_emb')    # (ds, type_dist_len, dim_emb)

        sc_len = self.input_tensor_dict['sc_len']  # (ds,)
        pw_len = self.input_tensor_dict['pw_len']  # (ds, sc_max_len)
        p_len = self.input_tensor_dict['p_len']  # (ds, sc_max_len)
        """ pre-process, adding explicit type distribution into p_emb and pw_emb """
        if self.use_ans_type_dist:
            LogInfo.logs('Induced answer types put into use ...')
            sc_len, pw_emb, pw_len, p_emb, p_len = self.apply_type_dist(
                type_dist_emb=type_dist_emb, old_sc_len=sc_len,
                old_pw_emb=pw_emb, old_pw_len=pw_len,
                old_p_emb=p_emb, old_p_len=p_len
            )

        """ Seg Task """
        v_hidden = None
        # v_hidden, seg_logits, seg_loglik, best_seg = self.seg_kernel.forward(
        #     v_emb=v_emb, v_len=self.input_tensor_dict['v_len'],
        #     tag_indices=self.input_tensor_dict['tag_indices'], mode=mode)

        """ Connection """
        dim_conn_emb = self.dim_emb  # the dimension of embedding in the connection part
        seg_emb = None
        LogInfo.logs('Connection data flow: %s', self.conn_name)
        if self.conn_name == 'Vemb':
            seg_emb = v_emb  # (ds, q_max_len, dim_conn_emb == dim_emb)
        elif self.conn_name == 'Vhidden':
            seg_emb = v_hidden              # imagine 2-layer RNN
        elif self.conn_name == 'Vresidual':
            assert v_emb.get_shape().as_list()[-1] == v_hidden.get_shape().as_list()[-1]
            seg_emb = v_emb + v_hidden      # imagine 2-layer RNN + Residual

        qw_el_emb, qw_rm_emb = self.connection(seg_emb=seg_emb, clause_emb=clause_emb, dim_conn_emb=dim_conn_emb)
        # qw_el_emb: (ds, q_max_len, dim_conn_emb)
        # qw_rm_emb: (ds, qw_max_len, dim_conn_emb), pay attention to the differences

        """ EL Task """
        el_score, el_feats_concat, el_raw_score = self.el_kernel.forward(
            qw_emb=qw_el_emb, qw_len=self.input_tensor_dict['v_len'],       # Note: no chunk or placeholders for EL
            el_type_emb=el_type_emb, el_feats=self.input_tensor_dict['el_feats'],
            el_len=self.input_tensor_dict['el_len'], mode=mode
        )
        # el_score: (ds, )
        # el_feats_concat: (ds, el_max_len, dim_el_hidden + el_feat_size)
        # el_raw_score: (ds, el_max_len)

        """ RM Task """
        rm_ret_dict = self.rm_kernel.forward(
            qw_emb=qw_rm_emb, qw_len=self.input_tensor_dict['qw_len'],
            sc_len=sc_len, p_emb=p_emb, pw_emb=pw_emb,
            p_len=p_len, pw_len=pw_len, mode=mode
        )   # possible items: rm_score, rm_path_score, rm_att_mat, rm_q_weight, rm_path_weight
        # rm_score: (ds, )
        # rm_path_score: (ds, sc_max_len)
        # rm_att_mat: (ds, sc_max_len, qw_max_len, merge_max_len)
        # rm_q_weight: (ds, sc_max_len, qw_max_len)
        # rm_path_weight: (ds, sc_max_len, merge_max_len)
        LogInfo.logs('rm_ret_dict: %s', rm_ret_dict.keys())

        """ Full Task"""
        rm_score = rm_ret_dict['rm_score']
        with tf.variable_scope('full_task', reuse=tf.AUTO_REUSE):
            rich_el_score = tf.expand_dims(el_score, axis=-1, name='rich_el_score')
            rich_rm_score = tf.expand_dims(rm_score, axis=-1, name='rich_rm_score')
            rich_feats_concat = tf.concat([rich_el_score, rich_rm_score,
                                           self.input_tensor_dict['extra_feats']],
                                          axis=-1, name='rich_feats_concat')        # (ds, 2 + extra_feat_size)

            """
            180315: try force setting the initial parameters .
            RM feature is 4 times more effective than EL feature.
            """
            final_feat_len = 2 + self.extra_feat_size
            init_w_value = np.zeros((final_feat_len, 1), dtype='float32')
            init_w_value[0] = 0.25
            init_w_value[1] = 1

            with tf.variable_scope('final_fc', reuse=tf.AUTO_REUSE):
                weights = tf.get_variable(name='weights', initializer=tf.constant(init_w_value))
                biases = tf.get_variable(name='biases', shape=(1,))
                full_score = tf.matmul(rich_feats_concat, weights) + biases         # (ds, 1)
            # full_score = tf.contrib.layers.fully_connected(
            #     inputs=rich_feats_concat, num_outputs=1,
            #     activation_fn=None, scope='final_fc', reuse=tf.AUTO_REUSE
            # )  # (ds, 1)
            full_score = tf.squeeze(full_score, axis=-1, name='full_score')  # (ds, )

        tensor_dict = {
            # 'v_hidden': v_hidden,
            # 'seg_logits': seg_logits,
            # 'seg_loglik': seg_loglik,
            # 'best_seg': best_seg,
            'el_score': el_score,
            'el_feats_concat': el_feats_concat,
            'el_raw_score': el_raw_score,
            'rich_feats_concat': rich_feats_concat,
            'full_score': full_score
        }
        for k, v in rm_ret_dict.items():
            tensor_dict[k] = v

        LogInfo.end_track('%d tensors saved and return.', len(tensor_dict))
        return tensor_dict

    def apply_type_dist(self, type_dist_emb, old_sc_len, old_pw_emb, old_pw_len, old_p_emb, old_p_len):
        """
        put the type_dist_emb into p_emb and pw_emb, then updating all related length tensors.
        """
        """ First: calculate aggregate type embedding by weighted sum """
        agg_type_emb = tf.reduce_sum(
            type_dist_emb * tf.expand_dims(self.input_tensor_dict['type_dist_weight'], axis=-1),
            axis=1, name='agg_type_emb'
        )  # (ds, dim_emb)
        agg_type_emb = tf.reshape(agg_type_emb, shape=[-1, 1, 1, self.dim_emb],
                                  name='agg_type_emb')     # (ds, 1, 1, dim_emb)
        # show_tensor(agg_type_emb)

        """ Now: update sc_len, p_emb, pw_emb, p_len, pw_len """
        position_mask = tf.one_hot(indices=old_sc_len, depth=self.sc_max_len,
                                   dtype=tf.int32, name='position_mask')
        # (ds, sc_max_len) as int32, value=1 at the position where type_dist is added
        # show_tensor(position_mask)

        delta_p_len = tf.multiply(
            tf.stack([position_mask] * self.path_max_len, axis=-1),  # (ds, sc_max_len, path_max_len)
            self.p_dir_mask, name='delta_p_len'
        )
        # (ds, sc_max_len, path_max_len) as int32,
        # fit the position_mask to the same shape as p_len
        # delta_p_len[ds_i, sc_len_j, 0] = 1, when sc_len[ds_i] = sc_len_j
        delta_p_emb = tf.multiply(
            tf.expand_dims(tf.cast(delta_p_len, dtype=tf.float32),
                           axis=-1),       # (ds, sc_max_len, path_max_len, 1)
            agg_type_emb,                  # (ds, 1, 1, dim_emb)
            name='delta_p_emb'
        )   # (ds, sc_max_len, path_max_len, dim_emb)
        # show_tensor(delta_p_len)
        # show_tensor(delta_p_emb)

        delta_pw_len = tf.multiply(
            tf.stack([position_mask] * self.pword_max_len, axis=-1),
            self.pw_dir_mask, name='delta_pw_len'
        )       # (ds, sc_max_len, pword_max_len)
        delta_pw_emb = tf.multiply(
            tf.expand_dims(tf.cast(delta_pw_len, dtype=tf.float32), axis=-1),
            agg_type_emb, name='delta_pw_emb'
        )       # (ds, sc_max_len, pword_max_len, dim_emb)
        # show_tensor(delta_pw_len)
        # show_tensor(delta_pw_emb)

        sc_len = old_sc_len + 1
        pw_emb = old_pw_emb + delta_pw_emb      # (ds, sc_max_len, pw_max_len, dim_emb)
        pw_len = old_pw_len + position_mask     # (ds, sc_max_len)
        p_emb = old_p_emb + delta_p_emb         # (ds, sc_max_len, p_max_len, dim_emb)
        p_len = old_p_len + position_mask       # (ds, sc_max_len)
        return sc_len, pw_emb, pw_len, p_emb, p_len

    def connection(self, seg_emb, clause_emb, dim_conn_emb):
        """
        define the connector from input v to qw_rm and qw_el.
        :param seg_emb: (ds, q_max_len, dim_conn_emb)
                        the combined embedding of each word after segment kernel.
        :param clause_emb: (ds, q_max_len, dim_clause_emb)
                           the "clause-positional" embedding
        :param dim_conn_emb: the last dimension of seg_emb,
                             which is also the input dimension size of RM and EL kernel.
        :return: qw_rm_emb and qw_el_emb. Both are (ds, q_max_len, dim_conn_emb)
        """
        """
        Kangqi on 180212:
        A "chunk_mat" could help us convert word level embedding at phrase level.
        But is it necessary?
        We never use the mention phrase representation in the RM kernel (replaced by <E>),
        while in the EL kernel, yes we try to figure out the association between 
        a candidate entity and contexts (mention phrase and other words).
        However, since we adopt the attention mechanism (weighted sum strategy),
        if the phrase embedding is just the average embedding of words, 
        then: is "weighted phrases" really different from "weighted words"? I don't think so.
        Therefore, for easing our model, 
            * I decide not introducing "chunk_mat" now;
            * Input of EL kernel is the embedding at word level (not related to phrases);
            * gather() is only applied at the part of RM kernel.    
        """
        # chunk_emb = tf.matmul(self.chunk_mat, raw_emb, name='chunk_emb')     # (ds, q_max_len, dim_emb)
        # # working as batch matrix multiplication
        # # used for generating the input of EL kernel

        LogInfo.logs('dim_conn_emb = %d', dim_conn_emb)
        dim_active_emb = dim_conn_emb       # the real input dimension of EL and RM
        if self.use_clause:
            dim_active_emb += self.dim_clause_emb

        qw_el_emb = seg_emb     # As noted above, just use the embedding at word level
        if self.use_clause:
            LogInfo.logs('Concatenate clause_embedding with [qw_el_emb].')
            qw_el_emb = tf.concat([qw_el_emb, clause_emb], axis=-1, name='qw_el_emb')

        with tf.variable_scope('embedding_lookup', reuse=tf.AUTO_REUSE):
            e_ph_emb = tf.get_variable(dtype=tf.float32, shape=(dim_conn_emb,), name='e_ph_emb')
            tm_ph_emb = tf.get_variable(dtype=tf.float32, shape=(dim_conn_emb,), name='tm_ph_emb')
            # the embedding variable of <E> and <Tm>

        e_mask = tf.expand_dims(self.input_tensor_dict['e_mask'], axis=-1)       # (ds, q_max_len, 1)
        tm_mask = tf.expand_dims(self.input_tensor_dict['tm_mask'], axis=-1)     # (ds, q_max_len, 1)
        word_mask = 1.0 - e_mask - tm_mask
        seg_with_ph_emb = seg_emb * word_mask + e_ph_emb * e_mask + tm_ph_emb * tm_mask
        # (ds, q_max_len, dim_conn_emb): as the embedding of the question with placeholders
        # show_tensor(seg_with_ph_emb)

        if self.use_clause:
            LogInfo.logs('Concatenate clause embedding with [seg_with_ph_emb].')
            seg_with_ph_emb = tf.concat([seg_with_ph_emb, clause_emb], axis=-1, name='seg_with_ph_emb')
            # (ds, q_max_len, dim_active_emb == dim_conn_emb + dim_clause_emb)

        gather_pos = self.input_tensor_dict['gather_pos']
        ds = tf.shape(gather_pos)[0]
        gather_params = tf.reshape(seg_with_ph_emb, shape=[-1, dim_active_emb],
                                   name='gather_params')   # (ds * q_max_len, dim_active_emb)
        rg = tf.range(ds) * self.q_max_len      # [0*q_max_len, 1*q_max_len, 2*q_max_len, ...]
        offset = tf.stack([rg] * self.q_max_len, axis=-1)   # (ds, q_max_len)
        # [[0, 0, 0, ..., 0],
        #  [q_max_len, q_max_len, ..., q_max_len]],
        #  ......
        #  [(ds-1)*q_max_len, ..., (ds-1)*q_max_len]]
        gather_indices = tf.reshape(offset + gather_pos, shape=[-1],
                                    name='gather_indices')          # (ds * q_max_len)
        gather_value = tf.gather(
            params=gather_params,           # (ds * q_max_len, dim_active_emb)
            indices=gather_indices,         # (ds * q_max_len, )
            axis=0, name='gather_value'
        )       # (ds * q_max_len, dim_active_emb),
        # show_tensor(gather_params)
        # show_tensor(gather_indices)
        # show_tensor(gather_value)
        qw_rm_emb = tf.reshape(gather_value, shape=[-1, self.q_max_len, dim_active_emb],
                               name='qw_rm_emb')        # (ds, q_max_len, dim_conn_emb)

        return qw_el_emb, qw_rm_emb

    def get_pair_loss(self, optm_score, data_weight):
        """
        :param optm_score:  (ds, )
        :param data_weight: (n_pair, )
        Note: ds = n_pair * 2
        in TRAIN mode, we put positive and negative cases together into one tensor
        """
        pos_score, neg_score = tf.unstack(tf.reshape(optm_score, shape=[-1, 2]), axis=1)
        # if self.use_loss_func == 'hinge':
        margin_loss = tf.nn.relu(neg_score + self.margin - pos_score, name='margin_loss')
        weighted_loss = tf.multiply(margin_loss, data_weight, name='weighted_margin_loss')
        # else:       # ranknet
        #     logits = pos_score - neg_score
        #     rn_loss = -1. * tf.log(tf.clip_by_value(tf.nn.sigmoid(logits),
        #                                             clip_value_min=1e-6,
        #                                             clip_value_max=1.0))      # binary cross-entropy
        #     weighted_loss = tf.multiply(rn_loss, data_weight, name='weighted_ranknet_loss')
        return weighted_loss

    def build_update_summary(self, task, spec_var_list=None):
        collection_name = 'optm_%s' % task      # optm_seg, optm_rm, ...
        loss_name = '%s_loss' % task            # seg_loss, rm_loss, ...
        task_loss = getattr(self, loss_name)
        tf.summary.scalar(loss_name, getattr(self, loss_name), collections=[collection_name])
        if spec_var_list is None:
            update_step = self.optimizer(self.learning_rate).minimize(task_loss)
        else:
            update_step = self.optimizer(self.learning_rate).minimize(task_loss, var_list=spec_var_list)
        optm_summary = tf.summary.merge_all(collection_name)
        return update_step, optm_summary
