"""
Fighting for EMNLP18!!
Major changes: paths serving as vocabularies; MemNN-like structure are used
Unnecessary codes are removed.
"""

import tensorflow as tf

from .relation_matching_model import RelationMatchingKernel
from .entity_linking_kernel import EntityLinkingKernel
from .mem_entity_linking_kernel import MemEntityLinkingKernel
from ..u import show_tensor

from kangqi.util.LogUtil import LogInfo


class CompqMultiTaskModel:

    def __init__(self, path_max_size, qw_max_len, pw_max_len,
                 el_feat_size, extra_feat_size, n_words, dim_emb,
                 pw_voc_inputs, pw_voc_length, pw_voc_domain, entity_type_matrix,
                 rnn_config, att_config, seq_merge_mode, scoring_mode,
                 loss_func, optm_name, learning_rate, full_back_prop):
        LogInfo.begin_track('Build [compq_emnlp18.CompqMultiTaskModel]:')

        self.dim_emb = dim_emb
        self.qw_max_len = qw_max_len
        # self.path_max_size = self.el_max_len = sc_max_len

        self.dim_type = entity_type_matrix.shape[1]
        # self.sup_max_len = sup_max_len
        self.el_feat_size = el_feat_size
        self.extra_feat_size = extra_feat_size

        assert optm_name in ('Adam', 'Adadelta', 'Adagrad', 'GradientDescent')
        self.learning_rate = learning_rate
        self.optimizer = getattr(tf.train, optm_name + 'Optimizer')
        assert loss_func.startswith('H')
        self.margin = float(loss_func[1:])  # "H0.5" --> 0.5

        rnn_cell_class = rnn_config['cell_class']
        att_func = att_config['att_func']
        if rnn_cell_class == 'None':
            LogInfo.logs('RNN Cell: None')
        else:
            LogInfo.logs('RNN Cell: %s-%d', rnn_cell_class, rnn_config['num_units'])
        LogInfo.logs('Attention: %s', att_func)
        self.rnn_config = rnn_config if rnn_cell_class != 'None' else None
        self.att_config = att_config if att_func != 'noAtt' else None

        self.rm_kernel = RelationMatchingKernel(
            dim_emb=dim_emb, qw_max_len=qw_max_len,
            seq_merge_mode=seq_merge_mode, scoring_mode=scoring_mode,
            rnn_config=self.rnn_config, att_config=self.att_config
        )
        # self.el_kernel = EntityLinkingKernel(
        #     dim_emb=dim_emb, qw_max_len=qw_max_len, pw_max_len=pw_max_len,
        #     seq_merge_mode=seq_merge_mode, scoring_mode=scoring_mode,
        #     rnn_config=self.rnn_config, att_config=self.att_config
        # )
        self.el_kernel = MemEntityLinkingKernel(
            dim_emb=dim_emb, qw_max_len=qw_max_len,
            seq_merge_mode=seq_merge_mode, scoring_mode=scoring_mode,
            rnn_config=self.rnn_config, att_config=self.att_config
        )
        """ RM & EL kernel defined """

        self.input_tensor_dict = self.input_tensor_definition(
            qw_max_len=qw_max_len,
            path_max_size=path_max_size, el_max_size=path_max_size,
            el_feat_size=el_feat_size, extra_feat_size=extra_feat_size
        )
        LogInfo.logs('Global input tensors defined.')

        with tf.variable_scope('embedding_lookup', reuse=tf.AUTO_REUSE):
            with tf.device('/cpu:0'):
                self.w_embedding_init = tf.placeholder(dtype=tf.float32,
                                                       shape=(n_words, dim_emb), name='w_embedding_init')
                self.w_embedding = tf.get_variable(name='w_embedding', initializer=self.w_embedding_init)
                # self.m_embedding_init = tf.placeholder(dtype=tf.float32,
                #                                        shape=(n_mids, dim_emb), name='m_embedding_init')
                # self.m_embedding = tf.get_variable(name='m_embedding', initializer=self.m_embedding_init)
                """ Below: path & entity-type lookup table
                pw_voc_inputs: (P, pw_max_len)
                pw_voc_length/domain: (P, )
                et_lookup: (E, T) """
                self.pw_voc_inputs = tf.constant(pw_voc_inputs, dtype=tf.int32, name='pw_voc_inputs')
                self.pw_voc_length = tf.constant(pw_voc_length, dtype=tf.int32, name='pw_voc_length')
                self.pw_voc_domain = tf.constant(pw_voc_domain, dtype=tf.int32, name='pw_voc_domain')
                self.et_lookup = tf.constant(entity_type_matrix, dtype=tf.float32, name='et_lookup')

        """ Define tensors for different tasks """
        rm_only_names = ['rm_qw_input', 'rm_qw_len', 'path_size', 'path_ids']
        # el_only_names = ['el_qw_input', 'el_qw_len', 'el_size', 'el_ids',
        #                  'el_indv_feats', 'el_comb_feats', 'path_sup_ids', 'path_sup_size']
        el_only_names = ['el_qw_input', 'el_qw_len', 'el_size', 'el_ids',
                         'el_indv_feats', 'el_comb_feats', 'el_sup_mask', 'local_sup_lookup']
        self.rm_eval_input_names = self.rm_optm_input_names = rm_only_names
        self.el_eval_input_names = self.el_optm_input_names = el_only_names
        self.full_eval_input_names = self.full_optm_input_names = rm_only_names + el_only_names + ['extra_feats']

        """ Build the main graph for both optm and eval """
        self.eval_tensor_dict = self.build_graph(mode_str='eval')
        self.optm_tensor_dict = self.build_graph(mode_str='optm')

        """ Loss & Update """
        LogInfo.logs('[rm & el] loss function: Hinge-%.1f', self.margin)
        # Entity Linking task
        el_weighted_loss = self.get_pair_loss(optm_score=self.optm_tensor_dict['el_score'])
        self.el_loss = tf.reduce_mean(el_weighted_loss, name='el_loss')
        self.el_update, self.optm_el_summary = self.build_update_summary(task='el')
        LogInfo.logs('Loss & update defined: [Entity Linking]')

        # Relation Matching task
        rm_weighted_loss = self.get_pair_loss(optm_score=self.optm_tensor_dict['rm_score'])
        self.rm_loss = tf.reduce_mean(rm_weighted_loss, name='rm_loss')
        self.rm_update, self.optm_rm_summary = self.build_update_summary(task='rm')
        LogInfo.logs('Loss & update defined: [Relation Matching]')

        # # Full task
        # full_weighted_loss = self.get_pair_loss(optm_score=self.optm_tensor_dict['full_score'])
        # self.full_loss = tf.reduce_mean(full_weighted_loss, name='full_loss')
        # LogInfo.logs('Full Back Propagate: %s', full_back_prop)
        # if not full_back_prop:  # full task: only update the last FC layer
        #     with tf.variable_scope('', reuse=tf.AUTO_REUSE):
        #         spec_var_list = [
        #             tf.get_variable(name=param_name)
        #             for param_name in ['full_task/final_fc/weights',
        #                                'full_task/final_fc/biases']
        #         ]
        #     self.full_update, self.optm_full_summary = self.build_update_summary(
        #         task='full', spec_var_list=spec_var_list
        #     )
        # else:
        #     self.full_update, self.optm_full_summary = self.build_update_summary(task='full')
        # LogInfo.logs('Loss & update defined: [Full Task]')

        # """ For debug in batch = 1"""
        # self.optm_tensor_dict['pos_rm_score'] = tf.reduce_sum(pos_rm_score)
        # self.optm_tensor_dict['neg_rm_score'] = tf.reduce_sum(neg_rm_score)
        # """"""

        LogInfo.end_track('End of Model')

    @staticmethod
    def input_tensor_definition(qw_max_len, path_max_size, el_max_size,
                                el_feat_size, extra_feat_size):
        input_tensor_dict = {
            # Part 1: Relation Matching
            # note: only need path id, instead of detail path sequence
            'rm_qw_input': tf.placeholder(dtype=tf.int32, shape=[None, path_max_size, qw_max_len], name='rm_qw_input'),
            'rm_qw_len': tf.placeholder(dtype=tf.int32, shape=[None, path_max_size], name='rm_qw_len'),
            # TODO: Position Encoding
            'rm_dep_input': tf.placeholder(dtype=tf.int32, shape=[None, path_max_size, qw_max_len],
                                           name='rm_dep_input'),
            'rm_dep_len': tf.placeholder(dtype=tf.int32, shape=[None, path_max_size], name='rm_dep_len'),
            'path_size': tf.placeholder(dtype=tf.int32, shape=[None], name='path_size'),
            'path_ids': tf.placeholder(dtype=tf.int32, shape=[None, path_max_size], name='path_ids'),

            # Part 2: Entity Linking
            # note: including 'path_sup_ids', which may consume lots of memories
            'el_qw_input': tf.placeholder(dtype=tf.int32, shape=[None, path_max_size, qw_max_len], name='el_qw_input'),
            'el_qw_len': tf.placeholder(dtype=tf.int32, shape=[None, path_max_size], name='el_qw_len'),
            # TODO: Position Encoding
            'el_size': tf.placeholder(dtype=tf.int32, shape=[None], name='el_size'),
            'el_ids': tf.placeholder(dtype=tf.int32, shape=[None, el_max_size], name='el_ids'),
            'el_indv_feats': tf.placeholder(dtype=tf.float32, shape=[None, el_max_size, el_feat_size],
                                            name='el_indv_feats'),
            'el_comb_feats': tf.placeholder(dtype=tf.float32, shape=[None, 1], name='el_comb_feats'),
            # 'path_sup_ids': tf.placeholder(dtype=tf.int32, shape=[None, el_max_size, None], name='path_sup_ids'),
            # 'path_sup_size': tf.placeholder(dtype=tf.int32, shape=[None, el_max_size], name='path_sup_size'),
            # Commented above: would introduce (ds, el_max_size, sup_max_size) when working on RNN
            # which costs lots of wasted memory

            'el_sup_mask': tf.placeholder(dtype=tf.float32, shape=[None, el_max_size, None], name='el_sup_mask'),
            # (batch, el_max_size, local_mem_size)

            # TODO: enable local_lookup, if full path vocabulary doesn't work
            'local_sup_lookup': tf.placeholder(dtype=tf.int32, shape=[None], name='local_sup_lookup'),
            # IMPORTANT: the shape is not (batch,), but (local_mem_size,)
            # for each batch, we provide such lookup table, which can remove irrelevant paths w.r.t this batch

            # Part 3: Additional
            'extra_feats': tf.placeholder(dtype=tf.float32, shape=[None, extra_feat_size], name='extra_feats'),
        }
        return input_tensor_dict

    def build_graph(self, mode_str):
        LogInfo.begin_track('Build graph: [MT-%s]', mode_str)
        mode = tf.contrib.learn.ModeKeys.INFER if mode_str == 'eval' else tf.contrib.learn.ModeKeys.TRAIN

        # TODO: Currently, try first embedding lookup, then calculating RNN
        # TODO: If we can reduce useless calculations (RNN over masking sequences), then it would be better.

        """ Part 1: Extract related predicates from memory """
        with tf.device('/cpu:0'):
            """ For RM kernel """
            rm_qw_emb = tf.nn.embedding_lookup(params=self.w_embedding,
                                               ids=self.input_tensor_dict['rm_qw_input'],
                                               name='rm_qw_emb')  # (ds, path_max_size, qw_max_len, dim_emb)
            rm_dep_emb = tf.nn.embedding_lookup(params=self.w_embedding,
                                                ids=self.input_tensor_dict['rm_dep_input'],
                                                name='rm_dep_emb')  # (ds, path_max_size, qw_max_len, dim_emb)
            pw_input = tf.nn.embedding_lookup(params=self.pw_voc_inputs,
                                              ids=self.input_tensor_dict['path_ids'],
                                              name='pw_input')  # (ds, path_max_size, pw_max_len)
            pw_emb = tf.nn.embedding_lookup(params=self.w_embedding,
                                            ids=pw_input,
                                            name='pw_emb')      # (ds, path_max_size, pw_max_len, dim_emb)
            pw_len = tf.nn.embedding_lookup(params=self.pw_voc_length,
                                            ids=self.input_tensor_dict['path_ids'],
                                            name='pw_len')      # (ds, path_max_size)

            """ For EL kernel """
            el_qw_emb = tf.nn.embedding_lookup(params=self.w_embedding,
                                               ids=self.input_tensor_dict['el_qw_input'],
                                               name='el_qw_emb')  # (ds, el_max_size, qw_max_len, dim_emb)
            el_type_signa = tf.nn.embedding_lookup(params=self.et_lookup,
                                                   ids=self.input_tensor_dict['el_ids'],
                                                   name='el_type_signa')    # (ds, el_max_size, dim_type)
            """ Below: non-MemNN implementation """
            # pw_sup_input = tf.nn.embedding_lookup(params=self.pw_voc_inputs,
            #                                       ids=self.input_tensor_dict['path_sup_ids'],
            #                                       name='pw_sup_input')
            # # (ds, el_max_size, sup_max_size, pw_max_len)
            # pw_sup_emb = tf.nn.embedding_lookup(params=self.w_embedding,
            #                                     ids=pw_sup_input,
            #                                     name='pw_sup_emb')
            # # (ds, el_max_size, sup_max_size, pw_max_len, dim_emb)
            # pw_sup_len = tf.nn.embedding_lookup(params=self.pw_voc_length,
            #                                     ids=self.input_tensor_dict['path_sup_ids'],
            #                                     name='pw_sup_len')
            # # (ds, el_max_size, sup_max_size)
            # pw_sup_domain = tf.nn.embedding_lookup(params=self.pw_voc_domain,
            #                                        ids=self.input_tensor_dict['path_sup_ids'],
            #                                        name='pw_sup_domain')
            # # (ds, el_max_size, sup_max_size)

            """ Below: MemNN implementation """
            # TODO: local_mem_size == global_mem_size: costs more memory, but code is easier
            # local_pw_sup_inputs = self.pw_voc_inputs
            # local_pw_sup_length = self.pw_voc_length
            # local_pw_sup_domain = self.pw_voc_domain
            # TODO: different local support for different batches, need lookup
            local_pw_sup_inputs = tf.nn.embedding_lookup(
                params=self.pw_voc_inputs,
                ids=self.input_tensor_dict['local_sup_lookup'],
                name='local_sup_pw_inputs'
            )    # (local_mem_size, pw_max_len)
            local_pw_sup_length = tf.nn.embedding_lookup(
                params=self.pw_voc_length,
                ids=self.input_tensor_dict['local_sup_lookup'],
                name='local_sup_pw_length'
            )   # (local_mem_size,)
            local_pw_sup_domain = tf.nn.embedding_lookup(
                params=self.pw_voc_domain,
                ids=self.input_tensor_dict['local_sup_lookup'],
                name='local_sup_pw_domain'
            )  # (local_mem_size,)

            pw_sup_emb = tf.nn.embedding_lookup(
                params=self.w_embedding,
                ids=local_pw_sup_inputs,
                name='pw_sup_emb'
            )  # (local_mem_size, pw_max_len, dim_emb)

        """ Part 2: EL task (MemNN implementation) """
        """ IMPORTANT: calculate predicate encoding of the full vocabulary at once """
        type_trans = tf.one_hot(
            indices=local_pw_sup_domain, depth=self.dim_type,
            dtype=tf.float32, name='type_trans'
        )    # (local_mem_size, dim_type)
        el_score, el_final_feats = self.el_kernel.forward(
            el_size=self.input_tensor_dict['el_size'],
            qw_emb=el_qw_emb, qw_len=self.input_tensor_dict['el_qw_len'],
            pw_sup_emb=pw_sup_emb, pw_sup_len=local_pw_sup_length,
            el_sup_mask=self.input_tensor_dict['el_sup_mask'],
            type_trans=type_trans, el_type_signa=el_type_signa,
            el_indv_feats=self.input_tensor_dict['el_indv_feats'],
            el_comb_feats=self.input_tensor_dict['el_comb_feats'],
            mode=mode
        )

        # """ Part 2: EL task """
        # """ IMPORTANT: First calculate predicate encoding, then attention """
        # type_trans = tf.one_hot(indices=pw_sup_domain, depth=self.dim_type,
        #                         dtype=tf.float32, name='type_trans')    # (ds, el_max_size, sup_max_size, dim_type)
        # # show_tensor(type_trans)
        # # show_tensor(el_type_signa)
        # el_score, el_final_feats = self.el_kernel.forward(
        #     el_size=self.input_tensor_dict['el_size'],
        #     qw_emb=el_qw_emb, qw_len=self.input_tensor_dict['el_qw_len'],
        #     pw_sup_emb=pw_sup_emb, pw_sup_len=pw_sup_len,
        #     sup_size=self.input_tensor_dict['path_sup_size'],
        #     type_trans=type_trans, el_type_signa=el_type_signa,
        #     el_indv_feats=self.input_tensor_dict['el_indv_feats'],
        #     el_comb_feats=self.input_tensor_dict['el_comb_feats'],
        #     mode=mode
        # )

        """ Part 3: RM task """
        # qw_emb/pw_emb: (ds, path_max_size, qw_max_len, dim_emb)
        # qw_len/pw_len: (ds, path_max_size)  [Don't forget "path_max_size"!!]
        # path_size: (ds,)

        use_qw_emb = rm_qw_emb      # TODO: position encoding if possible
        use_dep_emb = rm_dep_emb

        # TODO: path repr
        # TODO: qw_repr given path
        # TODO: dep_repr given path
        # TODO: final_merge

        rm_ret_dict = self.rm_kernel.forward(
            qw_emb=rm_qw_emb, qw_len=self.input_tensor_dict['rm_qw_len'],
            pw_emb=pw_emb, pw_len=pw_len,
            path_size=self.input_tensor_dict['path_size'], mode=mode
        )   # Possible items: rm_score, rm_att_mat, rm_q_weight, rm_path_score
        LogInfo.logs('rm_ret_dict: %s', rm_ret_dict.keys())

        """ Part 4: Full task """
        # rm_score = rm_ret_dict['rm_score']
        # with tf.variable_scope('full_task', reuse=tf.AUTO_REUSE):
        #     rich_el_score = tf.expand_dims(el_score, axis=-1, name='rich_el_score')
        #     rich_rm_score = tf.expand_dims(rm_score, axis=-1, name='rich_rm_score')
        #     rich_feats_concat = tf.concat([rich_el_score, rich_rm_score,
        #                                   self.input_tensor_dict['extra_feats']],
        #                                   axis=-1, name='rich_feats_concat')  # (ds, 2 + extra_feat_size)
        #     full_score = tf.contrib.layers.fully_connected(
        #         inputs=rich_feats_concat, num_outputs=1,
        #         activation_fn=None, scope='final_fc', reuse=tf.AUTO_REUSE
        #     )  # (ds, 1)
        #     full_score = tf.squeeze(full_score, axis=-1, name='full_score')  # (ds, )

        """ Ready to return """
        tensor_dict = {
            'el_score': el_score,
            'el_final_feats': el_final_feats,
            # 'rich_feats_concat': rich_feats_concat,
            # 'full_score': full_score
        }
        for k, v in rm_ret_dict.items():
            tensor_dict[k] = v

        LogInfo.logs('%d tensors saved and return.', len(tensor_dict))
        LogInfo.logs('FINAL tensor_dict: %s', tensor_dict.keys())
        LogInfo.end_track()
        return tensor_dict

    def get_pair_loss(self, optm_score):
        """
        :param optm_score:  (ds, )
        in TRAIN mode, we put positive and negative cases together into one tensor
        """
        pos_score, neg_score = tf.unstack(tf.reshape(optm_score, shape=[-1, 2]), axis=1)
        # if self.use_loss_func == 'hinge':
        margin_loss = tf.nn.relu(neg_score + self.margin - pos_score, name='margin_loss')
        """ 180410: remove data_weight from the model """
        # weighted_loss = tf.multiply(margin_loss, data_weight, name='weighted_margin_loss')
        # else:       # ranknet
        #     logits = pos_score - neg_score
        #     rn_loss = -1. * tf.log(tf.clip_by_value(tf.nn.sigmoid(logits),
        #                                             clip_value_min=1e-6,
        #                                             clip_value_max=1.0))      # binary cross-entropy
        #     weighted_loss = tf.multiply(rn_loss, data_weight, name='weighted_ranknet_loss')
        return margin_loss
        # return weighted_loss

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
