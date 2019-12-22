# import numpy as np
import tensorflow as tf

from ..module.simple_attention import SimpleAttention
from ..module.seq_helper import seq_hidden_averaging, seq_encoding, \
    seq_encoding_with_aggregation, seq_hidden_max_pooling
from ..module.tcn import TemporalConvNet
from ..module.stack_conv_net import StackConvNet
from ..u import show_tensor

from xusheng.model.rnn_encoder import BidirectionalRNNEncoder
from kangqi.util.tf.cosine_sim import cosine_sim

from kangqi.util.LogUtil import LogInfo


class CompqMultiTaskModel:

    def __init__(self, path_max_size, qw_max_len, pw_max_len, pseq_max_len,
                 el_feat_size, extra_feat_size, drop_rate,
                 n_words, n_mids, n_paths, dim_emb,
                 rnn_config, scn_config, att_config,
                 w_emb_fix, path_usage, sent_usage,
                 seq_merge_mode, scoring_mode, final_func,
                 loss_func, optm_name, learning_rate, full_back_prop):
        LogInfo.begin_track('Build [compq_emnlp18.CompqMultiTaskModel]:')

        self.dim_emb = dim_emb
        self.qw_max_len = qw_max_len
        self.pw_max_len = pw_max_len
        self.path_max_size = path_max_size
        self.pseq_max_len = pseq_max_len
        self.el_feat_size = el_feat_size
        self.extra_feat_size = extra_feat_size

        self.path_usage = path_usage
        self.sent_usage = sent_usage
        self.seq_merge_mode = seq_merge_mode
        self.scoring_mode = scoring_mode
        self.final_func = final_func

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

        self.scn_config = scn_config
        if att_func != 'noAtt':
            self.att_config = att_config
        else:
            self.att_config = None

        if rnn_cell_class != 'None':
            self.rnn_config = rnn_config
            self.rnn_config['reuse'] = tf.AUTO_REUSE
            self.dim_hidden = 2 * rnn_config['num_units']
        else:           # no RNN, just directly using embedding
            self.rnn_config = None
            self.dim_hidden = self.dim_emb

        self.input_tensor_dict = self.input_tensor_definition(
            qw_max_len=qw_max_len,
            pw_max_len=pw_max_len,
            path_max_size=path_max_size,
            pseq_max_len=self.pseq_max_len,
            el_feat_size=el_feat_size,
            extra_feat_size=extra_feat_size
        )
        LogInfo.logs('Global input tensors defined.')

        with tf.variable_scope('embedding_lookup', reuse=tf.AUTO_REUSE):
            with tf.device('/cpu:0'):
                w_trainable = True if w_emb_fix == 'Upd' else False
                LogInfo.logs('Word embedding trainable: %s', w_trainable)
                self.w_embedding_init = tf.placeholder(tf.float32, [n_words, dim_emb], 'w_embedding_init')
                self.w_embedding = tf.get_variable(name='w_embedding', initializer=self.w_embedding_init,
                                                   trainable=w_trainable)
                self.m_embedding_init = tf.placeholder(tf.float32, [n_mids, dim_emb], 'm_embedding_init')
                self.m_embedding = tf.get_variable(name='m_embedding', initializer=self.m_embedding_init)
                self.p_embedding_init = tf.placeholder(tf.float32, [n_paths, dim_emb], 'p_embedding_init')
                self.p_embedding = tf.get_variable(name='p_embedding', initializer=self.p_embedding_init)
        self.dropout_layer = tf.layers.Dropout(drop_rate)
        LogInfo.logs('Dropout: %.2f', drop_rate)

        """ Define tensors for different tasks """
        rm_only_names = ['qw_input', 'qw_len', 'dep_input', 'dep_len',
                         'path_size', 'path_cates', 'path_ids',
                         'pw_input', 'pw_len', 'pseq_ids', 'pseq_len']
        el_only_names = ['el_indv_feats', 'el_comb_feats', 'el_mask']
        self.rm_eval_input_names = self.rm_optm_input_names = rm_only_names
        self.el_eval_input_names = self.el_optm_input_names = el_only_names
        self.full_eval_input_names = self.full_optm_input_names = rm_only_names + el_only_names + ['extra_feats']

        """ Build the main graph for both optm and eval """
        self.eval_tensor_dict = self.build_graph(mode_str='eval')
        self.optm_tensor_dict = self.build_graph(mode_str='optm')
        self.rm_eval_output_names = filter(lambda x: x in self.eval_tensor_dict, ['rm_score', 'rm_final_feats'])
        self.el_eval_output_names = filter(lambda x: x in self.eval_tensor_dict, ['el_score', 'el_final_feats'])
        self.full_eval_output_names = filter(lambda x: x in self.eval_tensor_dict, ['full_score', 'full_final_feats'])

        """ Loss & Update """
        LogInfo.logs('[rm] loss function: Hinge-%.1f', self.margin)

        # Relation matching task
        rm_weighted_loss = self.get_pair_loss(optm_score=self.optm_tensor_dict['rm_score'])
        self.rm_loss = tf.reduce_mean(rm_weighted_loss, name='rm_loss')
        self.rm_update, self.optm_rm_summary = self.build_update_summary(task='rm')
        LogInfo.logs('Loss & update defined: [Relation Matching]')

        # Entity linking task
        el_weighted_loss = self.get_pair_loss(optm_score=self.optm_tensor_dict['el_score'])
        self.el_loss = tf.reduce_mean(el_weighted_loss, name='el_loss')
        self.el_update, self.optm_el_summary = self.build_update_summary(task='el')
        LogInfo.logs('Loss & update defined: [Entity Linking]')

        # Full task
        full_weighted_loss = self.get_pair_loss(optm_score=self.optm_tensor_dict['full_score'])
        self.full_loss = tf.reduce_mean(full_weighted_loss, name='full_loss')
        LogInfo.logs('Full Back Propagate: %s', full_back_prop)
        if not full_back_prop:  # full task: only update the last FC layer
            with tf.variable_scope('', reuse=tf.AUTO_REUSE):
                spec_var_list = [
                    tf.get_variable(name=param_name)
                    for param_name in ['full_task/out_fc/weights',
                                       'full_task/out_fc/biases']
                ]
            self.full_update, self.optm_full_summary = self.build_update_summary(
                task='full', spec_var_list=spec_var_list
            )
        else:
            self.full_update, self.optm_full_summary = self.build_update_summary(task='full')
        LogInfo.logs('Loss & update defined: [Full Task]')

        LogInfo.end_track('End of Model')

    @staticmethod
    def input_tensor_definition(qw_max_len, pw_max_len, path_max_size, pseq_max_len,
                                el_feat_size, extra_feat_size):
        input_tensor_dict = {
            # Path input
            # 5 = Main + E + T + Tm + Ord
            'path_size': tf.placeholder(tf.int32, [None], 'path_size'),
            'path_cates': tf.placeholder(tf.float32, [None, path_max_size, 5], 'path_cates'),
            'path_ids': tf.placeholder(tf.int32, [None, path_max_size], 'path_ids'),
            'pw_input': tf.placeholder(tf.int32, [None, path_max_size, pw_max_len], 'pw_input'),
            'pw_len': tf.placeholder(tf.int32, [None, path_max_size], 'pw_len'),
            'pseq_ids': tf.placeholder(tf.int32, [None, path_max_size, pseq_max_len], 'pseq_ids'),
            'pseq_len': tf.placeholder(tf.int32, [None, path_max_size], 'pseq_len'),
            'pseq_wd_input': tf.placeholder(tf.int32, [None, path_max_size, pseq_max_len, pw_max_len], 'pseq_wd_input'),
            'pseq_wd_len': tf.placeholder(tf.int32, [None, path_max_size, pseq_max_len], 'pseq_wd_len'),
            
            # Sentential information
            'qw_input': tf.placeholder(tf.int32, [None, path_max_size, qw_max_len], 'qw_input'),
            'qw_len': tf.placeholder(tf.int32, [None, path_max_size], 'qw_len'),
            # TODO: Position Encoding
            
            # Dependency information
            'dep_input': tf.placeholder(tf.int32, [None, path_max_size, qw_max_len], 'dep_input'),
            'dep_len': tf.placeholder(tf.int32, [None, path_max_size], 'dep_len'),

            # Linking information
            'el_indv_feats': tf.placeholder(tf.float32, [None, path_max_size, el_feat_size], 'el_indv_feats'),
            'el_comb_feats': tf.placeholder(tf.float32, [None, 1], 'el_comb_feats'),
            'el_mask': tf.placeholder(tf.float32, [None, path_max_size], 'el_mask'),
            # el_mask can make the better use of qw/dep_input (for both RM and EL)

            # Extra information
            'extra_feats': tf.placeholder(tf.float32, [None, extra_feat_size], 'extra_feats')
        }
        return input_tensor_dict

    def build_graph(self, mode_str):
        LogInfo.begin_track('Build graph: [MT-%s]', mode_str)
        mode = tf.contrib.learn.ModeKeys.INFER if mode_str == 'eval' else tf.contrib.learn.ModeKeys.TRAIN
        training = False if mode_str == 'eval' else True

        with tf.device('/cpu:0'):
            qw_emb = tf.nn.embedding_lookup(params=self.w_embedding,
                                            ids=self.input_tensor_dict['qw_input'],
                                            name='qw_emb')  # (ds, path_max_size, qw_max_len, dim_emb)
            dep_emb = tf.nn.embedding_lookup(params=self.w_embedding,
                                             ids=self.input_tensor_dict['dep_input'],
                                             name='dep_emb')  # (ds, path_max_size, qw_max_len, dim_emb)
            pw_emb = tf.nn.embedding_lookup(params=self.w_embedding,
                                            ids=self.input_tensor_dict['pw_input'],
                                            name='pw_emb')  # (ds, path_max_size, pw_max_len, dim_emb)
            pseq_emb = tf.nn.embedding_lookup(params=self.m_embedding,
                                              ids=self.input_tensor_dict['pseq_ids'],
                                              name='pseq_emb')  # (ds, path_max_size, pseq_max_size, dim_emb)
            path_emb = tf.nn.embedding_lookup(params=self.p_embedding,
                                              ids=self.input_tensor_dict['path_ids'],
                                              name='path_emb')  # (ds, path_max_size, dim_emb)
        pw_len = self.input_tensor_dict['pw_len']
        pseq_len = self.input_tensor_dict['pseq_len']
        qw_len = self.input_tensor_dict['qw_len']
        dep_len = self.input_tensor_dict['dep_len']

        qw_emb = self.dropout_layer(qw_emb, training=training)
        dep_emb = self.dropout_layer(dep_emb, training=training)
        pw_emb = self.dropout_layer(pw_emb, training=training)
        pseq_emb = self.dropout_layer(pseq_emb, training=training)
        path_emb = self.dropout_layer(path_emb, training=training)
        LogInfo.logs('Dropout performed.')

        rnn_encoder = None
        if self.rnn_config is not None:
            encoder_args = {'config': self.rnn_config, 'mode': mode}
            rnn_encoder = BidirectionalRNNEncoder(**encoder_args)

        """ For RM kernel """
        with tf.variable_scope('rm_task', reuse=tf.AUTO_REUSE):
            path_repr = self.build_path_repr__single(pw_emb=pw_emb, pw_len=pw_len,
                                                     pseq_emb=pseq_emb, pseq_len=pseq_len,
                                                     path_emb=path_emb, rnn_encoder=rnn_encoder)

            """ BiGRU """
            qw_repr = self.build_question_seq_repr(seq_emb=qw_emb, seq_len=qw_len, path_repr=path_repr,
                                                   rnn_encoder=rnn_encoder, scope_name='qw_repr')
            dep_repr = self.build_question_seq_repr(seq_emb=dep_emb, seq_len=dep_len, path_repr=path_repr,
                                                    rnn_encoder=rnn_encoder, scope_name='dep_repr')

            """ Temporal Conv Net """
            # qw_repr = self.build_question_seq_repr__tcn(seq_emb=qw_emb, seq_len=qw_len,
            #                                             training=training, scope_name='qw_repr')
            # dep_repr = self.build_question_seq_repr__tcn(seq_emb=dep_emb, seq_len=dep_len,
            #                                              training=training, scope_name='dep_repr')

            """ Stacking Conv Net (with Attention) """
            # qw_repr = self.build_question_seq_repr__scn(seq_emb=qw_emb, seq_len=qw_len, path_repr=path_repr,
            #                                             training=training, scope_name='qw_repr')
            # dep_repr = self.build_question_seq_repr__scn(seq_emb=dep_emb, seq_len=dep_len, path_repr=path_repr,
            #                                              training=training, scope_name='dep_repr')

            rm_final_feats, rm_score = self.rm_final_merge(
                path_repr=path_repr, qw_repr=qw_repr, dep_repr=dep_repr,
                path_cates=self.input_tensor_dict['path_cates'],
                path_size=self.input_tensor_dict['path_size']
            )

        """ For EL kernel """
        with tf.variable_scope('el_task', reuse=tf.AUTO_REUSE):
            el_final_feats, el_score = self.el_forward(el_indv_feats=self.input_tensor_dict['el_indv_feats'],
                                                       el_comb_feats=self.input_tensor_dict['el_comb_feats'],
                                                       el_mask=self.input_tensor_dict['el_mask'])

        """ For Full task """
        with tf.variable_scope('full_task', reuse=tf.AUTO_REUSE):
            full_final_feats, full_score = self.full_forward(
                el_final_feats=el_final_feats,
                rm_final_feats=rm_final_feats,
                extra_feats=self.input_tensor_dict['extra_feats']
            )

        """ Ready to return """
        tensor_dict = {'rm_score': rm_score,
                       'el_score': el_score,
                       'full_score': full_score,
                       'rm_final_feats': rm_final_feats,
                       'el_final_feats': el_final_feats,
                       'full_final_feats': full_final_feats}
        LogInfo.logs('%d tensors saved and return: %s', len(tensor_dict), tensor_dict.keys())
        LogInfo.end_track()
        return tensor_dict

    def build_path_repr__pw_side(self, pw_emb, pw_len, rnn_encoder, pw_usage):
        """
        :param pw_emb: (ds, path_max_size, pw_max_len, dim_wd_emb)
        :param pw_len: (ds, path_max_size)
        :param rnn_encoder:
        :param pw_usage: X,B,R (None / BOW / RNN)
        """
        with tf.variable_scope('pw_repr', reuse=tf.AUTO_REUSE):
            pw_emb = tf.reshape(pw_emb, [-1, self.pw_max_len, self.dim_emb])
            pw_len = tf.reshape(pw_len, [-1])
            if pw_usage == 'B':
                pw_repr = seq_hidden_averaging(seq_hidden_input=pw_emb, len_input=pw_len)
                # (ds*path_max_size, dim_wd_emb), simply BOW
                pw_repr = tf.reshape(pw_repr, [-1, self.path_max_size, self.dim_emb], 'pw_repr')
            elif pw_usage == 'R':
                pw_repr = seq_encoding_with_aggregation(
                    emb_input=pw_emb, len_input=pw_len,
                    rnn_encoder=rnn_encoder,
                    seq_merge_mode=self.seq_merge_mode
                )   # (ds*path_max_size, dim_hidden)
                pw_repr = tf.reshape(pw_repr, [-1, self.path_max_size, self.dim_hidden], 'pw_repr')
                # (ds, path_max_size, dim_qw_hidden)
            else:
                assert pw_usage == 'X'
                pw_repr = None
        if pw_repr is not None:
            show_tensor(pw_repr)
        return pw_repr

    def build_path_repr__pseq_side(self, path_emb, pseq_emb, pseq_len, rnn_encoder, pseq_usage):
        """
        :param path_emb: (ds, path_max_size, dim_emb)
        :param pseq_emb: (ds, path_max_size, pseq_max_len, dim_emb)
        :param pseq_len: (ds, path_max_size)
        :param rnn_encoder:
        :param pseq_usage: X,B,R,H (None / BOW / RNN / wHole)
        """
        with tf.variable_scope('pseq_repr', reuse=tf.AUTO_REUSE):
            pseq_emb = tf.reshape(pseq_emb, [-1, self.pseq_max_len, self.dim_emb])
            pseq_len = tf.reshape(pseq_len, [-1])
            if pseq_usage == 'H':
                pseq_repr = path_emb        # (ds, path_max_size, dim_emb)
            elif pseq_usage == 'B':
                pseq_repr = seq_hidden_averaging(seq_hidden_input=pseq_emb, len_input=pseq_len)
                pseq_repr = tf.reshape(pseq_repr, [-1, self.path_max_size, self.dim_emb], 'pseq_repr')
                # (ds, path_max_size, dim_wd_emb)
            elif pseq_usage == 'R':
                pseq_repr = seq_encoding_with_aggregation(
                    emb_input=pseq_emb, len_input=pseq_len,
                    rnn_encoder=rnn_encoder,
                    seq_merge_mode=self.seq_merge_mode
                )   # (ds*path_max_size, dim_hidden)
                pseq_repr = tf.reshape(pseq_repr, [-1, self.path_max_size, self.dim_hidden], 'pseq_repr')
                # (ds, path_max_size, dim_hidden)
            else:
                assert pseq_usage == 'X'
                pseq_repr = None
        if pseq_repr is not None:
            show_tensor(pseq_repr)
        return pseq_repr

    def build_path_repr__single(self, pw_emb, pw_len, path_emb, pseq_emb, pseq_len, rnn_encoder):
        """
        :param pw_emb: (ds, path_max_size, pw_max_len, dim_emb)
        :param pw_len: (ds, path_max_size)
        :param path_emb: (ds, path_max_size, dim_emb)
        :param pseq_emb: (ds, path_max_size, pseq_max_len, dim_emb)
        :param pseq_len: (ds, path_max_size)
        :param rnn_encoder:
        """
        LogInfo.logs('build_path_repr: path_usage = [%s].', self.path_usage)
        assert len(self.path_usage) == 2
        pw_repr = self.build_path_repr__pw_side(
            pw_emb=pw_emb, pw_len=pw_len,
            rnn_encoder=rnn_encoder,
            pw_usage=self.path_usage[0]
        )
        pseq_repr = self.build_path_repr__pseq_side(
            path_emb=path_emb, pseq_emb=pseq_emb, pseq_len=pseq_len,
            rnn_encoder=rnn_encoder, pseq_usage=self.path_usage[1]
        )
        if pw_repr is None:
            assert pseq_repr is not None
            final_repr = pseq_repr
        elif pseq_repr is None:
            final_repr = pw_repr
        else:   # summation
            final_repr = pw_repr + pseq_repr
        return final_repr       # (ds, path_max_size, dim_emb or dim_hidden)

    def build_path_repr__2tier(self):
        pass

    def build_question_seq_repr(self, seq_emb, seq_len, path_repr, rnn_encoder, scope_name):
        """
        :param seq_emb: (ds, path_max_size, qw_max_len, dim_wd_emb)
        :param seq_len: (ds, path_max_size)
        :param path_repr: (ds, path_max_size, dim_path_hidden)
        :param rnn_encoder: RNN encoder (could be None)
        :param scope_name: variable_scope name
        """
        seq_emb = tf.reshape(seq_emb, [-1, self.qw_max_len, self.dim_emb])
        seq_len = tf.reshape(seq_len, [-1])
        dim_path_hidden = path_repr.get_shape().as_list()[-1]
        path_repr = tf.reshape(path_repr, [-1, dim_path_hidden])

        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
            if self.att_config is not None:
                dim_att_hidden = self.att_config['dim_att_hidden']
                att_func = self.att_config['att_func']
                LogInfo.logs('build_seq_repr: att_func = [%s].', att_func)
                seq_hidden = seq_encoding(emb_input=seq_emb, len_input=seq_len, encoder=rnn_encoder)
                # (ds*path_max_size, qw_max_len, dim_hidden)
                seq_mask = tf.sequence_mask(lengths=seq_len,
                                            maxlen=self.qw_max_len,
                                            dtype=tf.float32,
                                            name='seq_mask')  # (ds*path_max_size, qw_max_len)
                simple_att = SimpleAttention(lf_max_len=self.qw_max_len,
                                             dim_att_hidden=dim_att_hidden,
                                             att_func=att_func)
                seq_att_rep, att_mat, seq_weight = simple_att.forward(lf_input=seq_hidden,
                                                                      lf_mask=seq_mask,
                                                                      fix_rt_input=path_repr)
                # q_att_rep: (DS, dim_hidden)
                # att_mat:   (DS, qw_max_len)
                # q_weight:  (DS, qw_max_len)
                # att_mat = tf.reshape(att_mat, [-1, self.path_max_size, self.qw_max_len], 'att_mat')
                # # (ds, path_max_size, qw_max_len)
                # seq_weight = tf.reshape(seq_weight, [-1, self.path_max_size, self.qw_max_len], 'seq_weight')
                # # (ds, path_max_size, qw_max_len)
                seq_repr = seq_att_rep
            else:  # no attention, similar with above
                LogInfo.logs('build_seq_repr: att_func = [noAtt], seq_merge_mode = [%s].', self.seq_merge_mode)
                seq_repr = seq_encoding_with_aggregation(emb_input=seq_emb, len_input=seq_len,
                                                         rnn_encoder=rnn_encoder,
                                                         seq_merge_mode=self.seq_merge_mode)
            seq_repr = tf.reshape(seq_repr, [-1, self.path_max_size, self.dim_hidden], 'seq_repr')
        return seq_repr     # (ds, path_max_size, dim_hidden)

    def build_question_seq_repr__scn(self, seq_emb, seq_len, path_repr, training, scope_name):
        """
        :param seq_emb: (ds, path_max_size, qw_max_len, dim_wd_emb)
        :param seq_len: (ds, path_max_size)
        :param path_repr: (ds, path_max_size, dim_path_hidden)
        :param training: True / False
        :param scope_name: variable_scope name
        """
        LogInfo.logs('build_seq_repr by SCN: %s', self.scn_config or 'None')
        seq_emb = tf.reshape(seq_emb, [-1, self.qw_max_len, self.dim_emb])
        seq_len = tf.reshape(seq_len, [-1])
        dim_path_hidden = path_repr.get_shape().as_list()[-1]
        path_repr = tf.reshape(path_repr, [-1, dim_path_hidden])        # (ds*path_max_size, dim_path_hidden)

        n_layers = self.scn_config['n_layers']
        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
            stack_conv_net = StackConvNet(n_layers=n_layers,
                                          n_outputs=self.dim_hidden,
                                          dropout=self.scn_config['dropout'],
                                          residual=self.scn_config['residual'])
            conv_hidden_list = stack_conv_net.__call__(inputs=seq_emb, training=training)
            # a list of "n_layer" elements, each (ds*path_max_size, qw_max_len, dim_hidden)
            concat_conv_hidden = tf.concat(conv_hidden_list, axis=1, name='concat_conv_hidden')
            # (ds*path_max_size, n_layers*qw_max_len, dim_hidden)
            show_tensor(concat_conv_hidden)

            seq_mask = tf.sequence_mask(lengths=seq_len, maxlen=self.qw_max_len, dtype=tf.float32, name='seq_mask')
            # (ds*path_max_size, qw_max_len)
            concat_seq_mask = tf.concat([seq_mask]*n_layers, axis=1, name='concat_seq_mask')
            # (ds*path_max_size, n_layers*qw_max_len)
            show_tensor(concat_seq_mask)

            if self.att_config is not None:
                dim_att_hidden = self.att_config['dim_att_hidden']
                att_func = self.att_config['att_func']
                LogInfo.logs('  Attention available: att_func = [%s].', att_func)
                simple_att = SimpleAttention(lf_max_len=n_layers*self.qw_max_len,
                                             dim_att_hidden=dim_att_hidden,
                                             att_func=att_func)
                seq_att_rep, att_mat, seq_weight = simple_att.forward(lf_input=concat_conv_hidden,
                                                                      lf_mask=concat_seq_mask,
                                                                      fix_rt_input=path_repr)
                # seq_att_rep: (DS, dim_hidden)
                # att_mat:     (DS, n_layers*qw_max_len)
                # seq_weight:  (DS, n_layers*qw_max_len)
                seq_repr = seq_att_rep
            else:  # no attention, use max-pooling at the last layer
                LogInfo.logs('  Attention not available: max pooling over the last layer.')
                last_conv_hidden = conv_hidden_list[-1]     # (ds*path_max_size, qw_max_len, dim_hidden)
                seq_repr = seq_hidden_max_pooling(seq_hidden_input=last_conv_hidden,
                                                  len_input=seq_len)        # (ds*path_max_size, dim_hidden)

            seq_repr = tf.reshape(seq_repr, [-1, self.path_max_size, self.dim_hidden], 'seq_repr')
        return seq_repr  # (ds, path_max_size, dim_hidden)

    def build_question_seq_repr__tcn(self, seq_emb, seq_len, training, scope_name):
        """
        Encode question repr by TemporalConvNet
        :param seq_emb: (ds, path_max_size, qw_max_len, dim_emb)
        :param seq_len: (ds, path_max_size)
        :param training: True / False
        :param scope_name: variable_scope name
        """
        seq_emb = tf.reshape(seq_emb, [-1, self.qw_max_len, self.dim_emb])
        seq_len = tf.reshape(seq_len, [-1])
        # dim_path_hidden = path_repr.get_shape().as_list()[-1]
        # path_repr = tf.reshape(path_repr, [-1, dim_path_hidden])

        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
            tcn = TemporalConvNet(num_channels=[self.dim_hidden]*3, kernel_size=3, dropout=0.0)
            tcn_output = tcn.__call__(inputs=seq_emb, training=training)
            # (ds*path_max_size, qw_max_len, dim_hidden)

            """ Now pick the last state of each question """
            last_position = seq_len - 1
            last_pos_mask = tf.expand_dims(
                tf.one_hot(indices=last_position, depth=self.qw_max_len),
                axis=-1, name='last_pos_mask'
            )   # (ds*path_max_size, qw_max_len, 1)
            seq_repr_2d = tf.reduce_sum(
                tcn_output*last_pos_mask,   # (ds*path_max_size, qw_max_len, dim_hidden)
                axis=1, name='seq_repr_2d'
            )   # (ds*path_max_size, dim_hidden)
            seq_repr = tf.reshape(seq_repr_2d, [-1, self.path_max_size, self.dim_hidden], 'seq_repr')

        return seq_repr      # (ds, path_max_size, dim_hidden)

    def rm_final_merge(self, path_repr, qw_repr, dep_repr, path_cates, path_size):
        """
        :param path_repr: (ds, path_max_len, dim_path_hidden)
        :param qw_repr:   (ds, path_max_len, dim_hidden)
        :param dep_repr:  (ds, path_max_len, dim_hidden)
        :param path_cates: (ds, path_max_len, 5)    5 = Main + E + T + Tm + Ord
        :param path_size: (ds, )
        """
        LogInfo.logs('rm_final_merge: sent_usage = [%s], scoring_mode = [%s], final_func = [%s].',
                     self.sent_usage, self.scoring_mode, self.final_func)
        with tf.variable_scope('rm_final_merge', reuse=tf.AUTO_REUSE):
            if self.sent_usage in ('mWsum', 'cpWsum'):      # weighted sum over the two scores
                _, qw_rm_score = self._rm_final_merge(path_repr=path_repr, sent_repr=qw_repr,
                                                      path_cates=path_cates, path_size=path_size,
                                                      scope_name='qw_repr')
                _, dep_rm_score = self._rm_final_merge(path_repr=path_repr, sent_repr=dep_repr,
                                                       path_cates=path_cates, path_size=path_size,
                                                       scope_name='dep_repr')
                alpha = tf.get_variable(name='alpha', shape=[], dtype=tf.float32)
                weight = tf.nn.sigmoid(alpha)
                rm_score = weight * qw_rm_score + (1. - weight) * dep_rm_score
                rm_final_feats = tf.stack([qw_rm_score, dep_rm_score], axis=-1, name='rm_final_feats')
                # (ds, 2) as final feats
            else:           # merge qw & dep into one sent_repr, then got final score
                if self.sent_usage == 'qwOnly':
                    sent_repr = qw_repr
                elif self.sent_usage == 'depOnly':
                    sent_repr = dep_repr
                elif self.sent_usage in ('mFC', 'cpFC'):
                    concat_repr = tf.concat([qw_repr, dep_repr], axis=-1, name='concat_repr')
                    # (ds, path_max_len, 2*dim_hidden)
                    # TODO: add dropout here for concat_repr
                    sent_repr = tf.contrib.layers.fully_connected(
                        inputs=concat_repr,
                        num_outputs=self.dim_hidden,
                        activation_fn=tf.nn.relu,
                        scope='mFC',
                        reuse=tf.AUTO_REUSE
                    )   # (ds, path_max_len, dim_hidden)
                elif self.sent_usage in ('mMax', 'cpMax'):      # merge by max pooling
                    sent_repr = tf.reduce_max(tf.stack([qw_repr, dep_repr], axis=0), axis=0)
                else:       # mSum, merge by summation
                    assert self.sent_usage in ('mSum', 'cpSum')
                    sent_repr = tf.add(qw_repr, dep_repr)
                # TODO: add dropout here for sent_repr and path_repr
                rm_final_feats, rm_score = self._rm_final_merge(path_repr=path_repr, sent_repr=sent_repr,
                                                                path_cates=path_cates, path_size=path_size,
                                                                scope_name='sent_repr')
        return rm_final_feats, rm_score

    def _rm_final_merge(self, path_repr, sent_repr, path_cates, path_size, scope_name):
        """
        Kernel part of rm_final_merge.
        :param path_repr: (ds, path_max_len, dim_path_hidden)
        :param sent_repr: (ds, path_max_len, dim_hidden)
        :param path_cates: (ds, path_max_len, 5)
        :param path_size: (ds, )
        :param scope_name:
        """
        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
            dim_path_hidden = path_repr.get_shape().as_list()[-1]
            if self.scoring_mode == 'compact':
                sent_repr = seq_hidden_max_pooling(seq_hidden_input=sent_repr, len_input=path_size)
                path_repr = seq_hidden_max_pooling(seq_hidden_input=path_repr, len_input=path_size)
                # (ds, dim_xx_hidden)
            else:
                assert self.scoring_mode in ('separated', 'bao')
                sent_repr = tf.reshape(sent_repr, [-1, self.dim_hidden])
                path_repr = tf.reshape(path_repr, [-1, dim_path_hidden])
                # (ds*path_max_size, dim_xx_hidden)

            """ Now apply final functions """
            if self.final_func == 'dot':
                assert dim_path_hidden == self.dim_hidden
                merge_score = tf.reduce_sum(sent_repr*path_repr, axis=-1, name='merge_score')
            elif self.final_func == 'cos':
                assert dim_path_hidden == self.dim_hidden
                merge_score = cosine_sim(lf_input=sent_repr, rt_input=path_repr)
            elif self.final_func == 'bilinear':
                bilinear_mat = tf.get_variable(name='bilinear_mat',
                                               shape=[dim_path_hidden, self.dim_hidden],
                                               dtype=tf.float32,
                                               initializer=tf.contrib.layers.xavier_initializer())
                proj_repr = tf.matmul(path_repr, bilinear_mat, name='proj_repr')
                merge_score = tf.reduce_sum(sent_repr * proj_repr, axis=-1, name='merge_score')
            else:
                assert self.final_func.startswith('fc')
                hidden_size = int(self.final_func[2:])
                concat_repr = tf.concat([sent_repr, path_repr], axis=-1, name='concat_repr')
                concat_hidden = tf.contrib.layers.fully_connected(
                    inputs=concat_repr,
                    num_outputs=hidden_size,
                    activation_fn=tf.nn.relu,
                    scope='fc1',
                    reuse=tf.AUTO_REUSE
                )   # (ds / ds*path_max_len, 32)
                merge_score = tf.contrib.layers.fully_connected(
                    inputs=concat_hidden,
                    num_outputs=1,
                    activation_fn=None,
                    scope='fc2',
                    reuse=tf.AUTO_REUSE
                )   # (ds / ds*path_max_len, 1)
                merge_score = tf.squeeze(merge_score, axis=-1, name='merge_score')

            """ add scores together, if working in separated / bao mode """
            if self.scoring_mode == 'compact':
                rm_score = merge_score
                rm_final_feats = tf.expand_dims(rm_score, -1, 'rm_final_feats')     # (ds, 1)
            else:
                assert self.scoring_mode in ('separated', 'bao')
                merge_score = tf.reshape(merge_score, [-1, self.path_max_size])     # (ds, path_max_size)
                path_mask = tf.sequence_mask(
                    lengths=path_size, maxlen=self.path_max_size,
                    dtype=tf.float32, name='path_mask'
                )  # (ds, path_max_size) as mask
                if self.scoring_mode == 'separated':
                    rm_score = tf.reduce_sum(merge_score*path_mask, axis=-1, name='rm_score')  # (ds, )
                    rm_final_feats = tf.expand_dims(rm_score, -1, 'rm_final_feats')     # (ds, 1)
                else:   # Imitate Bao's implementation, care about the detail path category
                    mask_score_3d = tf.expand_dims(
                        merge_score * path_mask,
                        axis=1, name='mask_score_3d'
                    )  # (ds, 1, path_max_size)
                    rm_final_feats = tf.squeeze(
                        tf.matmul(mask_score_3d, path_cates),  # (ds, 1, 5)
                        axis=1, name='rm_final_feats'
                    )  # (ds, 5)
                    rm_score_2d = tf.contrib.layers.fully_connected(
                        inputs=rm_final_feats,
                        num_outputs=1,
                        activation_fn=None,
                        scope='out_fc',
                        reuse=tf.AUTO_REUSE
                    )  # (ds / ds*path_max_len, 1)
                    rm_score = tf.squeeze(rm_score_2d, axis=-1, name='rm_score')
        return rm_final_feats, rm_score

    @staticmethod
    def el_forward(el_indv_feats, el_comb_feats, el_mask):
        """
        :param el_indv_feats: (ds, path_max_size, el_feats)
        :param el_comb_feats: (ds, 1)
        :param el_mask: (ds, path_max_size)
        """
        """ Currently: no use of sentential & syntactic information """
        sum_indv_feats = tf.reduce_sum(
            el_indv_feats * tf.expand_dims(el_mask, axis=-1),
            axis=1, name='sum_indv_feats'
        )  # (ds, el_feat_size), first sum together
        el_final_feats = tf.concat([sum_indv_feats, el_comb_feats], axis=-1, name='final_feats')
        # (ds, el_max_size+1) --> indv_feats + comb_feat
        el_score_2d = tf.contrib.layers.fully_connected(
            inputs=el_final_feats,
            num_outputs=1,
            activation_fn=None,
            scope='out_fc',
            reuse=tf.AUTO_REUSE
        )  # (ds, 1)
        el_score = tf.squeeze(el_score_2d, axis=-1, name='el_score')   # (ds, )
        return el_final_feats, el_score

    @staticmethod
    def full_forward(el_final_feats, rm_final_feats, extra_feats):
        """
        :param el_final_feats: (ds, el_final_feat_size)
        :param rm_final_feats: (ds, rm_final_feat_size)
        :param extra_feats: (ds, extra_feat_size)
        """
        full_final_feats = tf.concat(
            [rm_final_feats, el_final_feats, extra_feats],
            axis=-1, name='full_final_feats'
        )   # (ds, full_final_feat_size)
        # """
        # 180315: try force setting the initial parameters .
        # RM feature is 4 times more effective than EL feature.
        # """
        # final_feat_len = 2 + self.extra_feat_size
        # init_w_value = np.zeros((final_feat_len, 1), dtype='float32')
        # init_w_value[0] = 0.25
        # init_w_value[1] = 1
        # with tf.variable_scope('final_fc', reuse=tf.AUTO_REUSE):
        #     weights = tf.get_variable(name='weights', initializer=tf.constant(init_w_value))
        #     biases = tf.get_variable(name='biases', shape=(1,))
        #     full_score = tf.matmul(rich_feats_concat, weights) + biases         # (ds, 1)
        # full_score = tf.squeeze(full_score, axis=-1, name='full_score')  # (ds, )
        full_score_2d = tf.contrib.layers.fully_connected(
            inputs=full_final_feats,
            num_outputs=1,
            activation_fn=None,
            scope='out_fc',
            reuse=tf.AUTO_REUSE
        )  # (ds, 1)
        full_score = tf.squeeze(full_score_2d, axis=-1, name='full_score')  # (ds, )
        return full_final_feats, full_score

    def get_pair_loss(self, optm_score):
        """
        :param optm_score:  (ds, )
        in TRAIN mode, we put positive and negative cases together into one tensor
        """
        pos_score, neg_score = tf.unstack(tf.reshape(optm_score, shape=[-1, 2]), axis=1)
        margin_loss = tf.nn.relu(neg_score + self.margin - pos_score, name='margin_loss')
        return margin_loss

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
