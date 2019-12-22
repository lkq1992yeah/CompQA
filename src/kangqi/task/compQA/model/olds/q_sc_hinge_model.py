# -*- coding:utf-8 -*-

import tensorflow as tf

from .base_model import BaseModel
from .u import show_tensor

import module.q_module as q_md
import module.sk_module as sk_md
import module.item_module as item_md
import module.merge_module as merge_md

from kangqi.util.LogUtil import LogInfo


class QScHingeModel(BaseModel):

    def __init__(self, sess, n_words, dim_wd_emb, n_entities, n_preds, dim_kb_emb,
                 q_max_len, item_max_len, path_max_len, sc_max_len,
                 q_module_name, q_module_config, item_module_name, item_module_config,
                 sk_module_name, sk_module_config, merge_module_name, merge_module_config,
                 margin, optm_name, learning_rate, reuse=tf.AUTO_REUSE, verbose=0):

        LogInfo.begin_track('QScHingeModel Building ...')
        super(QScHingeModel, self).__init__(sess=sess, verbose=verbose)

        assert q_module_name in q_md.__all__
        assert item_module_name in item_md.__all__
        assert sk_module_name in sk_md.__all__
        assert merge_module_name in merge_md.__all__

        self.q_module = getattr(q_md, q_module_name)(**q_module_config)
        self.sk_module = getattr(sk_md, sk_module_name)(**sk_module_config)
        self.item_module = getattr(item_md, item_module_name)(**item_module_config)
        self.merge_module = getattr(merge_md, merge_module_name)(**merge_module_config)

        LogInfo.logs('Sub-Modules declared.')

        assert optm_name in ('Adam', 'Adadelta', 'Adagrad', 'GradientDescent')
        optm_name += 'Optimizer'

        self.item_max_len = item_max_len
        self.path_max_len = path_max_len
        self.sc_max_len = sc_max_len

        q_input = tf.placeholder(dtype=tf.int32,
                                 shape=[None, q_max_len],
                                 name='q_input')
        q_len_input = tf.placeholder(dtype=tf.int32,
                                     shape=[None],
                                     name='q_len_input')
        self.optm_input_tf_list = [q_input, q_len_input]

        sc_tensor_groups = []
        for cate in ('pos', 'neg'):
            sc_len_input = tf.placeholder(dtype=tf.int32,
                                          shape=[None],
                                          name=cate+'_sc_len_input')
            focus_kb_input = tf.placeholder(dtype=tf.int32,
                                            shape=[None, sc_max_len],
                                            name=cate+'_focus_kb_input')
            focus_item_input = tf.placeholder(dtype=tf.int32,
                                              shape=[None, sc_max_len, item_max_len],
                                              name=cate+'_focus_item_input')
            focus_item_len_input = tf.placeholder(dtype=tf.int32,
                                                  shape=[None, sc_max_len],
                                                  name=cate+'_focus_item_len_input')
            path_len_input = tf.placeholder(dtype=tf.int32,
                                            shape=[None, sc_max_len],
                                            name=cate+'_path_len_input')
            path_kb_input = tf.placeholder(dtype=tf.int32,
                                           shape=[None, sc_max_len, path_max_len],
                                           name=cate + '_path_kb_input')
            path_item_input = tf.placeholder(dtype=tf.int32,
                                             shape=[None, sc_max_len, path_max_len, item_max_len],
                                             name=cate+'_path_item_input')
            path_item_len_input = tf.placeholder(dtype=tf.int32,
                                                 shape=[None, sc_max_len, path_max_len],
                                                 name=cate+'_path_item_len_input')
            tensor_group = [
                sc_len_input, focus_kb_input,
                focus_item_input, focus_item_len_input,
                path_len_input, path_kb_input,
                path_item_input, path_item_len_input
            ]
            sc_tensor_groups.append(tensor_group)
            self.optm_input_tf_list += tensor_group

        LogInfo.begin_track('Showing input tensors:')
        for tensor in self.optm_input_tf_list:
            show_tensor(tensor)
        LogInfo.end_track()

        with tf.name_scope('Optimization'):

            with tf.variable_scope('Embedding_Lookup', reuse=reuse):
                with tf.device('/cpu:0'):
                    self.w_embedding_init = tf.placeholder(dtype=tf.float32,
                                                           shape=(n_words, dim_wd_emb),
                                                           name='w_embedding_init')
                    self.e_embedding_init = tf.placeholder(dtype=tf.float32,
                                                           shape=(n_entities, dim_kb_emb),
                                                           name='e_embedding_init')
                    self.p_embedding_init = tf.placeholder(dtype=tf.float32,
                                                           shape=(n_preds, dim_kb_emb),
                                                           name='p_embedding_init')

                    self.w_embedding = tf.get_variable(name='w_embedding', initializer=self.w_embedding_init)
                    self.e_embedding = tf.get_variable(name='e_embedding', initializer=self.e_embedding_init)
                    self.p_embedding = tf.get_variable(name='p_embedding', initializer=self.p_embedding_init)

                    q_embedding = tf.nn.embedding_lookup(
                        params=self.w_embedding,
                        ids=q_input,
                        name='q_embedding'
                    )   # (batch, q_max_len, dim_wd_emb)

            with tf.name_scope('Question'):
                LogInfo.begin_track('Question:')
                q_hidden = self.q_module.forward(
                    q_embedding=q_embedding,        # (batch, q_max_len, dim_wd_emb)
                    q_len=q_len_input,              # (batch, )
                    reuse=reuse
                )       # (batch, q_max_len, dim_q_hidden)
                show_tensor(q_hidden)
                LogInfo.end_track()

            logits_list = []
            for cate, sc_tensor_group in zip(('pos', 'neg'), sc_tensor_groups):
                LogInfo.begin_track('Calculate score at %s side:', cate)
                logits = self.work_on_logits(
                    q_hidden=q_hidden,
                    q_len=q_len_input,
                    sc_tensor_group=sc_tensor_group,
                    reuse=reuse
                )
                logits_list.append(logits)
                LogInfo.end_track()

            with tf.name_scope('Loss_Update'):
                pos_logits, neg_logits = logits_list
                margin_loss = tf.nn.relu(neg_logits + margin - pos_logits,
                                         name='margin_loss')
                self.avg_loss = tf.reduce_mean(margin_loss, name='avg_loss')
                tf.summary.scalar('avg_loss', self.avg_loss, collections=['optm'])
                optimizer = getattr(tf.train, optm_name)
                self.optm_step = optimizer(learning_rate).minimize(self.avg_loss)
                LogInfo.logs('* avg_loss and optm_step defined.')

        self.optm_summary = tf.summary.merge_all(key='optm')
        LogInfo.end_track()

    def work_on_logits(self, q_hidden, q_len, sc_tensor_group, reuse):
        [sc_len_input, focus_kb_input,
         focus_item_input, focus_item_len_input,
         path_len_input, path_kb_input,
         path_item_input, path_item_len_input] = sc_tensor_group

        with tf.variable_scope('Embedding_Lookup', reuse=reuse):
            with tf.device('/cpu:0'):
                focus_item_embedding = tf.nn.embedding_lookup(
                    params=self.w_embedding,
                    ids=tf.reshape(focus_item_input, [-1, self.item_max_len]),
                    name='focus_item_embedding'
                )  # (batch * sc_max_len, item_max_len, dim_wd_emb)
                focus_kb_hidden = tf.nn.embedding_lookup(
                    params=self.e_embedding,
                    ids=tf.reshape(focus_kb_input, (-1,)),
                    name='focus_kb_hidden'
                )  # (batch * sc_max_len, dim_kb_emb)

                path_item_embedding = tf.nn.embedding_lookup(
                    params=self.w_embedding,
                    ids=tf.reshape(path_item_input, [-1, self.item_max_len]),
                    name='path_item_embedding'
                )  # (batch * sc_max_len * path_max_len, item_max_len, dim_wd_emb)
                path_kb_hidden = tf.nn.embedding_lookup(
                    params=self.p_embedding,
                    ids=tf.reshape(path_kb_input, [-1, self.path_max_len]),
                    name='path_kb_hidden'
                )  # (batch * sc_max_len, path_max_len, dim_kb_emb)

        with tf.name_scope('Item'):
            LogInfo.begin_track('Item:')
            focus_wd_hidden = self.item_module.forward(
                item_wd_embedding=focus_item_embedding,
                item_len=tf.reshape(focus_item_len_input, [-1]),
                reuse=reuse
            )  # (batch * sc_max_len, dim_item_hidden), consistent with focus_kb_hidden
            show_tensor(focus_wd_hidden)
            raw_path_wd_hidden = self.item_module.forward(
                item_wd_embedding=path_item_embedding,
                item_len=tf.reshape(path_item_len_input, [-1]),
                reuse=reuse
            )  # (batch * sc_max_len * path_max_len, dim_item_hidden)
            path_wd_hidden = tf.reshape(
                tensor=raw_path_wd_hidden,
                shape=(-1, self.path_max_len, self.item_module.dim_item_hidden),
                name='path_wd_hidden'
            )  # (batch * sc_max_len, path_max_len, dim_item_hidden), consistent with path_kb_hidden
            show_tensor(path_wd_hidden)
            LogInfo.end_track()

        with tf.name_scope('Skeleton'):
            LogInfo.begin_track('Skeleton:')
            sk_hidden = self.sk_module.forward(
                path_wd_hidden=path_wd_hidden,  # (batch * sc_max_len, path_max_len, dim_item_hidden)
                path_kb_hidden=path_kb_hidden,  # (batch * sc_max_len, path_max_len, dim_kb_emb)
                path_len=tf.reshape(path_len_input, (-1,)),  # (batch * sc_max_len, )
                focus_wd_hidden=focus_wd_hidden,  # (batch * sc_max_len, dim_item_hidden)
                focus_kb_hidden=focus_kb_hidden,  # (batch * sc_max_len, dim_kb_hidden)
                reuse=reuse
            )  # (batch * sc_max_len, dim_sk_hidden)
            show_tensor(sk_hidden)
            sc_hidden = tf.reshape(
                sk_hidden,
                shape=[-1, self.sc_max_len, self.sk_module.dim_sk_hidden],
                name='sc_hidden'
            )  # (batch, sc_max_len, dim_sk_hidden)
            show_tensor(sc_hidden)
            LogInfo.end_track()

        with tf.name_scope('Merging'):
            LogInfo.begin_track('Merging:')
            merge_output = self.merge_module.forward(
                q_hidden=q_hidden,  # (batch, q_max_len, dim_q_hidden)
                q_len=q_len,        # (batch, )
                sc_hidden=sc_hidden,    # (batch, sc_max_len, dim_sk_hidden)
                sc_len=sc_len_input,    # (batch, )
                reuse=reuse
            )
            logits = merge_output[0]    # (batch * pn_size, ), don't need the remaining data
            LogInfo.end_track()

        return logits
