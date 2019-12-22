# -*- coding:utf-8 -*-

import numpy as np
import tensorflow as tf

from .base_model import BaseModel

import module.q_module as q_md
import module.sk_module as sk_md
import module.item_module as item_md
import module.merge_module as merge_md

from .u import show_tensor

from ..dataset.xy_eval_dyn_dataloader import QScEvalDynamicDataLoader

from kangqi.util.LogUtil import LogInfo


# The model works with xy_eval_dyn_dataloader
class QScDynamicEvalModel(BaseModel):

    def __init__(self, sess, n_words, dim_wd_emb, n_entities, n_preds, dim_kb_emb,
                 q_max_len, item_max_len, path_max_len, sc_max_len,
                 q_module_name, q_module_config, item_module_name, item_module_config,
                 sk_module_name, sk_module_config, merge_module_name, merge_module_config,
                 reuse=tf.AUTO_REUSE, verbose=0):

        LogInfo.begin_track('QScDynamicBaseModel Building ...')
        super(QScDynamicEvalModel, self).__init__(sess=sess, verbose=verbose)

        assert q_module_name in q_md.__all__
        assert item_module_name in item_md.__all__
        assert sk_module_name in sk_md.__all__
        assert merge_module_name in merge_md.__all__

        self.q_module = getattr(q_md, q_module_name)(**q_module_config)
        self.sk_module = getattr(sk_md, sk_module_name)(**sk_module_config)
        self.item_module = getattr(item_md, item_module_name)(**item_module_config)
        self.merge_module = getattr(merge_md, merge_module_name)(**merge_module_config)

        LogInfo.logs('Sub-Modules declared.')

        q_input = tf.placeholder(dtype=tf.int32,
                                 shape=[None, q_max_len],
                                 name='q_input')
        q_len_input = tf.placeholder(dtype=tf.int32,
                                     shape=[None],
                                     name='q_len_input')

        sc_len_input = tf.placeholder(dtype=tf.int32,
                                      shape=[None, None],
                                      name='sc_len_input')
        focus_kb_input = tf.placeholder(dtype=tf.int32,
                                        shape=[None, None, sc_max_len],
                                        name='focus_kb_input')
        focus_item_input = tf.placeholder(dtype=tf.int32,
                                          shape=[None, None, sc_max_len, item_max_len],
                                          name='focus_item_input')
        focus_item_len_input = tf.placeholder(dtype=tf.int32,
                                              shape=[None, None, sc_max_len],
                                              name='focus_item_len_input')
        path_len_input = tf.placeholder(dtype=tf.int32,
                                        shape=[None, None, sc_max_len],
                                        name='path_len_input')
        path_kb_input = tf.placeholder(dtype=tf.int32,
                                       shape=[None, None, sc_max_len, path_max_len],
                                       name='path_kb_input')
        path_item_input = tf.placeholder(dtype=tf.int32,
                                         shape=[None, None, sc_max_len, path_max_len, item_max_len],
                                         name='path_item_input')
        path_item_len_input = tf.placeholder(dtype=tf.int32,
                                             shape=[None, None, sc_max_len, path_max_len],
                                             name='path_item_len_input')

        self.eval_input_tf_list = [
            q_input, q_len_input, sc_len_input,
            focus_kb_input, focus_item_input, focus_item_len_input,
            path_len_input, path_kb_input,
            path_item_input, path_item_len_input
        ]
        LogInfo.begin_track('Showing input tensors:')
        for tensor in self.eval_input_tf_list:
            show_tensor(tensor)
        LogInfo.end_track()

        with tf.name_scope('Evaluation'):
            dyn_pn_size = tf.shape(focus_kb_input)[1]

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
                    show_tensor(self.w_embedding_init)
                    show_tensor(self.e_embedding_init)
                    show_tensor(self.p_embedding_init)

                    w_embedding = tf.get_variable(name='w_embedding', initializer=self.w_embedding_init)
                    e_embedding = tf.get_variable(name='e_embedding', initializer=self.e_embedding_init)
                    p_embedding = tf.get_variable(name='p_embedding', initializer=self.p_embedding_init)

                    q_embedding = tf.nn.embedding_lookup(
                        params=w_embedding,
                        ids=q_input,
                        name='q_embedding'
                    )   # (batch, q_max_len, dim_wd_emb)

                    focus_item_embedding = tf.nn.embedding_lookup(
                        params=w_embedding,
                        ids=tf.reshape(focus_item_input, [-1, item_max_len]),
                        name='focus_item_embedding'
                    )   # (batch * pn_size * sc_max_len, item_max_len, dim_wd_emb)
                    focus_kb_hidden = tf.nn.embedding_lookup(
                        params=e_embedding,
                        ids=tf.reshape(focus_kb_input, (-1,)),
                        name='focus_kb_hidden'
                    )  # (batch * pn_size * sc_max_len, dim_kb_emb)

                    path_item_embedding = tf.nn.embedding_lookup(
                        params=w_embedding,
                        ids=tf.reshape(path_item_input, [-1, item_max_len]),
                        name='path_item_embedding'
                    )   # (batch * pn_size * sc_max_len * path_max_len, item_max_len, dim_wd_emb)
                    path_kb_hidden = tf.nn.embedding_lookup(
                        params=p_embedding,
                        ids=tf.reshape(path_kb_input, [-1, path_max_len]),
                        name='path_kb_hidden'
                    )   # (batch * pn_size * sc_max_len, path_max_len, dim_kb_emb)

            with tf.name_scope('Question'):
                LogInfo.begin_track('Question:')
                q_hidden = self.q_module.forward(
                    q_embedding=q_embedding,        # (batch, q_max_len, dim_wd_emb)
                    q_len=q_len_input,              # (batch, )
                    reuse=reuse
                )       # (batch, q_max_len, dim_q_hidden)
                show_tensor(q_hidden)
                q_hidden_dup = tf.reshape(
                    tf.tile(
                        input=q_hidden,  # (batch, q_max_len, dim_q_hidden)
                        multiples=[1, dyn_pn_size, 1]
                    ),  # (batch, pn_size * q_max_len, dim_q_hidden)
                    shape=[-1, q_max_len, self.q_module.dim_q_hidden],
                    name='q_hidden_dup'
                )       # (batch * pn_size, q_max_len, dim_q_hidden)
                show_tensor(q_hidden_dup)
                q_len_dup = tf.reshape(
                    tf.tile(
                        input=tf.expand_dims(q_len_input, 1),   # (batch_size, 1)
                        multiples=[1, dyn_pn_size]
                    ),                                          # (batch_size, pn_size)
                    shape=[-1],
                    name='q_len_dup'
                )       # (batch * pn_size,)
                show_tensor(q_len_dup)
                LogInfo.end_track()

            with tf.name_scope('Item'):
                LogInfo.begin_track('Item:')
                focus_wd_hidden = self.item_module.forward(
                    item_wd_embedding=focus_item_embedding,     # (batch*pn_size*sc_max_len, item_max_len, dim_wd_emb)
                    item_len=tf.reshape(focus_item_len_input, [-1]),    # (batch * pn_size * sc_max_len, )
                    reuse=reuse
                )       # (batch * pn_size * sc_max_len, dim_item_hidden), consistent with focus_kb_hidden
                show_tensor(focus_wd_hidden)
                raw_path_wd_hidden = self.item_module.forward(
                    item_wd_embedding=path_item_embedding,
                    item_len=tf.reshape(path_item_len_input, [-1]),     # (batch*pn_size*sc_max_len*path_max_len, )
                    reuse=reuse
                )       # (batch * pn_size * sc_max_len * path_max_len, dim_item_hidden)
                path_wd_hidden = tf.reshape(
                    tensor=raw_path_wd_hidden,
                    shape=(-1, path_max_len, self.item_module.dim_item_hidden),
                    name='path_wd_hidden'
                )       # (batch * pn_size * sc_max_len, path_max_len, dim_item_hidden), consistent with path_kb_hidden
                show_tensor(path_wd_hidden)
                LogInfo.end_track()

            with tf.name_scope('Skeleton'):
                LogInfo.begin_track('Skeleton:')
                sk_hidden = self.sk_module.forward(
                    path_wd_hidden=path_wd_hidden,  # (batch * pn_size * sc_max_len, path_max_len, dim_item_hidden)
                    path_kb_hidden=path_kb_hidden,  # (batch * pn_size * sc_max_len, path_max_len, dim_kb_emb)
                    path_len=tf.reshape(path_len_input, (-1,)),     # (batch * pn_size * sc_max_len, )
                    focus_wd_hidden=focus_wd_hidden,    # (batch * pn_size * sc_max_len, dim_item_hidden)
                    focus_kb_hidden=focus_kb_hidden,    # (batch * pn_size * sc_max_len, dim_kb_hidden)
                    reuse=reuse
                )       # (batch * pn_size * sc_max_len, dim_sk_hidden)
                show_tensor(sk_hidden)
                sc_hidden = tf.reshape(
                    sk_hidden,
                    shape=[-1, sc_max_len, self.sk_module.dim_sk_hidden],
                    name='sc_hidden'
                )       # (batch * pn_size, sc_max_len, dim_sk_hidden)
                show_tensor(sc_hidden)
                LogInfo.end_track()

            with tf.name_scope('Merging'):
                LogInfo.begin_track('Merging:')
                logits, sc_final_rep, att_tensor, q_weight, sc_weight = self.merge_module.forward(
                    q_hidden=q_hidden_dup,      # (batch * pn_size, q_max_len, dim_q_hidden)
                    q_len=q_len_dup,            # (batch * pn_size, )
                    sc_hidden=sc_hidden,        # (batch * pn_size, sc_max_len, dim_sk_hidden)
                    sc_len=tf.reshape(sc_len_input, [-1]),      # (batch * pn_size, )
                    reuse=reuse
                )           # (batch * pn_size, )
                self.score_mat = tf.reshape(logits, [-1, dyn_pn_size], name='score_mat')  # (batch, pn_size)
                dyn_dim_sc = tf.shape(sc_final_rep)[-1]
                self.sc_final_rep = tf.reshape(sc_final_rep, [-1, dyn_pn_size, dyn_dim_sc],
                                               name='sc_final_rep')     # (batch, pn_size, dim_sc)
                self.att_tensor = tf.reshape(att_tensor, [-1, dyn_pn_size, q_max_len, sc_max_len],
                                             name='att_mat')       # (batch, pn_size, q_max_len, sc_max_len)
                self.q_weight = tf.reshape(q_weight, [-1, dyn_pn_size, q_max_len], name='q_weight')
                self.sc_weight = tf.reshape(sc_weight, [-1, dyn_pn_size, sc_max_len], name='sc_weight')

                show_tensor(self.score_mat)
                tf.summary.histogram('score_mat', self.score_mat, collections=['eval'])
                LogInfo.end_track()

        self.eval_summary = tf.summary.merge_all(key='eval')
        LogInfo.end_track()

    # Here the data_loader should be xy_eval_dyn_dataloader
    def evaluate(self, data_loader, epoch_idx, ob_batch_num=10, detail_fp=None, summary_writer=None):
        if data_loader is None or len(data_loader) == 0:    # empty eval data
            return 0.

        assert isinstance(data_loader, QScEvalDynamicDataLoader)
        self.prepare_data(data_loader=data_loader)
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()

        scan_size = 0
        ret_q_score_dict = {}           # <q, [(schema, score)]>
        for batch_idx in range(data_loader.n_batch):
            point = (epoch_idx - 1) * data_loader.n_batch + batch_idx
            local_data_list, local_indices = data_loader.get_next_batch()
            local_size = len(local_data_list[0])    # the first dimension is always batch size
            fd = {input_tf: local_data for input_tf, local_data in zip(self.eval_input_tf_list, local_data_list)}

            score_mat, sc_final_rep, att_tensor, q_weight, sc_weight, summary = \
                self.sess.run([self.score_mat, self.sc_final_rep, self.att_tensor,
                               self.q_weight, self.sc_weight, self.eval_summary],
                              feed_dict=fd,
                              options=run_options,
                              run_metadata=run_metadata)
            scan_size += local_size
            if (batch_idx + 1) % ob_batch_num == 0:
                LogInfo.logs('[eval-%s-B%d/%d] scanned = %d/%d',
                             data_loader.mode,
                             batch_idx + 1,
                             data_loader.n_batch,
                             scan_size,
                             len(data_loader))

            if summary_writer is not None:
                summary_writer.add_summary(summary, point)
                if batch_idx == 0:
                    summary_writer.add_run_metadata(run_metadata, 'epoch-%d' % epoch_idx)

            for local_idx, score_vec, sc_mat, local_att_tensor, local_q_weight, local_sc_weight in zip(
                    local_indices, score_mat, sc_final_rep, att_tensor, q_weight, sc_weight):   # enumerate each row
                q_idx, cands = data_loader.q_cands_tup_list[local_idx]
                score_list = ret_q_score_dict.setdefault(q_idx, [])         # save all candidates with their scores
                for cand_idx, cand in enumerate(cands):                     # enumerate each candidate
                    score = score_vec[cand_idx]
                    sc_vec = sc_mat[cand_idx]
                    att_mat = local_att_tensor[cand_idx]
                    q_weight_vec = local_q_weight[cand_idx]
                    sc_weight_vec = local_sc_weight[cand_idx]
                    cand.run_info = {'score': score,
                                     'sc_vec': sc_vec,
                                     'att_mat': att_mat,
                                     'q_weight_vec': q_weight_vec,
                                     'sc_weight_vec': sc_weight_vec
                                     }      # save detail information within the cand.
                    score_list.append(cand)

        # After scanning all the batch, now count the final F1 result
        f1_list = []
        for q_idx, score_list in ret_q_score_dict.items():
            score_list.sort(key=lambda x: x.run_info['score'], reverse=True)       # sort by score DESC
            if len(score_list) == 0:
                f1_list.append(0.)
            else:
                f1_list.append(score_list[0].f1)         # pick the f1 of the highest scored schema
        ret_metric = np.mean(f1_list)

        if detail_fp is not None:
            data_loader.dataset.load_dicts()
            w_dict = data_loader.dataset.w_dict
            w_uhash = {v: k for k, v in w_dict.items()}        # word_idx --> word

            bw = open(detail_fp, 'w')
            LogInfo.redirect(bw)
            LogInfo.logs('Epoch-%d: avg_f1 = %.6f', epoch_idx, ret_metric)
            srt_q_idx_list = sorted(ret_q_score_dict.keys())
            for q_idx in srt_q_idx_list:
                LogInfo.begin_track('Q-%04d [%s]:', q_idx,
                                    data_loader.dataset.webq_list[q_idx].encode('utf-8'))
                q_words = data_loader.dataset.q_words_dict[q_idx]
                word_surf_list = map(lambda x: w_uhash[x].encode('utf-8'), q_words)

                srt_list = ret_q_score_dict[q_idx]      # already sorted
                best_label_f1 = np.max([sc.f1 for sc in srt_list])
                best_label_f1 = max(best_label_f1, 0.000001)
                for rank, sc in enumerate(srt_list):
                    path_sz = len(sc.path_list)
                    path_surf_list = map(lambda x: 'Path-%d' % x, range(path_sz))
                    if rank < 5 or sc.f1 == best_label_f1:
                        LogInfo.begin_track('#-%04d: F1=%.6f, score=%9.6f ', rank+1, sc.f1, sc.run_info['score'])
                        for path_idx, path in enumerate(sc.path_list):
                            LogInfo.logs('Path-%d: [%s]', path_idx, '-->'.join(path))
                        for path_idx, words in enumerate(sc.path_words_list):
                            LogInfo.logs('Path-Word-%d: [%s]', path_idx, ' | '.join(words).encode('utf-8'))
                        LogInfo.logs('Raw Attention Score:')
                        self.print_att_matrix(word_surf_list=word_surf_list,
                                              path_surf_list=path_surf_list,
                                              att_mat=sc.run_info['att_mat'])
                        LogInfo.logs('Q-side Attention Weight:')
                        self.print_weight_vec(item_surf_list=word_surf_list,
                                              weight_vec=sc.run_info['q_weight_vec'])
                        LogInfo.logs('SC-side Attention Weight:')
                        self.print_weight_vec(item_surf_list=path_surf_list,
                                              weight_vec=sc.run_info['sc_weight_vec'])
                LogInfo.end_track()
            LogInfo.logs('Epoch-%d: avg_f1 = %.6f', epoch_idx, ret_metric)
            LogInfo.stop_redirect()
            bw.close()
        return ret_metric

    @staticmethod
    def print_att_matrix(word_surf_list, path_surf_list, att_mat):
        """
        Given surfaces of words in Q, and the indicator of each skeleton,
        print the attention matrix in a nice format.
        """
        word_sz = len(word_surf_list)
        path_sz = len(path_surf_list)
        header = [' ' * 10]
        for col_idx in range(att_mat.shape[1]):
            if col_idx < path_sz:
                header.append('%9s' % path_surf_list[col_idx])
            else:
                header.append('  <EMPTY>')
            if col_idx == path_sz - 1:
                header.append('|')
        LogInfo.logs(' '.join(header))
        for row_idx, att_row in enumerate(att_mat):
            # scan each row of the raw attention matrix
            show_str_list = []
            if row_idx >= word_sz:
                wd = '<EMPTY>'
            else:
                wd = word_surf_list[row_idx]
            show_str_list.append('%10s' % wd)
            for col_idx, att_val in enumerate(att_row):
                show_str_list.append('%9.6f' % att_val)
                if col_idx == path_sz - 1:
                    show_str_list.append('|')
            show_str = ' '.join(show_str_list)
            LogInfo.logs(show_str)
            if row_idx == word_sz - 1:
                LogInfo.logs('-' * len(show_str))  # split useful and non-useful parts
        LogInfo.end_track()

    @staticmethod
    def print_weight_vec(item_surf_list, weight_vec):
        """
        Given word or skeleton name, show the weight vector at the particular side
        """
        sz = len(item_surf_list)
        for idx, val in enumerate(weight_vec):
            # item = item_surf_list[idx] if idx < sz else '   <EMPTY>'
            if idx >= sz:
                break
            item = item_surf_list[idx]
            LogInfo.logs('%10s %9.6f', item, val)
