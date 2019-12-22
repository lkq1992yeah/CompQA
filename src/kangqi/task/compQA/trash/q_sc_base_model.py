# -*- coding:utf-8 -*-

import numpy as np
import tensorflow as tf

from .base_model import BaseModel

import module.q_module as q_md
import module.sc_module as sc_md

from .module.merge_module import MergeModule
from .u import get_best_result

from ..dataset.xy_eval_dataloader import QScEvalDataLoader

from kangqi.util.LogUtil import LogInfo


class QScBaseModel(BaseModel):

    def __init__(self, sess, q_max_len, n_words, n_wd_emb,
                 sk_num, sk_max_len, n_entities, n_preds, n_kb_emb,
                 q_module_name, q_module_config, sc_module_name, sc_module_config,
                 n_merge_hidden, pn_size, verbose=0, reuse=None):
        super(QScBaseModel, self).__init__(sess=sess, verbose=verbose)

        assert q_module_name in q_md.__all__
        q_module_config['n_words'] = n_words
        q_module_config['n_wd_emb'] = n_wd_emb
        self.q_module = getattr(q_md, q_module_name)(**q_module_config)

        assert sc_module_name in sc_md.__all__
        sc_module_config['sk_num'] = sk_num
        sc_module_config['sk_max_len'] = sk_max_len
        sc_module_config['n_entities'] = n_entities
        sc_module_config['n_preds'] = n_preds
        sc_module_config['n_kb_emb'] = n_kb_emb
        self.sc_module = getattr(sc_md, sc_module_name)(**sc_module_config)

        n_q_hidden = q_module_config['rnn_config']['num_units'] * 2  # bidirectional
        n_sc_hidden = sc_module_config['n_sc_hidden']
        merge_module_config = {'n_q_hidden': n_q_hidden,
                               'n_sc_hidden': n_sc_hidden,
                               'n_merge_hidden': n_merge_hidden}
        self.merge_module = MergeModule(**merge_module_config)
        LogInfo.logs('Q & Sc & Merge Module declared.')

        eval_q_input = tf.placeholder(dtype=tf.int32,
                                      shape=[None, q_max_len],
                                      name='eval_q_input')
        eval_q_len_input = tf.placeholder(dtype=tf.int32,
                                          shape=[None],
                                          name='eval_q_len_input')
        eval_focus_input = tf.placeholder(dtype=tf.int32,
                                          shape=[None, pn_size, sk_num],
                                          name='eval_focus_input')
        eval_path_input = tf.placeholder(dtype=tf.int32,
                                         shape=[None, pn_size, sk_num, sk_max_len],
                                         name='eval_path_input')
        eval_path_len_input = tf.placeholder(dtype=tf.int32,
                                             shape=[None, pn_size, sk_num],
                                             name='eval_path_len_input')
        eval_mask_input = tf.placeholder(dtype=tf.float32,
                                         shape=[None, pn_size],
                                         name='eval_mask_input')
        eval_f1_input = tf.placeholder(dtype=tf.float32,
                                       shape=[None, pn_size],
                                       name='eval_f1_input')

        self.eval_input_tf_list = [eval_q_input,
                                   eval_q_len_input,
                                   eval_focus_input,
                                   eval_path_input,
                                   eval_path_len_input,
                                   eval_mask_input,
                                   eval_f1_input]

        LogInfo.begin_track('[QScBaseModel] Building Evaluation ...')
        with tf.name_scope('Evaluation'):
            with tf.name_scope('QuestionEncoding'):
                q_hidden = self.q_module.forward(eval_q_input,
                                                 eval_q_len_input,
                                                 reuse=reuse)  # q_hidden: (data_size, n_q_hidden)
                LogInfo.logs('* q_hidden built: %s', q_hidden.get_shape().as_list())
                # self.fuck_q_hidden = q_hidden

            with tf.name_scope('SchemaEncoding'):
                eval_sc_hidden = self.sc_module.forward(
                    focus_input=tf.reshape(eval_focus_input, [-1, sk_num]),
                    path_input=tf.reshape(eval_path_input, [-1, sk_num, sk_max_len]),
                    path_len_input=tf.reshape(eval_path_len_input, [-1, sk_num]),
                    reuse=reuse
                )       # eval_sc_hidden: (data_size * pn_size, n_sc_hidden)
                LogInfo.logs('* eval_sc_hidden built: %s', eval_sc_hidden.get_shape().as_list())
                # self.fuck_eval_sc_hidden = eval_sc_hidden

            with tf.name_scope('Merging'):
                q_hidden_dup = tf.reshape(tf.stack([q_hidden] * pn_size, axis=1),
                                          [-1, n_q_hidden],
                                          name='q_hidden_dup')  # (data_size * pn_size, n_q_hidden)
                eval_logits = self.merge_module.forward(
                    q_hidden=q_hidden_dup,
                    sc_hidden=eval_sc_hidden,
                    reuse=reuse
                )
                self.score_mat = tf.reshape(eval_logits, [-1, pn_size])      # (data_size, pn_size)
                LogInfo.logs('* score_mat built: %s', self.score_mat.get_shape().as_list())

            with tf.name_scope('Metric'):
                metric_vec = get_best_result(score_tf=self.score_mat,
                                             metric_tf=eval_f1_input,
                                             mask_tf=eval_mask_input)
                self.eval_metric_val = tf.reduce_mean(metric_vec, name='eval_metric_val')
                tf.summary.scalar('eval_f1', self.eval_metric_val, collections=['eval'])
                LogInfo.logs('* eval_metric_val defined.')

        self.eval_summary = tf.summary.merge_all(key='eval')
        LogInfo.end_track()

        self.optm_input_tf_list = None
        self.optm_step = None
        self.avg_loss = None
        self.optm_summary = None
        self.saver = None

        # All elements above needs to be defined in the __init__() of children classes.

    def evaluate(self, data_loader, epoch_idx, detail_fp=None, summary_writer=None):
        if data_loader is None:
            return 0.

        assert isinstance(data_loader, QScEvalDataLoader)
        scan_size = 0
        ret_metric = 0.
        if data_loader.dynamic or data_loader.np_data_list is None:
            data_loader.renew_data_list()
        if self.verbose > 0:
            LogInfo.logs('num of batch = %d.', data_loader.n_batch)

        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()

        ret_q_score_dict = {}  # <q, [(schema, score)]>
        for batch_idx in range(data_loader.n_batch):
            point = (epoch_idx - 1) * data_loader.n_batch + batch_idx
            local_data_list, local_indices = data_loader.get_next_batch()
            local_size = len(local_data_list[0])    # the first dimension is always batch size
            fd = {input_tf: local_data for input_tf, local_data in zip(self.eval_input_tf_list, local_data_list)}

            if summary_writer is None:
                score_mat, local_metric = self.sess.run([self.score_mat, self.eval_metric_val], feed_dict=fd)
                summary = None
            else:
                score_mat, local_metric, summary = self.sess.run(
                    [self.score_mat, self.eval_metric_val, self.eval_summary],
                    feed_dict=fd, options=run_options, run_metadata=run_metadata)
            ret_metric = (ret_metric * scan_size + local_metric * local_size) / (scan_size + local_size)
            scan_size += local_size
            if (batch_idx+1) % self.ob_batch_num == 0:
                LogInfo.logs('[eval-%s-B%d/%d] metric = %.6f, scanned = %d/%d',
                             data_loader.mode,
                             batch_idx+1,
                             data_loader.n_batch,
                             ret_metric,
                             scan_size,
                             len(data_loader))
            if summary_writer is not None:
                summary_writer.add_summary(summary, point)
                if batch_idx == 0:
                    summary_writer.add_run_metadata(run_metadata, 'epoch-%d' % epoch_idx)

            for local_idx, score_vec in zip(local_indices, score_mat):      # enumerate each row
                q_idx, cands = data_loader.q_cands_tup_list[local_idx]
                score_tup_list = ret_q_score_dict.setdefault(q_idx, [])
                for cand_idx, cand in enumerate(cands):  # enumerate each candidate
                    score = score_vec[cand_idx]
                    score_tup_list.append((cand, score))

        if detail_fp is not None:
            bw = open(detail_fp, 'w')
            LogInfo.redirect(bw)
            LogInfo.logs('Epoch-%d: avg_f1 = %.6f', epoch_idx, ret_metric)
            srt_q_idx_list = sorted(ret_q_score_dict.keys())
            for q_idx in srt_q_idx_list:
                LogInfo.begin_track('Q-%04d [%s]:', q_idx,
                                    data_loader.dataset.webq_list[q_idx].encode('utf-8'))
                score_tup_list = ret_q_score_dict[q_idx]
                score_tup_list.sort(key=lambda x: x[1], reverse=True)
                best_label_f1 = np.max([tup[0].f1 for tup in score_tup_list])
                best_label_f1 = max(best_label_f1, 0.000001)
                for rank, tup in enumerate(score_tup_list):
                    schema, score = tup
                    if rank < 5 or schema.f1 == best_label_f1:
                        LogInfo.logs('%4d: F1=%.6f, score=%9.6f, schema=[%s]',
                                     rank+1, schema.f1, score, schema.path_list_str)
                        schema.display_embedding_info()
                LogInfo.end_track()
            LogInfo.logs('Epoch-%d: avg_f1 = %.6f', epoch_idx, ret_metric)
            LogInfo.stop_redirect()
            bw.close()

        return ret_metric