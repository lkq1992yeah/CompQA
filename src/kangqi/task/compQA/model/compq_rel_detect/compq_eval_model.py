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
In the evaluation part, due to different schemas have different qwords (with placeholders),
so pn_size never appears in the model. (Just like what simpq_eval_model does)
"""

import tensorflow as tf
import numpy as np

from ..compq_overall.relation_matching_kernel import RelationMatchingKernel as CompqKernel

from ..u import show_tensor
from ..base_model import BaseModel

from ...dataset.kq_schema import CompqSchema
from ...dataset import CompqSingleDataLoader

from kangqi.util.LogUtil import LogInfo


class CompqEvalModel(BaseModel):

    def __init__(self, sess, compq_kernel):
        LogInfo.begin_track('CompqEvalModel Building ...')
        assert isinstance(compq_kernel, CompqKernel)
        verbose = compq_kernel.verbose
        super(CompqEvalModel, self).__init__(sess=sess, verbose=verbose)

        # ======== define tensors ======== #

        q_max_len = compq_kernel.q_max_len
        path_max_len = compq_kernel.path_max_len
        pword_max_len = compq_kernel.pword_max_len

        qwords_input = tf.placeholder(dtype=tf.int32,
                                      shape=[None, q_max_len],
                                      name='q_words_input')     # (data_size, q_max_len)
        qwords_len_input = tf.placeholder(dtype=tf.int32,
                                          shape=[None],
                                          name='q_words_len_input')     # (data_size, )
        sc_len_input = tf.placeholder(dtype=tf.int32,
                                      shape=[None],
                                      name='sc_len_input')  # (data_size, )
        preds_input = tf.placeholder(dtype=tf.int32,
                                     shape=[None, None, path_max_len],
                                     name='preds_input')    # (data_size, sc_max_len, path_max_len)
        preds_len_input = tf.placeholder(dtype=tf.int32,
                                         shape=[None, None],
                                         name='preds_len_input')    # (data_size, sc_max_len)
        pwords_input = tf.placeholder(dtype=tf.int32,
                                      shape=[None, None, pword_max_len],
                                      name='pwords_input')  # (data_size, sc_max_len, pword_max_len)
        pwords_len_input = tf.placeholder(dtype=tf.int32,
                                          shape=[None, None],
                                          name='pwords_len_input')  # (data_size, sc_max_len)
        self.eval_input_tf_list = [
            qwords_input, qwords_len_input, sc_len_input,
            preds_input, preds_len_input,
            pwords_input, pwords_len_input
        ]
        LogInfo.begin_track('Showing %d input tensors:', len(self.eval_input_tf_list))
        for tensor in self.eval_input_tf_list:
            show_tensor(tensor)
        LogInfo.end_track()

        # ======== start building model ======== #

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
        _, _, self.score = compq_kernel.get_score(
            mode=tf.contrib.learn.ModeKeys.INFER,
            qwords_embedding=qwords_embedding, qwords_len=qwords_len_input, sc_len=sc_len_input,
            preds_embedding=preds_embedding, preds_len=preds_len_input,
            pwords_embedding=pwords_embedding, pwords_len=pwords_len_input
        )       # (data_size, ), also ignore the attention matrix information
        show_tensor(self.score)

        self.eval_summary = tf.summary.merge_all(key='eval')
        LogInfo.logs('* final score defined.')

        LogInfo.end_track()

    # The main code copies from q_sc_dyn_eval_model
    def evaluate(self, data_loader, epoch_idx, ob_batch_num=20,
                 detail_fp=None, result_fp=None, summary_writer=None):
        if data_loader is None or len(data_loader) == 0:  # empty eval data
            return 0.

        assert isinstance(data_loader, CompqSingleDataLoader)
        self.prepare_data(data_loader=data_loader)
        run_options = run_metadata = None
        if summary_writer is not None:
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()

        scan_size = 0
        ret_q_score_dict = {}  # <q, [(schema, score)]>
        for batch_idx in range(data_loader.n_batch):
            local_data_list, local_indices = data_loader.get_next_batch()
            local_size = len(local_data_list[0])  # the first dimension is always batch size
            fd = {input_tf: local_data for input_tf, local_data in zip(self.eval_input_tf_list, local_data_list)}

            # score_mat, sc_final_rep, att_tensor, q_weight, sc_weight, summary = \
            # local_score_list, summary = self.sess.run(
            local_score_list = self.sess.run(
                # [self.score_mat, self.sc_final_rep, self.att_tensor,
                #  self.q_weight, self.sc_weight, self.eval_summary],
                # [self.score, self.eval_summary],
                self.score,
                feed_dict=fd,
                options=run_options,
                run_metadata=run_metadata
            )
            scan_size += local_size
            if (batch_idx + 1) % ob_batch_num == 0:
                LogInfo.logs('[eval-%s-B%d/%d] scanned = %d/%d',
                             data_loader.mode,
                             batch_idx + 1,
                             data_loader.n_batch,
                             scan_size,
                             len(data_loader))

            if summary_writer is not None:
                if batch_idx == 0:
                    summary_writer.add_run_metadata(run_metadata, 'epoch-%d' % epoch_idx)

            # for local_idx, score_vec, sc_mat, local_att_tensor, local_q_weight, local_sc_weight in zip(
            #         local_indices, score_mat, sc_final_rep, att_tensor, q_weight, sc_weight):  # enumerate each row
            for local_idx, score in zip(local_indices, local_score_list):
                q_idx, cand = data_loader.cand_tup_list[local_idx]
                assert isinstance(cand, CompqSchema)
                score_list = ret_q_score_dict.setdefault(q_idx, [])  # save all candidates with their scores
                cand.run_info = {
                    'score': score,
                    # 'sc_vec': sc_vec,
                    # 'att_mat': att_mat,
                    # 'q_weight_vec': q_weight_vec,
                    # 'sc_weight_vec': sc_weight_vec
                }  # save detail information within the cand.
                score_list.append(cand)
                assert isinstance(cand, CompqSchema)

        # After scanning all the batch, now count the final F1 result
        f1_list = []
        for q_idx, score_list in ret_q_score_dict.items():
            score_list.sort(key=lambda x: x.run_info['score'], reverse=True)  # sort by score DESC
            if len(score_list) == 0:
                f1_list.append(0.)
            else:
                f1_list.append(score_list[0].f1)  # pick the f1 of the highest scored schema
        LogInfo.logs('Predict %d out of %d questions.', len(f1_list), data_loader.question_size)
        ret_metric = np.sum(f1_list) / data_loader.question_size

        if result_fp is not None:
            srt_q_idx_list = sorted(ret_q_score_dict.keys())
            with open(result_fp, 'w') as bw:        # write question --> selected schema
                for q_idx in srt_q_idx_list:
                    srt_list = ret_q_score_dict[q_idx]
                    ori_idx = -1
                    f1 = 0.
                    if len(srt_list) > 0:
                        best_sc = srt_list[0]
                        ori_idx = best_sc.ori_idx
                        f1 = best_sc.f1
                    bw.write('%d\t%d\t%.6f\n' % (q_idx, ori_idx, f1))

        if detail_fp is not None:
            bw = open(detail_fp, 'w')
            LogInfo.redirect(bw)
            LogInfo.logs('Epoch-%d: avg_f1 = %.6f', epoch_idx, ret_metric)
            srt_q_idx_list = sorted(ret_q_score_dict.keys())
            for q_idx in srt_q_idx_list:
                LogInfo.begin_track('Q-%04d [%s]:', q_idx,
                                    data_loader.dataset.q_list[q_idx].encode('utf-8'))
                # q_words = data_loader.dataset.q_words_dict[q_idx]
                # word_surf_list = map(lambda x: w_uhash[x].encode('utf-8'), q_words)

                srt_list = ret_q_score_dict[q_idx]  # already sorted
                best_label_f1 = np.max([sc.f1 for sc in srt_list])
                best_label_f1 = max(best_label_f1, 0.000001)
                for rank, sc in enumerate(srt_list):
                    # path_sz = len(sc.path_list)
                    # path_surf_list = map(lambda x: 'Path-%d' % x, range(path_sz))
                    if rank < 5 or sc.f1 == best_label_f1:
                        LogInfo.begin_track('#-%04d: F1=%.6f, score=%9.6f ', rank + 1, sc.f1, sc.run_info['score'])
                        for path_idx, path in enumerate(sc.path_list):
                            LogInfo.logs('Path-%d: [%s]', path_idx, '-->'.join(path))
                        for path_idx, words in enumerate(sc.path_words_list):
                            LogInfo.logs('Path-Word-%d: [%s]', path_idx, ' | '.join(words).encode('utf-8'))
                        # LogInfo.logs('Raw Attention Score:')
                        # self.print_att_matrix(word_surf_list=word_surf_list,
                        #                       path_surf_list=path_surf_list,
                        #                       att_mat=sc.run_info['att_mat'])
                        # LogInfo.logs('Q-side Attention Weight:')
                        # self.print_weight_vec(item_surf_list=word_surf_list,
                        #                       weight_vec=sc.run_info['q_weight_vec'])
                        # LogInfo.logs('SC-side Attention Weight:')
                        # self.print_weight_vec(item_surf_list=path_surf_list,
                        #                       weight_vec=sc.run_info['sc_weight_vec'])
                        LogInfo.end_track()
                LogInfo.end_track()
            LogInfo.logs('Epoch-%d: avg_f1 = %.6f', epoch_idx, ret_metric)
            LogInfo.stop_redirect()
            bw.close()
        return ret_metric
