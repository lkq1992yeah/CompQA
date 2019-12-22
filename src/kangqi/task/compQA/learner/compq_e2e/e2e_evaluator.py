"""
Author: Kangqi Luo
Date: 180206
Goal: the wrapper for evaluating different kernels
"""

import tensorflow as tf
import numpy as np

from kangqi.util.LogUtil import LogInfo


class BaseEvaluator:

    def __init__(self, task_name, compq_mt_model, sess, ob_batch_num=100,
                 detail_disp_func=None, summary_writer=None):
        self.task_name = task_name
        self.sess = sess
        self.compq_mt_model = compq_mt_model
        self.ob_batch_num = ob_batch_num

        input_tensor_names = getattr(compq_mt_model, '%s_eval_input_names' % task_name)
        self.active_input_tensor_dict = {k: compq_mt_model.input_tensor_dict[k] for k in input_tensor_names}
        self.output_tensor_names = getattr(compq_mt_model, '%s_eval_output_names' % task_name)
        self.output_tensor_list = [compq_mt_model.eval_tensor_dict[k] for k in self.output_tensor_names]

        self.detail_disp_func = detail_disp_func
        self.summary_writer = summary_writer
        self.run_options = self.run_metadata = None
        if self.summary_writer is not None:
            self.run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            self.run_metadata = tf.RunMetadata()

        self.scan_data = 0
        self.scan_batch = 0
        self.tb_point = 0       # x-axis in tensorboard
        self.eval_detail_dict = {}      # store evaluation detail of each data

    """ Reset informations for calculating the final result """
    def reset_eval_info(self):
        self.scan_data = self.scan_batch = 0
        self.eval_detail_dict = {}

    def evaluate(self, eval_dl, batch_idx):
        local_data, local_size = eval_dl.get_batch(batch_idx=batch_idx)
        active_input_names = set(self.active_input_tensor_dict.keys()) & set(local_data.keys())
        fd = {self.active_input_tensor_dict[key]: local_data[key] for key in active_input_names}
        local_output_list = self.sess.run(self.output_tensor_list,
                                          feed_dict=fd,
                                          options=self.run_options,
                                          run_metadata=self.run_metadata)

        local_eval_detail_dict = fd
        local_eval_detail_dict.update({k: v for k, v in zip(self.output_tensor_names, local_output_list)})
        for tensor_name, batch_val in local_eval_detail_dict.items():
            for val in batch_val:
                self.eval_detail_dict.setdefault(tensor_name, []).append(val)
        # Collect all input / outputs of this batch, saving into eval_detail_dict (split by each data point)

        self.scan_data += local_size
        self.scan_batch += 1
        self.tb_point += 1
        if self.scan_batch % self.ob_batch_num == 0:
            LogInfo.logs('[%3s][eval-%s-B%d/%d] scanned = %d/%d',
                         self.task_name,
                         eval_dl.mode,
                         self.scan_batch,
                         eval_dl.n_batch,
                         self.scan_data,
                         len(eval_dl))
        # if self.summary_writer is not None:
        #     self.summary_writer.add_summary(summary, self.tb_point)
        #     # if batch_idx == 0:
        #     #     summary_writer.add_run_metadata(run_metadata, 'epoch-%d' % epoch_idx)

    def evaluate_all(self, eval_dl, detail_fp=None, result_fp=None):
        self.reset_eval_info()
        for batch_idx in range(eval_dl.n_batch):
            self.evaluate(eval_dl=eval_dl, batch_idx=batch_idx)
        ret_f1 = self.post_process(eval_dl=eval_dl, detail_fp=detail_fp, result_fp=result_fp)
        LogInfo.logs('[%3s] %s_F1 = %.6f', self.task_name, eval_dl.mode, ret_f1)      # [ rm] train_F1 = xx
        return ret_f1

    # ====================== Below: functions to override ====================== #

    # Given all the evaluation detail, calculate the final result.
    def post_process(self, eval_dl, detail_fp, result_fp):
        assert len(self.eval_detail_dict) > 0

        ret_q_score_dict = {}
        for scan_idx, (q_idx, cand) in enumerate(eval_dl.eval_sc_tup_list):
            cand.run_info = {k: data_values[scan_idx] for k, data_values in self.eval_detail_dict.items()}
            ret_q_score_dict.setdefault(q_idx, []).append(cand)
            # put all output results into sc.run_info

        score_key = '%s_score' % self.task_name
        f1_key = 'f1' if self.task_name == 'full' else '%s_f1' % self.task_name
        f1_list = []
        for q_idx, score_list in ret_q_score_dict.items():
            score_list.sort(key=lambda x: x.run_info[score_key], reverse=True)  # sort by score DESC
            if len(score_list) == 0:
                f1_list.append(0.)
            else:
                f1_list.append(getattr(score_list[0], f1_key))
        LogInfo.logs('[%3s] Predict %d out of %d questions.', self.task_name, len(f1_list), eval_dl.total_questions)
        ret_metric = np.sum(f1_list).astype('float32') / eval_dl.total_questions

        if detail_fp is not None:
            schema_dataset = eval_dl.schema_dataset
            bw = open(detail_fp, 'w')
            LogInfo.redirect(bw)
            np.set_printoptions(threshold=np.nan)
            LogInfo.logs('Avg_%s_f1 = %.6f', self.task_name, ret_metric)
            srt_q_idx_list = sorted(ret_q_score_dict.keys())
            for q_idx in srt_q_idx_list:
                qa = schema_dataset.qa_list[q_idx]
                q = qa['utterance']
                LogInfo.begin_track('Q-%04d [%s]:', q_idx, q.encode('utf-8'))
                srt_list = ret_q_score_dict[q_idx]  # already sorted
                best_label_f1 = np.max([getattr(sc, f1_key) for sc in srt_list])
                best_label_f1 = max(best_label_f1, 0.000001)
                for rank, sc in enumerate(srt_list):
                    cur_f1 = getattr(sc, f1_key)
                    if rank < 20 or cur_f1 == best_label_f1:
                        LogInfo.begin_track('#-%04d [%s_F1 = %.6f] [row_in_file = %d]',
                                            rank+1, self.task_name, cur_f1, sc.ori_idx)
                        LogInfo.logs('%s: %.6f', score_key, sc.run_info[score_key])
                        if self.detail_disp_func is not None:
                            self.detail_disp_func(sc=sc, qa=qa, schema_dataset=schema_dataset)
                        else:
                            LogInfo.logs('Current: not output detail.')
                        LogInfo.end_track()
                LogInfo.end_track()
            LogInfo.logs('Avg_%s_f1 = %.6f', self.task_name, ret_metric)

            np.set_printoptions()  # reset output format
            LogInfo.stop_redirect()
            bw.close()

        """ Save detail information """
        if result_fp is not None:
            srt_q_idx_list = sorted(ret_q_score_dict.keys())
            with open(result_fp, 'w') as bw:  # write question --> selected schema
                for q_idx in srt_q_idx_list:
                    srt_list = ret_q_score_dict[q_idx]
                    ori_idx = -1
                    task_f1 = 0.
                    if len(srt_list) > 0:
                        best_sc = srt_list[0]
                        ori_idx = best_sc.ori_idx
                        task_f1 = getattr(best_sc, f1_key)
                    bw.write('%d\t%d\t%.6f\n' % (q_idx, ori_idx, task_f1))

        return ret_metric
