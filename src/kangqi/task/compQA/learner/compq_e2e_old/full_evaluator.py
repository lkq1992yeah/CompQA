import numpy as np

from ..base_evaluator import BaseEvaluator

from kangqi.util.LogUtil import LogInfo


class FullTaskEvaluator(BaseEvaluator):

    def __init__(self, compq_mt_model, sess, ob_batch_num=10,
                 name='full', summary_writer=None):
        self.compq_mt_model = compq_mt_model
        self.input_name_list = compq_mt_model.full_eval_input_names
        self.output_name_list = ['full_score', 'rich_feats_concat']
        concern_name_list = self.output_name_list
        self.ret_q_score_dict = {}      # saving all related schemas, for the use of pyltr

        input_tensor_list = [compq_mt_model.input_tensor_dict[k] for k in self.input_name_list]
        eval_output_tensor_list = [compq_mt_model.eval_tensor_dict[k] for k in self.output_name_list]
        BaseEvaluator.__init__(self, name=name, sess=sess,
                               input_name_list=self.input_name_list,
                               output_name_list=self.output_name_list,
                               input_tensor_list=input_tensor_list,
                               eval_output_tensor_list=eval_output_tensor_list,
                               concern_name_list=concern_name_list,
                               metric_name_list=['F1'],
                               ob_batch_num=ob_batch_num,
                               summary_writer=summary_writer)

    def post_process(self, eval_dl, detail_fp, result_fp):
        if len(self.eval_detail_dict) == 0:
            self.eval_detail_dict = {k: [] for k in self.concern_name_list}

        ret_q_score_dict = {}
        for scan_idx, (q_idx, cand) in enumerate(eval_dl.eval_sc_tup_list):
            cand.run_info = {k: self.eval_detail_dict[k][scan_idx] for k in self.concern_name_list}
            ret_q_score_dict.setdefault(q_idx, []).append(cand)
            # put all output results into sc.run_info

        f1_list = []
        for q_idx, score_list in ret_q_score_dict.items():
            score_list.sort(key=lambda x: x.run_info['full_score'], reverse=True)  # sort by score DESC
            if len(score_list) == 0:
                f1_list.append(0.)
            else:
                f1_list.append(score_list[0].f1)
        LogInfo.logs('[%3s] Predict %d out of %d questions.', self.name, len(f1_list), eval_dl.total_questions)
        ret_metric = np.sum(f1_list).astype('float32') / eval_dl.total_questions

        if detail_fp is not None:
            schema_dataset = eval_dl.schema_dataset
            bw = open(detail_fp, 'w')
            LogInfo.redirect(bw)
            np.set_printoptions(threshold=np.nan)
            LogInfo.logs('Avg_f1 = %.6f', ret_metric)
            srt_q_idx_list = sorted(ret_q_score_dict.keys())
            for q_idx in srt_q_idx_list:
                qa = schema_dataset.qa_list[q_idx]
                q = qa['utterance']
                LogInfo.begin_track('Q-%04d [%s]:', q_idx, q.encode('utf-8'))
                srt_list = ret_q_score_dict[q_idx]  # already sorted
                best_label_f1 = np.max([sc.f1 for sc in srt_list])
                best_label_f1 = max(best_label_f1, 0.000001)
                for rank, sc in enumerate(srt_list):
                    if rank < 20 or sc.f1 == best_label_f1:
                        LogInfo.begin_track('#-%04d [F1 = %.6f] [row_in_file = %d]', rank + 1, sc.f1, sc.ori_idx)
                        LogInfo.logs('full_score: %.6f', sc.run_info['full_score'])
                        show_overall_detail(sc)
                        LogInfo.end_track()
                LogInfo.end_track()
            LogInfo.logs('Avg_f1 = %.6f', ret_metric)
            np.set_printoptions()  # reset output format
            LogInfo.stop_redirect()
            bw.close()

        self.ret_q_score_dict = ret_q_score_dict
        return [ret_metric]


def show_overall_detail(sc):
    rich_feats_concat = sc.run_info['rich_feats_concat'].tolist()
    for category, gl_data, pred_seq in sc.raw_paths:
        LogInfo.logs('%s: link = [(#-%d) %s %s], pred_seq = %s',
                     category, gl_data.gl_pos, gl_data.comp, gl_data.value, pred_seq)
    show_str = '  '.join(['%6.3f' % x for x in rich_feats_concat])
    LogInfo.logs('rich_feats_concat = [%s]', show_str)
