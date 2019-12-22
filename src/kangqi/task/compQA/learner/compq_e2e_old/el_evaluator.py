import numpy as np

from ..base_evaluator import BaseEvaluator
from ...util.fb_helper import get_domain, get_range

from kangqi.util.LogUtil import LogInfo


class EntityLinkingEvaluator(BaseEvaluator):

    def __init__(self, compq_mt_model, sess, ob_batch_num=10,
                 name='el', summary_writer=None):
        self.compq_mt_model = compq_mt_model
        self.input_name_list = compq_mt_model.el_eval_input_names
        # self.output_name_list = ['el_score', 'el_feats_concat', 'el_raw_score']
        self.output_name_list = []
        self.try_list = ['el_score', 'el_feats_concat', 'el_raw_score', 'el_final_feats']
        # all possible displaying items (some from ACL18, some from EMNLP18)
        for val_name in self.try_list:
            if val_name in compq_mt_model.eval_tensor_dict:
                self.output_name_list.append(val_name)
        concern_name_list = self.output_name_list

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

        el_f1_list = []
        for q_idx, score_list in ret_q_score_dict.items():
            score_list.sort(key=lambda x: x.run_info['el_score'], reverse=True)  # sort by score DESC
            if len(score_list) == 0:
                el_f1_list.append(0.)
            else:
                el_f1_list.append(score_list[0].el_f1)
        LogInfo.logs('[%3s] Predict %d out of %d questions.', self.name, len(el_f1_list), eval_dl.total_questions)
        ret_metric = np.sum(el_f1_list).astype('float32') / eval_dl.total_questions

        if detail_fp is not None:
            schema_dataset = eval_dl.schema_dataset
            bw = open(detail_fp, 'w')
            LogInfo.redirect(bw)
            np.set_printoptions(threshold=np.nan)
            LogInfo.logs('Avg_el_f1 = %.6f', ret_metric)
            srt_q_idx_list = sorted(ret_q_score_dict.keys())
            for q_idx in srt_q_idx_list:
                qa = schema_dataset.qa_list[q_idx]
                q = qa['utterance']
                LogInfo.begin_track('Q-%04d [%s]:', q_idx, q.encode('utf-8'))
                srt_list = ret_q_score_dict[q_idx]  # already sorted
                best_label_f1 = np.max([sc.el_f1 for sc in srt_list])
                best_label_f1 = max(best_label_f1, 0.000001)
                for rank, sc in enumerate(srt_list):
                    if rank < 20 or sc.el_f1 == best_label_f1:
                        LogInfo.begin_track('#-%04d [el_F1 = %.6f] [row_in_file = %d]', rank + 1, sc.el_f1, sc.ori_idx)
                        LogInfo.logs('el_score: %.6f', sc.run_info['el_score'])
                        # show_el_detail(sc=sc)
                        LogInfo.logs('Current: not output detail.')
                        LogInfo.end_track()
                LogInfo.end_track()
            LogInfo.logs('Avg_el_f1 = %.6f', ret_metric)
            np.set_printoptions()  # reset output format
            LogInfo.stop_redirect()
            bw.close()

        return [ret_metric]


def show_el_detail(sc):
    el_feats_concat, el_raw_score = [
        sc.run_info[k].tolist() for k in ('el_feats_concat', 'el_raw_score')
    ]
    el_len = sc.input_np_dict['el_len']
    gl_tup_list = []
    for category, gl_data, pred_seq in sc.raw_paths:
        if category not in ('Entity', 'Main'):
            continue
        if category == 'Main':
            tp = get_domain(pred_seq[0])
        else:
            tp = get_range(pred_seq[-1])
        gl_tup_list.append((gl_data, tp))
    assert len(gl_tup_list) == el_len
    for el_idx in range(el_len):
        gl_data, tp = gl_tup_list[el_idx]
        LogInfo.begin_track('Entity %d / %d:', el_idx + 1, el_len)
        LogInfo.logs(gl_data.display())
        LogInfo.logs('Prominent type: [%s]  <---  (Ignore this line if type is not used.)', tp)
        LogInfo.logs('raw_score = %.6f', el_raw_score[el_idx])
        show_str = '  '.join(['%6.3f' % x for x in el_feats_concat[el_idx]])
        LogInfo.logs('el_feats_concat = %s', show_str)
        LogInfo.end_track()
