import numpy as np

from ..base_evaluator import BaseEvaluator

from kangqi.util.LogUtil import LogInfo


class RelationMatchingEvaluator(BaseEvaluator):

    def __init__(self, compq_mt_model, sess, ob_batch_num=10,
                 name='rm', summary_writer=None):
        self.compq_mt_model = compq_mt_model
        self.input_name_list = compq_mt_model.rm_eval_input_names
        # self.output_name_list = ['rm_score', 'rm_att_mat', 'rm_q_weight', 'rm_path_weight']
        self.output_name_list = []

        self.try_list = ['rm_score', 'rm_path_score', 'rm_att_mat', 'rm_q_weight', 'rm_path_weight',
                         'rm_fw_att_mat', 'rm_bw_att_mat', 'rm_fw_q_weight', 'rm_bw_q_weight']
        # all possible displaying items
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

        rm_f1_list = []
        for q_idx, score_list in ret_q_score_dict.items():
            score_list.sort(key=lambda x: x.run_info['rm_score'], reverse=True)  # sort by score DESC
            if len(score_list) == 0:
                rm_f1_list.append(0.)
            else:
                rm_f1_list.append(score_list[0].rm_f1)
        LogInfo.logs('[%3s] Predict %d out of %d questions.', self.name, len(rm_f1_list), eval_dl.total_questions)
        ret_metric = np.sum(rm_f1_list).astype('float32') / eval_dl.total_questions

        """ Save detail information """
        if result_fp is not None:
            srt_q_idx_list = sorted(ret_q_score_dict.keys())
            with open(result_fp, 'w') as bw:  # write question --> selected schema
                for q_idx in srt_q_idx_list:
                    srt_list = ret_q_score_dict[q_idx]
                    ori_idx = -1
                    rm_f1 = 0.
                    if len(srt_list) > 0:
                        best_sc = srt_list[0]
                        ori_idx = best_sc.ori_idx
                        rm_f1 = best_sc.rm_f1
                    bw.write('%d\t%d\t%.6f\n' % (q_idx, ori_idx, rm_f1))

        if detail_fp is not None:
            # use_p = self.compq_mt_model.rm_kernel.use_p
            # use_pw = self.compq_mt_model.rm_kernel.use_pw
            # path_usage = self.compq_mt_model.rm_kernel.path_usage
            # pw_max_len = self.compq_mt_model.pword_max_len
            # p_max_len = self.compq_mt_model.path_max_len
            schema_dataset = eval_dl.schema_dataset
            bw = open(detail_fp, 'w')
            LogInfo.redirect(bw)
            np.set_printoptions(threshold=np.nan)
            LogInfo.logs('Avg_rm_f1 = %.6f', ret_metric)
            srt_q_idx_list = sorted(ret_q_score_dict.keys())
            for q_idx in srt_q_idx_list:
                qa = schema_dataset.qa_list[q_idx]
                q = qa['utterance']
                LogInfo.begin_track('Q-%04d [%s]:', q_idx, q.encode('utf-8'))

                srt_list = ret_q_score_dict[q_idx]  # already sorted
                best_label_f1 = np.max([sc.rm_f1 for sc in srt_list])
                best_label_f1 = max(best_label_f1, 0.000001)
                for rank, sc in enumerate(srt_list):
                    if rank < 20 or sc.rm_f1 == best_label_f1:
                        LogInfo.begin_track('#-%04d [rm_F1 = %.6f] [row_in_file = %d]', rank + 1, sc.rm_f1, sc.ori_idx)
                        LogInfo.logs('rm_score: %.6f', sc.run_info['rm_score'])
                        # self.show_att_mat(sc=sc, qa=qa)
                        LogInfo.logs('Current: not output detail.')
                        LogInfo.end_track()
                LogInfo.end_track()
            LogInfo.logs('Avg_rm_f1 = %.6f', ret_metric)

            # LogInfo.logs('=================== Parameters ===================')
            # for param_name, param_result in zip(param_name_list, param_result_list):
            #     LogInfo.begin_track('%s: shape = %s ', param_name, param_result.shape)
            #     LogInfo.logs(param_result)
            #     LogInfo.end_track()

            np.set_printoptions()  # reset output format
            LogInfo.stop_redirect()
            bw.close()

        return [ret_metric]

    def show_att_mat(self, sc, qa):
        """
        Show detail: path, path words, attention matrix, or even induced answer type distribution
        :param sc:
        :param qa:
        :return:
        """
        # TODO: just support pwOnly
        raw_tok_list = [tok.token.lower() for tok in qa['tokens']]
        rm_tok_list = RelationMatchingEvaluator.prepare_rm_tok_list(sc=sc, raw_tok_list=raw_tok_list)
        qw_len = sc.input_np_dict['qw_len']
        q_max_len = len(sc.input_np_dict['gather_pos'])

        output_list = []
        for key in self.try_list:
            val = sc.run_info[key].tolist() if key in sc.run_info else None
            output_list.append(val)
        rm_score, rm_path_score, rm_att_mat, rm_q_weight, rm_path_weight, \
            rm_fw_att_mat, rm_bw_att_mat, rm_fw_q_weight, rm_bw_q_weight = output_list

        """ Build path_list and path_words_list, depending on whether using induced types or not """
        if not sc.use_ans_type_dist:
            show_path_list = sc.path_list
            show_path_words_list = sc.path_words_list
        else:       # pick non-Type paths, and adding the induce type information at the end.
            show_path_words_list = list(sc.path_words_list)
            show_path_list = []
            for raw_path, path in zip(sc.raw_paths, sc.path_list):
                if raw_path[0] != 'Type':
                    show_path_list.append(path)
            # ignored type constraints doesn't have the corresponding path word.
            # add induced type information at the ending
            induced_type_words = [str(sc.induced_type_weight_tups)]     # p_len = pw_len = 1
            show_path_words_list.append(induced_type_words)
            show_path_list.append(['induced_types'])

        show_path_size = len(show_path_list)
        if len(show_path_list) != len(show_path_words_list):
            LogInfo.begin_track('Warning: path_list and path_words_list mismatch!')
            for idx, path in enumerate(show_path_list):
                LogInfo.logs('Path-%d/%d: %s', idx + 1, len(show_path_list), path)
            for idx, path_words in enumerate(show_path_words_list):
                LogInfo.logs('Pword-%d/%d: %s', idx + 1, len(show_path_words_list), path_words)
            LogInfo.end_track()

        """ show the detail of each path one by one """
        for path_idx in range(show_path_size):
            LogInfo.begin_track('Showing att_mat at path-%d / %d:', path_idx + 1, show_path_size)
            pwords = show_path_words_list[path_idx]
            local_pw_len = len(pwords)
            LogInfo.logs('Path: [%s]', '-->'.join(show_path_list[path_idx]).encode('utf-8'))
            LogInfo.logs('Path-Word: [%s]', ' | '.join(pwords).encode('utf-8'))

            if rm_path_score is not None:
                LogInfo.logs('Path-score: %.6f', rm_path_score[path_idx])

            for att_name, active_att_mat, active_q_weight, active_path_weight in (
                ['Attention', rm_att_mat, rm_q_weight, rm_path_weight],     # attention matrix, or attention vector
                ['FW-Attention', rm_fw_att_mat, rm_fw_q_weight, None],      # forward attention vector
                ['BW-Attention', rm_bw_att_mat, rm_bw_q_weight, None]       # backward attention vector
            ):
                if active_att_mat is None:
                    continue
                LogInfo.logs('[%s]:', att_name)
                dim = len(np.array(active_att_mat).shape)
                if dim == 3:        # cross attention: (sc_max_len, qw_max_len, pw_max_len)
                    pwords_str = '%14s  %12s' % ('', '*')
                    for wd in pwords:
                        pwords_str += '  %12s' % wd
                    LogInfo.logs(pwords_str.encode('utf-8'))
                    if active_path_weight is not None:      # used in CrossAttention scenario
                        use_path_weight = active_path_weight[path_idx][:local_pw_len]
                        path_weight_str = '%14s  %12s' % ('*', '')
                        for score in use_path_weight:
                            path_weight_str += '  %12.3f' % score
                        LogInfo.logs(path_weight_str)
                    for tok_idx in range(min(qw_len+1, q_max_len)):     # output one more padding row if possible
                        att_vec = active_att_mat[path_idx][tok_idx][:local_pw_len+1]
                        # output one more padding column
                        disp_str = '%14s' % rm_tok_list[tok_idx]
                        if active_q_weight is not None:
                            disp_str += '  %12.3f' % active_q_weight[path_idx][tok_idx]
                        for score in att_vec:
                            disp_str += '  %12.3f' % score
                        LogInfo.logs(disp_str.encode('utf-8'))
                elif dim == 2:      # simple attention: (sc_max_len, qw_max_len)
                    for tok_idx in range(min(qw_len+1, q_max_len)):
                        tok = rm_tok_list[tok_idx]
                        att_val = active_att_mat[path_idx][tok_idx]
                        q_weight = active_q_weight[path_idx][tok_idx]
                        marks = int(round(q_weight * 50))
                        mark_str = '*' * marks + ' ' * (50 - marks)
                        disp_str = '%14s  %8.3f  %8.3f  [%s]' % (tok, att_val, q_weight, mark_str)
                        LogInfo.logs(disp_str.encode('utf-8'))
                else:
                    LogInfo.logs('Abnormal attention matrix: dimension = %d.', dim)
            LogInfo.end_track()

    @staticmethod
    def prepare_rm_tok_list(sc, raw_tok_list):
        rm_tok_list = []
        e_mask = sc.input_np_dict['e_mask']
        tm_mask = sc.input_np_dict['tm_mask']
        gather_pos = sc.input_np_dict['gather_pos']
        qw_len = sc.input_np_dict['qw_len']
        q_max_len = len(gather_pos)
        for idx in range(qw_len):
            original_pos = gather_pos[idx]
            if e_mask[original_pos] == 1:
                rm_tok_list.append('<E>')
            elif tm_mask[original_pos] == 1:
                rm_tok_list.append('<Tm>')
            else:
                rm_tok_list.append(raw_tok_list[original_pos])
        rm_tok_list += ['<PAD>'] * (q_max_len - qw_len)
        return rm_tok_list
