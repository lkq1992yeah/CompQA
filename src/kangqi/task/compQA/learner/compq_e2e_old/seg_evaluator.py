from ..base_evaluator import BaseEvaluator

from kangqi.util.LogUtil import LogInfo


class SegmentEvaluator(BaseEvaluator):

    def __init__(self, compq_mt_model, sess, ob_batch_num=10, name='seg', summary_writer=None):
        """
        input_tensor_list: v, v_len, tag_indices
        """
        input_tensor_list = [compq_mt_model.input_tensor_dict[k] for k in compq_mt_model.seg_input_names]
        eval_output_tensor_list = [compq_mt_model.eval_tensor_dict['best_seg'],
                                   compq_mt_model.eval_tensor_dict['seg_loglik']]
        BaseEvaluator.__init__(self, name=name,
                               input_tensor_list=input_tensor_list,
                               eval_output_tensor_list=eval_output_tensor_list,
                               sess=sess, ob_batch_num=ob_batch_num,
                               summary_writer=summary_writer)

    @staticmethod
    def pick_eval_detail(local_input_list, local_output_list):
        v, v_len, gold_indices = local_input_list
        predict_indices, loglik = local_output_list
        local_eval_detail_list = [v_len, gold_indices, predict_indices, loglik]
        return local_eval_detail_list

    def post_process(self, eval_dl, detail_fp):
        total_pred = 0      # total number of predicted chunks
        total_gold = 0      # total number of gold chunks
        total_correct = 0
        avg_log_lik = 0.

        data_size = 0
        for v_len, gold_tag, pred_tag, log_lik in zip(*self.eval_detail_list):
            data_size += 1
            gold_tag = gold_tag[: v_len]
            pred_tag = pred_tag[: v_len]
            gold_chunk_set = self.produce_chunk(gold_tag)
            pred_chunk_set = self.produce_chunk(pred_tag)
            total_pred += len(pred_chunk_set)
            total_gold += len(gold_chunk_set)
            total_correct += len(pred_chunk_set & gold_chunk_set)
            avg_log_lik += log_lik
            if data_size <= 5:
                LogInfo.begin_track('Check case-%d:', data_size)
                LogInfo.logs('seq_len = %d', v_len)
                LogInfo.logs('Gold: %s --> %s', gold_tag.tolist(), gold_chunk_set)
                LogInfo.logs('Pred: %s --> %s', pred_tag.tolist(), pred_chunk_set)
                LogInfo.logs('log_lik = %.6f', log_lik)
                LogInfo.end_track()
        p = 1. * total_correct / total_pred
        r = 1. * total_correct / total_gold
        f1 = 2.*p*r / (p+r) if (p+r) > 0. else 0.
        avg_log_lik /= data_size
        return f1, avg_log_lik

    @staticmethod
    def produce_chunk(tag_sequence):
        chunk_set = set([])
        seq_len = len(tag_sequence)
        st = -1
        for idx, value in enumerate(tag_sequence):
            if value == 0 or value == 2:    # B or O, check whether previous tags form a chunk
                if st != -1:
                    chunk = '%d_%d' % (st, idx)
                    chunk_set.add(chunk)
                    st = -1
                if value == 0:              # B, start of a new chunk
                    st = idx
            elif value == 1:                # I, chunk is not complete
                if st == -1:                # I not following any B, so we just treat it as the beginning of a chunk
                    st = idx
        if st != -1:
            chunk_set.add('%d_%d' % (st, seq_len))
        return chunk_set
