from ..base_optimizer import BaseOptimizer


class RelationMatchingOptimizer(BaseOptimizer):

    def __init__(self, compq_mt_model, sess, ob_batch_num=10, name='rm', summary_writer=None):
        """
        input_tensor_list: v, v_len, tag_indices
        eval_output_tensor_list: could be several tensors, but the first one must be seg_eval_output_seq
        """
        input_tensor_list = [compq_mt_model.input_tensor_dict[k] for k in compq_mt_model.rm_optm_input_names]
        BaseOptimizer.__init__(self, name=name,
                               input_tensor_list=input_tensor_list,
                               optm_step=compq_mt_model.rm_update,
                               loss=compq_mt_model.rm_loss,
                               extra_data=[],
                               # extra_data=[compq_mt_model.optm_tensor_dict['pos_rm_score'],
                               #             compq_mt_model.optm_tensor_dict['neg_rm_score']],
                               optm_summary=compq_mt_model.optm_rm_summary,
                               sess=sess, ob_batch_num=ob_batch_num,
                               summary_writer=summary_writer)
