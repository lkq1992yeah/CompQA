from ..base_optimizer import BaseOptimizer


class EntityLinkingOptimizer(BaseOptimizer):

    def __init__(self, compq_mt_model, sess, ob_batch_num=10, name='el', summary_writer=None):
        """
        input_tensor_list: v, v_len, tag_indices
        eval_output_tensor_list: could be several tensors, but the first one must be seg_eval_output_seq
        """
        input_tensor_list = [compq_mt_model.input_tensor_dict[k] for k in compq_mt_model.el_optm_input_names]
        BaseOptimizer.__init__(self, name=name,
                               input_tensor_list=input_tensor_list,
                               optm_step=compq_mt_model.el_update,
                               loss=compq_mt_model.el_loss,
                               extra_data=[],
                               optm_summary=compq_mt_model.optm_el_summary,
                               sess=sess, ob_batch_num=ob_batch_num,
                               summary_writer=summary_writer)
