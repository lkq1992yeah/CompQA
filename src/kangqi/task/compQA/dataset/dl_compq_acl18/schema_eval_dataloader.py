"""
Author: Kangqi Luo
Date: 180212
Goal: The data loader for schema-level tasks: rm (relation matching), el (entity linking), and all.
"""
import numpy as np
import tensorflow as tf

from ..base_dataloader import DataLoader
from ..dataset_acl18 import SchemaDatasetACL18
from ...model.compq_acl18 import CompqMultiTaskModel

from kangqi.util.LogUtil import LogInfo


class SchemaEvalDataLoader(DataLoader):
    def __init__(self, schema_dataset, compq_mt_model, q_evals_dict, task_name, mode, shuffle, batch_size):
        # q_evals_dict: <q, [sc]>
        assert isinstance(schema_dataset, SchemaDatasetACL18)
        assert isinstance(compq_mt_model, CompqMultiTaskModel)

        DataLoader.__init__(self, mode=mode, batch_size=batch_size, dynamic=False, shuffle=shuffle)
        """ dynamic=False: once we changed the train/eval data, we just change a new data loader """

        self.schema_dataset = schema_dataset
        self.total_questions = len(q_evals_dict)
        self.eval_sc_tup_list = []      # [(q_idx, sc)], used for tracing the original feed data# ]
        for q_idx, eval_list in q_evals_dict.items():
            for sc in eval_list:
                self.eval_sc_tup_list.append((q_idx, sc))
        total_size = len(self.eval_sc_tup_list)
        self.eval_sc_tup_list.sort(key=lambda _tup: _tup[0])  # just sort by q_idx

        wd_emb_util = schema_dataset.wd_emb_util
        input_tensor_names = getattr(compq_mt_model, '%s_eval_input_names' % task_name)
        global_input_dict = {k: [] for k in input_tensor_names}

        for q_idx, sc in self.eval_sc_tup_list:
            v_len = schema_dataset.v_len_vec[q_idx]
            v_input = schema_dataset.v_input_mat[q_idx]
            clause_input = schema_dataset.clause_input_mat[q_idx]
            sc_np_dict = sc.create_input_np_dict(
                qw_max_len=schema_dataset.q_max_len, sc_max_len=schema_dataset.sc_max_len,
                p_max_len=schema_dataset.path_max_len, pw_max_len=schema_dataset.pword_max_len,
                type_dist_len=schema_dataset.type_dist_len, q_len=v_len,
                word_idx_dict=wd_emb_util.load_word_indices(),
                mid_idx_dict=wd_emb_util.load_mid_indices()
            )
            for k, v in sc_np_dict.items():
                if k in global_input_dict:
                    global_input_dict[k].append(v)
            global_input_dict['v_len'].append(v_len)
            global_input_dict['v_input'].append(v_input)
            global_input_dict['clause_input'].append(clause_input)
        LogInfo.logs('%d schemas saved in dataloader [%s-%s].', total_size, task_name, mode)

        self.np_data_list = []
        for k in input_tensor_names:
            dtype = compq_mt_model.input_tensor_dict[k].dtype
            np_type = 'float32' if dtype == tf.float32 else 'int32'
            np_arr = np.array(global_input_dict[k], dtype=np_type)
            self.np_data_list.append(np_arr)

        self.update_statistics()
