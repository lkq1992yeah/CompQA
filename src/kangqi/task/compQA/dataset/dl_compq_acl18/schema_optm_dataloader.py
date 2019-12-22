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


class SchemaOptmDataLoader(DataLoader):

    def __init__(self, schema_dataset, compq_mt_model, q_optm_pairs_dict, task_name, mode, shuffle, batch_size):
        # q_optm_pairs_dict: <q, [(pos_sc, neg_sc, weight)]>
        assert isinstance(schema_dataset, SchemaDatasetACL18)
        assert isinstance(compq_mt_model, CompqMultiTaskModel)

        DataLoader.__init__(self, mode=mode, batch_size=batch_size, dynamic=False, shuffle=shuffle)
        """ dynamic=False: once we get new train/eval data, we just change a new data loader """

        self.schema_dataset = schema_dataset
        self.optm_pair_tup_list = []  # [(q_idx, pos_sc, neg_sc, weight)], used for tracing the original feed data
        for q_idx, optm_pairs in q_optm_pairs_dict.items():
            for pos_sc, neg_sc, weight in optm_pairs:
                self.optm_pair_tup_list.append((q_idx, pos_sc, neg_sc, weight))
        total_size = len(self.optm_pair_tup_list)
        self.optm_pair_tup_list.sort(key=lambda _tup: _tup[0])       # just sort by q_idx
        # np.random.shuffle(self.optm_pair_tup_list)  # we shuffle all <pos, neg> pairs here, instead of within dl.

        wd_emb_util = schema_dataset.wd_emb_util
        input_tensor_names = getattr(compq_mt_model, '%s_optm_input_names' % task_name)
        global_input_dict = {k: [] for k in input_tensor_names}

        for tup_idx, (q_idx, pos_sc, neg_sc, weight) in enumerate(self.optm_pair_tup_list):
            # LogInfo.logs('Data-%d/%d: q = %d, pos_rm_F1 = %.6f (row %d), neg_rm_F1 = %.6f (row %d), weight = %.6f',
            #              tup_idx+1, len(self.optm_pair_tup_list), q_idx,
            #              pos_sc.rm_f1, pos_sc.ori_idx,
            #              neg_sc.rm_f1, neg_sc.ori_idx, weight)
            v_len = schema_dataset.v_len_vec[q_idx]
            v_input = schema_dataset.v_input_mat[q_idx]
            clause_input = schema_dataset.clause_input_mat[q_idx]
            for sc in (pos_sc, neg_sc):     # even: pos_sc, odd: neg_sc
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
                global_input_dict['data_weights'].append(weight)
                """ Save data_weight twice, for keeping the same size of the first dimension"""
        LogInfo.logs('%d <pos, neg> pairs saved in dataloader [%s-%s].', total_size, task_name, mode)

        self.np_data_list = []
        for k in input_tensor_names:
            dtype = compq_mt_model.input_tensor_dict[k].dtype
            np_type = 'float32' if dtype == tf.float32 else 'int32'
            np_arr = np.array(global_input_dict[k], dtype=np_type)
            self.np_data_list.append(np_arr)

        self.update_statistics()
