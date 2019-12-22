"""
Author: Kangqi Luo
Date: 180212
Goal: The data loader for schema-level tasks: rm (relation matching), el (entity linking), and all.
"""

import numpy as np

from .emnlp_dataloader import EMNLPDataLoader

from ..dataset_emnlp18 import SchemaDatasetEMNLP18
from ...model.compq_emnlp18.compq_mt_model import CompqMultiTaskModel

from kangqi.util.LogUtil import LogInfo


class SchemaOptmDataLoader(EMNLPDataLoader):

    def __init__(self, schema_dataset, compq_mt_model,
                 q_optm_pairs_dict, task_name, mode, batch_size, shuffle, feat_gen):
        # q_optm_pairs_dict: <q, [(pos_sc, neg_sc, weight)]>
        assert isinstance(schema_dataset, SchemaDatasetEMNLP18)
        assert isinstance(compq_mt_model, CompqMultiTaskModel)

        EMNLPDataLoader.__init__(self, mode=mode, batch_size=batch_size, compq_mt_model=compq_mt_model)

        self.schema_dataset = schema_dataset
        self.optm_pair_tup_list = []  # [(q_idx, pos_sc, neg_sc, weight)], used for tracing the original feed data
        for q_idx, optm_pairs in q_optm_pairs_dict.items():
            for pos_sc, neg_sc, weight in optm_pairs:
                self.optm_pair_tup_list.append((q_idx, pos_sc, neg_sc, weight))
        total_size = len(self.optm_pair_tup_list)
        self.optm_pair_tup_list.sort(key=lambda _tup: _tup[0])       # just sort by q_idx
        if shuffle:
            np.random.shuffle(self.optm_pair_tup_list)
            # we shuffle all <pos, neg> pairs here, instead of within dl.

        input_tensor_names = getattr(compq_mt_model, '%s_optm_input_names' % task_name)
        global_input_dict = {}

        for tup_idx, (q_idx, pos_sc, neg_sc, weight) in enumerate(self.optm_pair_tup_list):
            # LogInfo.logs('Data-%d/%d: q = %d, pos_rm_F1 = %.6f (row %d), neg_rm_F1 = %.6f (row %d), weight = %.6f',
            #              tup_idx+1, len(self.optm_pair_tup_list), q_idx,
            #              pos_sc.rm_f1, pos_sc.ori_idx,
            #              neg_sc.rm_f1, neg_sc.ori_idx, weight)
            for sc in (pos_sc, neg_sc):     # even: pos_sc, odd: neg_sc
                sc_np_dict = feat_gen.input_gen(q_idx=q_idx, cand=sc)
                for k, v in sc_np_dict.items():
                    global_input_dict.setdefault(k, []).append(v)
        LogInfo.logs('%d <pos, neg> pairs saved in dataloader [%s-%s].', total_size, task_name, mode)

        self.prepare_np_input_list(global_input_dict=global_input_dict,
                                   n_rows=2*len(self.optm_pair_tup_list),
                                   input_tensor_names=input_tensor_names)
