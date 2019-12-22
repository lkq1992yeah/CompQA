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


class SchemaEvalDataLoader(EMNLPDataLoader):
    def __init__(self, schema_dataset, compq_mt_model,
                 q_evals_dict, task_name, mode, batch_size, shuffle, feat_gen):
        # q_evals_dict: <q, [sc]>
        assert isinstance(schema_dataset, SchemaDatasetEMNLP18)
        assert isinstance(compq_mt_model, CompqMultiTaskModel)

        EMNLPDataLoader.__init__(self, mode=mode, batch_size=batch_size, compq_mt_model=compq_mt_model)
        """ dynamic=False: once we changed the train/eval data, we just change a new data loader """

        self.schema_dataset = schema_dataset
        self.total_questions = len(q_evals_dict)
        self.eval_sc_tup_list = []      # [(q_idx, sc)], used for tracing the original feed data# ]
        for q_idx, eval_list in q_evals_dict.items():
            for sc in eval_list:
                self.eval_sc_tup_list.append((q_idx, sc))
        total_size = len(self.eval_sc_tup_list)
        self.eval_sc_tup_list.sort(key=lambda _tup: _tup[0])  # just sort by q_idx
        if shuffle:
            np.random.shuffle(self.eval_sc_tup_list)

        input_tensor_names = getattr(compq_mt_model, '%s_eval_input_names' % task_name)
        global_input_dict = {}

        for q_idx, sc in self.eval_sc_tup_list:
            sc_np_dict = feat_gen.input_gen(q_idx=q_idx, cand=sc)
            for k, v in sc_np_dict.items():
                global_input_dict.setdefault(k, []).append(v)
        LogInfo.logs('%d schemas saved in dataloader [%s-%s].', total_size, task_name, mode)

        self.prepare_np_input_list(global_input_dict=global_input_dict,
                                   n_rows=len(self.eval_sc_tup_list),
                                   input_tensor_names=input_tensor_names)
