import numpy as np

from .base_dataloader import BaseDataLoader
from kangqi.util.LogUtil import LogInfo


class BaseSchemaEvalDataLoader(BaseDataLoader):

    def __init__(self, q_evals_dict,
                 input_tensor_dict, schema_dataset,
                 task_name, mode, shuffle, batch_size):
        """
        :param q_evals_dict: <q, [sc]>
        :param input_tensor_dict: <input_name, tensor definition>
        :param task_name: el / rm / full, just for display
        :param mode: train / valid / test, just for display
        :param batch_size:
        :param shuffle: shuffle all candidate schemas or not
        """
        BaseDataLoader.__init__(self, batch_size=batch_size)

        self.task_name = task_name
        self.mode = mode
        self.schema_dataset = schema_dataset
        self.eval_sc_tup_list = []      # [(q_idx, sc)], used for tracing the original feed data
        self.total_questions = len(q_evals_dict)
        for q_idx, eval_list in q_evals_dict.items():
            for sc in eval_list:
                self.eval_sc_tup_list.append((q_idx, sc))
        n_rows = len(self.eval_sc_tup_list)

        self.eval_sc_tup_list.sort(key=lambda _tup: _tup[0])  # just sort by q_idx
        if shuffle:
            np.random.shuffle(self.eval_sc_tup_list)

        global_input_dict = {}
        for q_idx, sc in self.eval_sc_tup_list:
            sc_np_dict = sc.input_np_dict       # Already generated features
            for k, v in sc_np_dict.items():
                global_input_dict.setdefault(k, []).append(v)
        LogInfo.logs('%d schemas saved in dataloader [%s-%s].', n_rows, task_name, mode)

        self.prepare_np_input_list(global_input_dict=global_input_dict,
                                   input_tensor_dict=input_tensor_dict,
                                   n_rows=n_rows)

    # def batch_postprocess(self):      # do nothing
    #     pass
