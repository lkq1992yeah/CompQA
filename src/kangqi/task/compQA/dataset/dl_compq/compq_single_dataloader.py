"""
Author: Kangqi Luo
Goal: each data in the dataloader is a <q, sc> pair.
Due to different schema has different focus entities, so we got different qwords (with placeholder).
Therefore, multiple schemas cannot share one question representation.
"""


import numpy as np

from ..base_dataloader import DataLoader
from ..u import get_q_range_by_mode

from kangqi.util.LogUtil import LogInfo


class CompqSingleDataLoader(DataLoader):

    def __init__(self, dataset, mode,
                 batch_size, proc_ob_num,
                 dynamic=False, shuffle=False, verbose=0):
        super(CompqSingleDataLoader, self).__init__(batch_size=batch_size,
                                                    mode=mode,
                                                    proc_ob_num=proc_ob_num,
                                                    dynamic=dynamic,
                                                    shuffle=shuffle)
        self.dataset = dataset
        self.verbose = verbose

        self.question_size = None
        self.cand_tup_list = None       # [(q_idx, sc)]
        # store the candidate schemas of all related questions,
        # ** keeping the same order as what in np_data_list

    def renew_data_list(self):
        """
        Target: construct the np_data_list, storing all <q, sc> data
        np_data_list contains:
        1. q_words      (data_size, q_max_len)
        2. q_words_len  (data_size, )
        3. sc_len       (data_size, )
        4. preds        (data_size, sc_max_len, path_max_len)
        5. preds_len    (data_size, sc_max_len)
        6. pwords       (data_size, sc_max_len, pword_max_len)
        7. pwords_len   (data_size, sc_max_len)
        """
        if self.verbose >= 1:
            LogInfo.begin_track('[CompqSingleDataLoader] prepare data for [%s] ...', self.mode)
        q_idx_list = get_q_range_by_mode(data_name=self.dataset.data_name,
                                         mode=self.mode)
        filt_q_idx_list = filter(lambda q: q in self.dataset.q_cand_dict, q_idx_list)
        self.question_size = len(filt_q_idx_list)
        # Got all the related questions

        emb_pools = []              # [ qwords, qwords_len, sc_len, preds, preds_len, pwords, pwords_len ]
        for _ in range(self.dataset.array_num):
            emb_pools.append([])
        """
            Different from original complex DataLoader.
            No schemas share the same qwords.
        """
        self.cand_tup_list = []
        for scan_idx, q_idx in enumerate(filt_q_idx_list):
            if self.verbose >= 1 and scan_idx % self.proc_ob_num == 0:
                LogInfo.logs('%d / %d prepared.', scan_idx, len(filt_q_idx_list))
            cand_list = self.dataset.q_cand_dict[q_idx]

            # now store schema input into the corresponding position in the np_data_list
            for sc in cand_list:
                self.cand_tup_list.append((q_idx, sc))
                sc_tensor_inputs = self.dataset.get_schema_tensor_inputs(sc)
                for local_list, sc_tensor in zip(emb_pools, sc_tensor_inputs):
                    local_list.append(sc_tensor)
            # now the detail input of the schema is copied into the memory of the dataloader
            # including qwords, preds, pwords

        # Finally: merge inputs together, and produce the final np_data_list
        self.np_data_list = []
        for target_list in emb_pools:
            self.np_data_list.append(np.array(target_list, dtype='int32'))
        self.update_statistics()
        assert len(self.cand_tup_list) == self.np_data_list[0].shape[0]

        if self.verbose >= 1:
            LogInfo.logs('%d <q, sc> data collected.', self.np_data_list[0].shape[0])
            LogInfo.end_track()
