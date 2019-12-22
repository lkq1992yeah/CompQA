"""
Author: Kangqi Luo
Goal: each data in the dataloader is a <q, r> pair, which is very simple
"""

import numpy as np

from .base_dataloader import DataLoader
from .u import get_q_range_by_mode

from kangqi.util.LogUtil import LogInfo


class SimpqSingleDataLoader(DataLoader):

    def __init__(self, dataset, mode, q_max_len,
                 batch_size, proc_ob_num,
                 dynamic=False, shuffle=False, verbose=0):
        super(SimpqSingleDataLoader, self).__init__(batch_size=batch_size,
                                                    mode=mode,
                                                    proc_ob_num=proc_ob_num,
                                                    dynamic=dynamic,
                                                    shuffle=shuffle)
        self.dataset = dataset
        self.verbose = verbose

        self.q_max_len = q_max_len

        self.cand_tup_list = None       # [(q_idx, sc)]
        # store the candidate schemas of all related questions,
        # ** keeping the same order as what in np_data_list

    def renew_data_list(self):
        """
        Target: construct the np_data_list, storing all <q, r> data
        np_data_list contains:
        1. q_words      (data_size, q_max_len)
        2. q_words_len  (data_size, )
        3. preds        (data_size, path_max_len)
        4. preds_len    (data_size, )
        5. pwords       (data_size, pword_max_len)
        6. pwords_len   (data_size, )
        """
        if self.verbose >= 1:
            LogInfo.begin_track('[SimpqSingleDataLoader] prepare data for [%s] ...', self.mode)
        q_idx_list = get_q_range_by_mode(data_name=self.dataset.data_name,
                                         mode=self.mode)
        filt_q_idx_list = filter(lambda q: q in self.dataset.q_cand_dict, q_idx_list)
        # Got all the related questions

        q_words_input = []          # (data_size, q_max_len)
        q_words_len_input = []      # (data_size, )
        emb_pools = []              # [ preds, preds_len, pwords, pwords_len ]
        for _ in range(self.dataset.array_num):
            emb_pools.append([])

        self.cand_tup_list = []
        for scan_idx, q_idx in enumerate(filt_q_idx_list):
            if self.verbose >= 1 and scan_idx % self.proc_ob_num == 0:
                LogInfo.logs('%d / %d prepared.', scan_idx, len(filt_q_idx_list))
            q_words = self.dataset.q_words_dict[q_idx]
            q_words_len = len(q_words)
            word_idx_vec = np.zeros((self.q_max_len, ), dtype='int32')
            word_idx_vec[:q_words_len] = q_words        # get the word sequence of the question

            cand_list = self.dataset.q_cand_dict[q_idx]
            cand_size = len(cand_list)
            q_words_input += [word_idx_vec] * cand_size
            q_words_len_input += [q_words_len] * cand_size  # duplicate storage in Q side

            # now store schema input into the corresponding position in the np_data_list
            for sc in cand_list:
                self.cand_tup_list.append((q_idx, sc))
                sc_tensor_inputs = self.dataset.get_schema_tensor_inputs(sc)
                for local_list, sc_tensor in zip(emb_pools, sc_tensor_inputs):
                    local_list.append(sc_tensor)
                # now the detail input of the schema is copied into the memory of the dataloader

        # Finally: merge inputs together, and produce the final np_data_list
        self.np_data_list = [
            np.array(q_words_input, dtype='int32'),
            np.array(q_words_len_input, dtype='int32')
        ]
        for target_list in emb_pools:
            self.np_data_list.append(np.array(target_list, dtype='int32'))
        self.update_statistics()
        assert len(self.cand_tup_list) == self.np_data_list[0].shape[0]

        if self.verbose >= 1:
            LogInfo.logs('%d <q, r> data collected.', self.np_data_list[0].shape[0])
            LogInfo.end_track()
