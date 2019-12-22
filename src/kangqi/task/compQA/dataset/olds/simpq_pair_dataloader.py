"""
Author: Kangqi Luo
Goal: The dataloader is used in SimpQ experiments (relation detection experiment)
      Maintain all the <q, r+, r-> training data
"""

import numpy as np

from .base_dataloader import DataLoader
from .u import get_q_range_by_mode

from kangqi.util.LogUtil import LogInfo


class SimpqPairDataLoader(DataLoader):

    def __init__(self, dataset, mode, q_max_len,
                 batch_size, proc_ob_num,
                 dynamic=False, shuffle=True, verbose=0):
        super(SimpqPairDataLoader, self).__init__(batch_size=batch_size,
                                                  mode=mode,
                                                  proc_ob_num=proc_ob_num,
                                                  dynamic=dynamic,
                                                  shuffle=shuffle)
        self.dataset = dataset
        self.verbose = verbose

        self.q_max_len = q_max_len

    def renew_data_list(self):
        """
        Target: construct the np_data_list, storing all <q, r+, r-> pairs
        np_data_list contains:
        1. q_words              (data_size, q_max_len)
        2. q_words_len          (data_size, )
        3. pos_preds            (data_size, path_max_len)
        4. pos_preds_len        (data_size, )
        5. pos_pwords           (data_size, pword_max_len)
        6. pos_pwords_len       (data_size, )
        7-10. neg_xxxxxx
        """
        if self.verbose >= 1:
            LogInfo.begin_track('[SimpqPairDataLoader] prepare data for [%s] ...', self.mode)
        q_idx_list = get_q_range_by_mode(data_name=self.dataset.data_name,
                                         mode=self.mode)
        filt_q_idx_list = filter(lambda q: q in self.dataset.q_cand_dict, q_idx_list)
        # Got all the related questions

        q_words_input = []          # (data_size, q_max_len)
        q_words_len_input = []      # (data_size, )
        pos_emb_pools = []          # [ pos_preds, pos_preds_len, pos_pwords, pos_pwords_len ]
        neg_emb_pools = []          # [ neg_xxx ... ]
        for _ in range(self.dataset.array_num):
            pos_emb_pools.append([])
            neg_emb_pools.append([])

        for scan_idx, q_idx in enumerate(filt_q_idx_list):
            if self.verbose >= 1 and scan_idx % self.proc_ob_num == 0:
                LogInfo.logs('%d / %d prepared.', scan_idx, len(filt_q_idx_list))
            q_words = self.dataset.q_words_dict[q_idx]
            q_words_len = len(q_words)
            word_idx_vec = np.zeros((self.q_max_len, ), dtype='int32')
            word_idx_vec[:q_words_len] = q_words        # get the word sequence of the question

            cand_list = self.dataset.q_cand_dict[q_idx]
            pos_cands = filter(lambda _sc: _sc.f1 == 1.0, cand_list)
            neg_cands = filter(lambda _sc: _sc.f1 == 0.0, cand_list)
            pairs = []
            for pos in pos_cands:
                for neg in neg_cands:       # enumerate each <r+, r-> pair
                    pairs.append((pos, neg))        # just collect all the combinations
            pair_size = len(pairs)
            q_words_input += [word_idx_vec] * pair_size
            q_words_len_input += [q_words_len] * pair_size

            # now store the positive / negative input into the corresponding position in the np_data_list
            for direction, emb_pools in [(0, pos_emb_pools), (1, neg_emb_pools)]:
                sc_list = [pair[direction] for pair in pairs]     # lhs / rhs schema list
                for sc in sc_list:
                    sc_tensor_inputs = self.dataset.get_schema_tensor_inputs(sc)
                    for local_list, sc_tensor in zip(emb_pools, sc_tensor_inputs):
                        local_list.append(sc_tensor)
                    # now the detail input of the schema is copied into the memory of the dataloader

        # Finally: merge inputs together, and produce the final np_data_list
        self.np_data_list = [
            np.array(q_words_input, dtype='int32'),
            np.array(q_words_len_input, dtype='int32')
        ]
        for emb_pools in (pos_emb_pools, neg_emb_pools):
            for target_list in emb_pools:
                self.np_data_list.append(np.array(target_list, dtype='int32'))
        self.update_statistics()

        if self.verbose >= 1:
            LogInfo.logs('%d <q, r+, r-> data collected.', self.np_data_list[0].shape[0])
            LogInfo.end_track()
