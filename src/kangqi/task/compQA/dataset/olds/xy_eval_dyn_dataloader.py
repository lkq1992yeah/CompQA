# -*- coding:utf-8 -*-

import numpy as np

from .base_dataloader import DataLoader
from .u import get_q_range_by_mode

from kangqi.util.LogUtil import LogInfo


# Dynamic: one <Q, Schemas> data would be sperated into several small <Q, Schemas> rows
# Therefore we don't need to specify the detail number of pn_size
class QScEvalDynamicDataLoader(DataLoader):

    def __init__(self, dataset, mode,
                 q_max_len, sc_max_len, path_max_len, item_max_len,
                 batch_size, group_per_row=400,
                 dynamic=False, shuffle=False, verbose=0):
        super(QScEvalDynamicDataLoader, self).__init__(batch_size=batch_size,
                                                       mode=mode,
                                                       dynamic=dynamic,
                                                       shuffle=shuffle)
        self.dataset = dataset
        self.verbose = verbose

        self.q_max_len = q_max_len
        self.sc_max_len = sc_max_len
        self.path_max_len = path_max_len
        self.item_max_len = item_max_len

        self.group_per_row = group_per_row
        self.np_data_list = None
        self.q_cands_tup_list = None    # [ (q, [schema]) ]
        # all the candidates of one question could be split into several rows
        # These variables will be updated at renew_data_list() function

    # split the candidate list into several groups (without semantic meaning)
    # make sure the size each group is neither too long or too short
    def candidate_split(self, cand_list):
        if len(cand_list) == 0:
            return []           # zero groups
        cand_groups = []
        while True:
            if len(cand_list) <= 1.5 * self.group_per_row:
                # 1. the maximum length of each row is fixed;
                # 2. if the candidate list is long, the split strategy won't
                #    incidentally produce fragments with few candidates
                cand_groups.append(cand_list)
                break
            cand_groups.append(cand_list[:self.group_per_row])
            cand_list = cand_list[self.group_per_row:]
        return cand_groups

    def renew_data_list(self):
        if self.verbose >= 1:
            LogInfo.begin_track('[QScEvalDynamicDataLoader] prepare data for [%s] ...', self.mode)
        q_idx_list = get_q_range_by_mode(data_name=self.dataset.data_name, mode=self.mode)
        filt_q_idx_list = filter(
            lambda _x: _x in self.dataset.q_cand_dict,
            q_idx_list)        # pick those loaded questions

        if self.verbose >= 1:
            LogInfo.logs('%d questions founded in [%s] part.',
                         len(filt_q_idx_list), self.mode)

        self.q_cands_tup_list = []
        q_input_list = []           # Target: (ds, q_max_len) as int
        q_len_list = []             # Target: (ds, ) as int

        # sc_len_list = []            # Target: [ (, ) ] as int
        # focus_kb_list = []          # Target: [ (pn_size, sc_max_len, ) ] as int
        # focus_item_list = []        # Target: [ (pn_size, sc_max_len, item_max_len) ] as int
        # focus_item_len_list = []    # Target: [ (pn_size, sc_max_len, ) ] as int
        #
        # path_len_list = []          # Target: [ (pn_size, sc_max_len, ) ] as int
        # path_kb_list = []           # Target: [ (pn_size, sc_max_len, path_max_len) ] as int
        # path_item_list = []         # Target: [ (pn_size, sc_max_len, path_max_len, item_max_len) ] as int
        # path_item_len_list = []     # Target: [ (pn_size, sc_max_len, path_max_len) ] as int

        global_sc_info_lists = []
        for _ in range(self.dataset.array_num):
            global_sc_info_lists.append([])
        # global_sc_info_lists = [sc_len_list, focus_kb_list, focus_item_list, focus_item_len_list,
        #                         path_len_list, path_kb_list, path_item_list, path_item_len_list]

        # Note: pn_size is dynamic among different rows

        for pos, q_idx in enumerate(filt_q_idx_list):
            if self.verbose >= 1 and pos % self.proc_ob_num == 0:
                LogInfo.logs('%d / %d prepared.', pos, len(filt_q_idx_list))
            if self.verbose >= 2:
                LogInfo.begin_track('#%d Q-%d: ', pos, q_idx)

            q_words = self.dataset.q_words_dict[q_idx]
            q_len = len(q_words)
            q_words_vec = np.zeros((self.q_max_len,), dtype='int32')
            q_words_vec[:q_len] = q_words

            # cand_list = filter(lambda x: x.is_schema_ok(self.sk_num, self.sk_max_len),
            #                    self.dataset.q_cand_dict[q_idx])
            cand_list = self.dataset.q_cand_dict[q_idx]     # already filtered
            np.random.shuffle(cand_list)    # avoid the largest F

            cand_groups = self.candidate_split(cand_list)
            if self.verbose >= 2:
                LogInfo.logs('Total candidates = %d, split into %d groups.', len(cand_list), len(cand_groups))
                for grp_idx, local_cand_list in enumerate(cand_groups):
                    LogInfo.logs('  Group-%d: size = %d', grp_idx, len(local_cand_list))

            for local_cand_list in cand_groups:
                q_cands_tup = (q_idx, local_cand_list)
                self.q_cands_tup_list.append(q_cands_tup)

                q_input_list.append(q_words_vec)
                q_len_list.append(q_len)

                local_sc_info_lists = []        # saving tensor information of all candidates in this question
                for _ in range(len(global_sc_info_lists)):
                    local_sc_info_lists.append([])

                for cand_idx, schema in enumerate(local_cand_list):     # enumerate each candidate
                    sc_tensor_inputs = self.dataset.get_schema_tensor_inputs(schema)
                    for local_list, sc_tensor in zip(local_sc_info_lists, sc_tensor_inputs):
                        local_list.append(sc_tensor)

                for global_list, local_list in zip(global_sc_info_lists, local_sc_info_lists):
                    global_list.append(np.stack(local_list, axis=0))    # local info becomes a tensor

            if self.verbose >= 2:
                LogInfo.end_track()

        focus_kb_list = global_sc_info_lists[1]
        self.np_data_list = [np.array(q_input_list, dtype='int32'),
                             np.array(q_len_list, dtype='int32')]
        self.np_data_list += global_sc_info_lists
        self.update_statistics()                            # for updating statistics
        self.indices_list.sort(key=lambda x: len(focus_kb_list[x]))
        # sort by each PN_size ASC, which is VERY IMPORTANT!!

        if self.verbose >= 1 and len(self.q_cands_tup_list) > 0:
            group_len_vec = np.array([len(focus_kb_list[idx]) for idx in self.indices_list], dtype='int32')
            LogInfo.logs('Total candidate schemas = %d, group size = %d, avg pn size = %.2f',
                         group_len_vec.sum(), len(self.q_cands_tup_list),
                         1. * group_len_vec.sum() / len(self.q_cands_tup_list))
            assert len(group_len_vec) == len(self.q_cands_tup_list)
            for percentile in range(0, 110, 10):
                LogInfo.logs('Percentile-%d: PN_size = %.2f',
                             percentile, np.percentile(group_len_vec, percentile))
        if self.verbose >= 1:
            LogInfo.end_track()

    def get_next_batch(self):
        if self.cur_batch_idx in self.history_batch_dict:
            local_data_list, local_indices = self.history_batch_dict[self.cur_batch_idx]
        else:
            st_idx = self.cur_batch_idx * self.batch_size
            ed_idx = st_idx + self.batch_size
            local_indices = self.indices_list[st_idx:ed_idx]
            local_size = len(local_indices)

            # [q_input, q_len_input,
            #  sc_len_list, focus_kb_list, focus_item_list, focus_item_len_list,
            #  path_len_list, path_kb_list, path_item_list, path_item_len_list
            # ] = self.np_data_list

            q_input = self.np_data_list[0]
            q_len_input = self.np_data_list[1]
            eval_sc_info_lists = self.np_data_list[2:]
            focus_kb_list = eval_sc_info_lists[1]

            local_len_vec = np.array([len(focus_kb_list[local_idx]) for local_idx in local_indices], dtype='int32')
            local_pn_size = np.max(local_len_vec)                   # pn size of the current batch
            # LogInfo.logs('cur_batch_idx = %d, local_pn_size = %d', self.cur_batch_idx, local_pn_size)

            local_q_input = q_input[local_indices]
            local_q_len_input = q_len_input[local_indices]

            local_sc_len_input = np.zeros(
                (local_size, local_pn_size), dtype='int32')
            local_focus_kb_input = np.zeros(
                (local_size, local_pn_size, self.sc_max_len), dtype='int32')
            local_focus_item_input = np.zeros(
                (local_size, local_pn_size, self.sc_max_len, self.item_max_len), dtype='int32')
            local_focus_item_len_list = np.zeros(
                (local_size, local_pn_size, self.sc_max_len), dtype='int32')

            local_path_len_input = np.zeros(
                (local_size, local_pn_size, self.sc_max_len), dtype='int32')
            local_path_kb_input = np.zeros(
                (local_size, local_pn_size, self.sc_max_len, self.path_max_len), dtype='int32')
            local_path_item_input = np.zeros(
                (local_size, local_pn_size, self.sc_max_len, self.path_max_len, self.item_max_len), dtype='int32')
            local_path_item_len_list = np.zeros(
                (local_size, local_pn_size, self.sc_max_len, self.path_max_len), dtype='int32')
            local_sc_info_input_list = [local_sc_len_input, local_focus_kb_input,
                                        local_focus_item_input, local_focus_item_len_list,
                                        local_path_len_input, local_path_kb_input,
                                        local_path_item_input, local_path_item_len_list]

            for pos, local_idx in enumerate(local_indices):         # dynamic build numpy array for the current batch
                cands = len(focus_kb_list[local_idx])
                for local_input, eval_list in zip(local_sc_info_input_list, eval_sc_info_lists):
                    local_input[pos, :cands] = eval_list[local_idx]

            local_data_list = [local_q_input,
                               local_q_len_input] + local_sc_info_input_list
            # for local_data in local_data_list:
            #     LogInfo.logs('%s', local_data.shape)
            self.history_batch_dict[self.cur_batch_idx] = (local_data_list, local_indices)

        self.cur_batch_idx = (self.cur_batch_idx + 1) % self.n_batch
        return local_data_list, local_indices
