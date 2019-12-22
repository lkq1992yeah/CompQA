# -*- coding:utf-8 -*-

import numpy as np

from .base_dataloader import DataLoader
from .u_xianyang import get_webq_range_by_mode

from kangqi.util.LogUtil import LogInfo


# Load XY's train/xxx-yyy/zzz format file
# Note: currently we use a simple representation: regard a schema as a list of paths
class QScEvalDataLoader(DataLoader):

    def __init__(self, dataset, mode,
                 q_max_len, sk_num, sk_max_len,
                 pn_size, batch_size,
                 dynamic=False, shuffle=False, verbose=0):
        super(QScEvalDataLoader, self).__init__(batch_size=batch_size,
                                                mode=mode,
                                                dynamic=dynamic,
                                                shuffle=shuffle)
        self.dataset = dataset
        self.verbose = verbose

        self.q_max_len = q_max_len
        self.sk_num = sk_num
        self.sk_max_len = sk_max_len
        self.pn_size = pn_size

        self.q_cands_tup_list = None
        self.np_data_list = None
        # These variables will be updated at renew_data_list() function

    def renew_data_list(self):
        if self.verbose >= 1:
            LogInfo.begin_track('[QScEvalDataLoader] prepare data for [%s] ...', self.mode)
        q_idx_list = get_webq_range_by_mode(self.mode)
        filt_q_idx_list = filter(
            lambda _x: _x in self.dataset.q_cand_dict,
            q_idx_list)        # pick those loaded questions

        data_size = len(filt_q_idx_list)
        if self.verbose >= 1:
            LogInfo.logs('%d questions founded in [%s] part.', data_size, self.mode)

        # q_emb_input = np.zeros((data_size, self.q_max_len, self.n_wd_emb), dtype='float32')
        q_input = np.zeros((data_size, self.q_max_len), dtype='int32')
        q_len_input = np.zeros((data_size,), dtype='int32')
        eval_focus_input = np.zeros((data_size, self.pn_size, self.sk_num), dtype='int32')
        eval_path_input = np.zeros((data_size, self.pn_size, self.sk_num, self.sk_max_len), dtype='int32')
        eval_path_len_input = np.zeros((data_size, self.pn_size, self.sk_num), dtype='int32')
        eval_mask_input = np.zeros((data_size, self.pn_size), dtype='float32')
        eval_f1_input = np.zeros((data_size, self.pn_size), dtype='float32')

        self.q_cands_tup_list = []
        for pos, q_idx in enumerate(filt_q_idx_list):
            if self.verbose >= 1 and pos % self.proc_ob_num == 0:
                LogInfo.logs('%d / %d prepared.', pos, len(filt_q_idx_list))
            q_words = self.dataset.q_words_dict[q_idx]
            q_len = len(q_words)
            q_input[pos, :q_len] = q_words
            q_len_input[pos] = q_len

            cand_list = filter(lambda x: x.is_schema_ok(self.sk_num, self.sk_max_len),
                               self.dataset.q_cand_dict[q_idx])
            # LogInfo.logs('available candidate size = %d', len(cand_list))
            if len(cand_list) > self.pn_size:
                np.random.shuffle(cand_list)
                cand_list = cand_list[:self.pn_size]
            # truncate operation: first random shuffle, then picking
            # TODO: or maybe we could first keep skeletons, and then care about schemas
            # Now we've selected candidates that we use in the data_loader

            self.q_cands_tup_list.append((q_idx, cand_list))
            for cand_idx, schema in enumerate(cand_list):                   # enumerate each candidate
                eval_mask_input[pos, cand_idx] = 1
                eval_f1_input[pos, cand_idx] = schema.f1
                focus_input, path_input, path_len_input = \
                    schema.build_embedding_index(sk_num=self.sk_num,
                                                 sk_max_len=self.sk_max_len)
                eval_focus_input[pos, cand_idx] = focus_input
                eval_path_input[pos, cand_idx] = path_input
                eval_path_len_input[pos, cand_idx] = path_len_input

        self.np_data_list = [q_input,                   # (data_size, q_max_len) as int
                             # q_emb_input,               # (data_size, q_max_len, n_wd_emb)
                             q_len_input,               # (data_size, ) as int
                             eval_focus_input,          # (data_size, pn_size, sk_num) as int
                             eval_path_input,           # (data_size, pn_szie, sk_num, sk_max_len) as int
                             eval_path_len_input,       # (data_size, pn_size, sk_num) as int
                             eval_mask_input,           # (data_size, pn_size)
                             eval_f1_input]             # (data_size, pn_size)
        self.update_statistics()                            # for updating statistics

        if self.verbose >= 1:
            np_name_list = [
                'q_input', 'q_len_input', 'eval_focus_emb_input',
                'eval_path_emb_input', 'eval_path_len_input',
                'eval_mask_input', 'eval_f1_input'
            ]
            for name, np_data in zip(np_name_list, self.np_data_list):
                LogInfo.logs('%s: %s', name, np_data.shape)
            LogInfo.end_track()
