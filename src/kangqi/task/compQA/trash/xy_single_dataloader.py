# -*- coding:utf-8 -*-

import numpy as np

from .base_dataloader import DataLoader
from .u import weighted_sampling
from .u_xianyang import get_webq_range_by_mode

from kangqi.util.LogUtil import LogInfo


# SingleDataLoader: each <q, sc> is a single data, used in log-reg model
class QScSingleDataLoader(DataLoader):

    def __init__(self, dataset, mode, webq_list,
                 q_max_len, sk_num, sk_max_len, pn_size,
                 n_wd_emb, n_kb_emb, batch_size,
                 neg_rate=3, high_rate=2, max_pos_size=5,
                 dynamic=True, shuffle=True, verbose=0):
        super(QScSingleDataLoader, self).__init__(batch_size=batch_size,
                                                  mode=mode,
                                                  dynamic=dynamic,
                                                  shuffle=shuffle)
        self.dataset = dataset
        self.webq_list = webq_list
        self.verbose = verbose
        self.neg_rate = neg_rate        # number of negative data
        self.high_rate = high_rate      # number of negative data with F1 > 0
        self.max_pos_size = max_pos_size        # maximum number of positive data for each question

        self.q_max_len = q_max_len
        self.sk_num = sk_num
        self.sk_max_len = sk_max_len
        self.pn_size = pn_size
        self.n_wd_emb = n_wd_emb
        self.n_kb_emb = n_kb_emb

        self.np_data_list = None
        # These variables will be updated at renew_data_list() function

    def renew_data_list(self):
        if self.verbose >= 1:
            LogInfo.begin_track('[QScSingleDataLoader] prepare data for [%s] ...', self.mode)
        q_idx_list = get_webq_range_by_mode(self.mode)
        filt_q_idx_list = filter(
            lambda _x: _x in self.dataset.q_cand_dict,
            q_idx_list)  # pick those loaded questions

        q_input_list = []           # (data_size, q_max_len) as int
        # q_emb_list = []             # (data_size, q_max_len, n_wd_emb)
        q_len_list = []             # (data_size, ) as int
        optm_focus_emb_list = []    # (data_size, sk_num, n_kb_emb)
        optm_path_emb_list = []     # (data_size, sk_num, sk_max_len, n_kb_emb)
        optm_path_len_list = []     # (data_size, sk_num) as int
        optm_label_list = []        # (data_size, )

        for scan_idx, q_idx in enumerate(filt_q_idx_list):
            if self.verbose >= 1 and scan_idx % self.proc_ob_num == 0:
                LogInfo.logs('%d / %d prepared.', scan_idx, len(filt_q_idx_list))
            if self.verbose >= 2:
                LogInfo.begin_track('Q-%d [%s]: ', q_idx, self.webq_list[q_idx].encode('utf-8'))
            q_words = self.dataset.q_words_dict[q_idx]
            q_len = len(q_words)
            word_idx_vec = np.zeros((self.q_max_len,), dtype='int32')
            word_idx_vec[:q_len] = q_words
            # emb_matrix = self.dataset.q_emb_dict[q_idx]
            # q_len = emb_matrix.shape[0]
            # fit_emb_matrix = np.zeros((self.q_max_len, self.n_wd_emb), dtype='float32')
            # fit_emb_matrix[:q_len, :] = emb_matrix      # fit the matrix size

            cand_list = filter(lambda x: x.is_schema_ok(self.sk_num, self.sk_max_len),
                               self.dataset.q_cand_dict[q_idx])
            pos_cand_list, neg_cand_list = self.pos_neg_split(cand_list=cand_list)
            pick_pos_cand_list = self.pick_pos_data(pos_cand_list=pos_cand_list)
            pick_neg_cand_list = self.pick_neg_data(neg_cand_list=neg_cand_list,
                                                    positive_size=len(pick_pos_cand_list))
            if self.verbose >= 2:
                for pos_sc in pick_pos_cand_list:
                    LogInfo.logs('[POS] F1=%.6f, sc=[%s]', pos_sc.f1, pos_sc.path_list_str)
                for neg_sc in pick_neg_cand_list:
                    LogInfo.logs('[NEG] F1=%.6f, sc=[%s]', neg_sc.f1, neg_sc.path_list_str)

            use_cand_list = pick_pos_cand_list + pick_neg_cand_list
            use_size = len(use_cand_list)
            optm_label_list += (
                [1.] * len(pick_pos_cand_list) +
                [0.] * len(pick_neg_cand_list))
            q_input_list += [word_idx_vec] * use_size
            # q_emb_list += [fit_emb_matrix] * use_size
            q_len_list += [q_len] * use_size
            for schema in use_cand_list:
                focus_emb_input, path_emb_input, path_len_input = \
                    schema.build_embedding(kb_emb_util=self.dataset.kb_emb_util,
                                           sk_num=self.sk_num,
                                           sk_max_len=self.sk_max_len)
                optm_focus_emb_list.append(focus_emb_input)
                optm_path_emb_list.append(path_emb_input)
                optm_path_len_list.append(path_len_input)
            if self.verbose >= 2:
                LogInfo.end_track()

        self.np_data_list = [
            np.array(q_input_list, dtype='int32'),              # (data_size, q_max_len) as int
            # np.array(q_emb_list, dtype='float32'),            # (data_size, q_max_len, n_wd_emb)
            np.array(q_len_list, dtype='int32'),                # (data_size, ) as int
            np.array(optm_focus_emb_list, dtype='float32'),     # (data_size, sk_num, n_kb_emb)
            np.array(optm_path_emb_list, dtype='float32'),      # (data_size, sk_num, sk_max_len, n_kb_emb)
            np.array(optm_path_len_list, dtype='int32'),        # (data_size, sk_num) as int
            np.array(optm_label_list, dtype='float32')          # (data_size, )
        ]
        self.update_statistics()  # for updating statistics

        if self.verbose >= 1:
            np_name_list = [
                'q_emb_input', 'q_len_input',
                'optm_focus_emb_input', 'optm_path_emb_input', 'optm_path_len_input',
                'optm_label_list'
            ]
            for name, np_data in zip(np_name_list, self.np_data_list):
                LogInfo.logs('%s: %s', name, np_data.shape)
            LogInfo.end_track()

    @staticmethod
    def pos_neg_split(cand_list):
        pos_cand_list = filter(lambda sc: sc.f1 >= 0.5, cand_list)
        neg_cand_list = filter(lambda sc: sc.f1 < 0.5, cand_list)
        if len(pos_cand_list) == 0 and neg_cand_list[0].f1 > 0.:
            # shift some negative data into positive side
            move_pos = 0
            top_f1 = neg_cand_list[0].f1
            while True:
                if move_pos < len(neg_cand_list) and neg_cand_list[move_pos].f1 == top_f1:
                    move_pos += 1
                else:
                    break
            pos_cand_list = neg_cand_list[:move_pos]
            neg_cand_list = neg_cand_list[move_pos:]
        return pos_cand_list, neg_cand_list

    # Let's sample by F1 score
    def pick_pos_data(self, pos_cand_list):
        if len(pos_cand_list) <= self.max_pos_size:
            return pos_cand_list
        weight_list = [sc.f1 for sc in pos_cand_list]
        return weighted_sampling(item_list=pos_cand_list,
                                 weight_list=weight_list,
                                 budget=self.max_pos_size)

    def pick_neg_data(self, neg_cand_list, positive_size):
        positive_size = max(positive_size, 1)       # although no positive data, we shall pick some negatives
        neg_budget = positive_size * self.neg_rate
        high_neg_budget = positive_size * self.high_rate

        pick_neg_list = []      # the returned negative data

        high_neg_list = filter(lambda sc: sc.f1 > 0., neg_cand_list)
        low_neg_list = filter(lambda sc: sc.f1 == 0., neg_cand_list)

        np.random.shuffle(high_neg_list)
        high_neg_budget = min(high_neg_budget, len(high_neg_list))  # the real budget for high_neg
        pick_neg_list += high_neg_list[:high_neg_budget]

        remain_budget = neg_budget - high_neg_budget
        pick_neg_list += low_neg_list[:remain_budget]

        return pick_neg_list
