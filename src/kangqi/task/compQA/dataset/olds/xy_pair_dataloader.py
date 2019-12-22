# -*- coding:utf-8 -*-

import math
import numpy as np

from .base_dataloader import DataLoader
from .u import weighted_sampling, get_q_range_by_mode

from kangqi.util.LogUtil import LogInfo


# PairDataLoader: each (<q, sc+, sc->) is a single data, used in hinge model
class QScPairDataLoader(DataLoader):

    def __init__(self, dataset, mode,
                 q_max_len, sc_max_len, path_max_len, item_max_len,
                 batch_size, sampling_config,
                 dynamic=True, shuffle=True, verbose=0):
        super(QScPairDataLoader, self).__init__(batch_size=batch_size,
                                                mode=mode,
                                                dynamic=dynamic,
                                                shuffle=shuffle)
        self.dataset = dataset
        self.verbose = verbose
        self.sampling_config = sampling_config

        sample_func_name = self.sampling_config['name']
        assert sample_func_name in ['generate_pairs_by_gold_f1', 'generate_pairs_by_runtime_score']
        LogInfo.logs('Negative sampling function: %s', sample_func_name)
        self.neg_sample_func = getattr(self, sample_func_name)
        del self.sampling_config['name']

        self.q_max_len = q_max_len
        self.sc_max_len = sc_max_len
        self.path_max_len = path_max_len
        self.item_max_len = item_max_len

        self.np_data_list = None
        # These variables will be updated at renew_data_list() function

    def renew_data_list(self):
        if self.verbose >= 1:
            LogInfo.begin_track('[QScPairDataLoader] prepare data for [%s] ...', self.mode)
        q_idx_list = get_q_range_by_mode(data_name=self.dataset.data_name, mode=self.mode)
        filt_q_idx_list = filter(
            lambda _x: _x in self.dataset.q_cand_dict,
            q_idx_list)  # pick those loaded questions

        q_input_list = []               # (data_size, q_max_len) as int
        q_len_list = []                 # (data_size, ) as int

        pos_emb_pools = []
        neg_emb_pools = []
        # (sc_len, focus_kb, focus_item, focus_item_len, path_len, path_kb, path_item, path_item_len)
        for _ in range(self.dataset.array_num):
            pos_emb_pools.append([])
            neg_emb_pools.append([])

        for scan_idx, q_idx in enumerate(filt_q_idx_list):
            if self.verbose >= 1 and scan_idx % self.proc_ob_num == 0:
                LogInfo.logs('%d / %d prepared.', scan_idx, len(filt_q_idx_list))
            if self.verbose >= 2:
                LogInfo.begin_track('Q-%d [%s]: ', q_idx,
                                    self.dataset.webq_list[q_idx].encode('utf-8'))
            q_words = self.dataset.q_words_dict[q_idx]
            q_len = len(q_words)
            word_idx_vec = np.zeros((self.q_max_len,), dtype='int32')
            word_idx_vec[:q_len] = q_words

            # cand_list = filter(lambda x: x.is_schema_ok(self.sk_num, self.sk_max_len),
            #                    self.dataset.q_cand_dict[q_idx])
            cand_list = self.dataset.q_cand_dict[q_idx]
            self.sampling_config['cand_list'] = cand_list
            pair_data_list = self.neg_sample_func(**self.sampling_config)
            pair_size = len(pair_data_list)
            q_input_list += [word_idx_vec] * pair_size
            q_len_list += [q_len] * pair_size

            if self.verbose >= 2:
                for lf_cand, rt_cand in pair_data_list:
                    LogInfo.logs('[LF] F1=%.6f, sc=[%s]', lf_cand.f1, lf_cand.disp())
                    LogInfo.logs('[RT] F1=%.6f, sc=[%s]', rt_cand.f1, rt_cand.disp())
                    LogInfo.logs('========')

            # store embedding into pos & neg embedding list, separately
            lf_cand_list = [tup[0] for tup in pair_data_list]
            rt_cand_list = [tup[1] for tup in pair_data_list]
            for cand_list, emb_pools in [(lf_cand_list, pos_emb_pools), (rt_cand_list, neg_emb_pools)]:
                for schema in cand_list:
                    sc_tensor_inputs = self.dataset.get_schema_tensor_inputs(schema)
                    for local_list, sc_tensor in zip(emb_pools, sc_tensor_inputs):
                        local_list.append(sc_tensor)
            if self.verbose >= 2:
                LogInfo.end_track()

        self.np_data_list = [
            np.array(q_input_list, dtype='int32'),                  # (data_size, q_max_len) as int
            np.array(q_len_list, dtype='int32'),                    # (data_size, ) as int
        ]
        for emb_pools in (pos_emb_pools, neg_emb_pools):
            for target_list in emb_pools:
                self.np_data_list.append(np.array(target_list, dtype='int32'))
        self.update_statistics()  # for updating statistics

        if self.verbose >= 1:
            # np_name_list = [
            #     'q_input', 'q_len_input',
            #     'optm_pos_focus_input', 'optm_pos_path_input', 'optm_pos_path_len_input',
            #     'optm_neg_focus_input', 'optm_neg_path_input', 'optm_neg_path_len_input'
            # ]
            # for name, np_data in zip(np_name_list, self.np_data_list):
            #     LogInfo.logs('%s: %s', name, np_data.shape)
            LogInfo.logs('%d paired data collected.', self.np_data_list[0].shape[0])
            # for np_data in self.np_data_list:
            #     LogInfo.logs('%s', np_data.shape)
            LogInfo.end_track()

    @staticmethod
    def generate_pairs_by_gold_f1(cand_list, pos_size=5, neg_per_pos_size=10,
                                  raise_factor=0.75, delta=0.8, zero_shrink=0.01):
        """
        # raise_factor: (0, 1)
        # Why need a raise factor: raise the weight a little bit for those schemas with too small F1
        # Referred from negative sampling strategy in w2v, this may increase the diversity of the picked schemas
        # delta: (0, 1], which indicates the F1 gap between sc+ and sc-
        # zero_shrink: "shrink" the data size of schemas with F1=0, make sure that the majority of sc- has some F1
        """
        pos_cand_list = filter(lambda sc: sc.f1 > 0., cand_list)     # This is all the dataset with F1 > 0
        zero_cand_list = filter(lambda sc: sc.f1 == 0., cand_list)
        pair_data_list = []

        # Step 1: Pick sc+ from those schemas having F1 > 0
        pos_weight_list = map(lambda sc: math.pow(sc.f1, raise_factor), pos_cand_list)
        pick_pos_cand_list = weighted_sampling(item_list=pos_cand_list,
                                               weight_list=pos_weight_list,
                                               budget=pos_size)
        lf_cand_freq_dict = {}          # just count the frequency of each distinct sc+
        for lf_cand in pick_pos_cand_list:
            lf_cand_freq_dict.setdefault(lf_cand, 0)
            lf_cand_freq_dict[lf_cand] += 1

        # Step 2: Pick sc-, part of which comes from pos_cand_list, and the other part comes from zero_cand_list
        for lf_cand, freq in lf_cand_freq_dict.items():
            neg_budget = neg_per_pos_size * freq        # duplicate counts
            small_cand_list = filter(lambda sc: sc.f1 < lf_cand.f1 * delta,
                                     pos_cand_list)
            small_weight_list = map(lambda sc: math.pow(sc.f1, raise_factor), small_cand_list)
            # small_cand_list: the list of schemas with a positive F1,
            # but much smaller (controlled by delta) then the current sc+.

            if len(small_cand_list) == 0 and len(zero_cand_list) == 0:
                continue            # there's no negative data at all!!

            small_sample_ratio = 1. * len(small_cand_list) / (len(small_cand_list) + zero_shrink * len(zero_cand_list))
            small_neg_budget = int(neg_budget * small_sample_ratio)
            pick_small_cand_list = weighted_sampling(item_list=small_cand_list,
                                                     weight_list=small_weight_list,
                                                     budget=small_neg_budget)
            zero_neg_budget = neg_budget - small_neg_budget
            np.random.shuffle(zero_cand_list)
            pick_zero_cand_list = zero_cand_list[:zero_neg_budget]      # just random shuffle and pick

            rt_cand_list = pick_small_cand_list + pick_zero_cand_list
            pair_data_list += [(lf_cand, rt_cand) for rt_cand in rt_cand_list]

        return pair_data_list

    @staticmethod
    def generate_pairs_by_runtime_score(cand_list, pos_size=5, neg_per_pos_size=10,
                                        raise_factor=0.75, cool_down=1.,
                                        delta=0.8, zero_shrink=1.):
        """
        For generating pairs by score
        :param cand_list: candidate schema list
        :param pos_size: slot for positives
        :param neg_per_pos_size: negative slots for each positive
        :param raise_factor: raising the prob. of selecting low-f1 positive item
        :param cool_down: cool down factor for score-based negative sampling
        :param delta: determine the minimum distance between pos and neg.
        :param zero_shrink: shrink F1=0. data points
        :return: <POS, NEG> pairs.
        """
        pos_cand_list = filter(lambda sc: sc.f1 > 0., cand_list)  # This is all the dataset with F1 > 0
        zero_cand_list = filter(lambda sc: sc.f1 == 0., cand_list)
        pair_data_list = []

        # Step 1: Pick sc+ from those schemas having F1 > 0
        # the same as gold-f1-based method
        pos_weight_list = map(lambda sc: math.pow(sc.f1, raise_factor), pos_cand_list)
        pick_pos_cand_list = weighted_sampling(item_list=pos_cand_list,
                                               weight_list=pos_weight_list,
                                               budget=pos_size)
        lf_cand_freq_dict = {}  # just count the frequency of each distinct sc+
        for lf_cand in pick_pos_cand_list:
            lf_cand_freq_dict.setdefault(lf_cand, 0)
            lf_cand_freq_dict[lf_cand] += 1

        # Step 2: Pick sc-, part of which comes from pos_cand_list, and the other part comes from zero_cand_list
        for lf_cand, freq in lf_cand_freq_dict.items():
            neg_budget = neg_per_pos_size * freq  # duplicate counts
            small_cand_list = filter(lambda sc: sc.f1 < lf_cand.f1 * delta,
                                     pos_cand_list)
            np.random.shuffle(zero_cand_list)
            zero_shrink_size = int(len(zero_cand_list) * zero_shrink)
            neg_pool = small_cand_list + zero_cand_list[:zero_shrink_size]
            # Now we've filtered some candidates from zero_list, next sample by score.

            if len(neg_pool) == 0:
                continue  # there's no negative data at all!!
            tmp_score_list = []
            for cand in neg_pool:
                if cand.run_info is not None and 'score' in cand.run_info:
                    tmp_score_list.append(cand.run_info['score'] * cool_down)
                    # the score is calculated by the last iteration
                else:
                    tmp_score_list.append(0.)    # must be the first time
            max_score = np.max(tmp_score_list)
            neg_weight_list = np.exp(np.array(tmp_score_list) - max_score)
            # un-normalized exponential value

            rt_cand_list = weighted_sampling(item_list=neg_pool,
                                             weight_list=neg_weight_list,
                                             budget=neg_budget)
            pair_data_list += [(lf_cand, rt_cand) for rt_cand in rt_cand_list]

        return pair_data_list
