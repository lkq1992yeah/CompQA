"""
Author: Kangqi Luo
Goal: The dataloader is used in CompQ experiments (ignoring entity linking, but just relation detection)
      Maintain all the <q, sc+, sc-> training data
** Important: since different schemas have different focus entities,
              we prepare q+ and q- for the different pos/neg schema in each data point.
              So the actual data should be <q+, sc+, q- sc->
"""

import numpy as np

from ..base_dataloader import DataLoader
from ..u import get_q_range_by_mode

from kangqi.util.LogUtil import LogInfo


class CompqPairDataLoader(DataLoader):

    def __init__(self, dataset, mode,
                 batch_size, proc_ob_num,
                 f1_delta=0.75, neg_per_pos=100,
                 poor_f1_threshold=0.1,
                 poor_contribution=20,
                 poor_max_sample=100,
                 dynamic=True, shuffle=True, verbose=0):
        super(CompqPairDataLoader, self).__init__(batch_size=batch_size,
                                                  mode=mode,
                                                  proc_ob_num=proc_ob_num,
                                                  dynamic=dynamic,
                                                  shuffle=shuffle)
        self.dataset = dataset
        self.verbose = verbose

        self.f1_delta = f1_delta
        self.neg_per_pos = neg_per_pos
        # used in the simple generating strategy
        # deprecated on 180104

        self.poor_f1_threshold = poor_f1_threshold
        self.poor_contribution = poor_contribution
        self.poor_max_sample = poor_max_sample
        LogInfo.logs('poor_f1_threshold = %.3f', self.poor_f1_threshold)
        LogInfo.logs('poor_contribution = %d', self.poor_contribution)
        LogInfo.logs('poor_max_sample = %d', self.poor_max_sample)
        # used in partial_order generating strategy
        # put into use on 180106

    def renew_data_list(self):
        """
        Target: construct the np_data_list, storing all <q+, sc+, q-, sc-> pairs
        np_data_list contains:
        1. pos_qwords           (data_size, q_max_len)
        2. pos_qwords_len       (data_size, )
        3. pos_sc_len           (data_size, )
        4. pos_preds            (data_size, path_max_len)
        5. pos_preds_len        (data_size, )
        6. pos_pwords           (data_size, pword_max_len)
        7. pos_pwords_len       (data_size, )
        8. efeats_input         (data_size, e_max_size, e_feat_len)
        9. etypes_input         (data_size, e_max_size)
        10. emask_input         (data_size, e_max_size)
        11. extra_input         (data_size, extra_len)
        12-22. neg_xxxxxx
        """
        if self.verbose >= 1:
            LogInfo.begin_track('[CompqPairDataLoader] prepare data for [%s] ...', self.mode)
        q_idx_list = get_q_range_by_mode(data_name=self.dataset.data_name,
                                         mode=self.mode)
        filt_q_idx_list = filter(lambda q: q in self.dataset.q_cand_dict, q_idx_list)
        # Got all the related questions

        pos_emb_pools = []
        neg_emb_pools = []
        # [ qwords, qwords_len, sc_len, preds, preds_len, pwords, pwords_len ]
        data_weights = []

        for _ in range(self.dataset.array_num):     # "qwords", "qwords_len" are contained in the dataset
            pos_emb_pools.append([])
            neg_emb_pools.append([])

        tup_idx = 0
        for scan_idx, q_idx in enumerate(filt_q_idx_list):
            if self.verbose >= 1 and scan_idx % self.proc_ob_num == 0:
                LogInfo.logs('%d / %d prepared.', scan_idx, len(filt_q_idx_list))
            cand_list = self.dataset.q_cand_dict[q_idx]
            pos_neg_tup_list = self.generate_pairs__partial_order(cand_list)
            # now store the positive / negative input into the corresponding position in the np_data_list
            for pos_sc, neg_sc, weight in pos_neg_tup_list:
                tup_idx += 1
                LogInfo.logs('Data-%d: q = %d, pos_rm_F1 = %.6f (row %d), neg_rm_F1 = %.6f (row %d), weight = %.6f',
                             tup_idx, q_idx, pos_sc.f1, pos_sc.ori_idx, neg_sc.f1, neg_sc.ori_idx, weight)
                data_weights.append(weight)
                for sc, emb_pools in [(pos_sc, pos_emb_pools), (neg_sc, neg_emb_pools)]:
                    sc_tensor_inputs = self.dataset.get_schema_tensor_inputs(sc)
                    for local_list, sc_tensor in zip(emb_pools, sc_tensor_inputs):
                        local_list.append(sc_tensor)
                    # now the detail input of the schema is copied into the memory of the dataloader

        # Finally: merge inputs together, and produce the final np_data_list
        self.np_data_list = []
        for emb_pools in (pos_emb_pools, neg_emb_pools):
            for target_list in emb_pools:
                self.np_data_list.append(np.array(target_list, dtype='int32'))
        self.np_data_list.append(np.array(data_weights, dtype='float32'))
        self.update_statistics()

        if self.verbose >= 1:
            LogInfo.logs('%d <q, sc+, sc-> data collected.', self.np_data_list[0].shape[0])
            LogInfo.end_track()

    def generate_pairs__simple(self, cand_list):
        """
        Given the candidate list, generate positive and negative pairs.
        The positive schema could be any schemas with F1 > 0.
        Then for each schema, pick "neg_per_pos" schemas whose F1 is far from positive schema
        ** Deprecated on 180104: this generating strategy is a bit stupid.
        It cannot effectively penalize those schemas with non-zero but poor F1
        """
        pos_neg_tup_list = []       # [(sc+, sc-, weight)]
        for pos_sc in cand_list:
            pos_f1 = pos_sc.f1
            if pos_f1 < 1e-6:        # F1 = 0.0, won't be a positive
                continue
            neg_f1_ths = self.f1_delta * pos_f1
            neg_pool = filter(lambda sc: sc.f1 < neg_f1_ths, cand_list)
            np.random.shuffle(neg_pool)
            neg_pick = neg_pool[: self.neg_per_pos]     # random pick a limited number of schemas
            for neg_sc in neg_pick:
                pos_neg_tup_list.append((pos_sc, neg_sc, 1.))       # each pair weights equally
        return pos_neg_tup_list

    def generate_pairs__partial_order(self, cand_list):
        """
        Given the candidate list, generate positive and negative pairs.
        All good schemas (F1 >= poor_f1_threshold) are kept.
        Randomly sample up to "poor_sample_size" schemas from poor candidates.
        Then construct all <sc+, sc-> pairs from the picked sets.
        """
        pos_neg_tup_list = []       # [(sc+, sc-, weight)]
        good_cands = filter(lambda sc: sc.f1 >= self.poor_f1_threshold, cand_list)
        poor_cands = filter(lambda sc: sc.f1 < self.poor_f1_threshold, cand_list)

        np.random.shuffle(poor_cands)
        # TODO: Sample poor schemas in a smarter way (based on current score)
        poor_dilute_rate = 1.
        pick_poor_cands = poor_cands[: self.poor_max_sample]
        ppc_len = len(pick_poor_cands)
        if ppc_len > 0:
            poor_dilute_rate = 1. * min(ppc_len, self.poor_contribution) / ppc_len
        # dilute the contribution of each poor schema

        pick_cands = good_cands + pick_poor_cands
        for sc1 in pick_cands:
            for sc2 in pick_cands:
                if sc1.f1 < self.poor_f1_threshold:
                    continue
                if sc1.f1 > sc2.f1:
                    weight = 1. if sc2.f1 >= self.poor_f1_threshold else poor_dilute_rate
                    # tune down the contribution when a poor schema acts as a negative schema.
                    pos_neg_tup_list.append((sc1, sc2, weight))
        return pos_neg_tup_list
