"""
Author: Kangqi Luo
Date: 180118
Goal: Generate Structural Data
"""

import numpy as np
from .dataset_schema_reader import schema_classification

from kangqi.util.discretizer import Discretizer
from kangqi.util.LogUtil import LogInfo


ans_size_disc = Discretizer([2, 3, 5, 10, 50])      # 5+1
# ans < 2
# 2 <= ans < 3
# 3 <= ans < 5
# 5 <= ans < 10
# 10 <= ans < 50
# ans >= 50


def build_structural_data(all_cands_tup_list):
    """
    Given all the candidate schemas, build structural-based data
    """
    cand_size = len(all_cands_tup_list)
    LogInfo.begin_track('Build Structural Data for %d candidates:', cand_size)

    data_list = []      # store all extra list of candidates
    for data_idx, q_idx, sc in all_cands_tup_list:
        if data_idx % 50000 == 0:
            LogInfo.logs('%d / %d schemas scanned.', data_idx, cand_size)
        local_list = []

        ans_size = sc.ans_size
        disc_ans_size_list = ans_size_disc.convert(score=ans_size).tolist()
        local_list += disc_ans_size_list

        sc_class = schema_classification(sc=sc)
        disc_sc_class_list = [0] * 4
        disc_sc_class_list[sc_class] = 1
        local_list += disc_sc_class_list

        ce_at_ans = ce_at_med = 0
        """
        Currently, just focus on whether a constraint entity is linked to med / ans.
        Avoid bringing rare features.
        """
        for raw_path in sc.raw_paths:
            path_cate, focus, pred_seq = raw_path
            if path_cate in ('Entity', 'Time'):
                constr_len = len(pred_seq)
                if constr_len == 1:     # predicate length = 1, must be adding at answer
                    ce_at_ans = 1
                else:                   # adding at mediator
                    ce_at_med = 1

        local_list.append(ce_at_ans)
        local_list.append(ce_at_med)

        data_list.append(local_list)

    extra_arr = np.asarray(data_list, dtype='int32')        # (data_size, extra_feat_len)
    extra_len = extra_arr.shape[1]
    LogInfo.logs('Adding tensor with shape %s ... ', extra_arr.shape)
    LogInfo.end_track()

    np_data = [extra_arr]
    return extra_len, np_data
