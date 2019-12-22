"""
Author: Kangqi Luo
Date: 180118
Goal: Generate Entity Linking based data
"""

import math
import numpy as np

from ..kq_schema import CompqSchema
from ...util.fb_helper import get_domain, get_range

from kangqi.util.LogUtil import LogInfo


def build_entity_linking_data(all_cands_tup_list, e_max_size, mid_dict):
    """
    Goal: given all the candidate schemas, build entity-linking-related np data
    """
    cand_size = len(all_cands_tup_list)
    LogInfo.begin_track('Build Entity Linking Data for %d candidates:', cand_size)

    """ Step 1: Allocate Memory of Entity Linking """

    e_feat_len = 0
    for data_idx, q_idx, sc in all_cands_tup_list:
        if e_feat_len != 0:
            break
        for raw_path in sc.raw_paths:
            path_cate, focus, _ = raw_path
            if path_cate not in ('Main', 'Entity'):
                continue
            e_feat_len = len(focus.link_feat)
            break
    np_data = [
        np.zeros(shape=(cand_size, e_max_size, e_feat_len), dtype='float32'),    # efeats_input
        np.zeros(shape=(cand_size, e_max_size), dtype='int32'),                  # etypes_input
        np.zeros(shape=(cand_size, e_max_size), dtype='float32')                 # emask_input
    ]
    for item in np_data:
        LogInfo.logs('Adding tensor with shape %s ... ', item.shape)
    [efeats, etypes, emask] = np_data

    """ Step 2: Scan & Fill Data """

    for data_idx, q_idx, sc in all_cands_tup_list:
        if data_idx % 50000 == 0:
            LogInfo.logs('%d / %d schemas scanned.', data_idx, cand_size)
        assert isinstance(sc, CompqSchema)  # not Schema, but CompqSchema
        assert sc.use_idx == data_idx   # keep consistent with relation matching
        ent_idx = 0
        for raw_path, mid_seq in zip(sc.raw_paths, sc.path_list):
            path_cate, focus, _ = raw_path
            link_feat = focus.link_feat
            if path_cate not in ('Main', 'Entity'):
                continue        # only focus on entity
            assert focus.category == 'Entity'
            assert len(link_feat) == e_feat_len
            if path_cate == 'Main':
                out_pred = mid_seq[0]
                etype = get_domain(pred=out_pred)   # actual type: domain of the first predicate
            else:
                in_pred = mid_seq[-1]
                etype = get_range(pred=in_pred)     # actual type: range of the last predicate
            if etype == '':
                etype = '#unknown#'
            # efeats[data_idx, ent_idx, :] = link_feat
            efeats[data_idx, ent_idx, 0] = math.log(link_feat['score'])
            # for S-MART based EL result
            etypes[data_idx, ent_idx] = mid_dict[etype]
            emask[data_idx, ent_idx] = 1.
            ent_idx += 1

    LogInfo.end_track()
    return e_feat_len, np_data
