# -*- coding: utf-8 -*-

#==============================================================================
# Author: Kangqi Luo
# Goal: Generate extended skeleton (path + possible type constraints)
#==============================================================================

from .constraint_filter_basic import ground_query, get_implicit_information, is_type_implicit_contained
from .cand_constructor import CandidateConstructor

from ..u import is_linking_overlapped
from ..query_struct import Constraint

from kangqi.util.LogUtil import LogInfo

# The code is really similar with constraint_filtering(......)
def skeleton_extension(sk, tc_dict, driver, vb=0):
    if vb >= 1:
        LogInfo.begin_track('Running ground intention: ')
    touches_list, touch_types_list, query_ret_len = ground_query(sk, driver, vb=vb)
    # output all the ground instances of the query
    if vb >= 1:
        LogInfo.end_track()

    if vb >= 1:
        LogInfo.begin_track('Get implicit information: ')
    implicit_info_list = get_implicit_information(sk)
    # output the direct type implication and super type implication
    if vb >= 1:
        LogInfo.logs('Touch type list: %s', touch_types_list)
        LogInfo.logs('Implicit type list: %s', implicit_info_list[0])
        LogInfo.logs('Super implicit type list: %s', implicit_info_list[1])
        LogInfo.end_track()

    status_list = ['tc_conflict', 'tc_miss_type', 'tc_implicit', 'tc_keep']
    ret_constraint_list = []
    col_len = len(touches_list)
    focus_item = sk.focus_item

    for col_idx in range(col_len):
        var_pos = col_idx + 1
        status_dict = {st: 0 for st in status_list}
        for tl_item, t in tc_dict.items():
            if is_linking_overlapped(tl_item, focus_item):
                status_dict['tc_conflict'] += 1
                continue
            if t not in touch_types_list[col_idx]:
                status_dict['tc_miss_type'] += 1
                continue
            if is_type_implicit_contained(t, col_idx, implicit_info_list, 'Implicit_Super_Type'):
                status_dict['tc_implicit'] += 1     # filter implicit inferred constraints
                continue
            ret_constraint_list.append(Constraint(
                    var_pos, 'type.object.type', '==', t, 'Type', tl_item))
            status_dict['tc_keep'] += 1

        if vb >= 1:
            for k, v in sorted(status_dict.items(), key=lambda x: x[0]):
                if v == 0: continue
                LogInfo.logs('col-%d: %s = %d', col_idx, k, v)
    if vb >= 1:
        for constr in ret_constraint_list: LogInfo.logs(constr.to_string())
        LogInfo.logs('Candidate type constraints = %d.', len(ret_constraint_list))

    # Collect all the possible type constraints and then perform ext_sk construction
    cand_cons = CandidateConstructor()
    ext_sk_list = cand_cons.construct(sk, ret_constraint_list, driver=driver, vb=vb)
    return ext_sk_list