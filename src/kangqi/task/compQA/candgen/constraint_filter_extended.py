# -*- coding: utf-8 -*-

#==============================================================================
# Author: Kangqi Luo
# Goal: Pick all possible constriants that can be applied to one skeleton.
# Input: a **extended** skeleton: predicate path + (optional: type constraints)
#        therefore we could get the more precise implicit type set
# Output: a list of possible constraints that can be added to the extended skeleton.
#==============================================================================

from ..query_struct import Constraint, Schema
from .constraint_filter_basic import ground_query, get_implicit_information, is_predicate_semantically_legal

from ..u import is_linking_overlapped_with_list

from kangqi.util.LogUtil import LogInfo



# Given the extended skeleton and one candidate constraint,
# we are going to check how many ground instances are kept once we applied this constraint.
# We focus on two constraints here: ordinal constraint and time constraint.
# Since we don't care about the object value here (we can understand as, I added a solid constraint <?, p, ?>),
# the object of the constraint is always None.
def edge_representative_test(ext_sk, test_constr, driver, raw_ret_len, vb=0):
    # First prepare a new schema for ground query testing
    new_sc = Schema()
    new_sc.focus_item = ext_sk.focus_item
    new_sc.path = ext_sk.path
    new_sc.constraints = list(ext_sk.constraints)

    new_sc.constraints.append(test_constr)

    sparql_str_list, var_list = new_sc.get_sparql_str()
    query_str = ' '.join(sparql_str_list)        # Now we've built the query structure.
    query_ret = driver.perform_query(query_str)
    ret_len = len(query_ret)

    ratio = 1.0 * ret_len / raw_ret_len      # Return the ratio of ground query results kept after adding the constraint.
    if vb >= 1: LogInfo.logs('>>> Ratio = %.6f (%d / %d)', ratio, ret_len, raw_ret_len)
    return ratio


# Given the extended skeleton, return all the possible candidate constraints
# that can be applied on it.
# Note:
# - though we take tc_dict as input, we never use it, just for the consistency.
# - We use Implicit_Super_Type strict level for Entity/Time/Ordinal constraint extraction.

def constraint_filtering(ext_sk,
                         ec_dict, tc_dict,
                         tmc_set, tmv_dict,
                         oc_set, rank_set,
                         driver, min_ratio=0.0, vb=0):
    if vb >= 1:
        LogInfo.begin_track('Get implicit information: ')
    implicit_info_list = get_implicit_information(ext_sk)
    if vb >= 1:
        LogInfo.logs('Implicit type list: %s', implicit_info_list[0])
        LogInfo.logs('Super implicit type list: %s', implicit_info_list[1])
        LogInfo.end_track()

    if vb >= 1:
        LogInfo.begin_track('Running ground query: ')
    touches_list, touch_types_list, raw_ret_len = ground_query(ext_sk, driver, vb=vb)
    if vb >= 1:
        LogInfo.end_track()

    status_list = [
        'ec_conflict', 'ec_miss_type', 'ec_miss_touch', 'ec_keep',
        'tmc_conflict', 'tmc_miss_type', 'tmc_non_rep', 'tmc_keep',
        'oc_conflict', 'oc_miss_type', 'oc_non_rep', 'oc_keep'
    ]    # record the status of each possible constraint
    # We won't consider type constraints here, since they are added into ext_sk.
    total_status_dict = {st: 0 for st in status_list}

    ret_constraint_list = []
    col_len = len(touches_list)
    linking_item_list = []
    if ext_sk.focus_item is not None:
        linking_item_list.append(ext_sk.focus_item)
    for constr in ext_sk.constraints:
        linking_item_list.append(constr.linking_item)
    # We need to consider both starting entity and type constraints

    for col_idx in range(col_len):
        var_pos = col_idx + 1   # the variable position starts from x1.
        status_dict = {st: 0 for st in status_list} # initiate the status dict.

        for el_item, anchor_pred_list in ec_dict.items():
            # First check whether the linking item conflicts with the focus or not.
            if is_linking_overlapped_with_list(el_item, linking_item_list):
                status_dict['ec_conflict'] += len(anchor_pred_list)
                continue
            obj_mid = el_item.entity.id
            for pred in anchor_pred_list:
                # Second judge whether the predicate makes sense.
                if not is_predicate_semantically_legal(pred, col_idx,
                           touch_types_list, implicit_info_list, 'Implicit_Super_Type'): # type irrlevant
                    status_dict['ec_miss_type'] += 1
                    continue
                # After filtering lots of type nonsense predicates,
                # we judge whether the <P, O> has a direct touch.
                ec_touch_list = driver.query_subject_given_pred_obj(pred, obj_mid)
                ec_touch_set = set(ec_touch_list)
                if len(ec_touch_set & touches_list[col_idx]) == 0:
                    # no touch could link to constraint entity
                    status_dict['ec_miss_touch'] += 1
                    continue
                # Now we've found a suitable entity constraint
                ret_constraint_list.append(Constraint(
                        var_pos, pred, '==', obj_mid, 'Entity', el_item))
                status_dict['ec_keep'] += 1

        if len(tmv_dict) != 0:  # if there's no time linking value, then we just skip time constraint extraction.
            for tmc_pred in tmc_set:
                # First judge whether the predicate makes sense
                if not is_predicate_semantically_legal(
                        tmc_pred, col_idx, touch_types_list, implicit_info_list, 'Implicit_Super_Type'):
                    status_dict['tmc_miss_type'] += len(tmv_dict)
                    continue
                # Second judge whether the predicsate is representative
                test_constr = Constraint(var_pos, tmc_pred, None, None, 'Time', None)
                ratio = edge_representative_test(ext_sk, test_constr, driver, raw_ret_len, vb=vb)
                if ratio < min_ratio:
                    status_dict['tmc_non_rep'] += len(tmv_dict)
                    continue
                # The predicate is OK, now let's check each possible tml_item
                for tml_item, tmv in tmv_dict.items():
                    if is_linking_overlapped_with_list(tml_item, linking_item_list):
                        status_dict['tmc_conflict'] += 1
                        continue
                    # Now we've found a suitable time linking item
                    comp, tm_value = tmv
                    ret_constraint_list.append(Constraint(
                            var_pos, tmc_pred, comp, tm_value, 'Time', tml_item))
                    status_dict['tmc_keep'] += 1

        if len(rank_set) != 0:  # The same as the policy above.
            for oc_pred in oc_set:
                # First judge whether the predicate makes sense
                if not is_predicate_semantically_legal(oc_pred, col_idx,
                       touch_types_list, implicit_info_list, 'Implicit_Super_Type'):
                    status_dict['oc_miss_type'] += 2 * len(rank_set)
                    continue
                # Second judge whether the predicate is representative
                test_constr = Constraint(var_pos, oc_pred, None, None, 'Ordinal', None)
                ratio = edge_representative_test(ext_sk, test_constr, driver, raw_ret_len, vb=vb)
                if ratio < min_ratio:
                    status_dict['oc_non_rep'] += 2 * len(rank_set)
                    continue
                # The predicate is OK, now let's check each possible rank_item
                for rank_item in rank_set:
                    if is_linking_overlapped_with_list(rank_item, linking_item_list):
                        status_dict['oc_conflict'] += 2
                        continue
                    # Now we've found a suitable rank item
                    rank_num = rank_item.rank
                    for order in ['ASC', 'DESC']:
                        ret_constraint_list.append(Constraint(
                            var_pos, oc_pred, order, rank_num, 'Ordinal', rank_item))
                    status_dict['oc_keep'] += 2

        for k, v in status_dict.items(): total_status_dict[k] += v
        if vb >= 1:
            for k, v in sorted(status_dict.items(), key=lambda x: x[0]):
                if v == 0: continue
                LogInfo.logs('col-%d: %s = %d', col_idx, k, v)
    # End of enumerating each column

    if vb >= 1:
        for constr in ret_constraint_list: LogInfo.logs(constr.to_string())
        LogInfo.logs(
            'In total %d = %d entity + %d time + %d ordinal constraints kept.',
             len(ret_constraint_list), total_status_dict['ec_keep'],
             total_status_dict['tmc_keep'], total_status_dict['oc_keep'])
    return ret_constraint_list

