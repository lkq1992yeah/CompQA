# -*- coding: utf-8 -*-

#==============================================================================
# Author: Kangqi Luo
# Given the skeleton with a list of constraints extracted from the sentence,
# find a list of suitable constraints that can be applied to the skeleton.
# ** Note: I just move the related codes from cand_gen.py to here.
#==============================================================================

import random
from ..query_struct import Constraint
from ..u import load_domain_expect, load_super_type_dict, is_linking_overlapped

from kangqi.util.LogUtil import LogInfo


# ================= Beginning of commonly used codes ================= #

# Given the skeleton (could be extended), query the ground graph, and all possible types at each node.
def ground_query(ext_sk, driver, vb=0):
    path_size = len(ext_sk.path)
    touches_list = []       # [ set(touch entity) ]
    touch_types_list = []   # [ set(touch entity type) ]
    for _ in range(path_size):
        touches_list.append(set([]))
        touch_types_list.append(set([]))
    # Note: the focus is fixed, so the number of columns in the query result
    # is the same as the number of predicates in the path.

    sparql_str_list, var_list = ext_sk.get_sparql_str()
    var_num = len(var_list)
    query_str = ' '.join(sparql_str_list)        # Now we've built the query structure.
    query_ret = driver.perform_query(query_str)
    ''' Note: The output has x + 1 columns, and the first column is the name of the answer node. '''
    query_ret_len = len(query_ret)

    if vb >= 1:
        LogInfo.logs('%d ground result extracted, showing examples:', len(query_ret))
        LogInfo.logs('%s', query_ret[:10])

    for row in query_ret:
        if len(row) < var_num:  # avoid the affect by '\n'
            query_ret_len -= 1  # remove one invalid row.
            continue
        for col_idx in range(path_size):
            ret_col_idx = col_idx + 1   # ignore the first column: name variable
            touches_list[col_idx].add(row[ret_col_idx])
    # Now touches_list prepared, next prepare touch_types_list
    # Due to some cases with too many ground graphs, we have to sample a list of touch entities.

    for col_idx in range(path_size):
        local_touches = list(touches_list[col_idx])
        if len(local_touches) > 1000:      # we sample at most 1000 distinct touches
            local_touches = local_touches[:1000]        # no need to random sample: entities come from a set
        for touch in local_touches:
            if not touch.startswith('m.'):
                continue
            t_list = driver.query_type_given_entity(touch)
            touch_types_list[col_idx].update(t_list)
        if vb >= 1:
            LogInfo.logs(
                'col-#%d: distinct entities = %d, types = %d.',
                col_idx, len(touches_list[col_idx]), len(touch_types_list[col_idx]))
    return touches_list, touch_types_list, query_ret_len


# Given the extended skeleton, return the implicit types at each variable.
# These types can be induced from either the skeleton, or the branch type constriant.
# return [set(implicit_types)]
def get_implicit_type_list(ext_sk):
    path = ext_sk.path
    n_var = len(path)
    pred_domain_dict, pred_expect_dict = load_domain_expect()
    implicit_type_list = [set([]) for _ in range(n_var)]

    # First scan the skeleton to find implicit types
    for col_idx in range(n_var):
        var_pos = col_idx + 1
        point_to_pred = path[var_pos - 1]   # the predicate pointing to this var_pos
        point_to_expect_type = pred_expect_dict.get(point_to_pred)
        if point_to_expect_type is not None:
            implicit_type_list[col_idx].add(point_to_expect_type)
        if var_pos < len(path):
            point_from_pred = path[var_pos] # the predicate starts from this var_pos
            point_from_domain_type = point_from_pred[0 : point_from_pred.rfind('.')]
            implicit_type_list[col_idx].add(point_from_domain_type)
    # Second scan all branch types.
    for constr in ext_sk.constraints:
        if constr.constr_type != 'Type':
            continue
        var_pos = constr.x
        col_idx = var_pos - 1
        implicit_type_list[col_idx].add(constr.o) # add the branch type to the implicit type list
    return implicit_type_list

# extend the implicit type set by using super types.
def get_super_implicit_type_list(implicit_type_list):
    super_type_dict = load_super_type_dict()
    super_implicit_type_list = []
    for implicit_type_set in implicit_type_list:
        ext_implicit_type_set = set(implicit_type_set)
        for tp in implicit_type_set:
            sup_tp_set = super_type_dict.get(tp)    # get super types for each implicit type
            if sup_tp_set is not None:
                ext_implicit_type_set.update(sup_tp_set)
        super_implicit_type_list.append(ext_implicit_type_set)
    return super_implicit_type_list

# Returns implicit type lists, either with or without super types.
def get_implicit_information(ext_sk):
    implicit_type_list = get_implicit_type_list(ext_sk)
    super_implicit_type_list = get_super_implicit_type_list(implicit_type_list)
    return implicit_type_list, super_implicit_type_list


# Judge whether a type constraint at some place is
# semantically implicit contained in the current implicit info.
def is_type_implicit_contained(tp, col_idx, implicit_info_list, strict_level):
    assert strict_level in ('Implicit_Super_Type', 'Implicit')

    implicit_type_list, super_implicit_type_list = implicit_info_list
    if strict_level == 'Implicit':
        return tp in implicit_type_list[col_idx]
    elif strict_level == 'Implicit_Super_Type':
        return tp in super_implicit_type_list[col_idx]
    else:
        return False

# Given a constraint predicate, judge whether this predicate is semantically consistent.
# We list the conditions that a legal predicate should satisfy, in 3 different semantic levels.
# The main focus is the domain type of the predicate, short for "pred_tp"
# Touch: (Most relaxed) pred_tp matches the type of some touch entity;
# Implicit_Super_Type: pred_tp matches the implicit type, or its super types.
# Implicit: (Most strict) pred_tp must be the implicit type inferred from the path.
def is_predicate_semantically_legal(pred, col_idx,
               touch_types_list, implicit_info_list, strict_level):
    pred_domain_type = pred[0 : pred.rfind('.')]
    assert strict_level in ('Touch', 'Implicit_Super_Type', 'Implicit')

    use_type_set = None
    implicit_type_list, super_implicit_type_list = implicit_info_list
    if strict_level == 'Touch':
        use_type_set = touch_types_list[col_idx]
    elif strict_level == 'Implicit':
        use_type_set = implicit_type_list[col_idx]
    elif strict_level == 'Implicit_Super_Type':
        use_type_set = super_implicit_type_list[col_idx]

    return pred_domain_type in use_type_set


# ====================== End of commonly used codes ========================= #


# Filter lots of semantically inconsistent constraint candidates.
# Note: we didn't use min_ratio here, just for the consistency of the function.
def constraint_filtering(sk,
                         ec_dict, tc_dict,
                         tmc_set, tmv_dict,
                         oc_set, rank_set,
                         driver, min_ratio=0.0, vb=0):
    if vb >= 1: LogInfo.begin_track('Running ground query: ')
    touches_list, touch_types_list, query_ret_len = ground_query(sk, driver, vb=vb)
    if vb >= 1: LogInfo.end_track()

    if vb >= 1:
        LogInfo.begin_track('Get implicit information: ')
    implicit_info_list = get_implicit_information(sk)
    if vb >= 1:
        LogInfo.logs('Touch type list: %s', touch_types_list)
        LogInfo.logs('Implicit type list: %s', implicit_info_list[0])
        LogInfo.logs('Super implicit type list: %s', implicit_info_list[1])
        LogInfo.end_track()


    status_list = [
        'ec_conflict', 'ec_miss_type', 'ec_miss_touch', 'ec_keep',
        'tc_conflict', 'tc_miss_type', 'tc_implicit', 'tc_keep',
        'tmc_conflict', 'tmc_miss_type', 'tmc_keep',
        'oc_conflict', 'oc_miss_type', 'oc_keep'
    ]    # record the status of each possible constraint
    ret_constraint_list = []
    col_len = len(touches_list)

    total_status_dict = {st: 0 for st in status_list}
    for col_idx in range(col_len):
        var_pos = col_idx + 1   # the variable position starts from x1.
        status_dict = {st: 0 for st in status_list} # initiate the status dict.

        for el_item, anchor_pred_list in ec_dict.items():
            if is_linking_overlapped(el_item, sk.focus_item):
                status_dict['ec_conflict'] += len(anchor_pred_list)
                continue
            obj_mid = el_item.entity.id
            for pred in anchor_pred_list:
                if not is_predicate_semantically_legal(pred, col_idx,
                           touch_types_list, implicit_info_list, 'Touch'): # type irrlevant
                    status_dict['ec_miss_type'] += 1
                    continue
                ec_touch_list = driver.query_subject_given_pred_obj(pred, obj_mid)
                ec_touch_set = set(ec_touch_list)
                if len(ec_touch_set & touches_list[col_idx]) == 0:
                    # no touch could link to constraint entity
                    status_dict['ec_miss_touch'] += 1
                    continue
                ret_constraint_list.append(Constraint(
                        var_pos, pred, '==', obj_mid, 'Entity', el_item))
                status_dict['ec_keep'] += 1

        for tl_item, t in tc_dict.items():
            if is_linking_overlapped(tl_item, sk.focus_item):
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

        for tml_item, tmv in tmv_dict.items():
            if is_linking_overlapped(tml_item, sk.focus_item):
                status_dict['tmc_conflict'] += len(tmc_set)
                continue
            for tmc_pred in tmc_set:
                if not is_predicate_semantically_legal(tmc_pred, col_idx,
                                   touch_types_list, implicit_info_list, 'Touch'):
                    status_dict['tmc_miss_type'] += 1
                    continue
                comp, tm_value = tmv
                ret_constraint_list.append(Constraint(
                        var_pos, tmc_pred, comp, tm_value, 'Time', tml_item))
                status_dict['tmc_keep'] += 1

        for rank_item in rank_set:
            rank_num = rank_item.rank
            if is_linking_overlapped(rank_item, sk.focus_item):
                continue
            for oc_pred in oc_set:
                if not is_predicate_semantically_legal(oc_pred, col_idx,
                       touch_types_list, implicit_info_list, 'Implicit_Super_Type'):
                    status_dict['oc_miss_type'] += 2
                    continue
                for order in ['ASC', 'DESC']:
                    ret_constraint_list.append(Constraint(
                        var_pos, oc_pred, order, rank_num, 'Ordinal', rank_item))
                status_dict['oc_keep'] += 2

        for k, v in status_dict.items(): total_status_dict[k] += v
        if vb >= 1:
            for k, v in sorted(status_dict.items(), lambda x, y: cmp(x[0], y[0])):
                if v == 0: continue
                LogInfo.logs('col-%d: %s = %d', col_idx, k, v)
    # End of enumerating each column

    if vb >= 1:
        for constr in ret_constraint_list: LogInfo.logs(constr.to_string())
        LogInfo.logs(
            'In total %d = %d entity + %d type + %d time + %d ordinal constraints kept.',
             len(ret_constraint_list),
             total_status_dict['ec_keep'], total_status_dict['tc_keep'],
             total_status_dict['tmc_keep'], total_status_dict['oc_keep'])
    return ret_constraint_list


#============================================================================== #
#============================================================================== #


# Deprecated codes
## Given a constraint type added on some position,
## judge whether this extra imformation can be omitted or not.
#def bak_is_type_implicit_contained(self, tp, col_idx, path, strict_level):
#    assert strict_level in ('Implicit_Super_Type', 'Implicit')
#
#    implicit_type_set = self.get_implicit_type_set(col_idx, path)
#    if strict_level == 'Implicit':
#        return tp in implicit_type_set
#    elif strict_level == 'Implicit_Super_Type':
#        ext_implicit_type_set = self.extend_implicit_type_set(implicit_type_set)
#        return tp in ext_implicit_type_set
#    else:
#        return False
#
## Given a constraint predicate, judge whether this predicate is semantically consistent.
## We list the conditions that a legal predicate should satisfy, in 3 different semantic levels.
## The main focus is the domain type of the predicate, short for "pred_tp"
## Touch: (Most relaxed) pred_tp matches the type of some touch entity;
## Implicit_Super_Type: pred_tp matches the implicit type, or its super types.
## Implicit: (Most strict) pred_tp must be the implicit type inferred from the path.
#def bak_is_predicate_semantically_legal(self, pred, col_idx, path, touch_types_list, strict_level):
#    pred_domain_type = pred[0 : pred.rfind('.')]
#    assert strict_level in ('Touch', 'Implicit_Super_Type', 'Implicit')
#
#    if strict_level == 'Touch':
#        touch_type_set = touch_types_list[col_idx]
#        return pred_domain_type in touch_type_set
#    else:
#        implicit_type_set = self.get_implicit_type_set(col_idx, path)
#        # Now we've got implicit types from the path information.
#        if strict_level == 'Implicit':
#            return pred_domain_type in implicit_type_set
#        elif strict_level == 'Implicit_Super_Type':     # expand implicit type set with super types
#            ext_implicit_type_set = self.extend_implicit_type_set(implicit_type_set)
#            # LogInfo.logs('exp_imp_type_set = %s', ext_implicit_type_set)
#            return pred_domain_type in ext_implicit_type_set
#        else:
#            return False





#==============================================================================
# Deprecate codes, moved from cand_gen.py
#==============================================================================


#    # Given the skeleton, query the ground graph, and all possible types at each node.
#    def ground_query(self, focus_mid, path, vb = 0):
#        path_size = len(path)
#        var_size = path_size + 1
#
#        touches_list = []       # [ set(touch entity) ]
#        touch_types_list = []   # [ set(touch entity type) ]
#        for _ in range(path_size):
#            touches_list.append(set([]))
#            touch_types_list.append(set([]))
#        # Note: the focus is fixed, so the number of columns in the query result
#        # is the same as the number of predicates in the path.
#
#        query_str_list = []        # Now we construct a SPARQL query.
#        query_str_list.append('PREFIX fb: <http://rdf.freebase.com/ns/>')
#        var_line = 'SELECT DISTINCT'
#        for var_idx in range(1, var_size): var_line += (' ?x%d' %var_idx)
#        query_str_list.append(var_line)
#        query_str_list.append('WHERE {')
#        for path_idx in range(path_size):
#            s = ('fb:%s' %focus_mid) if path_idx == 0 else ('?x%d' %path_idx)
#            p = 'fb:%s' %path[path_idx]
#            o = '?x%d' %(path_idx + 1)
#            query_str_list.append('%s %s %s .' %(s, p, o))
#        query_str_list.append('}')
#        query_str = ' '.join(query_str_list)        # Now we've built the query structure.
#        query_ret = self.driver.perform_query(query_str)
#
#        if vb >= 1: LogInfo.logs('%d ground result extracted.', len(query_ret))
##        LogInfo.logs('%s', query_ret)
#        for col_idx in range(path_size):
#            for row in query_ret: touches_list[col_idx].add(row[col_idx])
#            for touch in touches_list[col_idx]:
#                if not touch.startswith('m.'): continue
#                t_list = self.driver.query_type_given_entity(touch)
#                touch_types_list[col_idx].update(t_list)
#            if vb >= 1:
#                LogInfo.logs(
#                    'col-#%d: distinct entities = %d, types = %d.',
#                    col_idx, len(touches_list[col_idx]), len(touch_types_list[col_idx]))
#        return touches_list, touch_types_list
#
#    # get implicit type from the predicates pointing from / to the current position.
#    def get_implicit_type_set(self, col_idx, path):
#        var_pos = col_idx + 1
#        pred_domain_dict, pred_expect_dict = load_domain_expect()
#        implicit_type_set = set([])         # the types inherited from path structure
#        point_to_pred = path[var_pos - 1]   # the predicate pointing to this var_pos
#        point_to_expect_type = pred_expect_dict.get(point_to_pred)
#        if point_to_expect_type is not None:
#            implicit_type_set.add(point_to_expect_type)
#        if var_pos < len(path):
#            point_from_pred = path[var_pos] # the predicate starts from this var_pos
#            implicit_type_set.add(point_from_pred[0 : point_from_pred.rfind('.')])
#        # LogInfo.logs('col_idx = %d, imp_type_set = %s', col_idx, implicit_type_set)
#        return implicit_type_set
#
#    # extend the implicit type set by using super types.
#    def extend_implicit_type_set(self, implicit_type_set):
#        super_type_dict = load_super_type_dict()
#        ext_implicit_type_set = set(implicit_type_set)
#        for tp in implicit_type_set:
#            sup_tp_set = super_type_dict.get(tp)    # get super types for each implicit type
#            if sup_tp_set is not None:
#                ext_implicit_type_set.update(sup_tp_set)
#        return ext_implicit_type_set
#
#    # Given a constraint type added on some position,
#    # judge whether this extra imformation can be omitted or not.
#    def is_type_implicit_contained(self, tp, col_idx, path, strict_level):
#        assert strict_level in ('Implicit_Super_Type', 'Implicit')
#
#        implicit_type_set = self.get_implicit_type_set(col_idx, path)
#        if strict_level == 'Implicit':
#            return tp in implicit_type_set
#        elif strict_level == 'Implicit_Super_Type':
#            ext_implicit_type_set = self.extend_implicit_type_set(implicit_type_set)
#            return tp in ext_implicit_type_set
#        else:
#            return False
#
#    # Given a constraint predicate, judge whether this predicate is semantically consistent.
#    # We list the conditions that a legal predicate should satisfy, in 3 different semantic levels.
#    # The main focus is the domain type of the predicate, short for "pred_tp"
#    # Touch: (Most relaxed) pred_tp matches the type of some touch entity;
#    # Implicit_Super_Type: pred_tp matches the implicit type, or its super types.
#    # Implicit: (Most strict) pred_tp must be the implicit type inferred from the path.
#    def is_predicate_semantically_legal(self, pred, col_idx, path, touch_types_list, strict_level):
#        pred_domain_type = pred[0 : pred.rfind('.')]
#        assert strict_level in ('Touch', 'Implicit_Super_Type', 'Implicit')
#
#        if strict_level == 'Touch':
#            touch_type_set = touch_types_list[col_idx]
#            return pred_domain_type in touch_type_set
#        else:
#            implicit_type_set = self.get_implicit_type_set(col_idx, path)
#            # Now we've got implicit types from the path information.
#            if strict_level == 'Implicit':
#                return pred_domain_type in implicit_type_set
#            elif strict_level == 'Implicit_Super_Type':     # expand implicit type set with super types
#                ext_implicit_type_set = self.extend_implicit_type_set(implicit_type_set)
#                # LogInfo.logs('exp_imp_type_set = %s', ext_implicit_type_set)
#                return pred_domain_type in ext_implicit_type_set
#            else:
#                return False
#
#    # Filter lots of semantically inconsistent constraint candidates.
#    def constraint_filtering(self, sk, ec_dict, tc_dict, tmc_set, tmv_dict, oc_set, rank_set, vb = 0):
#        focus_mid = sk.focus_item.entity.id
#        path = sk.path
#
#        if vb >= 1: LogInfo.begin_track('Running ground query: ')
#        touches_list, touch_types_list = self.ground_query(focus_mid, path, vb = vb)
#        if vb >= 1: LogInfo.end_track()
#        # We use this type information to quickly filter irrelevant predicates
#
#        status_list = [
#            'ec_conflict', 'ec_miss_type', 'ec_miss_touch', 'ec_keep',
#            'tc_conflict', 'tc_miss_type', 'tc_implicit', 'tc_keep',
#            'tmc_conflict', 'tmc_miss_type', 'tmc_keep',
#            'oc_conflict', 'oc_miss_type', 'oc_keep'
#        ]    # record the status of each possible constraint
#        ret_constraint_list = []
#        col_len = len(touches_list)
#
#        total_status_dict = {st: 0 for st in status_list}
#        for col_idx in range(col_len):
#            var_pos = col_idx + 1   # the variable position starts from x1.
#            status_dict = {st: 0 for st in status_list} # initiate the status dict.
#
#            for el_item, anchor_pred_list in ec_dict.items():
#                if is_linking_overlapped(el_item, sk.focus_item):
#                    status_dict['ec_conflict'] += len(anchor_pred_list)
#                    continue
#                obj_mid = el_item.entity.id
#                for pred in anchor_pred_list:
#                    if not self.is_predicate_semantically_legal(
#                            pred, col_idx, path, touch_types_list, 'Touch'):  # type irrelevant
#                        status_dict['ec_miss_type'] += 1
#                        continue
#                    ec_touch_list = self.driver.query_subject_given_pred_obj(pred, obj_mid)
#                    ec_touch_set = set(ec_touch_list)
#                    if len(ec_touch_set & touches_list[col_idx]) == 0:
#                        # no touch could link to constraint entity
#                        status_dict['ec_miss_touch'] += 1
#                        continue
#                    ret_constraint_list.append(Constraint(
#                            var_pos, pred, '==', obj_mid, 'Entity', el_item))
#                    status_dict['ec_keep'] += 1
#
#            for tl_item, t in tc_dict.items():
#                if is_linking_overlapped(tl_item, sk.focus_item):
#                    status_dict['tc_conflict'] += 1
#                    continue
#                if t not in touch_types_list[col_idx]:
#                    status_dict['tc_miss_type'] += 1
#                    continue
#                if self.is_type_implicit_contained(t, col_idx, path, 'Implicit_Super_Type'):
#                    status_dict['tc_implicit'] += 1     # filter implicit inferred constraints
#                    continue
#                ret_constraint_list.append(Constraint(
#                        var_pos, 'type.object.type', '==', t, 'Type', tl_item))
#                status_dict['tc_keep'] += 1
#
#            for tml_item, tmv in tmv_dict.items():
#                if is_linking_overlapped(tml_item, sk.focus_item):
#                    status_dict['tmc_conflict'] += len(tmc_set)
#                    continue
#                for tmc_pred in tmc_set:
#                    if not self.is_predicate_semantically_legal(
#                            tmc_pred, col_idx, path, touch_types_list, 'Touch'):
#                        status_dict['tmc_miss_type'] += 1
#                        continue
#                    comp, tm_value = tmv
#                    ret_constraint_list.append(Constraint(
#                            var_pos, tmc_pred, comp, tm_value, 'Time', tml_item))
#                    status_dict['tmc_keep'] += 1
#
#            for rank_item in rank_set:
#                rank_num = rank_item.rank
#                if is_linking_overlapped(rank_item, sk.focus_item):
#                    continue
#                for oc_pred in oc_set:
#                    if not self.is_predicate_semantically_legal(
#                            oc_pred, col_idx, path, touch_types_list, 'Implicit_Super_Type'):
#                        status_dict['oc_miss_type'] += 2
#                        continue
#                    for order in ['ASC', 'DESC']:
#                        ret_constraint_list.append(Constraint(
#                            var_pos, oc_pred, order, rank_num, 'Ordinal', rank_item))
#                    status_dict['oc_keep'] += 2
#
#            for k, v in status_dict.items(): total_status_dict[k] += v
#            if vb >= 1:
#                for k, v in sorted(status_dict.items(), lambda x, y: cmp(x[0], y[0])):
#                    if v == 0: continue
#                    LogInfo.logs('col-%d: %s = %d', col_idx, k, v)
#        # End of enumerating each column
#
#        if vb >= 1:
#            for constr in ret_constraint_list: LogInfo.logs(constr.to_string())
#            LogInfo.logs(
#                'In total %d = %d entity + %d type + %d time + %d ordinal constraints kept.',
#                 len(ret_constraint_list),
#                 total_status_dict['ec_keep'], total_status_dict['tc_keep'],
#                 total_status_dict['tmc_keep'], total_status_dict['oc_keep'])
#        return ret_constraint_list

