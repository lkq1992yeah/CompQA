from ..util.fb_helper import get_range, load_sup_sub_types

from kangqi.util.LogUtil import LogInfo


"""
Given the entity linking & type linking result, we filter out unlikely types
Mainly by topical filtering:
- For each possible entity, we collect all 1-hop and 2-hop predicates, and
- then check whether a candidate type is the sub-/super- type of the expected_type of the predicate.
In this way, we can filter out topical irrelevant types.
"""


def type_filtering(el_result, tl_result, sparql_driver, is_type_extend=True, vb=0):
    if vb >= 1:
        LogInfo.begin_track('Type Filtering:')
    relevant_preds = set([])
    for el in el_result:
        mid = el.entity.id
        local_relevant_preds = collect_relevant_predicate(mid, sparql_driver)
        relevant_preds |= local_relevant_preds
    if vb >= 1:
        LogInfo.logs('%d relevant predicates collected.', len(relevant_preds))

    topical_consistent_types = prepare_topical_consistent_types(
        relevant_pred_set=relevant_preds,
        is_type_extended=is_type_extend, vb=vb)
    filt_tl_result = filter(lambda tl: tl.entity.id in topical_consistent_types, tl_result)
    LogInfo.logs('Type Filter: %d / %d types are kept.', len(filt_tl_result), len(tl_result))
    if vb >= 1:
        LogInfo.end_track()

    return filt_tl_result


def collect_relevant_predicate(mid, sparql_driver):
    """
    Given a mid, query and return all its 1-hop and 2-hop predicates.
    """
    sparql_str = (
        '%s SELECT DISTINCT ?p1 ?p2 WHERE { '
        'fb:%s ?p1 ?o1 . '
        '?o1 ?p2 ?o2 . '
        '}' % (sparql_driver.query_prefix, mid)
    )
    query_ret = sparql_driver.perform_query(sparql_str)
    relevant_pred_set = set([])
    for row in query_ret:
        for pred in row:
            if pred.startswith('type.') or pred.startswith('common.'):      # ignore useless things
                continue
            relevant_pred_set.add(pred)
    # if vb >= 1:
    #     pred_list = list(relevant_pred_set)
    #     pred_list.sort()
    #     LogInfo.logs('Relevant predicates: %s', pred_list)
    return relevant_pred_set


def prepare_topical_consistent_types(relevant_pred_set, is_type_extended, vb):
    """
    Given a set of relevant predicates, return all expected types.
    If is_type_extended == True, then we also consider expand types to its super/sub types.
    """
    range_type_set = set([])
    for pred in relevant_pred_set:
        range_type = get_range(pred)
        if range_type is not None:
            range_type_set.add(range_type)
    if vb >= 1:
        LogInfo.logs('%d range types collected.', len(range_type_set))
    if not is_type_extended:
        return range_type_set

    extended_type_set = set(range_type_set)
    sup_type_dict, sub_type_dict = load_sup_sub_types()
    for range_type in range_type_set:
        extended_type_set |= sup_type_dict.get(range_type, set([]))
        extended_type_set |= sub_type_dict.get(range_type, set([]))
    if vb >= 1:
        LogInfo.logs('%d extended types collected.', len(extended_type_set))
    return extended_type_set
