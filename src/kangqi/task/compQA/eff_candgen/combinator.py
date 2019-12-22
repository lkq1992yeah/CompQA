import time
from datetime import datetime
from collections import namedtuple

from .translator import convert_combination
from ..util.fb_helper import is_mediator_as_expect, get_domain, get_range, \
    is_type_contained_by, is_pred_ignored, is_mediator_type

from kangqi.util.LogUtil import LogInfo


LinkData = namedtuple('LinkData', ['detail', 'category', 'comparison', 'display', 'link_feat'])


def is_overlap(la, lb):
    """
    Judge whether two linking details have overlapping relation
    """
    if la.tokens[-1].index < lb.tokens[0].index:
        return False
    if lb.tokens[-1].index < la.tokens[0].index:
        return False
    return True


def construct_gather_linkings(el_result, tl_result, tml_result, tml_comp_result):
    # Put all E/T/Tm linkings together.
    gather_linkings = []
    for el in el_result:
        assert hasattr(el, 'link_feat')
        disp = 'E: [%d, %d) %s (%s) %.6f' % (el.tokens[0].index,
                                             el.tokens[-1].index + 1,
                                             el.entity.id.encode('utf-8'),
                                             el.name.encode('utf-8'),
                                             el.surface_score)
        gather_linkings.append(LinkData(el, 'Entity', '==', disp, el.link_feat))
    for tl in tl_result:
        disp = 'T: [%d, %d) %s (%s) %.6f' % (tl.tokens[0].index,
                                             tl.tokens[-1].index + 1,
                                             tl.entity.id.encode('utf-8'),
                                             tl.name.encode('utf-8'),
                                             tl.surface_score)
        gather_linkings.append(LinkData(tl, 'Type', '==', disp, []))
    for tml, comp in zip(tml_result, tml_comp_result):
        disp = 'Tm: [%d, %d) %s %s %.6f' % (tml.tokens[0].index,
                                            tml.tokens[-1].index + 1, comp,
                                            tml.entity.sparql_name().encode('utf-8'),
                                            tml.surface_score)
        gather_linkings.append(LinkData(tml, 'Time', comp, disp, []))
    sz = len(gather_linkings)
    LogInfo.begin_track('%d E + %d T + %d Tm = %d links.',
                        len(el_result), len(tl_result), len(tml_result), sz)
    for link_data in gather_linkings:
        LogInfo.logs(link_data.display)
    LogInfo.end_track()
    return gather_linkings


def make_combination(gather_linkings, sparql_driver, vb):
    """
    Given the E/T/Tm linkings, return all the possible combination of query structure.
    The only restrict: can't use multiple linkings with overlapped mention.
    ** Used in either WebQ or CompQ, not SimpQ **
    :param gather_linkings: list of named_tuple (detail, category, comparison, display)
    :param sparql_driver: the sparql query engine
    :param vb: verbose
    :return: the dictionary including all necessary information of a schema.
    """
    sz = len(gather_linkings)
    el_size = len(filter(lambda x: x.category == 'Entity', gather_linkings))

    # Step 1: Prepare conflict matrix
    conflict_matrix = []
    for i in range(sz):
        local_conf_list = []
        for j in range(sz):
            if is_overlap(gather_linkings[i].detail, gather_linkings[j].detail):
                local_conf_list.append(j)
            elif gather_linkings[i].category == 'Type' and gather_linkings[j].category == 'Type':
                local_conf_list.append(j)
                """ 180205: We add this restriction for saving time."""
                """ I thought there should be only one type constraint in the schema. """
                """ Don't make the task even more complex. """
        conflict_matrix.append(local_conf_list)

    # Step 2: start combination searching
    LogInfo.begin_track('Starting searching combination (total links = %d, entities = %d):',
                        len(gather_linkings), el_size)
    ground_comb_list = []       # [ (comb, path_len, sparql_query_ret) ]
    for path_len in (1, 2):
        for mf_idx, main_focus in enumerate(gather_linkings):
            if main_focus.category != 'Entity':
                continue
            visit_arr = [0] * sz                # indicating how many conflicts at the particular searching state
            state_marker = ['Path-%d||F%d/%d' % (path_len, mf_idx+1, el_size)]
            cur_comb = [(0, mf_idx)]            # indicating the focus entity
            for conf_idx in conflict_matrix[mf_idx]:
                visit_arr[conf_idx] += 1
            search_start(path_len=path_len,
                         gather_linkings=gather_linkings,
                         sparql_driver=sparql_driver,
                         cur_idx=-1, cur_comb=cur_comb,
                         conflict_matrix=conflict_matrix,
                         visit_arr=visit_arr,
                         ground_comb_list=ground_comb_list,
                         state_marker=state_marker,
                         vb=vb)
    LogInfo.end_track()
    return ground_comb_list


def search_start(path_len, gather_linkings, sparql_driver,
                 cur_idx, cur_comb, conflict_matrix, visit_arr,
                 ground_comb_list, state_marker, vb):
    """
    Main search step.
    For each searching state, we translate the combination into a SPARQL template.
    If we got some query result, then we can construct some concrete schemas and continue searching,
    otherwise we won't go deeper, and the searching process returns back.
    :param path_len: the length of the main skeleton
    :param gather_linkings: the E/T/Tm linking results
    :param sparql_driver: the sparql query engine
    :param cur_idx: the current linking to be considered
    :param cur_comb: the current state of combination
    :param conflict_matrix: the adj. list indicating conflicts between linkings (won't share the overlapped interval)
    :param visit_arr: indicating whether a linking can be used or not
    :param ground_comb_list: [ (comb, sparql_query_ret) ] as the full information that we need
    :param state_marker: indicating the current searching state
    :param vb: verbose switch
    """
    if vb >= 1:
        LogInfo.begin_track('[%s] (%s)', '||'.join(state_marker), datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    sparql_str, _, _ = convert_combination(comb=cur_comb, path_len=path_len,
                                           gather_linkings=gather_linkings,
                                           sparql_driver=sparql_driver)
    # no need to get path_list_template, and raw_paths
    if vb >= 1:
        LogInfo.logs(sparql_str)
    st = time.time()
    query_ret = sparql_driver.perform_query(sparql_str)
    filt_query_ret = predicate_filtering(query_ret=query_ret, path_len=path_len)
    if vb >= 1:
        LogInfo.logs('Filt_Query_Ret = %d / %d (%.3fs)', len(filt_query_ret), len(query_ret), time.time() - st)

    if len(filt_query_ret) == 0:
        if vb >= 1:
            LogInfo.end_track()
        return                      # no need to search deeper

    ground_comb_list.append((cur_comb, path_len, filt_query_ret))
    if len(ground_comb_list) % 20 == 0:
        LogInfo.logs('Accumulate: %d combinations with SPARQL query results ... ',
                     len(ground_comb_list))

    # Ready to search deeper
    link_sz = len(gather_linkings)
    for next_idx in range(cur_idx+1, link_sz):
        if visit_arr[next_idx] != 0:    # cannot be visited due to conflict
            continue
        for conf_idx in conflict_matrix[next_idx]:  # ready to enter the next state
            visit_arr[conf_idx] += 1
        for attach_idx in range(1, path_len+1):     # enumerate each possible attach position
            next_comb = list(cur_comb)
            next_comb.append((attach_idx, next_idx))
            state_marker.append('%d/%d-%d' % (next_idx+1, link_sz, attach_idx))
            search_start(path_len=path_len,
                         gather_linkings=gather_linkings,
                         sparql_driver=sparql_driver,
                         cur_idx=next_idx, cur_comb=next_comb,
                         conflict_matrix=conflict_matrix,
                         visit_arr=visit_arr,
                         ground_comb_list=ground_comb_list,
                         state_marker=state_marker,
                         vb=vb)
            del state_marker[-1]
        for conf_idx in conflict_matrix[next_idx]:  # return back
            visit_arr[conf_idx] -= 1
    # Ends of DFS
    if vb >= 1:
        LogInfo.end_track()


def predicate_filtering(query_ret, path_len):
    """
    Given the SPARQL query result,
    we filter out meaningless predicate rows from the raw query results.
    Rule 1. ignore common. / type. predicates
    Rule 2. the last predicate of the skeleton cannot point to a mediator.
    Rule 3. keeping strict / elegant / coherent schemas, removing general ones.
    """
    filt_query_ret = []
    for row in query_ret:
        """ Remove paths pointing to a mediator """
        last_pred = row[path_len - 1]
        if is_mediator_as_expect(last_pred):
            continue        # Filter by Rule 2

        """ Remove common / type predicates in main path or entity constraints """
        """ Won't filter legal type constraints, since 'type.object.type' never appears in the query_ret """
        keep = True
        for pred in row:
            if is_pred_ignored(pred):
                keep = False
                break       # Filter by Rule 1
        if not keep:
            continue

        # """ 180203: only keep strict / elegant / coherent paths, ignoring other general 2-hops """
        # final_flag = False
        # if path_len == 1:
        #     final_flag = True       # keeping all 1-hop predicates
        # elif path_len == 2:
        #     p1_range = get_range(row[0])
        #     p2_domain = get_domain(row[1])      # ready to compare the domain & range of the two predicates
        #     # LogInfo.logs('p1: %s, p2: %s, range1: %s, domain2: %s', row[0], row[1], p1_range, p2_domain)
        #     if p1_range == p2_domain:
        #         final_flag = True           # strict / elegant
        #     elif is_type_contained_by(p1_range, p2_domain):
        #         final_flag = True           # coherent
        # if final_flag:      # only general schemas are removed
        #     filt_query_ret.append(row)

        """ 180308: Strict only """
        final_flag = False
        if path_len == 1:
            final_flag = True
        elif path_len == 2:
            p1_range = get_range(row[0])
            p2_domain = get_domain(row[1])      # ready to compare the domain & range of the two predicates
            if p1_range == p2_domain and is_mediator_type(p1_range):
                final_flag = True
        if final_flag:
            filt_query_ret.append(row)

    return filt_query_ret
