from ..util.fb_helper import inverse_predicate

from kangqi.util.LogUtil import LogInfo

"""
Two kinds of translator:
1. Given combination, without grounded predicates: used for 1-hop / 2-hop query, and collect grounded predicates.
2. Given combination, with grounded predicates: used for detail P/R/F1 query, and construct concrete schemas.
"""


def convert_combination(comb, path_len, gather_linkings,
                        sparql_driver, ground_pred_list=None, vb=0):
    """
    Given the current combination and the fixed linkings, construct the SPARQL query
    gather_linkings: [(detail, category, comparison, display)]
    comparison: ==, >, <, >=, Max, Min
    If ground_pred_list is not given, then we generate the SPARQL query for collecting 1-hop or 2-hop predicates.
    If ground_pred_list is given, then we generate the SPARQL query for the target answer.
    :return: 1. SPARQL query
             2. path_list (or just a template)
             [DEPRECATED after 12/05/2017]
             3. raw_paths [(category, focus_idx, focus_mid, pred_seq)]
             [PUT INTO USE after 12/06/2017]
    """
    if vb >= 2:
        LogInfo.begin_track('Checking translate:')
        surface = ' '.join([
            '(%d--[%s])' % (attach_idx, gather_linkings[linking_idx].display)
            for attach_idx, linking_idx in comb
        ])
        LogInfo.logs('Comb: %s', surface)
    path_list = []
    raw_paths = []
    sparql_lines = [sparql_driver.query_prefix]
    where_lines = []        # for constraints + OPTIONAL + FILTER
    order_line = ''         # for ordinal constraint
    comb.sort(key=lambda kv: kv[0])

    has_ground = ground_pred_list is not None
    # the predicate to be used in SPARQL (either ?p1, ?p2 ... placeholder, or grounded predicates)
    if ground_pred_list is not None:
        pl_pred_list = list(ground_pred_list)                                   # used in path_list
        sparql_pred_list = map(lambda pred: 'fb:'+pred, ground_pred_list)   # used in SPARQL query
    else:
        pred_cnt = path_len
        for attach_idx, link_idx in comb:
            category = gather_linkings[link_idx].category
            if attach_idx > 0 and category != 'Type':
                pred_cnt += 1
        sparql_pred_list = map(lambda x: '?p%d' % (x+1), range(pred_cnt))
        pl_pred_list = list(sparql_pred_list)
    pl_pred_list.insert(0, '')          # the real predicates (real mid or ?pxxx)
    sparql_pred_list.insert(0, '')      # store predicates (fb:xxx or ?pxxx) in the SPARQL query
    # add the dummy node at the beginning, since the predicate index starts from 1

    add_p_idx = path_len        # additional predicate index (incremental)
    for attach_idx, link_idx in comb:
        link_data = gather_linkings[link_idx]
        category = link_data.category
        path = []
        # path: a list of predicate templates.
        # we first put leading predicates into the path
        path += get_leading_predicates(attach_idx=attach_idx,
                                       path_len=path_len,
                                       pred_list=pl_pred_list)
        if attach_idx == 0:
            # this path points to the focus entity
            path.append(link_data.detail.entity.id)
            where_lines.append('fb:%s %s ?o1 .' % (
                link_data.detail.entity.id, sparql_pred_list[1]))
            for p_idx in range(1, path_len):
                where_lines.append('?o%d %s ?o%d .' % (
                    p_idx, sparql_pred_list[p_idx+1], p_idx+1))
            forward_pred_seq = pl_pred_list[1: path_len+1]
            raw_path_tup = ('Main', link_idx, link_data.detail.entity.id, forward_pred_seq)
            # Note: in raw_path, the Main path goes from focus to the answer
        elif category == 'Entity':
            # this path points to another entity, we need additional predicate
            add_p_idx += 1
            path.append(pl_pred_list[add_p_idx])
            backward_pred_seq = list(path)          # not including the entity itself
            path.append(link_data.detail.entity.id)
            where_lines.append('?o%d %s fb:%s .' % (
                attach_idx, sparql_pred_list[add_p_idx], link_data.detail.entity.id))
            raw_path_tup = ('Entity', link_idx, link_data.detail.entity.id, backward_pred_seq)
        elif category == 'Type':
            # type constraint, must be "type.object.type"
            path.append('type.object.type')
            backward_pred_seq = list(path)  # not including the type itself
            path.append(link_data.detail.entity.id)     # it's a type
            where_lines.append('?o%d fb:type.object.type fb:%s .' % (
                attach_idx, link_data.detail.entity.id))
            raw_path_tup = ('Type', link_idx, link_data.detail.entity.id, backward_pred_seq)
        elif category == 'Time':
            add_p_idx += 1
            comp = link_data.comparison     # ==, >, <, >=
            year = link_data.detail.entity.sparql_name()
            path.append(pl_pred_list[add_p_idx])
            backward_pred_seq = list(path)  # not including the time & comparison
            path.append(comp)
            path.append(year)
            where_lines.append('?o%d %s ?o%d .' % (
                attach_idx, sparql_pred_list[add_p_idx], add_p_idx))
            build_time_filter(year=year, comp=comp,
                              add_o_idx=add_p_idx,
                              filter_list=where_lines)
            raw_path_tup = ('Time', link_idx, year, backward_pred_seq)
        else:   # Ordinal
            add_p_idx += 1
            comp = link_data.comparison     # MAX, MIN, not using ASC/DESC
            rank = link_data.detail.entity.id      # TODO: haven't implemented yet
            show_name = '%s-%s' % (comp, rank)
            path.append(pl_pred_list[add_p_idx])
            backward_pred_seq = list(path)      # not including the rank & order
            path.append(comp)
            path.append(rank)
            where_lines.append('?o%d %s ?o%d .' % (
                attach_idx, sparql_pred_list[add_p_idx], add_p_idx))
            rank_num = int(rank)
            order_line = 'LIMIT 1 OFFSET %d' % (rank_num - 1)
            raw_path_tup = ('Ordinal', link_idx, show_name, backward_pred_seq)
        path_list.append(path)
        raw_paths.append(raw_path_tup)

    assert add_p_idx + 1 == len(pl_pred_list)      # all predicates must be used

    select_line = 'SELECT DISTINCT'
    if not has_ground:
        for p_idx in range(add_p_idx):
            select_line += ' ?p%d' % (p_idx+1)  # output all the placeholder predicates
    else:
        select_line += ' ?o%d ?n%d' % (path_len, path_len)       # just output the target answer
        where_lines.append('OPTIONAL { ?o%d fb:type.object.name ?n%d } .' % (path_len, path_len))
        # We are outputting names
    select_line += ' WHERE {'
    sparql_lines.append(select_line)
    sparql_lines += where_lines
    sparql_lines.append('}')
    if order_line != '':
        sparql_lines.append(order_line)

    if vb >= 2:
        LogInfo.logs('SPARQL:')
        for line in sparql_lines:
            LogInfo.logs(line.encode('utf-8'))
        LogInfo.logs('Path Template:')
        for path in path_list:
            LogInfo.logs(', '.join(path))
        LogInfo.end_track()
    return ' '.join(sparql_lines), path_list, raw_paths


def get_leading_predicates(attach_idx, path_len, pred_list):
    """
    Given the attach index, path len along with the list of predicates we use in the combination,
    return the predicate sequence starting from the target answer, ending to the particular attached variable.
    """
    ret_list = []
    for idx in range(path_len, attach_idx, -1):
        pred = pred_list[idx]   # predicate position starts from 1, since we have the dummy node in the pred_list
        if pred.startswith('fb:'):
            pred = pred[3:]     # fb:xxxx is the predicate used in SPARQL query, we need to remove the prefix.
        inv_pred = inverse_predicate(pred)
        ret_list.append(inv_pred)   # from end to start, so we are reversing edges
    return ret_list


def build_time_filter(year, comp, add_o_idx, filter_list):
    """
    Given the year, the comparison and the additional predicate index,
    append FILTER sentence to the list.
    Check evernote 180204 for detail information.
    """
    year_num = int(year)
    first_day = '"%s-01-01"^^xsd:dateTime' % year_num
    last_day = '"%s-01-01"^^xsd:dateTime' % (year_num + 1)
    if comp == '==':    # within the current year
        filter_list.append('FILTER (xsd:datetime(?o%d) >= %s) .' % (add_o_idx, first_day))
        filter_list.append('FILTER (xsd:datetime(?o%d) < %s) .' % (add_o_idx, last_day))
    elif comp == '>=':  # starts from the current year
        filter_list.append('FILTER (xsd:datetime(?o%d) >= %s) .' % (add_o_idx, first_day))
    elif comp == '>':   # starts from the next year
        filter_list.append('FILTER (xsd:datetime(?o%d) >= %s) .' % (add_o_idx, last_day))
    elif comp == '<':   # before the current year
        filter_list.append('FILTER (xsd:datetime(?o%d) < %s) .' % (add_o_idx, first_day))
