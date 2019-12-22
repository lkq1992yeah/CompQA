import re

# from kangqi.util.LogUtil import LogInfo

"""
Target: Given the SPARQL query (just output answer mid and its potential name) 
and the detail schema structure, retrieve the answer (entity names or literal values).
The detail schema structure helps apply some strategy to filter answer entities. 
"""


year_re = re.compile(r'[1-2][0-9][0-9][0-9]')


def prepare_forbidden_mid_given_comb(comb, gather_linkings):
    forbidden_mid_set = set([])  # all entities used as focus / constraints in the schema
    for _, link_idx in comb:
        link_data = gather_linkings[link_idx]
        category = link_data.category
        if category == 'Entity':  # main focus, or entity constraint
            forbidden_mid_set.add(link_data.detail.entity.id)
    return forbidden_mid_set


def prepare_forbidden_mid_given_raw_paths(raw_paths, gather_linkings):
    forbidden_mid_set = set([])  # all entities used as focus / constraints in the schema
    for raw_path in raw_paths:
        _, link_idx, link_mid, _ = raw_path
        link_data = gather_linkings[link_idx]
        category = link_data.category
        if category == 'Entity':  # main focus, or entity constraint
            forbidden_mid_set.add(link_data.detail.entity.id)
    return forbidden_mid_set


def kernel_querying(sparql_str, sparql_driver, forbidden_mid_set):
    """
    given sparql query and the engine, return the final answer.
    :param sparql_str: the SRAPQL query
    :param sparql_driver: query driver
    :param forbidden_mid_set: the entities used in the query, should be ignored (if possible).
    :return: the set of the final answer
    """
    forbidden_name_set = set([])    # all the names occurred in the query result whose mid is forbidden
    normal_name_set = set([])       # all the remaining normal names

    query_ret = sparql_driver.perform_query(sparql_str)
    for row in query_ret:       # copied from loss_calc.py
        try:  # some query result has error (caused by \n in a string)
            target_mid = row[0]
            if target_mid.startswith('m.'):
                target_name = row[1]        # get its name in FB
                if target_name == '':       # ignore entities without a proper name
                    continue
                if target_mid in forbidden_mid_set:
                    forbidden_name_set.add(target_name)
                else:
                    normal_name_set.add(target_name)
            else:  # the answer may be a number, string or datetime
                ans_name = target_mid
                if re.match(year_re, ans_name[0: 4]):
                    ans_name = ans_name[0: 4]
                    # if we found a DT, then we just keep its year info.
                normal_name_set.add(ans_name)
        except IndexError:    # some query result has an IndexError (caused by \n in a string)
            pass

    if len(normal_name_set) > 0:    # the normal answers have a strict higher priority.
        final_ans_set = normal_name_set
    else:   # we take the forbidden answer as output, only if we have no other choice.
        final_ans_set = forbidden_name_set

    return final_ans_set
