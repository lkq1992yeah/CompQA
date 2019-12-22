"""
Author: Kangqi Luo
Goal: Perform auxiliary operations to CompqSchema
"""


def add_relation_only_metric(cand_list):
    """
    We focus on "Relation Only" mode (RO).
    For each candidate schema, we search all the other schemas with the same predicate sequences as this one.
    Afterwards, we set the P/R/F1 of each as the highest P/R/F1 among them.
    Which means, we just ignore the entity linking performance, only judge whether a relation is ok or not.
    :param cand_list: the list of candidate schemas
    :return: the number of schemas whose F1 has been changed
    """
    group_cand_dict = {}
    for cand in cand_list:
        raw_path_rep = cand.get_rm_key()
        group_cand_dict.setdefault(raw_path_rep, []).append(cand)       # group schemas by raw path representation

    change_size = 0
    for raw_path_rep in group_cand_dict:
        local_cand_list = group_cand_dict[raw_path_rep]
        best_metric = (0., 0., 0.)
        for cand in local_cand_list:
            if cand.f1 > best_metric[2]:
                best_metric = (cand.p, cand.r, cand.f1)
        for cand in local_cand_list:
            if cand.f1 != best_metric[2]:
                change_size += 1
            cand.p = best_metric[0]
            cand.r = best_metric[1]
            cand.f1 = best_metric[2]        # force change P/R/F1 into Relation Only mode
    return change_size
