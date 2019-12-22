# -*- coding: utf-8 -*-

#==============================================================================
# Author: Kangqi Luo
# Goal: Given a skeleton with all its possible constraints,
#       construct all the candidate query structures.
# Rules:
# 1. There should have any overlap between the mention of the focus and all selected constraints.
# 2. At most one ordinal constraint could be selected.
#==============================================================================

import numpy as np

from ..query_struct.schema import Schema
from ..u import is_linking_overlapped

from kangqi.util.LogUtil import LogInfo

def nextPick(ptr, status):
    n = len(status)
    for new_ptr in range(ptr + 1, n):
        if status[new_ptr] == 1:
            return new_ptr
    return -1

class CandidateConstructor(object):

    def __init__(self):
        pass

    # return a list of final candidate schemas
    # Input: the extended_skeleton (with a path and optional type constraints)
    # We ensure that each candidate constraint has no overlapping with
    # the other items existing in the ext_sk (focus and the explicit type constraints)
    def construct(self, ext_sk, cand_list, driver=None, vb=0):
        ret_schema_list = []
        cand_size = len(cand_list)
        mat = []
        # conflict matrix (adjacent format)

        for i in range(cand_size):
            vec = [i]
            for j in range(i + 1, cand_size):
                conflict = False
                if is_linking_overlapped(cand_list[i].linking_item, cand_list[j].linking_item):
                    conflict = True
                elif cand_list[i].constr_type == cand_list[j].constr_type == 'Ordinal':
                    conflict = True
                if conflict: vec.append(j)
            mat.append(vec)

        ret_comb_list = self.dfs_search_combinations(mat, vb=vb)
        ret_schema_list.append(ext_sk)
        for comb in ret_comb_list:
            sc = Schema()
            sc.focus_item = ext_sk.focus_item
            sc.path = ext_sk.path
            sc.constraints = list(ext_sk.constraints) # Transfer ext_sk's constraints here.
            for constr_idx in comb:
                sc.constraints.append(cand_list[constr_idx])
            ret_schema_list.append(sc)
        if vb >= 1: LogInfo.logs('%d schemas generated from %d candidate constraints.',
                     len(ret_schema_list), len(cand_list))

        if driver is not None:
            filt_schema_list = self.filter_by_query_ret(ret_schema_list, driver, vb=vb)
            if vb >= 1:
                LogInfo.logs('%d out of %d schemas kept after intention result filtering.',
                             len(filt_schema_list), len(ret_schema_list))
        else:
            filt_schema_list = ret_schema_list

        return filt_schema_list

    def filter_by_query_ret(self, raw_schema_list, driver, vb=0):
        filt_schema_list = []
        for raw_schema in raw_schema_list:
            sparql_str_list, var_list = raw_schema.get_sparql_str()
            query_str = ' '.join(sparql_str_list)        # Now we've built the query structure.
            query_ret = driver.perform_query(query_str)
            ret_len = len(query_ret)
            if ret_len > 0:
                filt_schema_list.append(raw_schema)
        return filt_schema_list



    # mat: conflict matrix (if j in mat[i], then it means i and j are not compatible)
    def dfs_search_combinations(self, mat, vb=0):
        ret_comb_list = []
        n = len(mat)
        status = np.ones((n,), dtype = 'int32')
        stack = []
        ptr = -1
        while True:
            ptr = nextPick(ptr, status)
            if ptr == -1:   # backtrace: restore status array
                if len(stack) == 0: break   # indicating the end of searching
                pop_idx = stack.pop()
                for item in mat[pop_idx]: status[item] += 1
                ptr = pop_idx
            else:
                stack.append(ptr)
                for item in mat[ptr]: status[item] -= 1
                comb = list(stack)
                ret_comb_list.append(comb)
                if vb >= 1: LogInfo.logs('Find combination: %s.', comb)
        if vb >= 1: LogInfo.logs('%d combination retrieved.', len(ret_comb_list))
        return ret_comb_list


if __name__ == '__main__':
    cc = CandidateConstructor()
    mat = [[0, 1, 2, 3, 4], [1], [2], [3], [4]]
    cc.dfs_search_combinations(mat)
