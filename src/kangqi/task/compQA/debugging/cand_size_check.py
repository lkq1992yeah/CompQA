"""
Author: Kangqi Luo
Goal: Check candidate size, and compare between old and new versions.
"""

import codecs

from kangqi.util.LogUtil import LogInfo


def traverse_questions(main_fp, q_start, q_end):
    cand_size_dict = {}
    LogInfo.begin_track('Scanning files in %s ...', main_fp)
    for q_idx in range(q_start, q_end):
        if q_idx % 500 == 0:
            LogInfo.logs('%d scanned.', q_idx)
        div = q_idx / 100
        sub_dir = '%d-%d' % (div*100, div*100+99)
        fp = '%s/data/%s/%d_schema' % (main_fp, sub_dir, q_idx)
        with codecs.open(fp, 'r', 'utf-8') as br:
            lines = br.readlines()
        cand_size_dict[q_idx] = len(lines)
    LogInfo.end_track()
    return cand_size_dict


def cmp_two(tup1, tup2):
    d1 = 1. * tup1[2] / tup1[1] if tup1[1] > 0 else 0.
    d2 = 1. * tup2[2] / tup2[1] if tup2[1] > 0 else 0.
    if d1 == d2:
        return tup1[2] - tup2[2]
    elif d1 > d2:
        return -1
    else:
        return 1


def main():
    old_fp = 'runnings/candgen_CompQ/180101_Lukov_eff'
    new_fp = 'runnings/candgen_CompQ/180101_Lukov_eff'
    old_dict, new_dict = [traverse_questions(fp, 0, 2100) for fp in (old_fp, new_fp)]

    joint_set = set(old_dict.keys()) & set(new_dict.keys())
    cmp_tups = [(k, old_dict[k], new_dict[k]) for k in joint_set]
    cmp_tups.sort(cmp=cmp_two)

    LogInfo.logs('Rank\tQ\tRatio\tOld\tNew')
    for rank, (q_idx, old_size, new_size) in enumerate(cmp_tups):
        LogInfo.logs('#%d\tQ-%d\t%.4f\t%d\t%d', rank+1, q_idx,
                     1. * new_size / old_size if old_size > 0 else 0.0,
                     old_size, new_size)


if __name__ == '__main__':
    main()
