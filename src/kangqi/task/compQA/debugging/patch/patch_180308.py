"""
Split SPARQL caches into groups
We merge 100 questions into one group,
then candidate generation process can work in 100% parallel between different groups.
"""

import os
import json
import codecs

from kangqi.util.LogUtil import LogInfo


def prepare_mid_set(data_fp):
    collected = 0
    mid_set_dict = {}  # <q_idx, mid_set>
    for q_idx in range(2100):
        sub_idx = q_idx / 100 * 100
        link_fp = '%s/%d-%d/%d_links' % (data_fp, sub_idx, sub_idx + 99, q_idx)
        mid_set = set([])
        if os.path.isfile(link_fp):
            with codecs.open(link_fp, 'r', 'utf-8') as br:
                for line in br.readlines():
                    tup_list = json.loads(line.strip())
                    ld_dict = {k: v for k, v in tup_list}
                    if ld_dict['category'] == 'Entity':
                        mid_set.add(ld_dict['value'])
        mid_set_dict[q_idx] = mid_set
        collected += len(mid_set)
    LogInfo.logs('%d mid collected. average %.3f.', collected, 1. * collected / len(mid_set_dict))
    return mid_set_dict


def main():
    data_fp = 'runnings/candgen_CompQ/180209_ACL18_SMART/data'
    fuzzy_fp = 'runnings/acl18_cache/ori_full/sparql.cache'
    q_sc_stat_fp = 'runnings/acl18_cache/ori_full/q_sc_stat.cache'

    local_fuzzy_dict = {}       # <q_idx, [fuzzy_result]>
    local_q_sc_stat_dict = {}
    mid_set_dict = prepare_mid_set(data_fp=data_fp)

    with codecs.open(fuzzy_fp, 'r', 'utf-8') as br:
        lines = br.readlines()
        LogInfo.logs('Fuzzy query loaded.')
        for line_idx, line in enumerate(lines):
            # if line_idx == 100:
            #     break
            if line_idx % 10000 == 0:
                LogInfo.logs('Fuzzy: %d / %d', line_idx, len(lines))
            sparql_mid_set = set([])
            tup = json.loads(line)
            sparql = tup[0]
            spt = sparql.split(' ')
            for s in spt:
                if s.startswith('fb:m.'):
                    sparql_mid_set.add(s[3:])
            for q_idx, mid_set in mid_set_dict.items():
                if mid_set & sparql_mid_set == sparql_mid_set:
                    local_fuzzy_dict.setdefault(q_idx, []).append(tup)

    with codecs.open(q_sc_stat_fp, 'r', 'utf-8') as br:
        lines = br.readlines()
        LogInfo.logs('Q_Sc query loaded.')
        for line_idx, line in enumerate(lines):
            # if line_idx == 100:
            #     break
            if line_idx % 10000 == 0:
                LogInfo.logs('Q_sc: %d / %d', line_idx, len(lines))
            tup = json.loads(line)
            info_spt = tup[0].split('|')
            q_idx_str, sparql = info_spt
            if 'ORDER' in sparql:
                continue
            if 'FILTER' in sparql:
                continue            # remove ordinal and time constraint - related schemas
            sparql = sparql.replace(' OPTIONAL { ', ' ')
            sparql = sparql.replace('} . }', '. }')   # remove OPTIONAL for retrieving names
            q_idx = int(q_idx_str.split('_')[-1])
            new_key = '|'.join([q_idx_str, sparql,
                                'None', '', '',     # tm_comp, tm_value, allow_forever
                                'None', '',         # ord_comp, ord_rank
                                'None'])            # aggregate
            new_tup = [new_key, tup[1]]
            local_q_sc_stat_dict.setdefault(q_idx, []).append(new_tup)

    bw_fuzzy_list = []
    bw_q_sc_stat_list = []
    for idx in range(21):
        out_fuzzy_fp = 'runnings/acl18_cache/group_cache/sparql.g%02d.cache' % idx
        out_q_sc_fp = 'runnings/acl18_cache/group_cache/q_sc_stat.g%02d.cache' % idx
        bw_fuzzy_list.append(codecs.open(out_fuzzy_fp, 'w', 'utf-8'))
        bw_q_sc_stat_list.append(codecs.open(out_q_sc_fp, 'w', 'utf-8'))

    LogInfo.logs('Saving fuzzy ...')
    for q_idx, fuzzy_cache_list in local_fuzzy_dict.items():
        if q_idx % 100 == 0:
            LogInfo.logs('Current: %d / 2100', q_idx)
        group_idx = q_idx / 100
        use_bw = bw_fuzzy_list[group_idx]
        for tup in fuzzy_cache_list:
            use_bw.write(json.dumps(tup) + '\n')

    LogInfo.logs('Saving q_sc ...')
    for q_idx, q_sc_cache_list in local_q_sc_stat_dict.items():
        if q_idx % 100 == 0:
            LogInfo.logs('Current: %d / 2100', q_idx)
        group_idx = q_idx / 100
        use_bw = bw_q_sc_stat_list[group_idx]
        for tup in q_sc_cache_list:
            use_bw.write(json.dumps(tup) + '\n')

    for bw in bw_fuzzy_list:
        bw.close()
    for bw in bw_q_sc_stat_list:
        bw.close()


if __name__ == '__main__':
    main()
