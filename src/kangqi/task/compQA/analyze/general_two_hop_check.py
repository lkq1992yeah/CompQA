"""
Author: Kangqi Luo
Goal: Investigate how much F1 gain do general 2-hop schemas bring to us.
strict: 1-hop or 2-hop with mediator found in mediators.tsv
elegant: allow pred1.range == pred2.domain
coherent: allow pred2.domain to be the super type of pred1.range
general: allow all 2-hop predicate sequences
"""

import codecs
import json
import sys

from ..candgen_acl18.global_linker import LinkData
from ..dataset.kq_schema import CompqSchema
from ..dataset.ds_helper.dataset_schema_reader import schema_classification

from ..util.fb_helper import get_pred_name
from kangqi.util.LogUtil import LogInfo
from kangqi.util.time_track import TimeTracker as Tt


# super_type_dict = load_super_type_dict()

STRICT = 0
ELEGANT = 1
COHERENT = 2
GENERAL = 3


# def schema_classification(sc):
#     for category, focus, pred_seq in sc.raw_paths:
#         if category != 'Main':
#             continue  # only consider main path
#         elif len(pred_seq) == 1:
#             return STRICT
#         elif is_mediator_as_expect(pred=pred_seq[0]):
#             return STRICT
#         else:
#             p1_range = get_range(pred_seq[0])
#             p2_domain = get_domain(pred_seq[1])
#             if p1_range == p2_domain:
#                 return ELEGANT
#             elif p1_range in super_type_dict and p2_domain in super_type_dict[p1_range]:
#                 return COHERENT
#             else:
#                 return GENERAL
#     return GENERAL

sc_len_list = [0] * 10


def read_schemas_from_single_file(q_idx, schema_fp, gather_linkings, sc_max_len):
    """
    Read schemas from file, seperate them into several groups
    """
    general_list = []
    strict_list = []            # 2-hop must be mediator
    elegant_list = []           # 2-hop allows pred1.range == pred2.domain
    coherent_list = []          # 2-hop allows pred1.range \in pred2.domain ()
    global_lists = [strict_list, elegant_list, coherent_list, general_list]

    used_pred_name_dict = {}
    with codecs.open(schema_fp, 'r', 'utf-8') as br:
        lines = br.readlines()
        for ori_idx, line in enumerate(lines):
            Tt.start('read_single_line')
            sc = CompqSchema.read_schema_from_json(q_idx, json_line=line,
                                                   gather_linkings=gather_linkings, use_ans_type_dist=False,
                                                   placeholder_policy='ActiveOnly', full_constr=True,
                                                   fix_dir=True)
            Tt.record('read_single_line')
            sc.ori_idx = ori_idx
            # Tt.start('construct_path')
            # sc.construct_path_list()        # create the path_list on-the-fly
            # Tt.record('construct_path')
            sc_len_list[len(sc.raw_paths)] += 1
            for _, _, pred_seq in sc.raw_paths:
                for pred in pred_seq:
                    if pred not in used_pred_name_dict:
                        used_pred_name_dict[pred] = get_pred_name(pred)

            if len(sc.raw_paths) > sc_max_len:
                continue
            Tt.start('classification')
            sc_class = schema_classification(sc)
            global_lists[sc_class].append(sc)
            Tt.record('classification')
            # if q_idx == 1353:
            #     LogInfo.logs('Q-idx = %4d, Line = %4d, category = %d', q_idx, ori_idx, sc_class)
    for i in range(3):
        global_lists[i+1] += global_lists[i]
    return global_lists, used_pred_name_dict


def working_in_data(data_dir, file_list, qa_list, sc_max_len=3):

    save_fp = data_dir + '/log.schema_check'
    bw = open(save_fp, 'w')
    LogInfo.redirect(bw)

    LogInfo.begin_track('Working in %s, with schemas in %s:', data_dir, file_list)
    with open(data_dir + '/' + file_list, 'r') as br:
        lines = br.readlines()

    used_pred_tup_list = []
    stat_tup_list = []
    for line_idx, line in enumerate(lines):
        if line_idx % 100 == 0:
            LogInfo.logs('Scanning %d / %d ...', line_idx, len(lines))
        line = line.strip()
        q_idx = int(line.split('/')[2].split('_')[0])
        # if q_idx != 1353:
        #     continue
        schema_fp = data_dir + '/' + line
        link_fp = schema_fp.replace('schema', 'links')
        Tt.start('read_linkings')
        gather_linkings = []
        with codecs.open(link_fp, 'r', 'utf-8') as br:
            for gl_line in br.readlines():
                tup_list = json.loads(gl_line.strip())
                ld_dict = {k: v for k, v in tup_list}
                gather_linkings.append(LinkData(**ld_dict))
        Tt.record('read_linkings')
        Tt.start('read_schema')
        global_lists, used_pred_name_dict = \
            read_schemas_from_single_file(q_idx, schema_fp, gather_linkings, sc_max_len)
        Tt.record('read_schema')
        stat_tup_list.append((q_idx, global_lists))
        used_pred_tup_list.append((q_idx, used_pred_name_dict))

    stat_tup_list.sort(key=lambda _tup: max_f1(_tup[1][ELEGANT]) - max_f1(_tup[1][STRICT]), reverse=True)

    LogInfo.logs('sc_len distribution: %s', sc_len_list)

    LogInfo.logs('Rank\tQ_idx\tstri.\teleg.\tcohe.\tgene.'
                 '\tstri._F1\teleg._F1\tcohe._F1\tgene._F1\tUtterance')
    for rank, stat_tup in enumerate(stat_tup_list):
        q_idx, global_lists = stat_tup
        size_list = ['%4d' % len(cand_list) for cand_list in global_lists]
        max_f1_list = ['%8.6f' % max_f1(cand_list) for cand_list in global_lists]
        size_str = '\t'.join(size_list)
        max_f1_str = '\t'.join(max_f1_list)
        LogInfo.logs('%4d\t%4d\t%s\t%s\t%s',
                     rank+1, q_idx, size_str, max_f1_str,
                     qa_list[q_idx]['utterance'].encode('utf-8'))

    q_size = len(stat_tup_list)
    f1_upperbound_list = []
    avg_cand_size_list = []
    found_entity_list = []
    for index in range(4):
        local_max_f1_list = []
        local_cand_size_list = []
        for _, global_lists in stat_tup_list:
            sc_list = global_lists[index]
            local_max_f1_list.append(max_f1(sc_list))
            local_cand_size_list.append(len(sc_list))
        local_found_entity = len(filter(lambda y: y > 0., local_max_f1_list))
        f1_upperbound_list.append(1.0 * sum(local_max_f1_list) / q_size)
        avg_cand_size_list.append(1.0 * sum(local_cand_size_list) / q_size)
        found_entity_list.append(local_found_entity)

    LogInfo.logs('  strict: avg size = %.6f, upp F1 = %.6f',
                 avg_cand_size_list[STRICT], f1_upperbound_list[STRICT])
    for name, index in zip(('strict', 'elegant', 'coherent', 'general'),
                           (STRICT, ELEGANT, COHERENT, GENERAL)):
        avg_size = avg_cand_size_list[index]
        ratio = 1. * avg_size / avg_cand_size_list[STRICT] if avg_cand_size_list[STRICT] > 0 else 0.
        found_entity = found_entity_list[index]
        upp_f1 = f1_upperbound_list[index]
        gain = upp_f1 - f1_upperbound_list[STRICT]
        LogInfo.logs('%8s: avg size = %.6f (%.2fx), found entity = %d, '
                     'upp F1 = %.6f, gain = %.6f',
                     name, avg_size, ratio, found_entity, upp_f1, gain)
    LogInfo.end_track()

    bw.close()
    LogInfo.stop_redirect()

    used_pred_fp = data_dir + '/log.used_pred_stat'
    bw = codecs.open(used_pred_fp, 'w', 'utf-8')
    LogInfo.redirect(bw)
    ratio_list = []
    for q_idx, used_pred_name_dict in used_pred_tup_list:
        unique_id_size = len(set(used_pred_name_dict.keys()))
        unique_name_size = len(set(used_pred_name_dict.values()))
        ratio = 1. * unique_name_size / unique_id_size if unique_id_size > 0 else 1.
        ratio_list.append(ratio)
        LogInfo.logs('Q = %4d, unique id = %d, unique name = %d, ratio = %.4f',
                     q_idx, unique_id_size, unique_name_size, ratio)
    avg_ratio = sum(ratio_list) / len(ratio_list)
    LogInfo.logs('avg_ratio = %.4f', avg_ratio)
    LogInfo.stop_redirect()


def max_f1(sc_list):
    return max([sc.f1 for sc in sc_list]) if len(sc_list) > 0 else 0.


def calc_f1_gain(tup, target_index):
    """
    Return the strict/elegant/coherent/general_f1_gain of these candidates
    """
    q_idx, global_lists = tup
    f1_gain = max_f1(global_lists[target_index]) - max_f1(global_lists[0])
    return f1_gain


def main(data_dir):
    from ..dataset.u import load_compq, load_webq, load_simpq
    # data_dir = 'runnings/candgen_CompQ/180308_SMART_Fyes'
    file_list = 'all_list'
    if 'CompQ' in data_dir:
        qa_list = load_compq()
    elif 'WebQ' in data_dir:
        qa_list = load_webq()
    else:
        assert 'SimpQ' in data_dir
        qa_list = load_simpq()
    working_in_data(data_dir, file_list, qa_list)


if __name__ == '__main__':
    main(data_dir=sys.argv[1])
    Tt.display()
