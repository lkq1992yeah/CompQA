"""
Goal: Compare relation matching results.
Output two files: top-1 compare, and annotation
"""
import re
import json
import codecs
import numpy as np

from ..dataset.u import load_compq
from ..candgen_acl18.global_linker import LinkData
from ..dataset.kq_schema import CompqSchema
from ..dataset.ds_helper.dataset_schema_reader import schema_classification

from kangqi.util.LogUtil import LogInfo
from kangqi.util.discretizer import Discretizer


def read_ours(detail_fp):
    ours_info_dict = {}
    cur_q_idx = -1
    with codecs.open(detail_fp, 'r', 'utf-8') as br:
        for line in br.readlines():
            if line.startswith('Q-'):
                cur_q_idx = int(line[2:].split(' ')[0])
            elif line.startswith('  #-0001'):
                first_rb_pos = line.find(']')
                cur_f1 = float(line[:first_rb_pos].split(' ')[-1])
                last_rb_pos = line.rfind(']')
                cur_sc_line = int(line[:last_rb_pos].split(' ')[-1])
                ours_info_dict[cur_q_idx] = {'rm_f1': cur_f1, 'line_no': cur_sc_line}
    return ours_info_dict


def retrieve_schema(data_dir, q_idx, line_no):
    if line_no == -1:
        return
    div = q_idx / 100
    sub_dir = '%d-%d' % (div * 100, div * 100 + 99)
    sc_fp = '%s/%s/%d_schema' % (data_dir, sub_dir, q_idx)
    link_fp = '%s/%s/%d_links' % (data_dir, sub_dir, q_idx)
    gather_linkings = []
    with codecs.open(link_fp, 'r', 'utf-8') as br:
        for gl_line in br.readlines():
            tup_list = json.loads(gl_line.strip())
            ld_dict = {k: v for k, v in tup_list}
            gather_linkings.append(LinkData(**ld_dict))

    sc_list = []
    with codecs.open(sc_fp, 'r', 'utf-8') as br:
        for line in br.readlines():
            json_line = line.strip()
            sc = CompqSchema.read_schema_from_json(q_idx=q_idx, json_line=json_line,
                                                   gather_linkings=gather_linkings,
                                                   use_ans_type_dist=False,
                                                   placeholder_policy='ActiveOnly',
                                                   full_constr=False, fix_dir=False)
            sc_list.append(sc)

    # LogInfo.logs('Answer size = %d', sc.ans_size)
    # LogInfo.logs('P / R / F1 = %.3f / %.3f / %.3f', sc.p, sc.r, sc.f1)
    target_sc = sc_list[line_no-1]
    for path_idx, raw_path in enumerate(target_sc.raw_paths):
        category, gl_data, pred_seq = raw_path
        # LogInfo.logs('Path-%d: [%s] [%s] [%s %s (%s)]',
        #              path_idx+1, category, gl_data.mention, gl_data.comp, gl_data.value, gl_data.name)
        # LogInfo.logs('        %s', pred_seq)
        LogInfo.logs('Path-%d: [%s] [%s] (Linking: ["%s" --> %s %s (%s)])',
                     path_idx+1, category, pred_seq,
                     gl_data.mention, gl_data.comp, gl_data.value, gl_data.name)
    # LogInfo.logs('SPARQL: %s', target_sc.build_sparql())

    strict_sc_list = filter(lambda _sc: schema_classification(_sc) == 0, sc_list)
    return max([sc.f1 for sc in strict_sc_list])


def work(exp_dir_1, exp_dir_2, data_dir_1, data_dir_2, out_detail_fp, out_anno_fp):
    qa_list = load_compq()

    detail_fp_1 = exp_dir_1 + '/detail/rm.t.best'
    detail_fp_2 = exp_dir_2 + '/detail/rm.t.best'
    qidx_meta_dict_1 = read_ours(detail_fp_1)
    qidx_meta_dict_2 = read_ours(detail_fp_2)

    bw_detail = codecs.open(out_detail_fp, 'w', 'utf-8')
    bw_anno = codecs.open(out_anno_fp, 'w', 'utf-8')
    LogInfo.redirect(bw_detail)

    for bw in (bw_detail, bw_anno):
        bw.write('detail_fp_1: [%s] --> [%s]\n' % (data_dir_1, detail_fp_1))
        bw.write('detail_fp_2: [%s] --> [%s]\n\n' % (data_dir_2, detail_fp_2))

    missing_list = []
    first_only_list = []
    second_only_list = []
    compare_list = []
    for q_idx in range(1300, 2100):
        if q_idx not in qidx_meta_dict_1 and q_idx not in qidx_meta_dict_2:
            missing_list.append(q_idx)
        elif q_idx not in qidx_meta_dict_2:
            first_only_list.append(q_idx)
        elif q_idx not in qidx_meta_dict_1:
            second_only_list.append(q_idx)
        else:
            compare_list.append(q_idx)

    LogInfo.logs('Missing questions: %s', missing_list)
    LogInfo.logs('First only questions: %s', first_only_list)
    LogInfo.logs('Second only questions: %s\n', second_only_list)

    time_f1_list = [[], []]
    nontime_f1_list = [[], []]
    mark_counter = {}
    disc = Discretizer(split_list=[-0.5, -0.1, -0.000001, 0.000001, 0.1, 0.5])
    compare_list.sort(key=lambda x: qidx_meta_dict_1[x]['rm_f1'] - qidx_meta_dict_2[x]['rm_f1'])
    for q_idx in compare_list:
        info_dict_1 = qidx_meta_dict_1[q_idx]
        info_dict_2 = qidx_meta_dict_2[q_idx]
        rm_f1_1 = info_dict_1['rm_f1']
        rm_f1_2 = info_dict_2['rm_f1']
        delta = rm_f1_1 - rm_f1_2
        disc.convert(delta)
        qa = qa_list[q_idx]
        LogInfo.logs('============================\n')
        LogInfo.begin_track('Q-%04d: [%s]', q_idx, qa['utterance'])
        LogInfo.logs('rm_f1_1 = %.6f, rm_f1_2 = %.6f, delta = %.6f', rm_f1_1, rm_f1_2, delta)
        upb_list = []
        for d_idx, (data_dir, info_dict) in enumerate([(data_dir_1, info_dict_1),
                                                       (data_dir_2, info_dict_2)]):
            LogInfo.begin_track('Schema-%d, line = %d', d_idx, info_dict['line_no'])
            upb = retrieve_schema(data_dir, q_idx, info_dict['line_no'])
            upb_list.append(upb)
            LogInfo.end_track()
        LogInfo.end_track()
        LogInfo.logs('')

        bw_anno.write('Q-%04d: [%s]\n' % (q_idx, qa['utterance']))
        bw_anno.write('rm_f1_1 = %.6f, rm_f1_2 = %.6f, delta = %.6f\n' % (rm_f1_1, rm_f1_2, delta))
        if abs(delta) >= 0.5:
            hml = 'H'
        elif abs(delta) >= 0.1:
            hml = 'M'
        elif abs(delta) >= 1e-6:
            hml = 'L'
        else:
            hml = '0'
        if delta >= 1e-6:
            sgn = '+'
        elif delta <= -1e-6:
            sgn = '-'
        else:
            sgn = ''
        bw_anno.write('# Change: [%s%s]\n' % (sgn, hml))
        has_time = 'N'
        for tok in qa['tokens']:
            if re.match('^[1-2][0-9][0-9][0-9]$', tok.token[:4]):
                has_time = 'Y'
                break
        if has_time == 'Y':
            time_f1_list[0].append(rm_f1_1)
            time_f1_list[1].append(rm_f1_2)
        else:
            nontime_f1_list[0].append(rm_f1_1)
            nontime_f1_list[1].append(rm_f1_2)
        bw_anno.write('# Time: [%s]\n' % has_time)
        upb1, upb2 = upb_list
        if upb1 - upb2 <= -1e-6:
            upb_mark = 'Less'
        elif upb1 - upb2 >= 1e-6:
            upb_mark = 'Greater'
        else:
            upb_mark = 'Equal'
        bw_anno.write('# Upb: [%s] (%.3f --> %.3f)\n' % (upb_mark, upb1, upb2))
        overall = '%s%s_%s_%s' % (sgn, hml, has_time, upb_mark)
        mark_counter[overall] = 1 + mark_counter.get(overall, 0)
        bw_anno.write('# Overall: [%s]\n' % overall)
        bw_anno.write('\n\n')

    disc.show_distribution()

    LogInfo.logs('')
    for has_time in ('Y', 'N'):
        LogInfo.logs('Related to DateTime: [%s]', has_time)
        LogInfo.logs('    \tLess\tEqual\tGreater')
        for hml in ('-H', '-M', '-L', '0', '+L', '+M', '+H'):
            line = '%4s' % hml
            for upb_mark in ('Less', 'Equal', 'Greater'):
                overall = '%s_%s_%s' % (hml, has_time, upb_mark)
                count = mark_counter.get(overall, 0)
                line += '\t%4d' % count
                # LogInfo.logs('[%s]: %d (%.2f%%)', overall, count, 100. * count / 800)
            LogInfo.logs(line)
        LogInfo.logs('')
    LogInfo.logs('DateTime-related F1: %.6f v.s. %.6f, size = %d',
                 np.mean(time_f1_list[0]), np.mean(time_f1_list[1]), len(time_f1_list[0]))
    LogInfo.logs('DateTime-not-related F1: %.6f v.s. %.6f, size = %d',
                 np.mean(nontime_f1_list[0]), np.mean(nontime_f1_list[1]), len(nontime_f1_list[0]))

    LogInfo.stop_redirect()

    bw_detail.close()
    bw_anno.close()


def main():
    exp_id_list = [
        # 'CompQ-M069'
        'CompQ-E025'
    ]
    exp_dir_list = [
        # '180316_all_strict/MultiTask_rm__180308_SMART_Fhalf/N0.100-20__NoAtt_avg_H0.2__b32'
        '180414_all_strict/MultiTask_full__180415_EM_Fhalf/N0.100-20__fSource__att_fwbw_H0.5__fbpFalse__b32'
    ]
    data_dir_list = [
        # '180308_SMART_Fhalf'
        '180415_EM_Fhalf'
    ]

    for exp_id, exp_dir, data_dir in zip(exp_id_list, exp_dir_list, data_dir_list):
        exp_dir_1 = 'runnings/CompQ/' + exp_dir
        data_dir_1 = 'runnings/candgen_CompQ/' + data_dir + '/data'
        out_detail_fp = 'logs/EMNLP18/rm_%s_M043.detail' % exp_id
        out_anno_fp = 'logs/EMNLP18/rm_%s_M043.anno' % exp_id

        work(exp_dir_1=exp_dir_1,
             exp_dir_2='runnings/CompQ/18_02/180221_all_strict/'
                       'MultiTask_rm_fix_direction/ab2_compact_dot_H0.5_b32_k1.0',
             data_dir_1=data_dir_1,
             data_dir_2='runnings/candgen_CompQ/180209_ACL18_SMART/data',
             out_detail_fp=out_detail_fp,
             out_anno_fp=out_anno_fp)

    # # exp_dir_1: CompQ-M044
    # # exp_dir_2: CompQ-M043
    # work(exp_dir_1='runnings/CompQ/180309_all_strict/MultiTask_rm/cFalse_N0.100__ab2_compact_pwOnly_dot_H0.5__b32',
    #      exp_dir_2='runnings/CompQ/180221_all_strict/MultiTask_rm_fix_direction/ab2_compact_dot_H0.5_b32_k1.0',
    #      data_dir_1='runnings/candgen_CompQ/180308_SMART_Fdyn-co/data',
    #      data_dir_2='runnings/candgen_CompQ/180209_ACL18_SMART/data',
    #      out_detail_fp='rm_M044_M043.detail',
    #      out_anno_fp='rm_M044_M043.anno')

    # # exp_dir_1: CompQ-M055
    # # exp_dir_2: CompQ-M043
    # work(exp_dir_1='runnings/CompQ/180312_all_strict/MultiTask_rm__180308_SMART_Fhalf/'
    #                'cFalse_N0.100__ab2_compact_atdFalse_pwOnly_dot_H0.5__b32',
    #      exp_dir_2='runnings/CompQ/180221_all_strict/MultiTask_rm_fix_direction/ab2_compact_dot_H0.5_b32_k1.0',
    #      data_dir_1='runnings/candgen_CompQ/180308_SMART_Fhalf/data',
    #      data_dir_2='runnings/candgen_CompQ/180209_ACL18_SMART/data',
    #      out_detail_fp='rm_M055_M043.detail',
    #      out_anno_fp='rm_M055_M043.anno')

    # # exp_dir_1: CompQ-M056
    # # exp_dir_2: CompQ-M043
    # work(exp_dir_1='runnings/CompQ/180312_all_strict/MultiTask_rm__180308_SMART_Fhalf-co/'
    #                'cFalse_N0.100__ab2_compact_atdFalse_pwOnly_dot_H0.5__b32',
    #      exp_dir_2='runnings/CompQ/180221_all_strict/MultiTask_rm_fix_direction/ab2_compact_dot_H0.5_b32_k1.0',
    #      data_dir_1='runnings/candgen_CompQ/180308_SMART_Fhalf-co/data',
    #      data_dir_2='runnings/candgen_CompQ/180209_ACL18_SMART/data',
    #      out_detail_fp='rm_M056_M043.detail',
    #      out_anno_fp='rm_M056_M043.anno')

    # # exp_dir_1: CompQ-M059
    # # exp_dir_2: CompQ-M043
    # work(exp_dir_1='runnings/CompQ/180313_all_strict/MultiTask_rm__180308_SMART_Fhalf/'
    #                'N0.100-10000__ab2_compact_atdFalse_pwOnly_dot_H0.2__b32',
    #      exp_dir_2='runnings/CompQ/180221_all_strict/MultiTask_rm_fix_direction/ab2_compact_dot_H0.5_b32_k1.0',
    #      data_dir_1='runnings/candgen_CompQ/180308_SMART_Fhalf/data',
    #      data_dir_2='runnings/candgen_CompQ/180209_ACL18_SMART/data',
    #      out_detail_fp='logs/COLING18/rm_M059_M043.detail',
    #      out_anno_fp='logs/COLING18/rm_M059_M043.anno')

    # # exp_dir_1: CompQ-M065
    # # exp_dir_2: CompQ-M043
    # work(exp_dir_1='runnings/CompQ/180315_all_strict/MultiTask_rm__180314_SMART_Fhalf_enrich/'
    #                'N0.100-20__phActiveOnly_H0.2__b32',
    #      exp_dir_2='runnings/CompQ/180221_all_strict/MultiTask_rm_fix_direction/ab2_compact_dot_H0.5_b32_k1.0',
    #      data_dir_1='runnings/candgen_CompQ/180314_SMART_Fhalf_enrich/data',
    #      data_dir_2='runnings/candgen_CompQ/180209_ACL18_SMART/data',
    #      out_detail_fp='logs/COLING18/rm_M065_M043.detail',
    #      out_anno_fp='logs/COLING18/rm_M065_M043.anno')

    # # exp_dir_1: CompQ-M072
    # # exp_dir_2: CompQ-M043
    # work(exp_dir_1='runnings/CompQ/180316_all_strict/MultiTask_rm__180314_SMART_Fhalf_enrich/'
    #                'N0.100-20__NoAtt_avg_H0.2__b32',
    #      exp_dir_2='runnings/CompQ/180221_all_strict/MultiTask_rm_fix_direction/ab2_compact_dot_H0.5_b32_k1.0',
    #      data_dir_1='runnings/candgen_CompQ/180314_SMART_Fhalf_enrich/data',
    #      data_dir_2='runnings/candgen_CompQ/180209_ACL18_SMART/data',
    #      out_detail_fp='logs/COLING18/rm_M072_M043.detail',
    #      out_anno_fp='logs/COLING18/rm_M072_M043.anno')

# TODO: show evidence of each possible schema ??
# TODO: upb_rm_f1, predict_rm_f1, delta, positive supports of path, positive supports of sub-path


if __name__ == '__main__':
    main()
