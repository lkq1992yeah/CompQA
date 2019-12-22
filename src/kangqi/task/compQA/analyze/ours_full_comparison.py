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
                ours_info_dict[cur_q_idx] = {'f1': cur_f1, 'line_no': cur_sc_line}
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

    sc = sc_list[line_no-1]
    for path_idx, raw_path in enumerate(sc.raw_paths):
        category, gl_data, pred_seq = raw_path
        LogInfo.logs('Path-%d: [%s] [%s] (Linking: ["%s" --> %s %s (%s)])',
                     path_idx+1, category, pred_seq,
                     gl_data.mention, gl_data.comp, gl_data.value, gl_data.name)
    LogInfo.logs('SPARQL: %s', sc.build_sparql())

    strict_sc_list = filter(lambda _sc: schema_classification(_sc) == 0, sc_list)
    return max([sc.f1 for sc in strict_sc_list])


def work(data_name, exp_dir_1, data_dir_1, exp_dir_2, data_dir_2, out_detail_fp, out_anno_fp):
    qa_list = load_compq()
    detail_fp_1 = exp_dir_1 + '/detail/full.t.best'
    detail_fp_2 = exp_dir_2 + '/detail/full.t.best'
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
    if data_name == 'WebQ':
        range_list = range(3778, 5810)
    else:
        assert data_name == 'CompQ'
        range_list = range(1300, 2100)
    for q_idx in range_list:
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
    compare_list.sort(key=lambda x: qidx_meta_dict_1[x]['f1'] - qidx_meta_dict_2[x]['f1'])
    for q_idx in compare_list:
        info_dict_1 = qidx_meta_dict_1[q_idx]
        info_dict_2 = qidx_meta_dict_2[q_idx]
        f1_1 = info_dict_1['f1']
        f1_2 = info_dict_2['f1']
        delta = f1_1 - f1_2
        disc.convert(delta)
        qa = qa_list[q_idx]
        LogInfo.logs('============================\n')
        LogInfo.begin_track('Q-%04d: [%s]', q_idx, qa['utterance'])
        LogInfo.logs('f1_1 = %.6f, f1_2 = %.6f, delta = %.6f', f1_1, f1_2, delta)
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
        bw_anno.write('f1_1 = %.6f, f1_2 = %.6f, delta = %.6f\n' % (f1_1, f1_2, delta))
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
            time_f1_list[0].append(f1_1)
            time_f1_list[1].append(f1_2)
        else:
            nontime_f1_list[0].append(f1_1)
            nontime_f1_list[1].append(f1_2)
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
    # 1: good, 2: bad
    data_name = 'CompQ'
    exp_dir_1 = 'runnings/CompQ/180519_strict/all__full__180508_K05_Fhalf__dsFalse__drop0.0/' \
                'NFix-20__GRU150_BH_mSum__b32__fbpFalse'
    data_dir_1 = 'runnings/candgen_CompQ/180508_K05_Fhalf/data'
    exp_dir_2 = 'runnings/CompQ/180520_strict/all__full__180508_K05_Fhalf__dsFalse__drop0.0/' \
                'NFix-20__GRU150_BH_mSum_separated__b32__fbpFalse'
    data_dir_2 = 'runnings/candgen_CompQ/180508_K05_Fhalf/data'
    out_detail_fp = 'logs/EMNLP18/CompQ_compact_separated.detail'
    out_anno_fp = 'logs/EMNLP18/CompQ_compact_separated.anno'
    work(data_name=data_name, exp_dir_1=exp_dir_1, data_dir_1=data_dir_1,
         exp_dir_2=exp_dir_2, data_dir_2=data_dir_2,
         out_detail_fp=out_detail_fp, out_anno_fp=out_anno_fp)


if __name__ == '__main__':
    main()
