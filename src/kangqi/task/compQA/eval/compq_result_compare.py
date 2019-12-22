import codecs
import json
import linecache

from ..candgen_acl18.global_linker import LinkData
from ..dataset.u import load_compq
from ..dataset.kq_schema import CompqSchema
from .official_eval import compute_f1
from kangqi.util.LogUtil import LogInfo


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
                ours_info_dict[cur_q_idx] = {'F1': cur_f1, 'sc_line': cur_sc_line}
    return ours_info_dict


def read_bao(test_predict_fp, qa_list):
    bao_info_dict = {}
    with codecs.open(test_predict_fp, 'r', 'utf-8') as br:
        lines = br.readlines()
    for line_idx, line in enumerate(lines):
        q_idx = line_idx + 1300
        spt = line.strip().split('\t')
        predict_list = json.loads(spt[1])
        gold_list = qa_list[q_idx]['targetValue']
        lower_predict_list = [x.lower() for x in predict_list]
        lower_gold_list = [x.lower() for x in gold_list]
        r, p, f1 = compute_f1(lower_gold_list, lower_predict_list)
        bao_info_dict[q_idx] = {'F1': f1, 'predict_list': predict_list}
    return bao_info_dict


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
    json_line = linecache.getline(sc_fp, lineno=line_no).strip()
    sc = CompqSchema.read_schema_from_json(q_idx=q_idx, json_line=json_line,
                                           gather_linkings=gather_linkings,
                                           use_ans_type_dist=False)
    LogInfo.logs('Answer size = %d', sc.ans_size)
    LogInfo.logs('P / R / F1 = %.3f / %.3f / %.3f', sc.p, sc.r, sc.f1)
    for path_idx, raw_path in enumerate(sc.raw_paths):
        category, gl_data, pred_seq = raw_path
        LogInfo.logs('Path-%d: [%s] [%s] [%s %s (%s)]',
                     path_idx+1, category, gl_data.mention, gl_data.comp, gl_data.value, gl_data.name)
        LogInfo.logs('        %s', pred_seq)
    LogInfo.logs('SPARQL: %s', sc.build_sparql())


def main(qa_list, detail_fp, out_eval_fp, out_anno_fp, bao_predict_fp, data_dir):

    bao_info_dict = read_bao(bao_predict_fp, qa_list)
    our_info_dict = read_ours(detail_fp)
    for q_idx in range(1300, 2100):
        if q_idx not in our_info_dict:
            our_info_dict[q_idx] = {'F1': 0., 'sc_line': -1}
    disp_list = range(1300, 2100)
    disp_list.sort(key=lambda idx: our_info_dict[idx]['F1'])
    with open(out_eval_fp, 'w') as bw_eval, open(out_anno_fp, 'w') as bw_anno:
        LogInfo.redirect(bw_eval)
        LogInfo.begin_track('Showing comparison: ')
        LogInfo.logs('Our: %s', detail_fp)
        LogInfo.logs('Bao: %s', bao_predict_fp)
        for q_idx in disp_list:
            LogInfo.logs('')
            LogInfo.logs('==================================================================')
            LogInfo.logs('')
            LogInfo.begin_track('Q-%d: [%s]', q_idx, qa_list[q_idx]['utterance'].encode('utf-8'))
            our_f1 = our_info_dict[q_idx]['F1']
            bao_f1 = bao_info_dict[q_idx]['F1']
            LogInfo.logs('F1_gain = %.6f, Our_F1 = %.6f, Bao_F1 = %.6f',
                         our_f1 - bao_f1, our_f1, bao_f1)
            LogInfo.logs('Gold: %s', qa_list[q_idx]['targetValue'])
            LogInfo.logs('Bao: %s', bao_info_dict[q_idx]['predict_list'])
            LogInfo.begin_track('Our sc_line: %d', our_info_dict[q_idx]['sc_line'])
            retrieve_schema(data_dir=data_dir, q_idx=q_idx, line_no=our_info_dict[q_idx]['sc_line'])
            LogInfo.end_track()     # end of schema display
            LogInfo.end_track()     # end of the question

            bw_anno.write('Q-%04d\t%s\t%s\n' % (q_idx, qa_list[q_idx]['utterance'].encode('utf-8'),
                                                qa_list[q_idx]['targetValue']))
            bw_anno.write('Our_F1: %.6f\n' % our_f1)
            bw_anno.write('1: []\n')
            bw_anno.write('2: []\n')
            bw_anno.write('3: []\n')
            bw_anno.write('4: []\n')
            bw_anno.write('5: []\n')
            bw_anno.write('\n\n')

        LogInfo.end_track()
        LogInfo.stop_redirect()


if __name__ == '__main__':
    exp_dir = 'runnings/CompQ/180221_all_strict/MultiTask_full_fix_direction/' \
              'tTrue__ab2_compact_dot_H0.5_b32_k1.0__fbpFalse'
    main(qa_list=load_compq(),
         detail_fp=exp_dir+'/detail/full.t.best',
         out_eval_fp=exp_dir+'/log.eval',
         out_anno_fp=exp_dir+'/log.anno',
         bao_predict_fp='codalab/CompQ/Bao2016/test_predict.txt',
         data_dir='runnings/candgen_CompQ/180209_ACL18_SMART/data')
