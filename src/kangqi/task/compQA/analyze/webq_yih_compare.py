import json
import codecs
import linecache

from ..dataset.u import load_webq
from ..dataset.kq_schema import CompqSchema
from ..candgen_acl18.global_linker import LinkData

from kangqi.util.discretizer import Discretizer
from kangqi.util.LogUtil import LogInfo

from kangqi.task.compQA.eval.official_eval import compute_f1


def retrieve_schema(q_idx, link_fp, schema_fp, line_no):
    gather_linkings = []
    with codecs.open(link_fp, 'r', 'utf-8') as br:
        for gl_line in br.readlines():
            tup_list = json.loads(gl_line.strip())
            ld_dict = {k: v for k, v in tup_list}
            gather_linkings.append(LinkData(**ld_dict))
    schema_line = linecache.getline(schema_fp, lineno=line_no).strip()
    sc = CompqSchema.read_schema_from_json(q_idx=q_idx, json_line=schema_line,
                                           gather_linkings=gather_linkings,
                                           use_ans_type_dist=False,
                                           placeholder_policy='ActiveOnly',
                                           full_constr=False, fix_dir=False)
    return sc


def single_question(q_idx, qa, data_dir, line_no, yih_answer_list, ours_f1):
    div = q_idx / 100
    sub_dir = '%d-%d' % (div * 100, div * 100 + 99)
    schema_fp = '%s/data/%s/%d_schema' % (data_dir, sub_dir, q_idx)
    link_fp = '%s/data/%s/%d_links' % (data_dir, sub_dir, q_idx)
    sc = retrieve_schema(q_idx=q_idx, link_fp=link_fp, schema_fp=schema_fp, line_no=line_no)
    if abs(ours_f1 - sc.f1) > 1e-6:
        LogInfo.logs('Warning: F1 mismatch, Q = %d, line_no = %d, '
                     'recorded f1 = %.6f, schema f1 = %.6f',
                     q_idx, line_no, ours_f1, sc.f1)

    LogInfo.begin_track('Q-%d: [%s]', q_idx, qa['utterance'])
    gold_list = qa['targetValue']
    _, _, yih_f1 = compute_f1(goldList=gold_list, predictedList=yih_answer_list)
    ours_f1 = sc.f1
    delta = ours_f1 - yih_f1
    LogInfo.logs('F1 Delta: Ours = %.6f, Yih = %.6f, Delta = %.6f', ours_f1, yih_f1, delta)
    LogInfo.logs('Gold Answer: %s', qa['targetValue'])
    LogInfo.logs(' Yih Answer: %s', yih_answer_list)
    LogInfo.begin_track('Our schema, line_no = %d', line_no)
    for path_idx, raw_path in enumerate(sc.raw_paths):
        category, gl_data, pred_seq = raw_path
        LogInfo.logs('Path-%d: [%s] [%s] (Linking: ["%s" --> %s %s (%s)])',
                     path_idx+1, category, pred_seq,
                     gl_data.mention, gl_data.comp, gl_data.value, gl_data.name)
    LogInfo.end_track()
    LogInfo.logs('SPARQL: %s', sc.build_sparql())
    LogInfo.end_track()
    LogInfo.logs('Anno: []')

    LogInfo.logs()
    LogInfo.logs('================================')
    LogInfo.logs()


def work(exp_dir, data_dir, best_epoch, qa_list, yih_ret_dict):
    log_fp = '%s/yih_compare_%03d.txt' % (exp_dir, best_epoch)

    pick_sc_dict = {q_idx: (-1, 0.) for q_idx in range(3778, 5810)}
    ret_fp = '%s/result/full.t.%03d' % (exp_dir, best_epoch)
    with open(ret_fp, 'r') as br:
        for line in br.readlines():
            spt = line.strip().split('\t')
            q_idx = int(spt[0])
            line_no = int(spt[1])
            ours_f1 = float(spt[2])
            pick_sc_dict[q_idx] = (line_no, ours_f1)

    disc = Discretizer([-0.99, -0.50, -0.25, -0.01, 0.01, 0.25, 0.50, 0.99])
    delta_tup_list = []
    avg_yih_f1 = 0.
    avg_ours_f1 = 0.
    for q_idx in range(3778, 5810):
        qa = qa_list[q_idx]
        q = qa['utterance']
        gold_answer_list = qa['targetValue']
        yih_answer_list = json.loads(yih_ret_dict[q])
        _, _, yih_f1 = compute_f1(goldList=gold_answer_list, predictedList=yih_answer_list)
        ours_f1 = pick_sc_dict[q_idx][1]
        avg_yih_f1 += yih_f1
        avg_ours_f1 += ours_f1
        delta = ours_f1 - yih_f1
        disc.convert(delta)
        delta_tup_list.append((q_idx, delta))
    avg_yih_f1 /= 2032
    avg_ours_f1 /= 2032

    delta_tup_list.sort(key=lambda _tup: _tup[1])
    LogInfo.logs('%d questions delta sorted.', len(delta_tup_list))

    total_size = len(delta_tup_list)
    worse_size = len(filter(lambda _tup: _tup[1] < 0., delta_tup_list))
    better_size = len(filter(lambda _tup: _tup[1] > 0., delta_tup_list))
    equal_size = total_size - worse_size - better_size

    bw = codecs.open(log_fp, 'w', 'utf-8')
    LogInfo.redirect(bw)
    LogInfo.logs('Avg_Yih_F1 = %.6f, Avg_Ours_F1 = %.6f', avg_yih_f1, avg_ours_f1)
    LogInfo.logs(' Worse cases = %d (%.2f%%)', worse_size, 100. * worse_size / total_size)
    LogInfo.logs(' Equal cases = %d (%.2f%%)', equal_size, 100. * equal_size / total_size)
    LogInfo.logs('Better cases = %d (%.2f%%)', better_size, 100. * better_size / total_size)
    disc.show_distribution()
    LogInfo.logs()
    for q_idx, _ in delta_tup_list:
        qa = qa_list[q_idx]
        line_no, ours_f1 = pick_sc_dict[q_idx]
        q = qa['utterance']
        yih_answer_list = json.loads(yih_ret_dict[q])
        if line_no == -1:
            continue
        single_question(q_idx=q_idx, qa=qa,
                        data_dir=data_dir, line_no=line_no,
                        yih_answer_list=yih_answer_list,
                        ours_f1=ours_f1)
    LogInfo.stop_redirect()


def main():
    qa_list = load_webq()
    yih_ret_fp = 'codalab/WebQ/acl2015-msr-stagg/test_predict.txt'
    yih_ret_dict = {}
    with codecs.open(yih_ret_fp, 'r', 'utf-8') as br:
        for line in br.readlines():
            k, v = line.strip().split('\t')
            yih_ret_dict[k] = v
    LogInfo.logs('Yih result collected.')

    exp_tup_list = [
        ('180514_strict/all__full__180508_K03_Fhalf__depSimulate/'
         'NFix-20__wUpd_RH_qwOnly_compact__b32__fbpFalse', '180508_K03_Fhalf', 10),
        ('180516_strict/all__full__180508_K03_Fhalf__Lemmatize/'
         'NFix-20__wUpd_RH_qwOnly_compact__b32__fbpFalse', '180508_K03_Fhalf', 15),
        ('180516_strict/all__full__180508_K03_Fhalf__Lemmatize/'
         'NFix-20__wUpd_BH_qwOnly_compact__b32__fbpFalse', '180508_K03_Fhalf', 12)
    ]
    for exp_suf, data_suf, best_epoch in exp_tup_list:
        exp_dir = 'runnings/WebQ/' + exp_suf
        data_dir = 'runnings/candgen_WebQ/' + data_suf
        LogInfo.begin_track('Dealing with [%s], epoch = %03d:', exp_suf, best_epoch)
        work(exp_dir=exp_dir, data_dir=data_dir, best_epoch=best_epoch,
             qa_list=qa_list, yih_ret_dict=yih_ret_dict)
        LogInfo.end_track()


if __name__ == '__main__':
    LogInfo.begin_track('[webq_yih_compare] starts ... ')
    main()
    LogInfo.end_track()
