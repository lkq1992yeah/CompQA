"""
Author: Kangqi Luo
Goal: Given testing schema selection, collect the final answers from ***_ans file
      Prepare for the final P/R/F1 calculation.
"""

import json
import codecs
import argparse
import linecache

from ..dataset.u import load_webq, load_compq

from kangqi.util.LogUtil import LogInfo


parser = argparse.ArgumentParser(description='Final Result Generation')

parser.add_argument('--data_dir')
parser.add_argument('--result_dir')
parser.add_argument('--epoch', type=int)
parser.add_argument('--data_name', default='WebQ')
parser.add_argument('--q_st', type=int, default=-1)
parser.add_argument('--q_ed', type=int, default=-1)


def answer_collection(args):
    assert args.data_name in ('WebQ', 'CompQ')
    result_dir = args.result_dir
    data_dir = args.data_dir
    epoch = args.epoch
    st = args.q_st
    ed = args.q_ed
    if args.data_name == 'WebQ':
        qa_list = load_webq()
        if st == ed == -1:
            st = 3778
            ed = 5810
    else:
        qa_list = load_compq()
        if st == ed == -1:
            st = 1300
            ed = 2100
    select_schema_fp = '%s/test_schema_%03d.txt' % (result_dir, epoch)
    select_dict = {}        # <q_idx, sc.ori_idx>
    test_predict_fp = '%s/test_predict_%03d.txt' % (result_dir, epoch)

    LogInfo.begin_track('Collecting answers from %s ...', select_schema_fp)
    LogInfo.logs('Original schema data: %s', data_dir)
    with open(select_schema_fp, 'r') as br:
        for line in br.readlines():
            spt = line.strip().split('\t')
            select_dict[int(spt[0])] = int(spt[1])

    save_tups = []
    for scan_idx, q_idx in enumerate(range(st, ed)):
        if scan_idx % 50 == 0:
            LogInfo.logs('%d questions scanned.', scan_idx)
        utterance = qa_list[q_idx]['utterance']
        sc_idx = select_dict.get(q_idx, -1)
        if sc_idx == -1:
            ans_list = []       # no answer at all
        else:
            div = q_idx / 100
            sub_dir = '%d-%d' % (div*100, div*100+99)
            ans_fp = '%s/%s/%d_ans' % (data_dir, sub_dir, q_idx)
            ans_line = linecache.getline(ans_fp, lineno=sc_idx+1).strip()
            # with codecs.open(ans_fp, 'r', 'utf-8') as br:
            #     for skip_tms in range(sc_idx):      # skip rows before the target schema line
            #         br.readline()
            #     ans_line = br.readline().strip()
            LogInfo.logs('q_idx=%d, lineno=%d, content=[%s]', q_idx, sc_idx+1, ans_line)
            if args.data_name == 'CompQ':
                ans_line = ans_line.lower()
            ans_list = json.loads(ans_line)
        save_tups.append((utterance, ans_list))

    with codecs.open(test_predict_fp, 'w', 'utf-8') as bw:
        for utterance, ans_list in save_tups:
            bw.write(utterance + '\t')
            json.dump(ans_list, bw)
            bw.write('\n')
    LogInfo.end_track('Predicting results saved to %s.', test_predict_fp)


if __name__ == '__main__':
    _args = parser.parse_args()
    answer_collection(_args)
