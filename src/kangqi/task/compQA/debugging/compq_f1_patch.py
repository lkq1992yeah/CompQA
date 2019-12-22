"""
Author: Kangqi Luo
Goal: Re-calculate the F1 of complex questions.
Due to inconsistent cases in CompQ's gold answer set, we have to remove cases.
"""

import json
import codecs

from ..dataset.u import load_compq
from kangqi.util.LogUtil import LogInfo


def calc(gold_set, pred_set):
    joint_set = pred_set & gold_set
    p = 1. * len(joint_set) / len(pred_set) if len(pred_set) > 0 else 0.
    r = 1. * len(joint_set) / len(gold_set) if len(gold_set) > 0 else 0.
    f1 = 2 * p * r / (p + r) if p > 0. else 0.
    return p, r, f1


def process_single_question(q_idx, sc_fp, lower_gold_set, ans_fp, new_sc_fp):
    with codecs.open(sc_fp, 'r', 'utf-8') as br_sc, \
            codecs.open(ans_fp, 'r', 'utf-8') as br_ans, \
            codecs.open(new_sc_fp, 'w', 'utf-8') as bw:
        sc_lines = br_sc.readlines()
        ans_lines = br_ans.readlines()
        for sc_idx, (sc_line, ans_line) in enumerate(zip(sc_lines, ans_lines)):
            if sc_idx % 1000 == 0:
                LogInfo.logs('Q-%d, schema %d/%d ...', q_idx, sc_idx, len(sc_lines))
            schema = json.loads(sc_line.strip())
            lower_ans_list = json.loads(ans_line.strip().lower())
            lower_ans_set = set(lower_ans_list)
            p, r, f1 = calc(gold_set=lower_gold_set, pred_set=lower_ans_set)
            schema['p'] = p
            schema['r'] = r
            schema['f1'] = f1
            json.dump(schema, bw)
            bw.write('\n')


def main():
    data_dir = 'runnings/candgen_CompQ/180101_Lukov_eff/data'
    qa_list = load_compq()
    for qa_idx, qa in enumerate(qa_list):
        div = qa_idx / 100
        sub_dir = '%d-%d' % (div*100, div*100+99)
        sc_fp = '%s/%s/%d_schema_cased' % (data_dir, sub_dir, qa_idx)
        ans_fp = '%s/%s/%d_ans' % (data_dir, sub_dir, qa_idx)
        new_sc_fp = '%s/%s/%d_schema' % (data_dir, sub_dir, qa_idx)
        gold_list = qa['targetValue']
        lower_gold_set = set([x.lower() for x in gold_list])
        process_single_question(qa_idx, sc_fp, lower_gold_set, ans_fp, new_sc_fp)


if __name__ == '__main__':
    main()
