"""
Author: Kangqi Luo
Goal: Direct compare between several predict results
Currently, we just compare two items
"""

import json
import codecs
import argparse
from ast import literal_eval

from ..dataset.u import load_webq

from kangqi.util.LogUtil import LogInfo


parser = argparse.ArgumentParser(description='Result Comparison')
parser.add_argument('--data_name', default='WebQ', choices=['WebQ', 'SimpQ', 'CompQ'])
parser.add_argument('--predict_fp_dict', help='A dictionary of <model_name, predict_answer> pairs')
parser.add_argument('--q_st', type=int)
parser.add_argument('--q_ed', type=int)


class CompareItem:

    def __init__(self, q_idx, q, gold_ans, predict_ans_list):
        self.q_idx = q_idx
        self.q = q
        self.predict_ans_list = predict_ans_list
        self.gold_ans = gold_ans

        self.predict_metric_list = []
        for predict_ans in self.predict_ans_list:
            rec, prec, f1 = computeF1(goldList=self.gold_ans, predictedList=predict_ans)
            self.predict_metric_list.append((prec, rec, f1))


def computeF1(goldList,predictedList):

  """Assume all questions have at least one answer"""
  if len(goldList)==0:
    raise Exception("gold list may not be empty")
  """If we return an empty list recall is zero and precision is one"""
  if len(predictedList)==0:
    return (0,1,0)
  """It is guaranteed now that both lists are not empty"""

  precision = 0
  for entity in predictedList:
    if entity in goldList:
      precision+=1
  precision = float(precision) / len(predictedList)

  recall=0
  for entity in goldList:
    if entity in predictedList:
      recall+=1
  recall = float(recall) / len(goldList)

  f1 = 0
  if precision+recall>0:
    f1 = 2*recall*precision / (precision + recall)
  return (recall,precision,f1)


def load_predict_detail(predict_fp):
    q_ans_dict = {}     # <utterance, [answer]>
    with codecs.open(predict_fp, 'r', 'utf-8') as br:
        for line in br.readlines():
            spt = line.strip().split('\t')
            q = spt[0]
            ans_list = json.loads(spt[1])
            q_ans_dict[q] = ans_list
    return q_ans_dict


def key_func(comp_item, our_index, other_index):
    f1_gain = comp_item.predict_metric_list[our_index][2] - comp_item.predict_metric_list[other_index][2]
    return f1_gain


def main(args):
    assert args.data_name == 'WebQ'
    qa_list = load_webq()

    predict_fp_dict = literal_eval(args.predict_fp_dict)
    model_name_list = predict_fp_dict.keys()
    LogInfo.logs('Model names: %s', model_name_list)
    model_name_dict = {model_name: idx for idx, model_name in enumerate(model_name_list)}
    other_idx = model_name_dict['STAGG']
    our_idx = 1 - other_idx

    predict_detail_dict = {model_name: load_predict_detail(predict_fp)
                           for model_name, predict_fp in predict_fp_dict.items()}
    # <model_name, <utterance, [answer]>>

    compare_list = []
    for q_idx in range(args.q_st, args.q_ed):
        qa = qa_list[q_idx]
        q = qa['utterance']
        gold_ans = qa['targetValue']
        predict_ans_list = []
        for model_name in model_name_list:
            predict_ans = predict_detail_dict[model_name].get(q, [])
            predict_ans_list.append(predict_ans)
        compare_list.append(CompareItem(q_idx=q_idx, q=q, gold_ans=gold_ans, predict_ans_list=predict_ans_list))
    compare_list.sort(key=lambda item: key_func(comp_item=item,
                                                our_index=our_idx,
                                                other_index=other_idx))

    LogInfo.begin_track('Showing detail: ')
    for comp_item in compare_list:
        f1_gain = key_func(comp_item, our_idx, other_idx)
        LogInfo.begin_track('Q-%d: [%s]', comp_item.q_idx, comp_item.q)
        LogInfo.logs('F1 gain = %.6f, Our = %s, Other = %s', f1_gain,
                     comp_item.predict_metric_list[our_idx],
                     comp_item.predict_metric_list[other_idx])
        LogInfo.logs('Gold: %s', comp_item.gold_ans)
        LogInfo.logs('Our: %s', comp_item.predict_ans_list[our_idx])
        LogInfo.logs('Other: %s', comp_item.predict_ans_list[other_idx])
        LogInfo.end_track()
    LogInfo.end_track()


if __name__ == '__main__':
    LogInfo.begin_track('Result Comparison Start ...')
    _args = parser.parse_args()
    main(_args)
    LogInfo.end_track('All Done.')
