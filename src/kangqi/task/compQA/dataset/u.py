# -*- coding:utf-8 -*-

import os
import time
import json
import random
import codecs
import cPickle
from ..linking.parser import CoreNLPParser

from kangqi.util.LogUtil import LogInfo


parser_ip = '202.120.38.146'
parser_port = 9601      # 8601
parser = CoreNLPParser('http://%s:%d/parse' % (parser_ip, parser_port))   # just open the parser
# parser.parse('who is the prime minister of japan ?')


def weighted_sampling(item_list, weight_list, budget):
    if len(item_list) == 0:
        return []

    data_size = len(item_list)
    acc_list = list(weight_list)
    for i in range(1, data_size):
        acc_list[i] += acc_list[i-1]
    norm = acc_list[-1]
    acc_list = map(lambda score: score / norm, acc_list)

    sample_list = []
    for _ in range(budget):
        x = random.random()
        pick_idx = -1
        for i in range(data_size):
            if acc_list[i] >= x:
                pick_idx = i
                break
        sample_list.append(item_list[pick_idx])

    return sample_list


def load_compq():
    compq_path = '/home/data/ComplexQuestions'
    pickle_fp = compq_path + '/compQ.all.cPickle'
    if os.path.isfile(pickle_fp):
        LogInfo.logs('Loading ComplexQuestions from cPickle ...')
        with open(pickle_fp, 'r') as br:
            qa_list = cPickle.load(br)
    else:
        LogInfo.logs('CompQ initializing ... ')
        qa_list = []
        for Tvt in ('train', 'test'):
            fp = '%s/compQ.%s.release' % (compq_path, Tvt)
            with codecs.open(fp, 'r', 'utf-8') as br:
                for line in br.readlines():
                    qa = {}
                    q, a_list_str = line.strip().split('\t')
                    qa['utterance'] = q
                    qa['targetValue'] = json.loads(a_list_str)
                    qa['parse'] = parser.parse(qa['utterance'])
                    qa['tokens'] = qa['parse'].tokens
                    qa_list.append(qa)
        with open(pickle_fp, 'w') as bw:
            cPickle.dump(qa_list, bw)
    LogInfo.logs('%d ComplexQuestions loaded.', len(qa_list))
    return qa_list


def load_webq():
    webq_path = '/home/data/Webquestions'
    pickle_fp = webq_path + '/webquestions.all.cPickle'
    if os.path.isfile(pickle_fp):
        LogInfo.logs('Loading Webquestions from cPickle ...')
        with open(pickle_fp, 'r') as br:
            qa_list = cPickle.load(br)
    else:
        LogInfo.logs('WebQ initializing ... ')
        webq_fp = webq_path + '/Json/webquestions.examples.json'
        with codecs.open(webq_fp, 'r', 'utf-8') as br:
            webq_data = json.load(br)
        qa_list = []
        for raw_info in webq_data:
            qa = {}
            target_value = []
            ans_line = raw_info['targetValue']
            ans_line = ans_line[7: -2]      # remove '(list (' and '))'
            for ans_item in ans_line.split(') ('):
                ans_item = ans_item[12:]    # remove 'description '
                if ans_item.startswith('"') and ans_item.endswith('"'):
                    ans_item = ans_item[1: -1]
                target_value.append(ans_item)
            qa['utterance'] = raw_info['utterance']
            qa['targetValue'] = target_value
            qa['parse'] = parser.parse(qa['utterance'])
            qa['tokens'] = qa['parse'].tokens
            qa_list.append(qa)
            if len(qa_list) % 1000 == 0:
                LogInfo.logs('%d / %d scanned.', len(qa_list), len(webq_data))
        with open(pickle_fp, 'w') as bw:
            cPickle.dump(qa_list, bw)
    LogInfo.logs('%d WebQuesetions loaded.', len(qa_list))
    return qa_list


def load_simpq():
    simpq_path = '/home/data/SimpleQuestions/SimpleQuestions_v2'
    pickle_fp = simpq_path + '/simpQ.all.cPickle'
    st = time.time()
    if os.path.isfile(pickle_fp):
        LogInfo.logs('Loading SimpleQuestions from cPickle ...')
        with open(pickle_fp, 'r') as br:
            qa_list = cPickle.load(br)
    else:
        LogInfo.logs('SimpQ initializing ... ')
        qa_list = []
        for Tvt in ('train', 'valid', 'test'):
            fp = '%s/annotated_fb_data_%s.txt' % (simpq_path, Tvt)
            with codecs.open(fp, 'r', 'utf-8') as br:
                for line in br.readlines():
                    qa = {}
                    s, p, o, q = line.strip().split('\t')
                    s = remove_simpq_header(s)
                    p = remove_simpq_header(p)
                    o = remove_simpq_header(o)
                    qa['utterance'] = q
                    qa['targetValue'] = (s, p, o)       # different from other datasets
                    # qa['tokens'] = parser.parse(qa['utterance']).tokens
                    qa['parse'] = parser.parse(qa['utterance'])
                    qa['tokens'] = qa['parse'].tokens
                    qa_list.append(qa)
                    if len(qa_list) % 1000 == 0:
                        LogInfo.logs('%d scanned.', len(qa_list))
        with open(pickle_fp, 'w') as bw:
            cPickle.dump(qa_list, bw)
    LogInfo.logs('[%.3fs] %d SimpleQuestions loaded.', time.time() - st, len(qa_list))
    return qa_list


def remove_simpq_header(mid):
    mid = mid[17:]  # remove www.freebase.com/
    mid = mid.replace('/', '.')
    return mid


def load_simpq_sp_dict(fb='FB2M'):
    sp_dict_fp = '/home/data/SimpleQuestions/SimpleQuestions_v2/freebase-subsets/SP-%s.txt' % fb
    sp_dict = {}
    with codecs.open(sp_dict_fp, 'r', 'utf-8') as br:
        for line in br.readlines():
            spt = line.strip().split('\t')
            subj = spt[0]
            pred_set = set(spt[1:])
            sp_dict[subj] = pred_set
    LogInfo.logs('%d <entity, pred_set> loaded from %s.', len(sp_dict), fb)
    return sp_dict


def get_q_range_by_mode(data_name, mode):
    assert mode in ('train', 'valid', 'test')
    assert data_name in ('SimpQ', 'WebQ', 'CompQ')

    q_idx_list = []
    if data_name == 'WebQ':             # 3023 / 3778 / 2032
        if mode == 'train':
            q_idx_list = range(3023)
        elif mode == 'valid':
            q_idx_list = range(3023, 3778)
        elif mode == 'test':
            q_idx_list = range(3778, 1000000)
    elif data_name == 'SimpQ':          # 75910 / 86755 / 108442
        if mode == 'train':
            q_idx_list = range(75910)
        elif mode == 'valid':
            q_idx_list = range(75910, 86755)
        elif mode == 'test':
            q_idx_list = range(86755, 1000000)
    elif data_name == 'CompQ':
        if mode == 'train':
            q_idx_list = range(1000)
        elif mode == 'valid':
            q_idx_list = range(1000, 1300)
        elif mode == 'test':
            q_idx_list = range(1300, 2100)
    return q_idx_list


def main():
    qa_list = load_simpq()
    for idx in range(4):
        qa = qa_list[idx]
        LogInfo.logs('utterance: %s', qa['utterance'].encode('utf-8'))
        LogInfo.logs('tokens: %s', [tok.token for tok in qa['tokens']])


if __name__ == '__main__':
    main()
