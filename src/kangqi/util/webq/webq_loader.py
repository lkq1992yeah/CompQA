# -*- coding: utf-8 -*-

#==============================================================================
# Author: Kangqi Luo
# Goal: Load Webquestions from Json File
#==============================================================================

import json

from kangqi.util.LogUtil import LogInfo

class Webq(object):

    def __init__(self, raw_item):
        self.focus = raw_item['url']
        if isinstance(self.focus, unicode):
            self.focus = self.focus.encode('utf-8')

        self.surface = raw_item['utterance']
        if isinstance(self.surface, unicode):
            self.surface = self.surface.encode('utf-8')

        self.answer_set = set([])
        self.lower_answer_set = set([])

        target_value = raw_item['targetValue']
        target_value = target_value[6 : -1]
        target_value = target_value.replace(') (', ')###(')
        spt = target_value.split('###')
        for item in spt:
            ans_str = item[13 : -1]
            if ans_str.startswith('"') and ans_str.endswith('"'):
                ans_str = ans_str[1 : -1]
            if isinstance(ans_str, unicode):
                ans_str = ans_str.encode('utf-8')
            self.answer_set.add(ans_str)
            self.lower_answer_set.add(ans_str.lower())

    def display(self):
        LogInfo.logs('Surface: %s', self.surface)
        LogInfo.logs('Focus: %s', self.focus)
        LogInfo.logs('Answer Set: %s', self.answer_set)
        LogInfo.logs('Lower Answer Set: %s', self.lower_answer_set)


def load_webq_from_json(json_fp):
    with open(json_fp, 'r') as br:
        raw_list = json.load(br)
    qa_list = []
    for raw_item in raw_list:
        qa = Webq(raw_item)
        qa_list.append(qa)
    LogInfo.logs('Collected %d QA from [%s]', len(qa_list), json_fp)
    return qa_list

def load_webq_all():
    return load_webq_from_json('/home/kangqi/Webquestions/Json/webquestions.examples.json')

if __name__ == '__main__':
    qa_list = load_webq_all()
    test_qa_list = qa_list[3778 : 5810]

    for idx in range(10):
        LogInfo.begin_track('Showing Q-%d: ', idx)
        qa = test_qa_list[idx]
        qa.display()
        LogInfo.end_track()