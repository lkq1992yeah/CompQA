# -*- coding: utf-8 -*-

#==============================================================================
# Author: Kangqi Luo
# Utility functions of question structure.
#==============================================================================

import json
from kangqi.util.LogUtil import LogInfo

class QA(object):

    def __init__(self, q, a_list_str, Tvt, q_idx):
        self.q = q
        self.tokens = None

        self.a_list_str = a_list_str
        self.a_list = json.loads(a_list_str)
        self.lower_a_list = [x.lower() for x in self.a_list]
        self.Tvt = Tvt
        self.q_idx = q_idx      # just for visualization, starting from 1

    def __key(self):
        return (self.q, self.a_list_str, self.Tvt, self.q_idx)

    def __eq__(self, another):
        return type(self) == type(another) and self.__key() == another.__key()

    def __hash__(self):
        return hash(self.__key())

    def to_string(self):
        return '%s-%d ==> ("%s", %s)' %(self.Tvt, self.q_idx, self.q, self.a_list)


def load_complex_questions(path='/home/data/ComplexQuestions'):
    train_qa_list = []
    test_qa_list = []
    for Tvt, qa_list in zip(['train', 'test'], [train_qa_list, test_qa_list]):
        fp = '%s/compQ.%s.release' %(path, Tvt)
        with open(fp, 'r') as br:
            q_idx = 0
            for line in br.readlines():
                q_idx += 1
                q, a_list_str = line.strip().split('\t')
                qa = QA(q, a_list_str, Tvt, q_idx)
                qa_list.append(qa)
        LogInfo.logs('%d %s QA loaded.', len(qa_list), Tvt)
    return train_qa_list, test_qa_list

def load_complex_questions_ordinal_only(path='/home/data/ComplexQuestions'):
    train_qa_list, test_qa_list = load_complex_questions(path=path)
    filt_train_qa_list = []
    filt_test_qa_list = []
    for Tvt, qa_list, filt_qa_list in zip(
            ['train', 'test'], [train_qa_list, test_qa_list],
            [filt_train_qa_list, filt_test_qa_list]):
        filt_fp = '%s/ordinal_picking/pick.%s' %(path, Tvt)
        with open(filt_fp, 'r') as br:
            for line in br.readlines():
                filt_idx = int(line.strip().split('\t')[0]) - 1
                filt_qa_list.append(qa_list[filt_idx])
        LogInfo.logs('%d %s ordinal QA filtered.', len(filt_qa_list), Tvt)
    return filt_train_qa_list, filt_test_qa_list
