# -*- coding: utf-8 -*-

# Copied from src.kangqi.task.compQA.data_prepare.data_saving
# Goal: convert a float number into 1-hot vector

import numpy as np

from kangqi.util.LogUtil import LogInfo


class Discretizer(object):

    def __init__(self, split_list, output_mode='numpy', name=''):
        assert output_mode in ('numpy', 'list')
        split_list.sort()
        self.output_mode = output_mode
        self.split_list = split_list
        self.len = len(split_list) + 1
        self.name = name
        self.distribution = np.zeros((self.len + 1,), dtype='int32')        # used for showing overall distribution

    def convert(self, score):
        if self.output_mode == 'numpy':
            ret_vec = np.zeros((self.len, ), dtype='int32')
        else:
            ret_vec = [0] * self.len
        if score is None:
            self.distribution[-1] += 1       # default: return nothing, and the last column records it.
            return ret_vec

        ret_idx = -1
        if score < self.split_list[0]:
            ret_idx = 0
        elif score >= self.split_list[-1]:
            ret_idx = self.len - 1
        else:
            for i in range(len(self.split_list) - 1):       # i \in [0, self.len - 2)
                if self.split_list[i] <= score < self.split_list[i + 1]:
                    ret_idx = i + 1
                    break
        ret_vec[ret_idx] = 1
        self.distribution[ret_idx] += 1
        return ret_vec

    def batch_convert(self, score_list):
        ret_matrix = np.zeros((len(score_list), self.len), dtype='int32')
        for idx in range(len(score_list)):
            ret_matrix[idx] = self.convert(score_list[idx])
        return ret_matrix

    # distribution_vec: sum of several discretized vectors
    def show_distribution(self):
        sz = np.sum(self.distribution)
        show_name = 'the' if self.name == '' else self.name
        LogInfo.begin_track('Showing %s distribution over %d data: ', show_name, sz)
        for idx in range(self.len):
            val = self.distribution[idx]
            LogInfo.logs('[%s, %s): %d / %d (%.3f%%)',
                         '-inf' if idx == 0 else str(self.split_list[idx - 1]),
                         str(self.split_list[idx]) if idx < self.len - 1 else 'inf',
                         int(val), sz, 100.0 * val / sz)
        nothing_val = self.distribution[-1]
        LogInfo.logs('Missing: %d / %d (%.3f%%)', nothing_val, sz, 100.0 * nothing_val / sz)
        LogInfo.end_track()
