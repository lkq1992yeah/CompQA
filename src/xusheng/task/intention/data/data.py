"""
data generator for current model
"""

from xusheng.util.data_util import BaseBatchGenerator
from xusheng.util.log_util import LogInfo

import codecs
import numpy as np


class DataLoader(object):

    def __init__(self, max_seq_len, vocab_index_dict):
        self.max_seq_len = max_seq_len
        self.dict = vocab_index_dict
        self.data = list()
        self.data_size = 0

    def decode_line(self, line):
        spt = line.strip().split("\t")
        context, pinlei = spt[0], spt[1]
        context_idx = [self.dict.get(_) for _ in context.strip().split(" ")]
        context_seq = len(context_idx)
        # padding
        for _ in range(self.max_seq_len - context_seq):
            context_idx.append(0)
        pinlei_idx = self.dict.get(pinlei.strip())

        return context_idx, context_seq, pinlei_idx

    def load(self, data_file, encoding):
        LogInfo.begin_track("Loading data from %s...", data_file)
        context_idxs, context_seqs, pinlei_idxs = list(), list(), list()
        cnt = 0
        with codecs.open(data_file, 'r', encoding=encoding) as fin:
            for line in fin:
                context_idx, context_seq, pinlei_idx = self.decode_line(line)
                context_idxs.append(context_idx)
                context_seqs.append(context_seq)
                pinlei_idxs.append(pinlei_idx)
                cnt += 1
                LogInfo.show_line(cnt, 10000)
        self.data = list(zip(context_idxs, context_seqs, pinlei_idxs))
        self.data_size = len(self.data)
        LogInfo.end_track()


class BatchGenerator(BaseBatchGenerator):

    def __init__(self, data, batch_size):
        super(BatchGenerator, self).__init__(data=data,
                                             batch_size=batch_size)

    def next_batch(self):
        if self.pointer + self.batch_size > self.data_all_size:
            enc_ip_batch, tf_idf_ip_batch, labels_batch, length_batch =\
                zip(*self.data[self.pointer: self.data_all_size])
            self.pointer = self.data_all_size
        else:
            enc_ip_batch, tf_idf_ip_batch, labels_batch, length_batch = \
                zip(*self.data[self.pointer: self.pointer + self.batch_size])
            self.pointer += self.batch_size

        enc_ip_batch = np.asarray(enc_ip_batch)
        tf_idf_ip_batch = np.asarray(tf_idf_ip_batch)
        labels_batch = np.asarray(labels_batch)
        length_batch = np.asarray(length_batch)

        return list(zip(enc_ip_batch, tf_idf_ip_batch, labels_batch, length_batch))


