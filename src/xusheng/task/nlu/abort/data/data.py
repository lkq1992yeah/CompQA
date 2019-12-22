"""
data generator for current model
"""

from xusheng.util.data_util import BaseBatchGenerator
from xusheng.util.log_util import LogInfo
from xusheng.util.struct_util import TopKRankedList

import codecs
import numpy as np
import os


def get_jaccard_score(set_a, set_b):
    if len(set_a) == 0 or len(set_b) == 0:
        return 0
    cnt = 0
    for elem in set_a:
        if elem in set_b:
            cnt += 1
    return float(cnt) / (len(set_a)+len(set_b)-cnt)


def fuzzy_match_name(mention, vocab, PN):
    """
    :param mention: list of strings
    :param vocab: list of (string, set) tuple
    :param PN: number of candidates = PN-1
    :return: list of strings with size PN-1
    """
    m_set = set()
    for ch in mention:
        m_set.add(ch)
    # LogInfo.begin_track("generate for %s [%s]...", mention, m_set)
    rank_list = TopKRankedList(PN-1)
    for name, c_set in vocab.items():
        score = get_jaccard_score(m_set, c_set)
        # LogInfo.logs("%s [%s] : %.4f", name, c_set, score)
        if score == 1.0:
            continue
        rank_list.push((name, score))
    LogInfo.logs("Cands for %s: [%s]", mention, "|".join(rank_list.top_names()))
    # LogInfo.end_track()
    return rank_list.top_names()


def fuzzy_match_id(mention, vocab_loader, PN):
    """

    :param mention: list of mention ids
    :param vocab_loader: vocab
    :return: list of candidate ids
    """
    return []


def candidate_generate(label_list, query_idx, query_len, vocab_loader, PN):
    new_query_idxs = list()
    new_query_lens = list()
    new_link_masks = list()
    new_entity_idxs = list()
    for label_line, query_line, qlen_line in zip(label_list, query_idx, query_len):
        i = 0
        while i < len(label_line):
            tag = label_line[i]
            if tag % 2 == 1:
                j = i + 1
                while label_line[j] == tag + 1:
                    j += 1
                entity_idx = fuzzy_match_id(query_line[i:j], vocab_loader, PN)
                link_mask = np.zeros(shape=[len(label_line)])
                for k in range(i, j):
                    link_mask[k] = 1
                new_query_idxs.append(query_line)
                new_query_lens.append(qlen_line)
                new_link_masks.append(link_mask)
                new_entity_idxs.append(entity_idx)
                i = j
    return new_query_idxs, new_query_lens, new_link_masks, new_entity_idxs


class DataLoader(object):

    def __init__(self, max_seq_len, vocab_index_dict):
        self.max_seq_len = max_seq_len
        self.dict = vocab_index_dict
        self.data = list()
        self.data_size = 0
        self.max = 0

    def decode_line(self, line):
        spt = line.strip().split("\t")
        query, label, intent, link_mask, entity = \
            spt[0], spt[1], int(spt[2]), spt[3], spt[4]
        # default = 0 means not found
        query_idx = [self.dict.get("[["+term+"]]", 0) for term in query.strip().split(" ")]
        label = [int(i) for i in label.split(" ")]
        link_mask = [int(i) for i in link_mask.split(" ")]
        entity_idx = [self.dict.get("[["+term+"]]", 0) for term in entity.strip().split(" ")]

        # actual length of query
        query_len = len(query_idx)
        self.max = max(query_len, self.max)

        # padding
        for _ in range(self.max_seq_len - query_len):
            query_idx.append(0)  # the last one in vocab is zero-vec
            label.append(0)  # 7 label tags
            link_mask.append(0)  # 1 means entity, 1 means context

        return query_idx, query_len, label, intent, link_mask, entity_idx

    def load(self, data_file, encoding):
        LogInfo.begin_track("Loading data from %s...", data_file)
        if os.path.isfile(data_file):
            LogInfo.begin_track("[Exist] Loading from %s...", data_file)
            query_idxs, query_lens, labels, intents, link_masks, entity_idxs \
                = list(), list(), list(), list(), list(), list()
            cnt = 0
            with codecs.open(data_file, 'r', encoding=encoding) as fin:
                for line in fin:
                    spt = line.strip().split("\t")
                    query_idxs.append([int(idx) for idx in spt[0].split(" ")])
                    query_lens.append(int(spt[1]))
                    labels.append([int(idx) for idx in spt[2].split(" ")])
                    intents.append(int(spt[3]))
                    link_masks.append([int(idx) for idx in spt[4].split(" ")])
                    entity_idxs.append([int(idx) for idx in spt[5].split(" ")])
                    cnt += 1
                    LogInfo.show_line(cnt, 1000000)
            LogInfo.end_track("Max_seq_len = %d.", self.max_seq_len)
        else:
            txt_data_file = data_file + ".name"
            LogInfo.begin_track("[Not Exist] Loading from %s...", txt_data_file)
            query_idxs, query_lens, labels, intents, link_masks, entity_idxs \
                = list(), list(), list(), list(), list(), list()
            cnt = 0
            fout = codecs.open(data_file, 'w', encoding=encoding)
            with codecs.open(txt_data_file, 'r', encoding=encoding) as fin:
                for line in fin:
                    query_idx, query_len, label, intent, link_mask, entity_idx\
                        = self.decode_line(line)
                    fout.write(" ".join([str(x) for x in query_idx]) + "\t" +
                               str(query_len) + "\t" +
                               " ".join([str(x) for x in label]) + "\t" +
                               str(intent) + "\t" +
                               " ".join([str(x) for x in link_mask]) + "\t" +
                               " ".join([str(x) for x in entity_idx]) + "\n")
                    query_idxs.append(query_idx)
                    query_lens.append(query_len)
                    labels.append(label)
                    intents.append(intent)
                    link_masks.append(link_mask)
                    entity_idxs.append(entity_idx)
                    cnt += 1
                    LogInfo.show_line(cnt, 1000000)
            fout.close()
            LogInfo.logs("Write into %s.", data_file)
            LogInfo.end_track("Max_seq_len = %d.", self.max)
        self.data = list(zip(query_idxs, query_lens, labels,
                             intents, link_masks, entity_idxs))
        self.data_size = len(self.data)
        LogInfo.end_track("Loaded. Size: %d.", self.data_size)


class BatchGenerator(BaseBatchGenerator):

    def __init__(self, data, batch_size):
        super(BatchGenerator, self).__init__(data=data,
                                             batch_size=batch_size)

    def next_batch(self):
        if self.pointer + self.batch_size > self.data_size:
            query_idx, query_len, label, intent, link_mask, entity_idx =\
                zip(*self.data[self.pointer: self.data_size])
            self.pointer = self.data_size
        else:
            query_idx, query_len, label, intent, link_mask, entity_idx = \
                zip(*self.data[self.pointer: self.pointer + self.batch_size])
            self.pointer += self.batch_size

        query_idx = np.array(query_idx)
        query_len = np.array(query_len)
        label = np.array(label)
        intent = np.array(intent)
        link_mask = np.array(link_mask)
        entity_idx = np.array(entity_idx)

        return [query_idx, query_len, label, intent, link_mask, entity_idx]




