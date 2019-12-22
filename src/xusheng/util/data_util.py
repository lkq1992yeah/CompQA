"""
Data utils
"""

import codecs
import numpy as np
import random

from xusheng.util.log_util import LogInfo


def load_kv_table(file_path):
    kv_table = dict()
    with codecs.open(file_path, 'r', encoding='utf-8') as fin:
        for line in fin:
            spt = line.strip().split()
            if len(spt) < 2:
                LogInfo.logs("[error] bad line: %s", line.strip())
            kv_table[spt[0]] = spt[1]
    return kv_table


def load_kkv_table(file_path):
    kkv_table = dict()
    with codecs.open(file_path, 'r', encoding='utf-8') as fin:
        for line in fin:
            spt = line.strip().split()
            if len(spt) < 3:
                LogInfo.logs("[error] bad line: %s", line.strip())
            kkv_table[spt[0] + ' ' + spt[1]] = spt[2]
    return kkv_table


class VocabularyLoader(object):
    """
    load embeddings from file
    index from 1-n, 0 represents padding (NULL <--> zero-vec)
    """
    def __init__(self):
        self.vocab_size = 0
        self.index_vocab_dict = list()
        self.vocab_index_dict = dict()
        self.vocab_embedding = list()

    def load_vocab(self, vocab_file, embedding_dim, encoding):
        LogInfo.begin_track("Loading vocab from %s...", vocab_file)
        self.vocab_size = 0
        self.index_vocab_dict.clear()
        self.vocab_index_dict.clear()
        self.vocab_embedding.clear()
        with codecs.open(vocab_file, 'r', encoding=encoding) as fin:
            index = 0
            # 0 embedding for not-found query term
            self.vocab_index_dict["[[NULL]]"] = index
            self.index_vocab_dict.append("[[NULL]]")
            self.vocab_embedding.append([0.0 for _ in range(embedding_dim)])
            index += 1
            for line in fin:
                spt = line.strip().split()
                self.vocab_index_dict[spt[0]] = index
                self.index_vocab_dict.append(spt[0])
                embedding = [float(spt[i].strip()) for i in range(1, len(spt))]
                self.vocab_embedding.append(embedding)
                index += 1
                LogInfo.show_line(index, 50000)
        self.vocab_size = len(self.vocab_embedding)
        self.vocab_embedding = np.array(self.vocab_embedding)
        LogInfo.end_track("Vocab loaded. Size: %d.", self.vocab_size)

    def load_vocab_name(self, vocab_file, encoding):
        LogInfo.begin_track("Loading vocab from %s...", vocab_file)
        self.vocab_size = 0
        self.index_vocab_dict.clear()
        self.vocab_index_dict.clear()
        with codecs.open(vocab_file, 'r', encoding=encoding) as fin:
            index = 0
            for line in fin:
                self.vocab_index_dict[line.strip()] = index
                self.index_vocab_dict.append(line.strip())
                index += 1
                LogInfo.show_line(index, 50000)

        self.vocab_size = index
        LogInfo.end_track("Vocab loaded. Size: %d.", self.vocab_size)

    def load_vocab_embedding(self, embedding_file, encoding):
        LogInfo.begin_track("Loading embeddings from %s...", embedding_file)
        vocab_embedding = len(self.vocab_index_dict) * [None]
        with codecs.open(embedding_file, 'r', encoding=encoding) as fin:
            count = 0
            for line in fin:
                strs = line.split()
                embedding = [float(strs[i].strip()) for i in range(1, len(strs))]
                vocab_embedding[self.vocab_index_dict[strs[0].strip()]] = embedding
                count += 1
                LogInfo.show_line(count, 50000)

        assert count == len(vocab_embedding)
        self.vocab_embedding = np.asarray(vocab_embedding)
        LogInfo.end_track("Vocab loaded. Size: %d.", self.vocab_size)


class BaseBatchGenerator(object):
    def __init__(self, data, batch_size):
        self.batch_size = batch_size
        self.data = data
        self.data_size = len(self.data)
        self.pointer = 0
        self.num_batches = 0
        self.create_batches()

    def reset_batch_pointer(self):
        # random.shuffle(self.data)
        self.pointer = 0

    def create_batches(self):
        if self.data_size % self.batch_size == 0:
            self.num_batches = int(self.data_size / self.batch_size)
        else:
            self.num_batches = int(self.data_size / self.batch_size) + 1

        # When the data (tensor) is too small, let's give them a better error message
        if self.num_batches == 0:
            assert False, "Not enough data. Make seq_length and batch_size small."
        LogInfo.logs("Batches created. (%d)", self.num_batches)

    def next_batch(self):
        pass


def decode_data_line(line, vocab_index_dict):
    strs = line.strip().split(u"")
    item_id, features_str, labels_str, length_str =\
        strs[0].strip(), strs[1].strip(), strs[2].strip(), strs[3].strip()
    tfidf_enc_str = features_str.split(u"")

    enc_ip = list(map(vocab_index_dict.get, tfidf_enc_str[0].strip().split(u" ")))

    tf_ip_strs = tfidf_enc_str[1].strip().split(u" ")
    idf_ip_strs = tfidf_enc_str[2].strip().split(u" ")
    tf_idf_ip_strs = tfidf_enc_str[3].strip().split(u" ")

    tf_idf_ip = []
    for i in range(len(tf_ip_strs)):
        tf_idf_ip.append([float(tf_ip_strs[i].strip()),
                          float(idf_ip_strs[i].strip()),
                          float(tf_idf_ip_strs[i].strip())])

    labels = [float(x.strip()) for x in labels_str.split(u" ")]

    return item_id, enc_ip, tf_idf_ip, labels, int(length_str)


def decoder_data_all(data_file, encoding, vocab_index_dict):
    item_id_all, enc_ip_all, tf_idf_ip_all, labels_all, length_all = \
        [], [], [], [], []
    with codecs.open(data_file, 'r', encoding=encoding) as fin:
        for line in fin:
            item_id, enc_ip, tf_idf_ip, labels, length = \
                decode_data_line(line, vocab_index_dict)

            item_id_all.append(item_id)
            enc_ip_all.append(enc_ip)
            tf_idf_ip_all.append(tf_idf_ip)
            labels_all.append(labels)
            length_all.append(length)

    return list(zip(item_id_all, enc_ip_all, tf_idf_ip_all, labels_all, length_all))


def normalize(data, data_mean=None, data_std=None):
    data = np.asarray(data)
    if data_mean is None:
        data_mean = np.mean(data, axis=0)
    if data_std is None:
        data_std = np.std(data, axis=0)

    data = (data - data_mean) / (data_std + 0.00001)

    return data.tolist(), data_mean, data_std