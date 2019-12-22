"""
TF-IDF library
add one document each time with a name and a list of words in that documents
u can get tf-idf scores, and other related useful scores
currently only for idf scores, unfinished lib

"""

import numpy as np
from kangqi.util.LogUtil import LogInfo

class TfIdf(object):

    def __init__(self):
        self.documents = list()
        self.idf_cnt = dict()
        self.word_freq = dict()

    def add_document(self, doc_name, word_list):
        self.documents.append(doc_name)
        # count word frequency
        for word in word_list:
            self.word_freq[word] = self.word_freq.get(word, 0.0) + 1.0
        # count document contains each word
        for word in set(word_list):
            self.idf_cnt[word] = self.idf_cnt.get(word, 0.0) + 1.0

    def get_tf(self, word_list):
        tf_score = dict()
        for word in word_list:
            tf_score[word] = tf_score.get(word, 0.0) + 1.0
        for word in tf_score:
            tf_score[word] / len(word_list)
        return tf_score

    def get_idf(self, word_list):
        idf_score = dict()
        for word in set(word_list):
            idf_score[word] = np.log(1 + len(self.documents) /
                                     (self.idf_cnt.get(word, 0.0)+1.0))  # smooth
        return idf_score

    def get_tfIdf(self, word_list):
        tf_score = self.get_tf(word_list)
        idf_score = self.get_idf(word_list)
        tfIdf_score = dict()
        for word in set(word_list):
            tfIdf_score[word] = tf_score[word] * idf_score[word]
        return tfIdf_score

    # get idf score for each word in corpus
    # threshold means word freq
    def get_idf(self, threshold = 1):
        idf_score = dict()
        for word, cnt in self.idf_cnt.items():
            if self.word_freq[word] < threshold:
                continue
            idf_score[word] = np.log(1 + len(self.documents) / cnt)
        return idf_score

if __name__=="__main__":
    tfidf = TfIdf()
    tfidf.add_document('a', ['hi', 'hello', 'dog', 'cat'])
    tfidf.add_document('b', ['hi', 'bye', 'dog', 'cat'])
    tfidf.add_document('c', ['hi', 'hello', 'eat'])
    tfidf.add_document('d', ['hi', 'bye', 'dog', 'tiger'])
    idf = tfidf.get_idf()
    ret = sorted(idf.items(), key=lambda x: x[1], reverse=True)
    LogInfo.logs(ret)
