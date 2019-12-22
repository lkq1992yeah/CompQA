# -*- coding: utf-8 -*-
# Rewrite by Xusheng
# Merge question representation with schema representation using different strategies
# Avg: average word representation in q side and skeleton representation in schema side
# Att: mutual attention between q and schema, based on the idea of ABCNN

class MergeBaseModule(object):

    def __init__(self, q_max_len, sc_max_len, dim_q_hidden, dim_sk_hidden):
        self.q_max_len = q_max_len
        self.sc_max_len = sc_max_len
        self.dim_q_hidden = dim_q_hidden
        self.dim_sk_hidden = dim_sk_hidden

    def forward(self, q_hidden, q_len, sc_hidden, sc_len, reuse=None):
        """
        :param q_hidden: (batch, q_max_len, dim_q_hidden) 
        :param q_len:  (batch, )
        :param sc_hidden:  (batch, sc_max_len, dim_sk_hidden)
        :param sc_len: (batch, )
        :param reuse: reuse parameters
        :return: (batch, ) final score
        """
        pass
