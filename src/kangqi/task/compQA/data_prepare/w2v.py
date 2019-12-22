# -*- coding: utf-8 -*-

import os
import numpy as np
import cPickle

from kangqi.util.LogUtil import LogInfo

class Word2Vec(object):

    def __init__(self, fp, n_emb):
        LogInfo.begin_track('Loading w2v from %s: ', fp)
        fp_pydump = fp + '.pydump'
        if not os.path.isfile(fp_pydump):
            LogInfo.logs('Mode: [txt]')
            self.wd_dict = {}       # <word, index>
            vec_list = []
            cnt = 0
            with open(fp, 'r') as br:
                for line in br.readlines():
                    cnt += 1
                    if cnt % 100000 == 0: LogInfo.logs('Current: %d lines loaded.', cnt)
                    spt = line.strip().split(' ')
                    assert len(spt) == n_emb + 1
                    wd = spt[0]
                    self.wd_dict[wd] = len(self.wd_dict)
                    vec = np.zeros((n_emb, ), dtype='float32')
                    for idx in range(n_emb):
                        vec[idx] = float(spt[idx + 1])
                    vec_list.append(vec)
            self.w2v_matrix = np.stack(vec_list)    # (total_words, n_emb)
            LogInfo.logs('%d words and %s embedding loaded.',
                         len(self.wd_dict), self.w2v_matrix.shape)
            LogInfo.logs('Saving to binary ... ')
            with open(fp_pydump, 'wb') as bw:
                cPickle.dump(self.wd_dict, bw)
                np.save(bw, self.w2v_matrix)
            LogInfo.logs('Saved binary to %s.', fp_pydump)
        else:
            LogInfo.logs('Mode: [pydump]')
            with open(fp_pydump, 'rb') as br:
                self.wd_dict = cPickle.load(br)
                self.w2v_matrix = np.load(br)
            LogInfo.logs('%d words and %s embedding loaded.',
                         len(self.wd_dict), self.w2v_matrix.shape)

#        sample_list = self.wd_dict.items()[0 : 100]
#        for wd, idx in sample_list:
#            LogInfo.logs('%3d --> %s', idx, wd)


        LogInfo.end_track()


    def get(self, wd):
        wd_idx = self.wd_dict.get(wd, -1)
        if wd_idx == -1: return None
        return self.w2v_matrix[wd_idx]
