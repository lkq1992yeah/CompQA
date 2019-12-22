# -*- coding: utf-8 -*-

# ==============================================================================
# Author: Kangqi Luo
# Goal: cache read / write.
# Currently we focus dictionary cache only.
# ==============================================================================

import os
import cPickle

from kangqi.util.LogUtil import LogInfo


class DictCache(object):

    def __init__(self, cache_fp=None):
        self.cache_dict = {}
        self.cache_fp = cache_fp
        self.use_flag = cache_fp is not None    # if False: not save anything

        if not self.use_flag:
            return
        if os.path.isfile(cache_fp):
            br = open(cache_fp, 'rb')
            while True:
                try:
                    k, v = cPickle.load(br)
                    self.cache_dict[k] = v
                except EOFError:
                    break
            br.close()
        else:
            cache_dir = os.path.dirname(cache_fp)
#            LogInfo.logs('cache_fp = %s, cache_dir = %s', cache_fp, cache_dir)
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)
        LogInfo.logs('Loaded %d records from cache: [%s].', len(self.cache_dict), cache_fp)

    def get(self, key):
        if not self.use_flag:
            return None
        return self.cache_dict.get(key)     # return value, or None.

    def put(self, key, val):
        if not self.use_flag:
            return None
        if val is None:
            return      # ignore None values
        self.cache_dict[key] = val
        tup = (key, val)
#        LogInfo.logs('key type: %s', type(key))
        with open(self.cache_fp, 'ab') as bw:
            cPickle.dump(tup, bw)
