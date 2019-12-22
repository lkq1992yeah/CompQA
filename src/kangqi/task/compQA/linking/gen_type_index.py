#==============================================================================
# Author: Kangqi Luo
# Goal: generate type link indices
#==============================================================================

import re
import cPickle

from . import parser
from ..u import load_stop_words

from kangqi.util.LogUtil import LogInfo

type_res_dir = '/home/kangqi/workspace/PythonProject/resources/compQA/type_link'
parser = parser.CoreNLPParser('http://localhost:4000/parse')



# get [(type, raw names)]
# two sources of raw name: from id directly, or from its name
def load_raw_names():
    tidx_tp_dict = {}       # <t_idx, type>
    tidx_name_dict = {}     # <t_idx, name>

    for fp, _dict in [('type_names.tsv', tidx_name_dict), ('type_dict.tsv', tidx_tp_dict)]:
        with open(type_res_dir + '/' + fp, 'r') as br:
            for line in br.readlines():
                spt = line.strip().split('\t')
                if len(spt) == 2:
                    idx, item = spt
                    _dict[int(idx)] = item
                else:
                    _dict[int(spt[0])] = ''
            LogInfo.logs('%d items loaded from %s.', len(_dict), fp)

    assert len(tidx_tp_dict) == len(tidx_name_dict)
    size = len(tidx_name_dict)

    type_name_dict = {}     # <type, real name> (type.object.name)
    raw_name_list = []
    for idx in range(1, size + 1):
        tp = tidx_tp_dict[idx]
        name = tidx_name_dict[idx]
        name_from_id = tp[tp.rfind('.') + 1 : ]
        type_name_dict[tp] = name if name != '' else name_from_id
        if name != '':
            raw_name_list.append((tp, name))
        raw_name_list.append((tp, name_from_id))

    LogInfo.logs('%d <type, raw names> loaded.', len(raw_name_list))
    return type_name_dict, raw_name_list

# remove stop words, replace characters, perform stemming
# the output ret_name is actually a BAG-OF-WORD
def refine_name(tp_name):
    # First step: remove punctuations
    tp_name = re.sub(r'(\/|_|,|-|\(|\))', ' ', tp_name)     # replace / , _ - ( ) as a blank
    tp_name = re.sub(r'\.', '', tp_name)                    # just remove dots
    tp_name = re.sub(r' +', ' ', tp_name)                   # shrink blanks

    # Second stop: lemmatize
    tokens = parser.parse(tp_name).tokens
    lemma_list = [token.lemma for token in tokens]

    # Third step: remove stop words
    stop_word_set = load_stop_words()
    nonstop_list = [wd for wd in lemma_list if wd not in stop_word_set]

    ret_name = '\t'.join(sorted(set(nonstop_list)))
    return ret_name



def prepare_type_index():
    # First: get raw names
    type_name_dict, raw_name_list = load_raw_names()

    # Second: name refine
    name_idx_dict = {}  # <phrase, index>
    name_list = []      # uhash of name_idx_dict
    idx_tp_dict = {}   # <phrase_idx, set(type)>
    for tp, name in raw_name_list:
        refined = refine_name(name)
        if refined == '': continue
        if refined not in name_idx_dict:
            new_idx = len(name_idx_dict)
            name_idx_dict[refined] = new_idx
            name_list.append(refined)
            idx_tp_dict[new_idx] = set([])
        phrase_idx = name_idx_dict[refined]
        idx_tp_dict[phrase_idx].add(tp)
    LogInfo.logs('names refined, generated %d distinct BOWs from %d <type, name> raw data.',
                 len(name_idx_dict), len(raw_name_list))

    # Third: build invert index
    inverted_index = {}     # <word, set(phrase_index)>
    for name, idx in name_idx_dict.items():
        spt = name.split('\t')
        for lemma in spt:
            if lemma not in inverted_index:
                inverted_index[lemma] = set([])
            inverted_index[lemma].add(idx)
    LogInfo.logs('%d entries of inverted index constructed.', len(inverted_index))
#    for lemma, name_idx_set in inverted_index.items():
#        LogInfo.logs('%s: %s', lemma, name_idx_set)
    with open(type_res_dir + '/type_link_inverted_index.pydump', 'wb') as bw:
        cPickle.dump(type_name_dict, bw)    # from type to its type.object.name
        cPickle.dump(idx_tp_dict, bw)       # from phrase index to types
        cPickle.dump(name_list, bw)         # from phrase index to phrases (BOW)
        cPickle.dump(inverted_index, bw)    # from word to phrase index
    LogInfo.logs('pickle complete.')





# same mapping: choose shortest (non-overlap)

if __name__ == '__main__':
    LogInfo.begin_track('[gen_type_index] starts ... ')
    prepare_type_index()
    LogInfo.end_track('Done.')