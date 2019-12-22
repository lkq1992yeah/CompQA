"""
Author: Kangqi Luo
Goal: Collect ALL lowercase words from WebQ, CompQ, SimpQ, CoNLL-2003, predicate name
      We generate such large word dictionary and it won't be changed among different experiments.
      A rough estimation: ~100K words (20 times smaller than GloVe)
      GloVe voc: 2196K
      Add <START> and <PAD> as virtual words
"""

import codecs
import cPickle
import numpy as np

from ...dataset.u import load_simpq, load_webq, load_compq
from ...util.fb_helper import adjust_name, load_domain_range, load_type_name, \
    inverse_predicate, get_item_name
from ...util.word_emb import WordEmbeddingUtil
from kangqi.util.LogUtil import LogInfo


def update_from_qa(word_dict, qa_list):
    for qa in qa_list:
        tok_list = [tok.token.lower() for tok in qa['tokens']]
        for wd in tok_list:
            word_dict.setdefault(wd, len(word_dict))


def update_from_conll_2003(word_dict, conll_fp):
    with codecs.open(conll_fp, 'r', 'utf-8') as br:
        for line in br.readlines():
            line = line.strip()
            if line == '':
                continue
            spt = line.split(' ')
            token = spt[0].lower()
            word_dict.setdefault(token, len(word_dict))


def update_from_mid_name(word_dict, mid_name_fp):
    with codecs.open(mid_name_fp, 'r', 'utf-8') as br:
        for line in br.readlines():
            spt = line.strip().split('\t')
            name = adjust_name(spt[-1])
            for wd in name.split(' '):
                word_dict.setdefault(wd, len(word_dict))


def create_new_emb_value(full_indices_fp, full_matrix_fp, target_indices_fp, out_target_matrix_fp):
    with open(full_indices_fp, 'r') as br:
        full_word_dict = cPickle.load(br)
    LogInfo.logs('Full size: %d', len(full_word_dict))
    with open(target_indices_fp, 'r') as br:
        target_word_dict = cPickle.load(br)
    LogInfo.logs('Target size: %d', len(target_word_dict))

    full_emb_mat = np.load(full_matrix_fp)
    LogInfo.logs('Full matrix: %s', full_emb_mat.shape)
    full_size, dim_emb = full_emb_mat.shape
    target_size = len(target_word_dict)
    target_emb_mat = np.random.uniform(
        low=-0.1, high=0.1, size=(target_size, dim_emb)).astype('float32')
    for wd in target_word_dict:
        if wd in ('<START>', '<PAD>', '<UNK>'):
            continue        # leave them alone
        target_idx = target_word_dict[wd]
        if wd in full_word_dict:
            full_idx = full_word_dict[wd]
            target_emb_mat[target_idx] = full_emb_mat[full_idx]
    LogInfo.logs('Ready for saving ...')
    np.save(out_target_matrix_fp, target_emb_mat)
    LogInfo.logs('Saving Done.')


def prepare_global_dict():
    word_dict = {}
    word_dict.setdefault('<PAD>', len(word_dict))
    word_dict.setdefault('<START>', len(word_dict))
    word_dict.setdefault('<UNK>', len(word_dict))

    word_dict.setdefault('<E>', len(word_dict))
    word_dict.setdefault('<T>', len(word_dict))
    word_dict.setdefault('<Tm>', len(word_dict))
    word_dict.setdefault('<Ord>', len(word_dict))

    word_dict.setdefault('min', len(word_dict))     # check kq_schema.py for the detail of virtual predicates
    word_dict.setdefault('max', len(word_dict))
    word_dict.setdefault('before', len(word_dict))
    word_dict.setdefault('after', len(word_dict))
    word_dict.setdefault('in', len(word_dict))
    word_dict.setdefault('since', len(word_dict))

    update_from_mid_name(word_dict, 'data/fb_metadata/PS-name.txt')
    update_from_mid_name(word_dict, 'data/fb_metadata/TS-name.txt')
    LogInfo.logs('After PS/TS-name: %d', len(word_dict))

    compq_list = load_compq()
    update_from_qa(word_dict, compq_list)
    LogInfo.logs('After CompQ: %d', len(word_dict))

    webq_list = load_webq()
    update_from_qa(word_dict, webq_list)
    LogInfo.logs('After WebQ: %d', len(word_dict))

    simpq_list = load_simpq()
    update_from_qa(word_dict, simpq_list)
    LogInfo.logs('After SimpQ: %d', len(word_dict))

    update_from_conll_2003(word_dict, '/home/data/conll03/eng.train')
    update_from_conll_2003(word_dict, '/home/data/conll03/eng.testa')
    update_from_conll_2003(word_dict, '/home/data/conll03/eng.testb')
    LogInfo.logs('After conll03: %d', len(word_dict))

    with open('data/compQA/word_emb_in_use/word_emb.indices', 'w') as bw:
        cPickle.dump(word_dict, bw)
    LogInfo.logs('Dump complete.')      # 84403


def prepare_mid_dict():
    mid_dict = {}
    mid_dict.setdefault('<PAD>', len(mid_dict))
    mid_dict.setdefault('<START>', len(mid_dict))
    mid_dict.setdefault('<UNK>', len(mid_dict))

    mid_dict.setdefault('m.__in__', len(mid_dict))
    mid_dict.setdefault('m.__before__', len(mid_dict))
    mid_dict.setdefault('m.__after__', len(mid_dict))
    mid_dict.setdefault('m.__since__', len(mid_dict))
    mid_dict.setdefault('m.__max__', len(mid_dict))
    mid_dict.setdefault('m.__min__', len(mid_dict))

    type_name_dict = load_type_name()
    for tp in type_name_dict:
        mid_dict.setdefault(tp, len(mid_dict))
    LogInfo.logs('After TS-name: %d', len(mid_dict))

    pred_domain_dict, pred_range_dict = load_domain_range()
    for pred, tp_domain in pred_domain_dict.items():
        mid_dict.setdefault(pred, len(mid_dict))
        mid_dict.setdefault(inverse_predicate(pred), len(mid_dict))
        mid_dict.setdefault(tp_domain, len(mid_dict))
    LogInfo.logs('After pred-domain: %d', len(mid_dict))

    for pred, tp_range in pred_range_dict.items():
        mid_dict.setdefault(pred, len(mid_dict))
        mid_dict.setdefault(inverse_predicate(pred), len(mid_dict))
        mid_dict.setdefault(tp_range, len(mid_dict))
    LogInfo.logs('After pred-range: %d', len(mid_dict))

    target_size = len(mid_dict)
    with open('data/compQA/word_emb_in_use/mid_emb.indices', 'w') as bw:
        cPickle.dump(mid_dict, bw)
    LogInfo.logs('Dump %d mid complete.', target_size)

    wd_emb_util = WordEmbeddingUtil('glove', 300)
    target_emb_mat = np.random.uniform(
        low=-0.1, high=0.1, size=(target_size, 300)).astype('float32')
    for mid in mid_dict:
        target_idx = mid_dict[mid]
        name = get_item_name(mid)
        if name != '':
            emb = wd_emb_util.get_phrase_emb(phrase=name)
            if emb is not None:
                target_emb_mat[target_idx] = emb
    LogInfo.logs('Ready for saving ...')
    np.save('data/compQA/word_emb_in_use/mid_emb.glove_300.npy', target_emb_mat)
    LogInfo.logs('Saving Done.')        # 149437


def main():
    prepare_global_dict()
    full_indices_fp = 'data/glove/vec/en_300.txt.indices'
    full_matrix_fp = 'data/glove/vec/en_300.txt.npy'
    target_indices_fp = 'data/compQA/word_emb_in_use/word_emb.indices'
    out_target_matrix_fp = 'data/compQA/word_emb_in_use/word_emb.glove_300.npy'
    create_new_emb_value(full_indices_fp, full_matrix_fp, target_indices_fp, out_target_matrix_fp)

    # prepare_mid_dict()


if __name__ == '__main__':
    main()
