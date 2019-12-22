"""
Author: Kangqi Luo
Goal: Extract "silver" entity mention position from SimpQ.
"""

import codecs
import numpy as np

from ..u import load_simpq
from ...linking.parser import CoreNLPParser
from kangqi.util.LogUtil import LogInfo


parser_ip = '202.120.38.146'
parser_port = 9601      # 8601
parser = CoreNLPParser('http://%s:%d/parse' % (parser_ip, parser_port))   # just open the parser
lexicon_fp = '/home/xusheng/wikipedia/en-extracted/m_e_lexicon_v2.txt'
anno_fp = '/home/data/SimpleQuestions/SimpleQuestions_v2/mention_extract/simpq_all_mentions.txt'


def load_concern_mid(qa_list):
    concern_set = set([])
    for qa in qa_list:
        concern_set.add(qa['targetValue'][0])
    LogInfo.logs('%d concern mid loaded.', len(concern_set))
    return concern_set


def load_lexicon(concern_mid_set):
    LogInfo.begin_track('Preparing wiki entries')

    LogInfo.begin_track('Scanning raw entry from [%s]: ', lexicon_fp)
    mid_surface_dict = {}
    with open(lexicon_fp, 'r') as br:
        lines = br.readlines()
        LogInfo.logs('%d raw lines loaded.', len(lines))
        for line_idx, line in enumerate(lines):
            if line_idx % 200000 == 0:
                LogInfo.logs('Current line: %d / %d', line_idx, len(lines))
            try:
                line_u = line.decode('utf-8')
            except UnicodeDecodeError:
                # LogInfo.logs('Skip the line %d due to UnicodeDecodeError: [%s]', line_idx, line)
                continue
            spt = line_u.strip().split('\t')
            if len(spt) != 8:
                continue
            surface = spt[0].lower()
            mid = spt[2]
            if mid not in concern_mid_set:
                continue
            mid_surface_dict.setdefault(mid, set([])).add(surface)
    LogInfo.logs('MID hit ratio = %.6f (%d / %d)', 1. * len(mid_surface_dict) / len(concern_mid_set),
                 len(mid_surface_dict), len(concern_mid_set))
    sz_before_dedup = sum([len(v) for v in mid_surface_dict.values()])
    LogInfo.end_track('%d different <mid, surface> extracted.', sz_before_dedup)

    LogInfo.begin_track('Surface tokenization ...')
    surface_std_dict = {}       # <surface, standard form (after tokenize)>
    proced = 0
    for surface_set in mid_surface_dict.values():
        for surface in surface_set:
            if surface in surface_std_dict:
                continue
            try:
                tok_list = [tok.token for tok in parser.parse(surface).tokens]
                std = ' '.join(tok_list)
            except:
                LogInfo.logs('Tokenize exception for [%s].', surface.encode('utf-8'))
                std = ''
            surface_std_dict[surface] = std
            proced += 1
            if proced % 10000 == 0:
                LogInfo.logs('Current surface: %d', proced)
    LogInfo.end_track('All %d distinct surfaces have tokenized.', len(surface_std_dict))

    LogInfo.begin_track('Now deduping ...')
    proced = 0
    for mid, surface_set in mid_surface_dict.items():
        std_set = set([])
        for surface in surface_set:
            std_set.add(surface_std_dict[surface])
        entry_list = []
        for std in std_set:
            if std == '':           # encounter exception when performing tokenization
                continue
            entry_list.append(std.split(' '))
        mid_surface_dict[mid] = entry_list
        # LogInfo.logs(entry_list)
        proced += 1
        if proced % 10000 == 0:
            LogInfo.logs('Current mid: %s / %s', proced, len(mid_surface_dict))
    sz_after_dedup = sum([len(v) for v in mid_surface_dict.values()])
    LogInfo.end_track('Dedup complete, <mid, surface> from %d to %d (ratio = %.6f)',
                      sz_before_dedup, sz_after_dedup, 1. * sz_after_dedup / sz_before_dedup)

    LogInfo.end_track()
    return mid_surface_dict


""" ================ Deal with questions ================ """


def process_questions(qa_list, mid_surface_dict, save_fp):
    LogInfo.begin_track('Now scanning all questions ...')
    with codecs.open(save_fp, 'w', 'utf-8') as bw:
        for qa_idx, qa in enumerate(qa_list):
            if qa_idx % 5000 == 0:
                LogInfo.logs('Current: %d / %d', qa_idx, len(qa_list))
            q = qa['utterance']
            mid = qa['targetValue'][0]
            if mid not in mid_surface_dict:
                continue
            tokens = parser.parse(q).tokens
            tok_list = [tok.token for tok in tokens]
            st, ed, span_len, jac, entry = silver_mention_extraction(tok_list=tok_list,
                                                                     entry_list=mid_surface_dict[mid])
            if span_len == 0:
                continue
            bw.write('%d\t%d\t%d\t%.3f\t%s\t%s\t%s\n' % (
                qa_idx, st, ed, jac,
                ' '.join(tok_list[st: ed]),
                ' '.join(entry),
                ' '.join(tok_list)
            ))
    LogInfo.end_track()


def silver_mention_extraction(tok_list, entry_list):
    """
    Given q and all available entries, find the most similar n-gram to one of the surfaces in the list.
    surface_list: [ [wd] ]
    """
    sz = len(tok_list)
    srt_list = []           # [ (st, ed, len, jac, entry) ]
    for i in range(sz):
        for j in range(i+1, sz+1):
            for entry in entry_list:
                cur_ngram = tok_list[i: j]
                jac = jaccard(entry, cur_ngram)
                if jac > 0:
                    srt_list.append((i, j, j - i, jac, entry))
    srt_list.sort(cmp=cmp_item, reverse=True)
    # for item in srt_list:
    #     st, ed, size, jac, entry = item
    #     ngram = ' '.join(tok_list[st: ed])
    #     entry = ' '.join(entry)
    #     LogInfo.logs('[%d, %d), entry = %s, ngram = %s, jac = %.6f',
    #                  st, ed, entry, ngram, jac)

    if len(srt_list) == 0:
        return -1, -1, 0, 0, []             # st, ed, len, jac, entry
    return srt_list[0]


def cmp_item(ta, tb):
    if ta[3] == tb[3]:
        return ta[2] - tb[2]        # len DESC
    return -1 if ta[3] < tb[3] else 1


def jaccard(entry, ngram):
    # Jaccard similarity, considering duplicate words.
    joins = 0
    le = len(entry)
    ln = len(ngram)
    # entry_str = ' '.join(entry)
    # ngram_str = ' '.join(ngram)
    for wd in entry:
        found = -1
        for idx, wd_ngram in enumerate(ngram):
            if wd == wd_ngram:
                found = idx
                break
        if found != -1:
            joins += 1
            del ngram[found]
        if len(ngram) == 0:
            break
    jac = 1. * joins / (le + ln - joins)
    # LogInfo.logs('entry = %s, ngram = %s, joins = %d, jac = %.6f', entry_str, ngram_str, joins, jac)
    return jac


""" ================== Read the annotation, pick the good cases ======================= """


def load_annotations_bio(word_dict, q_max_len):
    """ Read annotation, convert to B,I,O format, and store into numpy array """
    LogInfo.begin_track('Load SimpQ-mention annotation from [%s]:', anno_fp)
    raw_tup_list = []       # [(v, v_len, tag)]
    with codecs.open(anno_fp, 'r', 'utf-8') as br:
        for line_idx, line in enumerate(br.readlines()):
            spt = line.strip().split('\t')
            q_idx, st, ed = [int(x) for x in spt[:3]]
            jac = float(spt[3])
            if jac != 1.0:
                continue        # only pick the most accurate sentences
            tok_list = spt[-1].lower().split(' ')
            v_len = len(tok_list)
            v = [word_dict[tok] for tok in tok_list]        # TODO: make sure all word exists
            tag = [2] * st + [0] + [1] * (ed-st-1) + [2] * (v_len-ed)       # 0: B, 1: I, 2: O
            # if line_idx < 10:
            #     LogInfo.begin_track('Check case-%d: ', line_idx)
            #     LogInfo.logs('tok_list: %s', tok_list)
            #     LogInfo.logs('v: %s', v)
            #     LogInfo.logs('tag: %s', tag)
            #     LogInfo.end_track()
            assert len(tag) == len(v)
            raw_tup_list.append((v, v_len, tag))
    q_size = len(raw_tup_list)
    v_len_list = [tup[1] for tup in raw_tup_list]
    LogInfo.logs('%d high-quality annotation loaded.', q_size)
    LogInfo.logs('maximum length = %d (%.6f on avg)', np.max(v_len_list), np.mean(v_len_list))
    for pos in (25, 50, 75, 90, 95, 99, 99.9):
        LogInfo.logs('Percentile = %.1f%%: %.6f', pos, np.percentile(v_len_list, pos))

    filt_tup_list = filter(lambda _tup: _tup[1] <= q_max_len, raw_tup_list)
    LogInfo.logs('%d / %d sentence filtered by [q_max_len=%d].',
                 len(filt_tup_list), q_size, q_max_len)

    # idx = 0
    for v, _, tag in filt_tup_list:
        v += [0] * (q_max_len - len(v))
        tag += [2] * (q_max_len - len(tag))
        # if idx < 10:
        #     LogInfo.begin_track('Check formed case-%d ', idx)
        #     LogInfo.logs('v: %s', v)
        #     LogInfo.logs('tag: %s', tag)
        #     LogInfo.end_track()
        # idx += 1
    v_list, v_len_list, tag_list = [[tup[i] for tup in filt_tup_list]
                                    for i in range(3)]
    np_data_list = [
        np.array(v_list, dtype='int32'),        # (ds, q_max_len)
        np.array(v_len_list, dtype='int32'),    # (ds, )
        np.array(tag_list, dtype='int32')       # (ds, num_classes)
    ]
    for idx, np_data in enumerate(np_data_list):
        LogInfo.logs('np-%d: %s', idx, np_data.shape)
    LogInfo.end_track()
    return np_data_list


""" ====================================================== """


def test():
    # q = "what artist's creates riot grrrl music didn't happen"
    q = "Province, Territory, Centrally Administered Area, or Capital Territory"
    tok_list = [tok.token for tok in parser.parse(q).tokens]
    LogInfo.logs(tok_list)
    # entry_list = [['riot'],
    #               ['grrrl'],
    #               ['grrrl', 'music'],
    #               ['riot', 'grrrl', 'music']]


def main():
    qa_list = load_simpq()
    concern_mid_set = load_concern_mid(qa_list=qa_list)
    mid_surface_dict = load_lexicon(concern_mid_set=concern_mid_set)
    process_questions(qa_list=qa_list,
                      mid_surface_dict=mid_surface_dict,
                      save_fp=anno_fp)


if __name__ == '__main__':
    test()
