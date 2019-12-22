"""
Goal: given a list of target <mention, entity> pairs, check their position in the whole lexicon.
"""
import json
import codecs
import numpy as np
import xmlrpclib
from collections import namedtuple

from .lexicon import Lexicon, LexiconEntry
from ...candgen_acl18.global_linker import LinkData
from ...dataset.kq_schema import CompqSchema
from ...dataset.u import load_compq

from kangqi.util.LogUtil import LogInfo


def load_xs_word_indices():
    xs_indices_fp = '/home/xusheng/word2vec/vec/en_wiki_indices.txt'
    xs_word_indices = set([])
    found_bracket = 0
    with open(xs_indices_fp, 'r') as br:
        lines = br.readlines()
        for line_idx, line in enumerate(lines):
            line = line.strip()
            if not (line.startswith('[[') and line.endswith(']]')):
                continue
            found_bracket += 1
            try:
                line_u = line.decode('utf-8')
                if not (line_u.startswith('[[') and line_u.endswith(']]')):
                    LogInfo.logs('Warning: line_%d --> %s', line_idx+1, line)
                assert line_u.startswith('[[') and line_u.endswith(']]')
                xs_word_indices.add(line_u[2:-2])
            except UnicodeDecodeError:
                continue
    LogInfo.logs('%d raw indices --> %d brackets --> %d utf-8 indices.',
                 len(lines), found_bracket, len(xs_word_indices))
    return xs_word_indices


def kernel_collect(q_idx, link_fp, schema_fp, concern_dict):
    gather_linkings = []
    with codecs.open(link_fp, 'r', 'utf-8') as br:
        for gl_line in br.readlines():
            tup_list = json.loads(gl_line.strip())
            ld_dict = {k: v for k, v in tup_list}
            gather_linkings.append(LinkData(**ld_dict))

    with codecs.open(schema_fp, 'r', 'utf-8') as br:
        sc_lines = br.readlines()
        for ori_idx, sc_line in enumerate(sc_lines):
            sc = CompqSchema.read_schema_from_json(q_idx=q_idx, json_line=sc_line,
                                                   gather_linkings=gather_linkings)
            f1 = sc.f1
            for category, link_data, pred_seq in sc.raw_paths:
                if link_data.category == 'Entity':
                    mention = link_data.mention.lower()
                    std_name = link_data.name
                    mid = link_data.value
                    key = '%d\t%s\t%s\t%s' % (q_idx, mention, mid, std_name)
                    prev_f1 = concern_dict.get(key, 0.)
                    concern_dict[key] = max(prev_f1, f1)


def collect_concern_pairs(data_dir):        # Collect <mention, mid> from S-MART based CompQ schemas.
    concern_dict = {}
    for q_idx in range(2100):
        if q_idx % 100 == 0:
            LogInfo.logs('%d scanned, got %d different <mention, mid> pairs.', q_idx, len(concern_dict))
        div = q_idx / 100
        sub_dir = '%d-%d' % (div*100, div*100+99)
        link_fp = '%s/data/%s/%d_links' % (data_dir, sub_dir, q_idx)
        schema_fp = '%s/data/%s/%d_schema' % (data_dir, sub_dir, q_idx)
        kernel_collect(q_idx, link_fp, schema_fp, concern_dict)

    save_name = data_dir + '/lexicon_validate/mention_mid_pairs.txt'
    with codecs.open(save_name, 'w', 'utf-8') as bw:
        for k, f1 in concern_dict.items():
            bw.write('%s\t%.6f\n' % (k, f1))
    LogInfo.logs('Save complete.')
    return concern_dict


def retrieve_rank_from_lexicon(data_dir, lexicon):
    input_fp = data_dir + '/lexicon_validate/mention_mid_pairs.txt'
    output_fp = data_dir + '/lexicon_validate/srt_output.link_prob.txt'
    with codecs.open(input_fp, 'r', 'utf-8') as br, codecs.open(output_fp, 'w', 'utf-8') as bw:
        for line_idx, line in enumerate(br.readlines()):
            if line_idx % 200 == 0:
                LogInfo.logs('Current: %d lines.', line_idx)
            line = line.strip()
            q_idx_str, mention, mid, name, f1_str = line.split('\t')
            entry_list = lexicon.get(surface=mention)
            rank = -1
            for idx, entry in enumerate(entry_list):
                if entry.mid == mid:
                    rank = idx + 1
                    break
            bw.write('%s\t%d\n' % (line, rank))


def analyze_output(data_dir, sort_item):
    """ Check the rank distribution in a global perspective """
    rank_matrix = [[], [], [], []]      # focus on 4 tiers: F1 = 1.0, F1 >= 0.5, F1 >= 0.1, F1 > 0
    fp = '%s/lexicon_validate/srt_output.%s.txt' % (data_dir, sort_item)
    with codecs.open(fp, 'r', 'utf-8') as br:
        for line in br.readlines():
            spt = line.strip().split('\t')
            f1 = float(spt[-2])
            rank = int(spt[-1])
            if rank == -1:
                rank = 2111222333
            if f1 == 1.0:
                rank_matrix[0].append(rank)
            if f1 >= 0.5:
                rank_matrix[1].append(rank)
            if f1 >= 0.1:
                rank_matrix[2].append(rank)
            if f1 >= 1e-6:
                rank_matrix[3].append(rank)
    for ths, rank_list in zip((1.0, 0.5, 0.1, 1e-6), rank_matrix):
        LogInfo.begin_track('Show stat for F1 >= %.6f:', ths)
        rank_list = np.array(rank_list)
        case_size = len(rank_list)
        LogInfo.logs('Total cases = %d.', case_size)
        LogInfo.logs('MRR = %.6f', np.mean(1. / rank_list))
        for pos in (50, 60, 70, 80, 90, 95, 99, 99.9, 100):
            LogInfo.logs('Percentile = %.1f%%: %.6f', pos, np.percentile(rank_list, pos))
        LogInfo.end_track()


CheckItem = namedtuple('CheckItem', ['q_idx', 'mention', 'mid', 'std_name', 'f1', 'rank'])


def check_one_by_one(data_dir, sort_item, lexicon, xs_word_indices):
    qa_list = load_compq()
    fp = '%s/lexicon_validate/srt_output.%s.txt' % (data_dir, sort_item)
    check_tups = []
    got_emb = 0
    with codecs.open(fp, 'r', 'utf-8') as br:
        for line in br.readlines():
            spt = line.strip().split('\t')
            q_idx_str, mention, mid, std_name, f1_str, rank_str = spt
            q_idx = int(q_idx_str)
            f1 = float(f1_str)
            rank = int(rank_str)
            if f1 == 1.:
                check_tups.append(CheckItem(q_idx, mention, mid, std_name, f1, rank))
                check_name = std_name.lower().replace(' ', '_')
                if check_name in xs_word_indices:       # check whether XS's embedding contains this item
                    got_emb += 1
                else:
                    LogInfo.logs('W/O embedding: %s', check_name)
    check_tups.sort(key=lambda _tup: _tup.q_idx)

    LogInfo.begin_track('%d check items for analyze:', len(check_tups))
    miss = len(filter(lambda _tup: _tup.rank == -1, check_tups))
    case_idx = 0
    LogInfo.logs('Missing: %d / %d = %.6f%%', miss, len(check_tups), 100. * miss / len(check_tups))
    LogInfo.logs('Get_emb: %d / %d = %.6f%%', got_emb, len(check_tups), 100. * got_emb / len(check_tups))
    # for check_item in check_tups:
    #     if check_item.rank != -1:
    #         continue
    #     case_idx += 1
    #     LogInfo.begin_track('Checking case %d / %d:', case_idx, miss)
    #     LogInfo.logs('Q_idx = %d [%s]', check_item.q_idx, qa_list[check_item.q_idx]['utterance'].encode('utf-8'))
    #     LogInfo.logs('Mention: [%s] --> %s (%s)',
    #                  check_item.mention.encode('utf-8'),
    #                  check_item.mid,
    #                  check_item.std_name.encode('utf-8'))
    #     entry_list = lexicon.get(check_item.mention)
    #     for ent_idx, entry in enumerate(entry_list):
    #         if ent_idx >= 10:
    #             break
    #         lex_entry = LexiconEntry(**entry)
    #         LogInfo.logs('#%d/%d: %s', ent_idx + 1, len(entry_list), lex_entry.display())
    #     LogInfo.logs('Anno: []')
    #     LogInfo.end_track()
    #     # if case_idx >= 10:
    #     #     break

    LogInfo.end_track()


def main():
    data_dir = 'runnings/candgen_CompQ/180209_ACL18_SMART'
    # lexicon_fp = '/home/xusheng/wikipedia/en-extracted/m_e_lexicon_v2.txt'
    # type_name_fp = 'data/fb_metadata/TS-name.txt'
    # lexicon = Lexicon(lexicon_fp=lexicon_fp,
    #                   type_name_fp=type_name_fp,
    #                   link_prob_filter_num=0,
    #                   load_types=False)
    lexicon = xmlrpclib.ServerProxy('http://202.120.38.146:9602')
    LogInfo.logs('Lukov Lexicon Proxy binded.')

    # collect_concern_pairs(data_dir)
    # retrieve_rank_from_lexicon(data_dir, lexicon)
    # analyze_output(data_dir, 'link_prob')

    xs_word_indices = load_xs_word_indices()
    check_one_by_one(data_dir, 'link_prob', lexicon, xs_word_indices)


if __name__ == '__main__':
    main()
