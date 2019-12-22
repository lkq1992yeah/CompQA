"""
Author: Kangqi Luo
Goal: Entity/Type/Time linking based on n-grams.
Note: all linking procedure ignore cases.
"""

import os
import re
import codecs
import argparse

from ..parser import CoreNLPParser
from ...dataset.u import load_compq

from kangqi.util.LogUtil import LogInfo


punc_mask_str = "?!:', "
year_re = re.compile(r'^[1-2][0-9][0-9][0-9]$')

working_ip = '202.120.38.146'
parser_port_dict = {'Blackhole': 9601, 'Darkstar': 8601}
sparql_port_dict = {'Blackhole': 8999, 'Darkstar': 8699}

parser = argparse.ArgumentParser(description='Lukov Linker')
parser.add_argument('--machine', default='Blackhole', choices=['Blackhole', 'Darkstar'])
parser.add_argument('--data_name', choices=['WebQ', 'CompQ', 'SimpQ'])
parser.add_argument('--link_save_path')
parser.add_argument('--link_name')
parser.add_argument('--allow_alias', action='store_true')


class LukovLinkTuple:

    def __init__(self, surface, start, length, mid, name, score, popularity, category):
        self.surface = surface
        self.start = start
        self.length = length
        self.mid = mid
        self.name = name

        self.score = score
        self.popularity = popularity
        self.category = category
        """
            surface: the mention surface in the question
            start: the starting token index of the surface
            length: the length (token level) of the mention surface
            mid: the linked entity mid
            name: corresponding type.object.name
            popularity: number of facts that the entity occurs
            category: this mid is an Entity or Type
            
            score: entity linking score (haven't put into use yet)
        """

    @staticmethod
    def read_from_split_line(spt):
        tup = LukovLinkTuple(surface=spt[1],
                             start=int(spt[2]), length=int(spt[3]),
                             mid=spt[4], name=spt[5],
                             score=float(spt[6]),
                             popularity=int(spt[7]),
                             category=spt[8])
        return tup


class LukovLinker:

    def __init__(self, parser_ip, parser_port,
                 mid_name_fp='data/fb_metadata/S-NAP-ENO-triple.txt',
                 type_name_fp='data/fb_metadata/TS-name.txt',
                 pred_name_fp='data/fb_metadata/PS-name.txt',
                 e_pop_fp='data/fb_metadata/entity_pop_5m.txt',
                 t_pop_fp='data/fb_metadata/type_pop.txt',
                 allow_alias=False):
        self.parser = CoreNLPParser('http://%s:%d/parse' % (parser_ip, parser_port))    # just open the parser
        self.surface_mid_dict = {}      # <surface, set([mid])>
        self.mid_name_dict = {}         # <mid, type.object.name>
        self.type_set = set([])
        self.pred_set = set([])
        self.pop_dict = {}              # <mid, popularity>
        LogInfo.begin_track('Loading surface --> mid dictionary from [%s] ...', mid_name_fp)
        LogInfo.logs('Allow alias = %s', allow_alias)
        skip_domain_set = {'common', 'type', 'user', 'base', 'freebase', 'g'}

        with codecs.open(mid_name_fp, 'r', 'utf-8') as br:
            scan = 0
            while True:
                line = br.readline()
                if line is None or line == '':
                    break
                spt = line.strip().split('\t')
                if len(spt) < 3:
                    continue
                mid = spt[0]
                name = spt[2]
                surface = name.lower()      # save lowercase as searching entrance
                skip = False                # ignore some subjects at certain domain
                mid_prefix_pos = mid.find('.')
                if mid_prefix_pos == -1:
                    skip = True
                else:
                    mid_prefix = mid[: mid_prefix_pos]
                    if mid_prefix in skip_domain_set:
                        skip = True
                if not skip:
                    if spt[1] == 'type.object.name':
                        self.mid_name_dict[mid] = name
                    if spt[1] == 'type.object.name' or allow_alias:
                        self.surface_mid_dict.setdefault(surface, set([])).add(mid)
                scan += 1
                if scan % 100000 == 0:
                    LogInfo.logs('%d lines scanned.', scan)
                # if scan >= 10000:
                #     break
        LogInfo.logs('%d lines scanned.', scan)
        LogInfo.logs('%d <surface, mid_set> loaded.', len(self.surface_mid_dict))
        LogInfo.logs('%d <mid, name> loaded.', len(self.mid_name_dict))

        with codecs.open(type_name_fp, 'r', 'utf-8') as br:
            for line in br.readlines():
                spt = line.strip().split('\t')
                if len(spt) < 2:
                    continue
                type_mid, type_name = spt[0], spt[1]
                surface = type_name.lower().replace('(s)', '')
                type_prefix = type_mid[: type_mid.find('.')]
                if type_prefix not in skip_domain_set:
                    self.surface_mid_dict.setdefault(surface, set([])).add(type_mid)
                    self.mid_name_dict[type_mid] = type_name
                    self.type_set.add(type_mid)
        LogInfo.logs('After scanning %d types, %d <surface, mid_set> loaded.',
                     len(self.type_set), len(self.surface_mid_dict))

        with codecs.open(pred_name_fp, 'r', 'utf-8') as br:
            for line in br.readlines():
                spt = line.strip().split('\t')
                if len(spt) < 2:
                    continue
                self.pred_set.add(spt[0])
        LogInfo.logs('%d predicates scanned.', len(self.pred_set))

        for pop_fp in [e_pop_fp, t_pop_fp]:
            LogInfo.logs('Reading popularity from %s ...', pop_fp)
            with codecs.open(pop_fp, 'r', 'utf-8') as br:
                for line in br.readlines():
                    spt = line.strip().split('\t')
                    self.pop_dict[spt[0]] = int(spt[1])
        LogInfo.logs('%d <mid, popularity> loaded.', len(self.pop_dict))
        LogInfo.end_track()

    def link_single_question(self, q):
        link_tups = []
        parse_result = self.parser.parse(q)
        token_list = parse_result.tokens
        tokens = len(token_list)
        for i in range(tokens):
            for j in range(i+1, tokens):
                std_surface = ' '.join([tok.token for tok in token_list[i: j]])
                # LogInfo.begin_track('std_surface: %s', std_surface)
                lower_surface_list = build_lowercased_surface(token_list, i, j)
                match_set = set([])
                if re.match(year_re, std_surface):
                    match_set.add(std_surface)      # time value
                for surf in lower_surface_list:
                    match_set |= self.surface_mid_dict.get(surf, set([]))
                for match_mid in match_set:
                    if match_mid in self.pred_set:
                        continue        # won't match a mention into predicates
                    if re.match(year_re, match_mid):
                        mid_name = match_mid
                        pop = 1000      # set to a rather high value
                        category = 'Time'
                    else:
                        mid_name = self.mid_name_dict[match_mid]
                        pop = self.pop_dict.get(match_mid, 1)
                        category = 'Type' if match_mid in self.type_set else 'Entity'
                    # LogInfo.logs('match_mid: %s, name: %s', match_mid, mid_name)
                    tup = LukovLinkTuple(surface=std_surface,
                                         start=i, length=j-i,
                                         mid=match_mid, name=mid_name,
                                         popularity=pop,
                                         category=category)
                    link_tups.append(tup)
                # LogInfo.end_track()
        LogInfo.logs('%d link tuples retrieved.', len(link_tups))
        return link_tups


def build_lowercased_surface(token_list, st, ed):
    """
    Construct the surface form in [st, ed)
    Be careful: Their may have or may not have a blank before the token starting with a punctuation
                For example, "'s" is a token starting with a punctuation.
                It's hard to say whether we shall directly connect two the last token, or adding a blank
                But we can enumerate each possibility.
    Thus, the return value is a list of possible surfaces.
    """
    surface_list = ['']
    for idx in range(st, ed):
        tok = token_list[idx].token.lower()
        if idx == st:           # must not add blank
            new_surface_list = [x + tok for x in surface_list]
        elif idx > st and tok[0] not in punc_mask_str:  # must add blank
            new_surface_list = [x + ' ' + tok for x in surface_list]
        else:       # both add or not add are possible
            tmp1_list = [x + tok for x in surface_list]
            tmp2_list = [x + ' ' + tok for x in surface_list]
            new_surface_list = tmp1_list + tmp2_list
        surface_list = new_surface_list
    return surface_list


def load_lukov_link_result(link_fp):
    q_links_dict = {}
    LogInfo.begin_track('Loading lukov linking data from [%s] ...', link_fp)
    with codecs.open(link_fp, 'r', 'utf-8') as br:
        lines = br.readlines()
        for scan, line in enumerate(lines):
            if scan % 500000 == 0:
                LogInfo.logs('Scanned %d / %d rows.', scan, len(lines))
            spt = line.strip().split('\t')
            q_id = spt[0]
            q_idx = int(q_id[q_id.find('-') + 1:])
            tup = LukovLinkTuple.read_from_split_line(spt)
            q_links_dict.setdefault(q_idx, []).append(tup)
    LogInfo.end_track('Read %d linkings for %d questions from [%s].',
                 len(lines), len(q_links_dict), link_fp)
    return q_links_dict


def main(args):
    parser_port = parser_port_dict[args.machine]
    assert args.data_name == 'CompQ'

    qa_list = load_compq()
    linker = LukovLinker(parser_ip=working_ip,
                         parser_port=parser_port,
                         allow_alias=args.allow_alias)
    if not os.path.exists(args.link_save_path):
        os.makedirs(args.link_save_path)
    save_fp = args.link_save_path + '/' + args.link_name
    LogInfo.begin_track('Linking data save to: %s', save_fp)
    with codecs.open(save_fp, 'w', 'utf-8') as bw:
        for q_idx, qa in enumerate(qa_list):
            q = qa['utterance']
            LogInfo.begin_track('Entering Q-%d [%s]:', q_idx, q.encode('utf-8'))
            link_tups = linker.link_single_question(q)
            for tup in link_tups:
                bw.write('CompQ-%d\t%s\t%d\t%d\t%s\t%s\t%.6f\t%d\t%s\n' % (
                    q_idx, tup.surface, tup.start, tup.length,
                    tup.mid, tup.name, tup.score, tup.popularity, tup.category
                ))
            LogInfo.end_track()
    LogInfo.end_track()


if __name__ == '__main__':
    _args = parser.parse_args()
    main(_args)
