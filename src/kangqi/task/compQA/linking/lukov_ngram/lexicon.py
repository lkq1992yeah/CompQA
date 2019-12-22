"""
Author: Kangqi
Goal: Surface --> Entity Lexicon
"""

import codecs

from SimpleXMLRPCServer import SimpleXMLRPCServer
from SimpleXMLRPCServer import SimpleXMLRPCRequestHandler

from kangqi.util.LogUtil import LogInfo


class LexiconEntry(object):

    def __init__(self, wiki_name, fb_name, mid, category, stat_dict):
        self.wiki_name = wiki_name          # entity name in wiki
        self.fb_name = fb_name              # entity name in FB
        self.mid = mid                      # entity mid
        self.category = category            # Entity / Type
        self.stat_dict = stat_dict  # the dictionary storing all statistics

    def __str__(self):
        return 'mid = [%s], wiki_name = [%s], fb_name = [%s], category = [%s], stat_dict = %s' % (
            self.mid, self.wiki_name, self.fb_name, self.category, self.stat_dict
        )

    def display(self):
        disp = ''
        disp += '[%s] %s' % (self.category, self.mid)
        if self.wiki_name != '':
            disp += ' | wk_[%s]' % self.wiki_name
        if self.fb_name != '':
            disp += ' | fb_[%s]' % self.fb_name
        disp += ' | %s' % self.stat_dict
        return disp


class Lexicon:

    def __init__(self, lexicon_dir='/home/xusheng/wikipedia/en-extracted',
                 type_name_fp='data/fb_metadata/TS-name.txt',
                 link_prob_filter_num=5, load_types=True):
        self.stat_keys = ['link_prob', 'word_jac', 'ngram_jac', 'wiki_pop', 'fb_pop']
        skip_domain_set = {'common', 'type', 'user', 'base', 'freebase', 'g'}
        self.entry_dict = {}

        LogInfo.begin_track('Lexicon initializing:')
        LogInfo.logs('link_prob_filter = %d', link_prob_filter_num)

        self_fp = lexicon_dir + '/lexicon_entity.txt'
        anchor_fp = lexicon_dir + '/lexicon_anchor.txt'
        redirect_fp = lexicon_dir + '/lexicon_redirect.txt'
        disamb_fp = lexicon_dir + '/lexicon_disamb.txt'

        self_entry_dict = self.load_lexicon_from_fp(self_fp)
        redirect_entry_dict = self.load_lexicon_from_fp(redirect_fp)
        anchor_entry_dict = self.load_lexicon_from_fp(anchor_fp)
        disamb_entry_dict = self.load_lexicon_from_fp(disamb_fp)

        surface_set = set(self_entry_dict.keys() + redirect_entry_dict.keys() + anchor_entry_dict.keys())
        LogInfo.begin_track('Now perform entry filtering over %d surfaces:', len(surface_set))
        final_keep = 0
        for surface in surface_set:
            found_mid_set = set([])     # priority: self > redirect > disamb > anchor (top-K)
            """ Collect from self / redirect / disamb """
            for entry_dict in (self_entry_dict, redirect_entry_dict, disamb_entry_dict):
                for entry in entry_dict.get(surface, []):
                    mid = entry.mid
                    if mid not in found_mid_set:
                        found_mid_set.add(mid)
                        self.entry_dict.setdefault(surface, []).append(entry)
            """ Collect from anchor """
            entry_list = anchor_entry_dict.get(surface, [])
            entry_list.sort(key=lambda _entry: _entry.stat_dict['link_prob'], reverse=True)
            if link_prob_filter_num > 0:
                filt_entry_list = entry_list[: link_prob_filter_num]
            else:
                filt_entry_list = entry_list
            for entry in filt_entry_list:
                mid = entry.mid
                if mid not in found_mid_set:
                    found_mid_set.add(mid)
                    self.entry_dict.setdefault(surface, []).append(entry)
            final_keep += len(self.entry_dict[surface])
        LogInfo.end_track('%d entries (%.3f per surface) kept after filtering by link probability.',
                          final_keep, 1. * final_keep / len(surface_set))

        if load_types:
            with codecs.open(type_name_fp, 'r', 'utf-8') as br:
                keep_types = 0
                for line in br.readlines():
                    spt = line.strip().split('\t')
                    if len(spt) < 2:
                        continue
                    type_mid, type_name = spt[0], spt[1]
                    surface = type_name.lower().replace('(s)', '')
                    type_prefix = type_mid[: type_mid.find('.')]
                    if type_prefix not in skip_domain_set:
                        lex_entry = LexiconEntry(wiki_name='', fb_name=type_name, mid=type_mid,
                                                 category='Type', stat_dict={})
                        self.entry_dict.setdefault(surface, []).append(lex_entry)
                        keep_types += 1
            LogInfo.logs('Scanned %d type entries from [%s].', keep_types, type_name_fp)

        LogInfo.end_track()

    def load_lexicon_from_fp(self, lexicon_fp):
        LogInfo.begin_track('Loading wiki-anchor information from [%s]:', lexicon_fp)
        keep = 0
        entry_dict = {}
        with open(lexicon_fp, 'r') as br:
            lines = br.readlines()
            LogInfo.logs('%d raw lines loaded.', len(lines))
            for line_idx, line in enumerate(lines):
                if line_idx % 200000 == 0:
                    LogInfo.logs('Current: %d / %d', line_idx, len(lines))
                """
                Due to different codings in the lexicon file, we try convert to unicode through utf-8 manually
                """
                try:
                    line_u = line.decode('utf-8')
                except UnicodeDecodeError:
                    # LogInfo.logs('Skip the line %d due to UnicodeDecodeError: [%s]', line_idx, line)
                    continue
                spt = line_u.strip().split('\t')
                if len(spt) != 8:
                    continue
                surface, wiki_name, mid = spt[:3]  # all lowercased
                surface = surface.replace('\\', '')
                wiki_name = wiki_name.replace('\\', '')  # don't know why ...
                if not mid.startswith('m.'):
                    continue
                keep += 1
                stat_dict = {k: float(v) for k, v in zip(self.stat_keys, spt[3: 8])}
                lex_entry = LexiconEntry(wiki_name=wiki_name, fb_name='', mid=mid,
                                         category='Entity', stat_dict=stat_dict)
                entry_dict.setdefault(surface, []).append(lex_entry)
        LogInfo.end_track('%d out of %d raw entries kept.', keep, len(lines))
        return entry_dict

    def contains(self, surface):
        return surface in self.entry_dict

    def get(self, surface):
        return self.entry_dict.get(surface, [])


def local_test():
    Lexicon()


def start_service():

    # Restrict to a particular path.
    class RequestHandler(SimpleXMLRPCRequestHandler):
        rpc_paths = ('/RPC2',)

    srv_port = 9602
    server = SimpleXMLRPCServer(("0.0.0.0", srv_port), requestHandler=RequestHandler)
    server.register_introspection_functions()

    lexicon_dir = '/home/xusheng/wikipedia/en-extracted'
    type_name_fp = 'data/fb_metadata/TS-name.txt'
    lexicon = Lexicon(lexicon_dir=lexicon_dir, type_name_fp=type_name_fp,
                      link_prob_filter_num=10, load_types=False)

    # import math
    # from kangqi.util.discretizer import Discretizer
    # log_wiki_pop_disc = Discretizer([1, 2, 3, 4, 6], output_mode='list', name='log_wiki_pop')
    # log_fb_pop_disc = Discretizer([3, 4, 6, 8], output_mode='list', name='log_fb_pop')
    #
    # LogInfo.begin_track('Performing discretizing: ')
    # scan = 0
    # for entry_list in lexicon.entry_dict.values():
    #     for lex_entry in entry_list:
    #         if lex_entry.category != 'Entity':
    #             continue
    #         wiki_pop = lex_entry.stat_dict['wiki_pop']
    #         fb_pop = lex_entry.stat_dict['fb_pop']
    #         log_wiki_pop = None if wiki_pop <= 0 else math.log(wiki_pop)
    #         log_fb_pop = None if fb_pop <= 0 else math.log(fb_pop)
    #         log_wiki_pop_disc.convert(score=log_wiki_pop)
    #         log_fb_pop_disc.convert(score=log_fb_pop)
    #         scan += 1
    #         if scan % 500000 == 0:
    #             LogInfo.logs('%d Enity lex_entry scanned.', scan)
    # log_wiki_pop_disc.show_distribution()
    # log_fb_pop_disc.show_distribution()
    # LogInfo.end_track()

    server.register_function(lexicon.get)
    server.register_function(lexicon.contains)
    LogInfo.logs('Functions registered: %s', server.system_listMethods())

    # Run the server's main loop
    LogInfo.logs('Serving start ......')
    server.serve_forever()


if __name__ == '__main__':
    start_service()
