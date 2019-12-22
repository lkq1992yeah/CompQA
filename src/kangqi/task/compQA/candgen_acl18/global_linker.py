"""
Author: Kangqi Luo
Date: 180208
Goal: E/T/Tm/Ord linking
"""
import re
import json
import codecs
import numpy as np

from ..util.fb_helper import load_type_name, is_type_ignored

from kangqi.util.LogUtil import LogInfo
from kangqi.util.time_track import TimeTracker as Tt


class LinkData:

    def __init__(self, category, start, end, mention, comp, value, name, link_feat, gl_pos=None):
        self.gl_pos = gl_pos
        self.category = category
        self.start = start
        self.end = end
        self.mention = mention
        self.comp = comp
        self.value = str(value)
        self.name = name
        self.link_feat = link_feat      # dictionary

    def serialize(self):
        ret_list = []
        for key in ('category', 'start', 'end', 'mention', 'comp', 'value', 'name', 'link_feat'):
            ret_list.append((key, getattr(self, key)))
        if self.gl_pos is not None:
            ret_list = [('gl_pos', self.gl_pos)] + ret_list
        return ret_list

    def display(self):
        ret_str = ''
        if self.gl_pos is not None:
            ret_str += '#%02d ' % self.gl_pos
        ret_str += '%s: [%d, %d) (%s) %s %s ' % (
            self.category, self.start, self.end, self.mention, self.comp, self.value)
        if self.name != '':
            ret_str += '(%s) ' % self.name
        ret_str += str(self.link_feat)
        return ret_str


class GlobalLinker:

    def __init__(self, wd_emb_util, q_links_fp, ordinal_fp='data/fb_metadata/ordinal_fengli.tsv'):
        """
        Necessary information: word embedding helper, and S-MART / Lukov entity linking results
        """
        LogInfo.begin_track('GlobalLinker initializing ... ')
        self.wd_emb_util = wd_emb_util
        self.q_links_dict = {}          # <q_idx, [LinkData]>

        self.type_emb_dict = {}         # <type, (name, embedding of its name)>
        self.top_similar_types = 10
        self.year_re = re.compile(r'^[1-2][0-9][0-9][0-9]$')

        self.ordinal_word_dict = {}         # <ordinal word, its corresponding number>
        self.superlative_words = set([])        # contain both -est words and xx-th words

        with codecs.open(ordinal_fp, 'r', 'utf-8') as br:
            for line in br.readlines():
                spt = line.strip().split(' ')
                if spt[0] in ('max', 'min'):
                    self.superlative_words |= set(spt[1].split(','))
                else:
                    num = int(spt[0])
                    for wd in spt[1].split(','):
                        self.ordinal_word_dict[wd] = num
                        self.superlative_words.add(wd)
        LogInfo.logs('%d ordinal & %d superlative words loaded.',
                     len(self.ordinal_word_dict), len(self.superlative_words))

        LogInfo.begin_track('Load entity linking table from [%s]:', q_links_fp)
        with codecs.open(q_links_fp, 'r', 'utf-8') as br:
            for line in br.readlines():
                if line.startswith('#'):
                    continue
                spt = line.strip().split('\t')
                q_idx, st, ed, mention, mid, wiki_name, feats = spt
                q_idx = int(q_idx)
                st = int(st)
                ed = int(ed)
                feat_dict = json.loads(feats)
                for k in feat_dict:
                    v = float('%.6f' % feat_dict[k])
                    feat_dict[k] = v
                link_data = LinkData(category='Entity',
                                     start=st, end=ed,
                                     mention=mention, comp='==',
                                     value=mid, name=wiki_name,
                                     link_feat=feat_dict)
                self.q_links_dict.setdefault(q_idx, []).append(link_data)
        LogInfo.logs('%d questions of link data loaded.', len(self.q_links_dict))
        LogInfo.end_track()

        type_name_dict = load_type_name()
        for tp, tp_name in type_name_dict.items():
            self.type_emb_dict[tp] = (tp_name, self.wd_emb_util.get_phrase_emb(phrase=tp_name))
        LogInfo.logs('%d type embedding loaded.', len(self.type_emb_dict))

        LogInfo.end_track('Initialize complete.')

    def perform_linking(self, q_idx, tok_list, entity_only=False):
        tok_list = map(lambda x: x.lower(), tok_list)       # all steps ignore cases.
        Tt.start('entity')
        el_list = self.q_links_dict.get(q_idx, [])
        Tt.record('entity')

        if not entity_only:
            Tt.start('type')
            tl_list = self.type_linking(tok_list)
            Tt.record('type')
            Tt.start('time')
            tml_list = self.time_linking(tok_list)
            Tt.record('time')
            Tt.start('ordinal')
            ord_list = self.ordinal_linking(tok_list)
            Tt.record('ordinal')
            gather_linkings = el_list + tl_list + tml_list + ord_list
        else:       # For SimpleQuestions
            gather_linkings = el_list

        for idx in range(len(gather_linkings)):
            gather_linkings[idx].gl_pos = idx
        return gather_linkings

    def type_linking(self, tok_list):
        tl_data_list = []
        rank_tup_list = []      # [(st, ed, type, score)]
        tok_size = len(tok_list)
        for st in range(tok_size):
            for ed in range(st+1, tok_size+1):
                if ed - st > 3:
                    continue        # consider at most tri-gram
                phrase = ' '.join(tok_list[st: ed])
                phrase_emb = self.wd_emb_util.get_phrase_emb(phrase=phrase)
                if phrase_emb is None:
                    continue
                for tp, (type_name, type_emb) in self.type_emb_dict.items():
                    if is_type_ignored(tp):
                        continue
                    if type_emb is not None:
                        cosine = np.sum(type_emb * phrase_emb).astype(float)
                        cosine = float('%.6f' % cosine)
                        rank_tup_list.append((st, ed, tp, type_name, cosine))

        rank_tup_list.sort(key=lambda _tup: _tup[-1], reverse=True)
        pick_tup_list = []
        select_type_set = set([])
        for tup in rank_tup_list:       # pick top-k distinct similar types, no matter the position of span.
            cur_type = tup[2]
            if cur_type not in select_type_set:
                pick_tup_list.append(tup)
                select_type_set.add(cur_type)
            if len(select_type_set) >= self.top_similar_types:
                break
        for st, ed, tp, type_name, cosine in pick_tup_list:
            mention = ' '.join(tok_list[st: ed])
            tl_data_list.append(LinkData(category='Type', start=st, end=ed,
                                         mention=mention, comp='==', value=tp,
                                         name=type_name, link_feat={'sim': cosine}))
        return tl_data_list

    def time_linking(self, tok_list):
        tml_data_list = []
        for idx, tok in enumerate(tok_list):
            if re.match(self.year_re, tok[:4]):
                year = tok[:4]
                last_tok = tok_list[idx - 1] if idx > 0 else ''
                comp = '=='
                if last_tok == 'before':
                    comp = '<'
                elif last_tok == 'after':
                    comp = '>'
                elif last_tok == 'since':
                    comp = '>='
                st = idx
                ed = idx+1
                mention = ' '.join(tok_list[st: ed])
                tml_data_list.append(LinkData(category='Time', start=st, end=ed,
                                              mention=mention, comp=comp,
                                              value=year, name='', link_feat={}))
        return tml_data_list

    def ordinal_linking(self, tok_list):
        """
        consider 3 conditions:
            1. single xx-th word: first, 7th, most..
            2. single -est word: longest, largest
            3. xx-th -est combination: second largest
        """
        ordinal_data_list = []
        tok_size = len(tok_list)
        for idx in range(tok_size):
            token = tok_list[idx]
            if token in self.superlative_words:     # found a superlative word
                if idx+1 < tok_size and tok_list[idx+1] in self.superlative_words:
                    # make sure the current word is the last word in a potential ordinal span
                    continue
                # try to extract a direct ordinal number
                if idx > 0 and tok_list[idx-1] in self.ordinal_word_dict:
                    st = idx - 1
                    num = self.ordinal_word_dict[tok_list[idx-1]]       # xx-th xx-est combination
                else:
                    st = idx
                    if tok_list[idx] in self.ordinal_word_dict:         # single xx-th
                        num = self.ordinal_word_dict[tok_list[idx]]
                    else:                                               # single xx-est
                        num = 1                                         # default rank = 1
                ed = idx+1
                mention = ' '.join(tok_list[st: ed])
                for comp in ('max', 'min'):        # consider both directions
                    ordinal_data_list.append(LinkData(category='Ordinal', start=st, end=ed,
                                                      mention=mention, comp=comp,
                                                      value=str(num), name='', link_feat={}))
        return ordinal_data_list
