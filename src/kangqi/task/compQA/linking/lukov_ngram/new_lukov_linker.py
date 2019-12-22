"""
Author: Kangqi Luo
Goal: Lukov Linker + Lexicon
"""


import re
import json
import codecs
import xmlrpclib

from ..entity_linker import IdentifiedEntity, KBEntity, DateValue
from .lexicon import LexiconEntry
from ...dataset.u import load_compq, load_webq, load_simpq

from kangqi.util.LogUtil import LogInfo
from kangqi.util.discretizer import Discretizer


class LukovLinker:

    def __init__(self, lexicon):
        self.lexicon = lexicon
        self.year_re = re.compile(r'^[1-2][0-9][0-9][0-9]$')
        self.punc_str = u"?!:',."
        self.trivial_set = {'the', 'a', 'an', 'of', 'on', 'at', 'by'}
        # Lukov et al., Sec 2.2.1
        self.log_wiki_pop_disc = Discretizer([1, 2, 3, 4, 6], output_mode='list', name='log_wiki_pop')
        self.log_fb_pop_disc = Discretizer([3, 4, 6, 8], output_mode='list', name='log_fb_pop')
        # Following the configurations in lexicon.py

    def link_single_question(self, tokens):
        """
        :param tokens: question tokens
        :return: el_result, tl_result, tml_result
        """
        el_result = []
        tl_result = []
        tml_result = []

        """ Step 1: Find all possible mentions """
        # TODO: Make sure we can handle "'s" or "u.s." cases
        span_link_dict = {}
        token_size = len(tokens)
        for st in range(token_size):
            for ed in range(st + 1, token_size + 1):
                if ed - st > 5:
                    continue        # only consider at most 5-gram
                span = '%d_%d' % (st, ed)
                lower_surface = self.build_lowercased_surface(tokens=tokens, st=st, ed=ed)

                check_time_surface = lower_surface[:4]      # check whether the surface starts with a valid year
                if ed - st == 1 and re.match(self.year_re, check_time_surface):
                    # Time match: highest priority, but only allowing unigram (ed - st == 1)
                    tmv = DateValue(name=lower_surface, date=check_time_surface)
                    tml_item = IdentifiedEntity(tokens=tokens[st: ed],
                                                name=lower_surface,
                                                entity=tmv, score=0.,
                                                surface_score=0,
                                                perfect_match=True)
                    span_link_dict.setdefault(span, []).append(tml_item)
                elif self.lexicon.contains(lower_surface):
                    entry_list = self.lexicon.get(lower_surface)
                    for raw_entry in entry_list:
                        if isinstance(raw_entry, dict):     # raw_entry is retrieved by ServerProxy, resulting in dict
                            lex_entry = LexiconEntry(**raw_entry)
                        else:
                            lex_entry = raw_entry
                            assert isinstance(lex_entry, LexiconEntry)

                        # LogInfo.logs(str(lex_entry))
                        use_name = lex_entry.wiki_name if lex_entry.category == 'Entity' else lex_entry.fb_name
                        # Entity name: from Wiki; Type name: from FB

                        obj_item = KBEntity(name=use_name,
                                            identifier=lex_entry.mid,
                                            score=0,
                                            aliases=None)
                        perfect = (use_name.lower() == lower_surface)
                        el_item = IdentifiedEntity(tokens=tokens[st: ed],
                                                   name=use_name,
                                                   entity=obj_item, score=0.,
                                                   surface_score=0.,
                                                   perfect_match=perfect)
                        setattr(el_item, 'lex_entry', lex_entry)
                        # attach the detail entry to the item, will be used in link_feat generation
                        span_link_dict.setdefault(span, []).append(el_item)
                        """ Note: Here el_item could be either entity or type """

        """ Step 2: Span Filter by POS-tag """
        postag_available_groups = []  # store all st-ed pair satisfying pos-tag limitation
        for st_ed in span_link_dict.keys():
            st, ed = [int(x) for x in st_ed.split('_')]
            flag = False
            for idx in range(st, ed):
                postag = tokens[idx].pos
                if postag.startswith('NN') or postag.startswith('JJ'):
                    flag = True
                    break
            if flag:
                postag_available_groups.append((st, ed))

        """ Step 3: Longest Match (Following Lukov's Trick) """
        longest_match_groups = []
        sz = len(postag_available_groups)
        for i in range(sz):
            st_i, ed_i = postag_available_groups[i]
            filter_flag = False
            for j in range(sz):
                if i == j:
                    continue
                st_j, ed_j = postag_available_groups[j]
                if st_j <= st_i and ed_j >= ed_i and not tokens[st_j].token in self.trivial_set:
                    """
                    We found a set larger than the current [st_i, ed_i),
                    and the larger set doesn't starts with Lukov's trivial words.
                    Then we could filter the current span.
                    """
                    filter_flag = True
                    break
            if not filter_flag:
                longest_match_groups.append((st_i, ed_i))

        """ Step 4: Construct Link Feature, Add Into IdentifiedEntity """
        # Keep all Time/Type item, and generate link features for Entity item
        for st, ed in longest_match_groups:
            span = '%d_%d' % (st, ed)
            surface = ' '.join(tok.token for tok in tokens[st: ed]).encode('utf-8')
            LogInfo.begin_track('Check surface = [%s]:', surface)
            link_list = span_link_dict[span]
            for link_item in link_list:
                entity = link_item.entity
                if isinstance(entity, DateValue):   # Time
                    LogInfo.logs('Time: [%s]', entity.value.encode('utf-8'))
                    tml_result.append(link_item)
                else:
                    assert isinstance(entity, KBEntity)
                    LogInfo.logs('%s: [%s] [link_prob=%.3f] [%s]',
                                 link_item.lex_entry.category,
                                 link_item.entity.id.encode('utf-8'),
                                 link_item.lex_entry.stat_dict.get('link_prob', -1.),
                                 link_item.name.encode('utf-8'))
                    if link_item.lex_entry.category == 'Type':    # Type from Lexicon
                        tl_result.append(link_item)
                    else:   # Found entity, ready to extract link features
                        stat_dict = link_item.lex_entry.stat_dict
                        keys = ['link_prob', 'word_jac', 'ngram_jac', 'fb_pop', 'wiki_pop']
                        link_prob, ngram_jac, word_jac, fb_pop, wiki_pop = [stat_dict[k] for k in keys]
                        # log_fb_pop = None if fb_pop <= 0 else math.log(fb_pop)
                        # log_wiki_pop = None if wiki_pop <= 0 else math.log(wiki_pop)
                        # link_feat = (
                        # [link_prob, ngram_jac, word_jac] +
                        # self.log_wiki_pop_disc.convert(log_wiki_pop) +
                        # self.log_fb_pop_disc.convert(log_fb_pop)
                        # )
                        link_feat = {
                            'link_prob': link_prob,
                            'word_jac': word_jac,
                            'ngram_jac': ngram_jac,
                            'fb_pop': fb_pop,
                            'wiki_pop': wiki_pop
                        }
                        setattr(link_item, 'link_feat', link_feat)
                        LogInfo.logs('Entity Feature: %s', link_item.link_feat)
                        el_result.append(link_item)
            LogInfo.end_track()

        return el_result, tl_result, tml_result

    def build_lowercased_surface(self, tokens, st, ed):
        """
        ** Copied from lukov_basic_linker.py
        Construct the surface form in [st, ed)
        Be careful: Their may have or may not have a blank before the token starting with a punctuation
                    For example, "'s" is a token starting with a punctuation.
        In this version, we try not enumerating all the possible combinations,
        but just try using some rules.
        Rule 1: we not add blank before some token, if it starts with a punctuation.
                For example, "obama 's wife" "star trek: the movie"
        Rule 2: TBD (may focus on string like "u.s.")
        Therefore, we only return ONE surface form.
        """
        surface = ''
        for idx in range(st, ed):
            tok = tokens[idx].token.lower()
            if tok == '':
                continue        # avoid empty token
            begin_char = tok[0]
            if len(surface) > 0 and begin_char not in self.punc_str:
                surface += ' '      # adding blank before the token
            surface += tok
        return surface


def main():
    LogInfo.begin_track('Generating xs-lex EL data ...')
    data_name = 'simpq'

    if data_name == 'webq':
        qa_list = load_webq()
        out_fp = '/home/data/Webquestions/ACL18/webq.all.xs-lex.q_links'
    elif data_name == 'compq':
        qa_list = load_compq()
        out_fp = '/home/data/CompQuestions/ACL18/compQ.all.xs-lex.q_links'
    else:
        assert data_name == 'simpq'
        qa_list = load_simpq()
        out_fp = '/home/data/SimpleQuestions/SimpleQuestions_v2/ACL18/simpQ.all.xs-lex.q_links'

    lexicon = xmlrpclib.ServerProxy('http://202.120.38.146:9602')
    LogInfo.logs('Lukov Lexicon Proxy binded.')
    linker = LukovLinker(lexicon=lexicon)
    with codecs.open(out_fp, 'w', 'utf-8') as bw:
        for q_idx, qa in enumerate(qa_list):
            LogInfo.begin_track('Current q_idx = %d', q_idx)
            tokens = qa['tokens']
            el_result, _, _ = linker.link_single_question(tokens=tokens)
            for el_item in el_result:
                st = el_item.tokens[0].index
                ed = el_item.tokens[-1].index + 1
                mention = ' '.join([tok.token.lower() for tok in el_item.tokens])
                mid = el_item.entity.id
                wiki_name = el_item.entity.name
                feat_dict = el_item.link_feat
                bw.write('%04d\t%d\t%d\t%s\t%s\t%s\t%s\n' % (
                    q_idx, st, ed, mention, mid, wiki_name, json.dumps(feat_dict)
                ))
            LogInfo.end_track()
    LogInfo.end_track()

if __name__ == '__main__':
    main()
