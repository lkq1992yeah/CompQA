# -*- coding: utf-8 -*-

# ==============================================================================
# Just a wrapper for all the linking tasks: entity / type / time.
# Copied from Xianyang's code: <BH>/home/xianyang/wiki/process/process.py
# ==============================================================================


from . import entity_linker, surface_index_memory, parser, type_linker
from .entity_linker import IdentifiedEntity, KBEntity

from kangqi.util.LogUtil import LogInfo
from kangqi.util.discretizer import Discretizer


class LinkingWrapper:

    def __init__(self, base='/home/xianyang/aqqu/aqqu',
                 parser_ip='202.120.38.146', parser_port=9601,
                 linking_mode='Raw', q_links_dict=None, lukov_linker=None):
        self.base = base
        self.linking_mode = linking_mode
        self.q_links_dict = q_links_dict    # save S-MART results
        self.lukov_linker = lukov_linker
        assert linking_mode in ('Raw', 'S-MART', 'Lukov')
        if linking_mode == 'Lukov':
            assert self.lukov_linker is not None
        """
            Raw: the raw version, won't read anything from S-MART or our Lukov's implementation
            S-MART: read from S-MART result (only available in WebQ)
            Lukov: read from our lukov_ngram linker data
        """
        LogInfo.logs('Initiating parser ... ')
        self.parser = parser.CoreNLPParser('http://%s:%d/parse' % (parser_ip, parser_port))   # just open the parser

        self.is_data_loaded = False
        self.surface_index = None
        self.entity_linker = None
        self.type_linker = None

        self.smart_score_disc = Discretizer(split_list=[2, 3, 8, 50, 2000, 12500, 25000, 40000], output_mode='list')
        # the split distribution is manually designed by observing S-MART data in both CompQ & WebQ datasets

        self.pop_filter_num = 5
        # Only used in LukovLinker, for each span,
        # we just select top number of entities sorted by popularity

    def load_data(self):
        if self.is_data_loaded:
            return
        LogInfo.begin_track('EL-Wrapper initializing ... ')
        LogInfo.logs('Initiating index ...')
        self.surface_index = surface_index_memory.EntitySurfaceIndexMemory(
            self.base + '/data/entity-list',
            self.base + '/data/entity-surface-map',
            self.base + '/data/entity-index')
        LogInfo.logs('Initiating entity_linker')
        self.entity_linker = entity_linker.EntityLinker(self.surface_index, 7)

        LogInfo.logs('Initiating type_linker [KQ]')
        self.type_linker = type_linker.TypeLinker()

        LogInfo.end_track('Initialized.')
        self.is_data_loaded = True

    # Key function: return tokens, entities, types and times
    def link(self, q_idx, sentence):
        parse_result = self.parser.parse(sentence)
        tokens = parse_result.tokens
        linking_mode = self.linking_mode

        el_result = []
        tl_result = []
        tml_result = []

        if linking_mode in ('Raw', 'S-MART'):
            self.load_data()
            raw_result = self.entity_linker.identify_entities_in_tokens(tokens)     # entity & time
            for item in raw_result:
                if isinstance(item.entity, entity_linker.KBEntity):
                    el_result.append(item)
                elif isinstance(item.entity, entity_linker.DateValue):
                    tml_result.append(item)
            if linking_mode == 'S-MART':
                # won't use the previous results, but just read S-MART data
                el_result = []
                smart_list = self.q_links_dict.get(q_idx, [])
                for smart_item in smart_list:   # enumerate each candidate in S-MART result
                    use_tokens = []             # determine the token we use for the current EL result
                    start = smart_item.st_pos
                    end = smart_item.st_pos + smart_item.length
                    cur_pos = 0
                    for t in tokens:
                        if start <= cur_pos < end:
                            use_tokens.append(t)
                        cur_pos += len(t.token) + 1
                    obj_entity = KBEntity(name=smart_item.e_name,
                                          identifier=smart_item.mid,
                                          score=0,
                                          aliases=None)
                    perfect = (smart_item.e_name.lower().replace('_', ' ') == smart_item.surface_form.lower())
                    # Chicago_Ohio --> chicago ohio
                    el_item = IdentifiedEntity(tokens=use_tokens, name=smart_item.e_name,
                                               entity=obj_entity, score=0,
                                               surface_score=smart_item.score,
                                               perfect_match=perfect)
                    link_feat = self.smart_score_disc.convert(score=smart_item.score)
                    setattr(el_item, 'link_feat', link_feat)
                    el_result.append(el_item)
            tl_result = self.type_linker.identiy_types_in_tokens(tokens)        # type link
        elif linking_mode == 'Lukov':
            # All the Entity/Type/Time linking will be performed by the Lukov Linker
            el_result, tl_result, tml_result = self.lukov_linker.link_single_question(tokens)
            # if self.q_links_dict is None:
            #     self.q_links_dict = load_lukov_link_result(link_fp=self.aux_fp)
            # el_result = []
            # tl_result = []
            # tml_result = []
            # lukov_link_list = self.q_links_dict.get(q_idx, [])
            # group_link_dict = {}        # <st_ed, links>
            # """ separate link results into several groups by [st, ed) """
            # for link_tup in lukov_link_list:
            #     st = link_tup.start
            #     ed = st + link_tup.length
            #     key = '%s_%s' % (st, ed)
            #     group_link_dict.setdefault(key, []).append(link_tup)
            # """ judge tagging, at least one NN(P) and JJ occurs in the span """
            # postag_available_groups = []    # store all st-ed pair satisfying pos-tag limitation
            # for st_ed in group_link_dict.keys():
            #     st, ed = [int(x) for x in st_ed.split('_')]
            #     flag = False
            #     for idx in range(st, ed):
            #         postag = tokens[idx].pos
            #         if postag.startswith('NN') or postag.startswith('JJ'):
            #             flag = True
            #             break
            #     if flag:
            #         postag_available_groups.append((st, ed))
            # """ longest match filtering """
            # longest_match_groups = []
            # sz = len(postag_available_groups)
            # for i in range(sz):
            #     st_i, ed_i = postag_available_groups[i]
            #     filter_flag = False
            #     for j in range(sz):
            #         if i == j:
            #             continue
            #         st_j, ed_j = postag_available_groups[j]
            #         if st_j <= st_i and ed_j >= ed_i:       # [st_i, ed_i) \in [st_j, ed_j)
            #             filter_flag = True      # found a longer span, filter the current one
            #             break
            #     if not filter_flag:
            #         longest_match_groups.append((st_i, ed_i))
            # """ Popularity filtering at each position """
            # for st, ed in longest_match_groups:
            #     key = '%s_%s' % (st, ed)
            #     links = group_link_dict[key]
            #     links.sort(key=lambda tup: tup.popularity, reverse=True)        # E/T/Tm
            #     for link_tup in links[: self.pop_filter_num]:
            #         LogInfo.logs('[%s] [%d, %d): %s (%s)',
            #                      link_tup.category, link_tup.start,
            #                      link_tup.start + link_tup.length,
            #                      link_tup.mid, link_tup.name.encode('utf-8'))
            #         if link_tup.category in ('Entity', 'Type'):
            #             obj_item = KBEntity(name=link_tup.name,
            #                                 identifier=link_tup.mid,
            #                                 score=link_tup.score,
            #                                 aliases=None)
            #             perfect = (link_tup.name.lower() == link_tup.surface.lower())
            #             el_item = IdentifiedEntity(tokens=tokens[st: ed],
            #                                        name=link_tup.name,
            #                                        entity=obj_item, score=0.,
            #                                        surface_score=link_tup.score,
            #                                        perfect_match=perfect)
            #             if link_tup.category == 'Entity':
            #                 el_result.append(el_item)
            #             else:
            #                 tl_result.append(el_item)
            #         else:       # Time obj
            #             tmv = DateValue(name=link_tup.name, date=link_tup.mid)
            #             # either name or date is the year surface
            #             tml_item = IdentifiedEntity(tokens=tokens[st: ed],
            #                                         name=link_tup.name,
            #                                         entity=tmv, score=0.,
            #                                         surface_score=link_tup.score,
            #                                         perfect_match=True)
            #             tml_result.append(tml_item)

        return tokens, el_result, tl_result, tml_result

    def parse(self, sentence):
        return self.parser.parse(sentence)

    # contains the time identification
    def entity_identify_with_parse(self, tokens):
        self.load_data()
        return self.entity_linker.identify_entities_in_tokens(tokens)

    def time_identify_with_parse(self, tokens):
        self.load_data()
        return self.entity_linker.identify_dates(tokens)

    # ==== Used in SimpleQuestions, given an entity, return its mention ==== #

    def link_with_ground_truth(self, sentence, focus_name, focus_mid):
        """
        ** ONLY USED IN SIMPLEQUESTIONS SCENARIO **
        Given the focus entity name, return the most likely mention span.
        The best span would be:
        1. exact match the entity name
        2. the longest substring of the entity name
        We allow the mention starting with a useless "the"
        :param sentence: the question surface
        :param focus_name: the focus name
        :param focus_mid: the corresponding mid
        :return: the identified entities (but there should be only one)
        """
        tokens = self.parser.parse(sentence).tokens
        q_word_list = [tok.token.lower() for tok in tokens]

        focus_word_list = ['']          # the default list, just an empty string
        if focus_name != '':
            focus_tokens = self.parser.parse(focus_name).tokens
            focus_word_list = [tok.token.lower() for tok in focus_tokens]

        n = len(q_word_list)
        m = len(focus_word_list)
        st = ed = -1
        best_match_words = 0.
        best_match_chars = 0.
        for i in range(n):
            if best_match_words == m:
                break       # already found exact match
            for j in range(i+1, n+1):
                if best_match_words == m:
                    break   # already found exact match
                span = q_word_list[i: j]
                if self.is_contained(span, focus_word_list):
                    match_words = len(span)
                    match_chars = len(''.join(span))
                    if match_words < best_match_words:
                        continue
                    if match_words == best_match_words and match_chars < best_match_chars:
                        continue
                    # now update the interval
                    st = i
                    ed = j - 1  # close interval
                    best_match_words = match_words
                    best_match_chars = match_chars
        if st > 0 and q_word_list[st - 1] == 'the':
            st -= 1
        obj_entity = KBEntity(name=focus_name,
                              identifier=focus_mid,
                              score=0,
                              aliases=None)
        el_item = IdentifiedEntity(tokens=tokens[st: ed + 1],
                                   name=focus_name,
                                   entity=obj_entity,
                                   score=0,
                                   surface_score=1. * best_match_words / m,
                                   perfect_match=best_match_words == m)
        LogInfo.logs('Q surface: %s', q_word_list)
        LogInfo.logs('Focus surface: %s', focus_word_list)
        LogInfo.logs('EL result: [%d, %d] "%s" --> %s', st, ed,
                     ' '.join(q_word_list[st: ed+1]).encode('utf-8'),
                     focus_name.encode('utf-8'))
        if st == -1 or ed == -1:
            LogInfo.logs('Warning: no suitable span found.')
        el_result = [el_item]
        tl_result = []
        tml_result = []
        return tokens, el_result, tl_result, tml_result

    @staticmethod
    def is_contained(span, target_word_list):
        """
        Check whether the span is a sub word sequence in the target word list
        """
        len_diff = len(target_word_list) - len(span)
        if len_diff < 0:
            return False
        for st in range(len_diff + 1):
            flag = True
            for i in range(len(span)):
                if span[i] != target_word_list[st + i]:
                    flag = False
                    break
            if flag:
                return True
        return False

    # ===================== #


class RankItem:
    def __init__(self, rank, tokens, rank_method):
        self.rank = rank
        self.tokens = tokens
        self.rank_method = rank_method

    def to_string(self):
        wd_list = [x.token for x in self.tokens]
        return '(%d, %s, %s)' % (self.rank, ' '.join(wd_list), self.rank_method)


def main():
    q = "company-brand relationship"
    linker = LinkingWrapper()
    tokens = linker.parser.parse(q).tokens
    for token in tokens:
        LogInfo.logs('%s %s', token.token, token.lemma)

if __name__ == '__main__':
    main()
