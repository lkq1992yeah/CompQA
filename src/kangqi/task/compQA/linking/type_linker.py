# -*- coding: utf-8 -*-

import cPickle

from ..u import load_stop_words
from .entity_linker import IdentifiedEntity, KBEntity  # we treat types as a special entity

from kangqi.util.LogUtil import LogInfo

# Type linker: this code is written by myself.
class TypeLinker:

    def __init__(self,
             root_path = '/home/kangqi/workspace/PythonProject/resources/compQA/type_link',
             link_data_name = 'type_link_inverted_index.pydump'):
        link_data_fp = root_path + '/' + link_data_name
        with open(link_data_fp, 'rb') as br:
            self.type_name_dict = cPickle.load(br)  # <type, real_name> (type.object.name)
            self.idx_tp_dict = cPickle.load(br)     # <pharse_idx, set(types)>
            self.name_list = cPickle.load(br)       # [type names] (here the name is BOW)
            self.inverted_index = cPickle.load(br)  # <word, set(phrase_idx)>
        LogInfo.logs('Type link data loaded, with %d type names, ' +
                     '%d distinct BOWs and %d inv-index entries.',
                     len(self.type_name_dict), len(self.idx_tp_dict), len(self.inverted_index))
        assert len(self.name_list) == len(self.idx_tp_dict)

    # TODO: We could consider tf-idf for a more precise surface score.
    #==============================================================================
    # Given the tokens of the whole question, perform the type linking.
    # we first search all linkable mentions, then try to link each possible mention,
    # and at last we filter some sub-optimal linking choices.
    #
    # searching strategy:
    # * We start from st = ed = 0, we slide ed towards the end,
    #   and try to link types for this [st, ed] range.
    #   If we successfully linked to some type (all non-stop words in this range maps to a type name),
    #   then we could go on sliding ed,
    #   otherwise, we slide st, and reset ed = st, starting a new loop for type linking.
    #
    # filtering strategy:
    # * For each target type, we only keep the range with highest linking probability.
    #   If we got a tie, then we pick the range with the smallest length.
    #   If still got a tie, we pick the range which is more closer to the beginning of the sentence.
    #==============================================================================
    def identiy_types_in_tokens(self, tokens):
        tl_result = []
        n = len(tokens); st = ed = 0
        while st < n:
            if ed == n:     # can't slide ed any longer.
                st += 1; ed = st
                continue
            cur_tokens = tokens[st : ed + 1]
            cur_linked_items = self.kernel_link(cur_tokens)
            if len(cur_linked_items) == 0:
                st += 1; ed = st
            else:
                tl_result += cur_linked_items
                ed += 1

        # Now let's do some filtering
        filt_tl_result = []
        tl_grp_dict = {}        # group by different target types
        for tl_item in tl_result:
            tp = tl_item.entity.id
            if tp not in tl_grp_dict: tl_grp_dict[tp] = []
            tl_grp_dict[tp].append(tl_item)
        for tp, items in tl_grp_dict.items():
            items.sort(self.cmp_func)
            filt_tl_result.append(items[0])
        return filt_tl_result

    def cmp_func(self, x, y):
        ret = -cmp(x.surface_score, y.surface_score)
        if ret != 0: return ret

        len_x = x.tokens[-1].index - x.tokens[0].index + 1
        len_y = y.tokens[-1].index - y.tokens[0].index + 1
        ret = cmp(len_x, len_y)
        if ret != 0: return ret

        return cmp(x.tokens[0].index, y.tokens[0].index)



    # Now we perfrom the kernel part of type linking for this selected range.
    def kernel_link(self, tokens):
        stop_word_set = load_stop_words()
        hit_set_list = []       # store a list of hit sets
        for tok in tokens:
            wd = tok.lemma      # we use lemma here because the inverted index is built upon lemmas.
            if wd not in stop_word_set:
                hit_set = self.inverted_index.get(wd)
                if hit_set is None:
                    return []   # I found a unknown entry, and obviously I can't get any linked types.
                hit_set_list.append(hit_set)
        non_stop_len = len(hit_set_list)
        if non_stop_len == 0:
            return []           # we only found stop words in this range...
        joint_set = hit_set_list[0]
        for idx in range(1, non_stop_len):
            joint_set &= hit_set_list[idx]

        # OK, now we could get those phrases covering all non-stop words
        tl_item_list = []
        for phrase_idx in joint_set:
            phrase = self.name_list[phrase_idx]
            phrase_len = len(phrase.split('\t'))
            surface_score = 1.0 * non_stop_len / phrase_len
            perfect_match = True if surface_score == 1.0 else False
            for tp in self.idx_tp_dict[phrase_idx]:
                tp_name = self.type_name_dict[tp]
                type_object = KBEntity(tp_name, tp, 0, None)
                tl_item = IdentifiedEntity(tokens, tp_name,
                                           type_object, score = 0,
                                           surface_score = surface_score,
                                           perfect_match = perfect_match)
                tl_item_list.append(tl_item)
        return tl_item_list


if __name__ == '__main__':
    linker = TypeLinker()

    from .linking_wrapper import LinkingWrapper
    wrapper = LinkingWrapper()
    while True:
        q = raw_input('Enter a question: ')
        input_tokens = wrapper.parse(q).tokens
        tl_result = linker.identiy_types_in_tokens(input_tokens)
        for tl_item in tl_result:
            tokens = tl_item.tokens
            interval = '[%d, %d]' %(tokens[0].index, tokens[-1].index)
            wd_list = []
            for t in tokens: wd_list.append(t.token)
            token_surface = ' '.join(wd_list)
            LogInfo.logs(
                'T: Output: "%s", Tokens: "%s", interval: %s, surface_score: %g, ' +
                'score: %g, mid: "%s", perfect_match: %s.',
                tl_item.name.decode('utf-8'),
                token_surface, interval,
                tl_item.surface_score, tl_item.score,
                tl_item.entity.id, tl_item.perfect_match
            )
#==============================================================================
#     while True:
#         wd = raw_input('Enter a word: ')
#         if wd in linker.inverted_index:
#             phrase_idx_set = linker.inverted_index[wd]
#             LogInfo.logs('Got %d phrases: ', len(phrase_idx_set))
#             for phrase_idx in phrase_idx_set:
#                 LogInfo.logs('#%d: [%s] --> %s', phrase_idx,
#                              linker.name_list[phrase_idx],
#                              linker.idx_tp_dict[phrase_idx])
#         else:
#             LogInfo.logs('[%s] not exist in inv-index.', wd)
#==============================================================================
