# -*- coding: utf-8 -*-

# ==============================================================================
# Author: Kangqi Luo
# Goal: Given a question, return all candidate query structures.
# ==============================================================================

from . import constraint_filter_basic, constraint_filter_extended
from .cand_constructor import CandidateConstructor
from .ext_sk_gen import skeleton_extension

from ..query_struct import Schema
from ..linking.linking_wrapper import LinkingWrapper, RankItem
from ..sparql.sparql import SparqlDriver

from ..u import is_mediator_as_expect, extract_ordinal_predicates, \
                extract_time_predicates, load_ordinal_words

from kangqi.util.cache import DictCache
from kangqi.util.LogUtil import LogInfo


class CandidateGenerator(object):

    def __init__(self, parser_ip='202.120.38.146', parser_port=9601,
                 sparql_ip='202.120.38.146', sparql_port=8999,
                 root_path='/home/kangqi/workspace/PythonProject',
                 cache_dir='runnings/candgen/cache',
                 use_sparql_cache=True, check_only=False,
                 sparql_verbose=False,
                 k_hop=1, max_hops=2):
        LogInfo.begin_track('Initializing CandidateGenerator ... ')
        self.driver = None
        self.one_hop_cache = None

        self.linker = LinkingWrapper(parser_ip=parser_ip, parser_port=parser_port)
        self.cand_construct = CandidateConstructor()
        if cache_dir != 'None':
            self.degree_cache = DictCache('%s/%s/degree_cache' % (root_path, cache_dir))       # save <path, degree>
            self.linking_cache = DictCache('%s/%s/linking_cache' % (root_path, cache_dir))
            self.path_cache = DictCache('%s/%s/path_cache_%d_%d' % (root_path, cache_dir, k_hop, max_hops))
        else:
            self.degree_cache = DictCache()
            self.linking_cache = DictCache()
            self.path_cache = DictCache()
        self.k_hop = k_hop
        self.max_hops = max_hops
        # save <e, set(path)>, where path is a string

        self.driver = SparqlDriver(sparql_ip=sparql_ip,
                                   sparql_port=sparql_port,
                                   use_cache=use_sparql_cache,
                                   verbose=sparql_verbose)
        self.one_hop_cache = DictCache()    # default: do not save anything
        if not check_only and cache_dir != 'None':
            # If we just check schemas, then we could skip some big caches
            LogInfo.logs('Loading one_hop_cache ... ')
            self.one_hop_cache = DictCache('%s/%s/one_hop_cache' % (root_path, cache_dir))
            # save <s, <o, [p]>> for a single 1-hop query

        self.filter_set = {'base', 'freebase', 'user'}
        LogInfo.end_track()

    # ==============================================================================
    # Entity / Type / Time linking blocks
    # ==============================================================================
    @staticmethod
    def cmp_item(x, y):   # the same strategy as what is type_linker.py
        if x.score == y.score and len(x.tokens) == len(y.tokens):   # 3rd Priority: starting point ASC
            return cmp(x.tokens[0].index, y.tokens[0].index)
        if x.score == y.score:                              # 2nd Priority: interval ASC
            return cmp(len(x.tokens), len(y.tokens))
        return -cmp(x.score, y.score)                       # 1st Priority: score DESC

    # We'd like to perform both entity / type / time linking here.
    def linking(self, q_idx, q, min_surface_score, min_pop, s_mart, vb=0):
        linking_result = self.linking_cache.get(q)
        if linking_result is None:
            linking_result = self.linker.link(q_idx, q, s_mart=s_mart)
            self.linking_cache.put(q, linking_result)
        tokens, el_result, tl_result, tml_result = linking_result
        keep_el_result = self.entity_filter(
                            el_result=el_result,
                            min_surface_score=min_surface_score,
                            min_pop=min_pop)  # perform entity filtering
        linking_result = tokens, keep_el_result, tl_result, tml_result
        # re-construct the linking result, removing useless entities.

        if vb >= 0:
            tokens, el_result, tl_result, tml_result = linking_result
            self.entity_or_type_result_displaying(el_result, 'E')
            self.entity_or_type_result_displaying(tl_result, 'T')
            self.tml_result_displaying(tml_result)

        return linking_result

    # remove base. / freebase. / user. entities found by Bast's EL tool.
    # And also remove entities with low surface score or popularity.
    # 140417: for each candidate entity, we just keep only one mention,
    # sorted by surface_score, interval length, starting point.
    def entity_filter(self, el_result, min_surface_score, min_pop):
        keep_dict = {}      # <mid, el_item>
        final_el_result = []
        # First filter by popularity and surface score threshold
        for el_item in el_result:
            mid = el_item.entity.id
            dot_pos = mid.find('.')
            if dot_pos != -1:
                pref = mid[0: dot_pos]
                if pref in self.filter_set:
                    continue
            if el_item.score < min_pop:
                continue
            if el_item.surface_score < min_surface_score:
                continue
            if mid not in keep_dict:
                keep_dict[mid] = []
            keep_dict[mid].append(el_item)
        # Second remove duplicate entities by their surface score, interval length and starting point.
        for mid, item_list in keep_dict.items():
            item_list.sort(self.cmp_item)
            final_el_result.append(item_list[0])
        return final_el_result

    @staticmethod
    def entity_or_type_result_displaying(linking_result, mark):
        for e in linking_result:
            tokens = e.tokens
            interval = '[%d, %d]' % (tokens[0].index, tokens[-1].index)
            wd_list = []
            for t in tokens:
                wd_list.append(t.token)
            token_surface = ' '.join(wd_list).encode('utf-8')

            LogInfo.logs(
                '%s: Output: "%s", Tokens: "%s", interval: %s, surface_score: %g, ' +
                'score: %g, mid: "%s", perfect_match: %s.',
                mark, e.name.encode('utf-8'),
                token_surface, interval,
                e.surface_score, e.score,
                e.entity.id.encode('utf-8'), e.perfect_match
            )
            # since e.name is actually a Unicode string,
            # we'd better encode to UTF-8 for correctly shown in log, rather than decode

    @staticmethod
    def tml_result_displaying(tml_result):
        for tml_item in tml_result:
            wd_list = []
            for t in tml_item.tokens:
                wd_list.append(t.token)
            token_surface = ' '.join(wd_list).encode('utf-8')
            LogInfo.logs('Time: Output: "%s", Tokens: "%s", value: "%s", perfect_match: %s.',
                         tml_item.name.encode('utf-8'), token_surface,
                         tml_item.entity.sparql_name().encode('utf-8'),
                         tml_item.perfect_match)

    # ==============================================================================
    # Path Extraction Blocks
    # ==============================================================================

    # return a dict: <o, [predicate]>
    def one_hop_expansion(self, mid, vb=0):
        query_ret_dict = self.one_hop_cache.get(mid)
        if query_ret_dict is not None:
            return query_ret_dict

        if vb >= 1:
            LogInfo.begin_track('One-hop expansion on %s: ', mid)
        raw_output = self.driver.query_pred_obj_given_subj(mid)     # need filter
        if raw_output is None:
            raw_output = []
        query_ret_dict = {}
        for tup in raw_output:
            # LogInfo.logs('tuple: %s', tup)
            if len(tup) != 2:
                continue
            p, o = tup
            if p.startswith('common') or p.startswith('type'):
                continue
            if o not in query_ret_dict:
                query_ret_dict[o] = []
            query_ret_dict[o].append(p)

        if vb >= 1:     # Check expanding property distribution
            po = 0
            freq_dict = {}
            for o, p_list in query_ret_dict.items():
                po += len(p_list)
                for k in p_list:
                    if k not in freq_dict:
                        freq_dict[k] = 1
                    else:
                        freq_dict[k] += 1
            LogInfo.logs('PO size = %d, obj = %d, average %.3f predicates per object.',
                         po, len(query_ret_dict),
                         0.0 if len(query_ret_dict) == 0 else 1.0 * po / len(query_ret_dict))
            srt_list = sorted(freq_dict.items(), key=lambda x: x[1], reverse=True)
            for k, v in srt_list:
                LogInfo.logs('[%5d]: %s', v, k.encode('utf-8'))
        if vb >= 1:
            LogInfo.end_track()

        self.one_hop_cache.put(mid, query_ret_dict)
        return query_ret_dict

    # judge how many non-mediator-as-domain predicates are there in the path
    # which stands for the real hop of the path
    def path_degree(self, path):
        deg = self.degree_cache.get(path)
        if deg is not None:
            return deg

        deg = 0
        for pred in path.split('\t'):
            if not is_mediator_as_expect(pred):
                deg += 1
        self.degree_cache.put(path, deg)
        return deg

    # Given the mid, return all K-hop (with cvt merged) paths
    # max hops: the limitation of path (nothing merged)
    # Return type: set(path)
    def path_extraction(self, focus, vb=0):
        ret_path_set = self.path_cache.get(focus)
        if ret_path_set is None:
            ret_path_set = set([])
            interm_dict = {focus: {''}}   # <intermediate entity, set(paths)>

            cur_len = 0
            while len(interm_dict) > 0:     # if still can expand
                cur_len += 1
                next_interm_dict = {}   # save the paths to be processed next time
                if vb >= 1:
                    LogInfo.begin_track('Current len = %d, expanding entities = %d: ',
                                        cur_len, len(interm_dict))
                for mid, path_set in interm_dict.items():
                    one_hop_dict = self.one_hop_expansion(mid, vb=vb)
                    for obj, pred_list in one_hop_dict.items():     # iterate each target object
                        for path in path_set:       # now we get focus --path--> mid,
                            for pred in pred_list:  # and get mid --pred--> obj.
                                # So here we just need to combine paths,
                                # and judge whether we need to expand or not.
                                new_path = pred if path == '' else path + '\t' + pred
                                item_sz = len(new_path.split('\t'))     # get the raw length of this pattern
                                deg = self.path_degree(new_path)
                                if deg < self.k_hop and item_sz < self.max_hops:
                                    # we could expand "focus --new_path--> obj", with two conditions:
                                    # 1. degree doesn't reach K
                                    # 2. the raw length of the path couldn't be too large
                                    #    (avoid finding repeated med. predicates when facing some fucked up cases)
                                    if obj not in next_interm_dict:
                                        next_interm_dict[obj] = set([])
                                    next_interm_dict[obj].add(new_path)
                                # If the last predicate is not pointing to a mediator,
                                # then we will pick the current path as the candidate.
                                last_pred = new_path.split('\t')[-1]
                                if not is_mediator_as_expect(last_pred):
                                    ret_path_set.add(new_path)
                interm_dict = next_interm_dict  # ready for the next loop
                if vb >= 1:
                    LogInfo.end_track(
                        'Scanned len = %d, collected paths = %d, next expanding entities = %d.',
                        cur_len, len(ret_path_set), len(interm_dict)
                    )
            self.path_cache.put(focus, ret_path_set)

        if vb >= 1:
            for path in ret_path_set:
                LogInfo.logs('[%s]', ', '.join(path.split('\t')).encode('utf-8'))
        LogInfo.logs('%d distinct path extracted for [%s].', len(ret_path_set), focus)
        return ret_path_set

    # ==============================================================================
    # Constraint Extraction Blocks
    # ==============================================================================

    # Given all linked entities, return all possible entity constraints
    def entity_constraint_extraction(self, el_result, vb=0):
        ec_dict = {}        # <el_item, [anchor predicate]>
        total_size = 0
        for el_item in el_result:
            mid = el_item.entity.id
            raw_anchor_pred_list = self.driver.query_pred_given_object(mid)
            anchor_pred_list = [pred for pred in raw_anchor_pred_list if not (
                                    pred.startswith('common') or pred.startswith('type')
                               )]   # filter common / type predicates
            ec_dict[el_item] = anchor_pred_list
            total_size += len(anchor_pred_list)
        if vb == 1:
            for el_item, p_list in ec_dict.items():
                LogInfo.begin_track(
                    '%d anchors for obj = %s (%s): ', len(p_list),
                    el_item.entity.id.encode('utf-8'),
                    el_item.name.encode('utf-8'))
                for p in p_list:
                    LogInfo.logs(p)
                LogInfo.end_track()
        LogInfo.logs('In total %d candidate objects and ' +
                     '%d <obj_mid, anchor_pred> extracted.',
                     len(ec_dict), total_size)
        return ec_dict, total_size

    # given type linking results, I just return all <tl_item, id> info
    @staticmethod
    def type_constraint_extraction(tl_result):
        tl_dict = {}
        for tl_item in tl_result:
            tl_dict[tl_item] = tl_item.entity.id
        LogInfo.logs('In total %d <IsA, type> extracted.', len(tl_dict))
        return tl_dict

    # recognizat all time value information
    # consider words like "before / after / since"
    @staticmethod
    def time_value_constraint_extraction(tml_result, tokens, vb=0):
        tmv_dict = {}
        for tml_item in tml_result:
            tm_value = tml_item.entity.sparql_name()
            st_pos = tml_item.tokens[0].index
            last_lemma = tokens[st_pos - 1].lemma if st_pos > 0 else ''
            comp = '=='
            if last_lemma == 'before':
                comp = '<'
            elif last_lemma == 'after':
                comp = '>'
            elif last_lemma == 'since':
                comp = '>='
            tmv_dict[tml_item] = (comp, tm_value)
            if vb >= 1:
                LogInfo.logs('%s %s', comp, tm_value)
        LogInfo.logs('In total %d time values extracted.', len(tmv_dict))
        return tmv_dict

    # retrieve all rank-related words, and turn into a possible candidate rank set.
    # each rank info contains the rank number along with its mention and position
    # for example: (1, 'first', 'Ordinal')
    # Note: we should avoid treating a year as a rank value, therefore we ignore tokens in tml_result
    # 170417: Remove duplicate rank items, just keeping the first rank item found in the sentence.
    @staticmethod
    def rank_extraction(tml_result, tokens, vb=0):
        ignore_idx_set = set([])        # we won't extract rank values in those word indices.
        for tml_item in tml_result:
            for token in tml_item.tokens:
                ignore_idx_set.add(token.index)
        rank_set = set([])
        used_rank_word_set = set([])
        ordinal_word_dict = load_ordinal_words()
        for w_idx in range(len(tokens)):
            if w_idx in ignore_idx_set:
                continue    # skip a datetime word
            w_token = tokens[w_idx]
            word = w_token.token
            # 170417: Remove duplicate rank items
            if word in used_rank_word_set:
                continue
            used_rank_word_set.add(word)

            # first check whether this word is just a number or not
            try:
                num_rank = int(word)
            except ValueError:
                num_rank = -1
            if num_rank != -1:
                rank_set.add(RankItem(num_rank, [w_token], 'Number'))
                continue
            # second check whether this word is an known ordinal word
            known_rank = ordinal_word_dict.get(word, None)
            if known_rank is not None:
                rank_set.add(RankItem(known_rank, [w_token], 'Ordinal'))
                continue
            # third check whether this word is an xx-est or not
            # This check may bring noise, but we don't care right now.
            if word.endswith('est'):
                rank_set.add(RankItem(1, [w_token], 'Superlative'))
                continue
        if vb >= 1:
            for rank_item in rank_set:
                LogInfo.logs('(rank = %d, word = %s, type = %s)',
                             rank_item.rank, rank_item.tokens[0].token, rank_item.rank_method)
        LogInfo.logs('%d rank info extracted.', len(rank_set))
        return rank_set

    # ==============================================================================
    # Constraint Filtering & Merging Blocks. (Moved to constraint_filter_basic.py)
    # ==============================================================================

    # ==============================================================================
    # External Calling function
    # ==============================================================================
    def run_candgen(self, q_idx, q, min_surface_score=0.0, min_pop=0,
                    use_ext_sk=False, min_ratio=0.0,
                    s_mart=False, vb=0):
        LogInfo.begin_track('[S1] Entity / Type / Time linking: ')
        tokens, el_result, tl_result, tml_result = \
            self.linking(q_idx=q_idx, q=q,
                         min_surface_score=min_surface_score,
                         min_pop=min_pop, s_mart=s_mart, vb=vb)
        el_size = len(el_result)
        tl_size = len(tl_result)
        tml_size = len(tml_result)
        LogInfo.logs('In total %d E + %d T + %d times linked.', el_size, tl_size, tml_size)
        LogInfo.end_track()     # End of Entity linking

        LogInfo.begin_track('[S2] Skeleton extraction: ')
        LogInfo.logs('Need to traverse %d <mention, entity> candidates.', el_size)
        skeleton_list = []  # store all candidate skeletons (simple query structure)
        # skeleton_list.append(Schema())  # add the simplest schema (without any other predicates)
        for el_idx in range(el_size):
            el_item = el_result[el_idx]
            mid = el_item.entity.id.encode('utf-8')
            e_name = el_item.name.encode('utf-8')
            if vb >= 1:
                LogInfo.begin_track('Path extraction on %d / %d [%s (%s)]: ',
                                    el_idx + 1, el_size, mid, e_name)
            path_set = self.path_extraction(mid, vb=vb)
            for path in path_set:
                sk = Schema()
                sk.focus_item = el_item
                sk.path = path.split('\t')
                skeleton_list.append(sk)
            if vb >= 1:
                LogInfo.end_track()
        sk_size = len(skeleton_list)
        LogInfo.logs('In total %d skeletons collected.', sk_size)
        LogInfo.end_track()     # End of Path extraction

        LogInfo.begin_track('[S3] Constraint extraction: ')     # There should be multiple types.
        # 1. entity constraint from EL
        if vb >= 1:
            LogInfo.begin_track('[S3-1] Entity constraint: ')
        ec_dict, ec_size = self.entity_constraint_extraction(el_result, vb=vb)
        if vb >= 1:
            LogInfo.end_track()

        # 2. type constraint from types
        if vb >= 1:
            LogInfo.begin_track('[S3-2] Type constraint: ')
        tc_dict = self.type_constraint_extraction(tl_result)
        if vb >= 1:
            LogInfo.end_track()

        # 3. explicit time constraint
        # (if there has inplicit time constraint, I think we need to transform the original sentence)
        if vb >= 1:
            LogInfo.begin_track('[S3-3] Time constraint: ')
        tmc_set = extract_time_predicates()     # just extract predicates with type.datetime as range.
        tmv_dict = self.time_value_constraint_extraction(tml_result, tokens, vb=vb)
        # extract all time values (like "in 2012", "before 2012", "since 2012")
        if vb >= 1:
            LogInfo.end_track()

        # 4. ordinal constraints
        if vb >= 1:
            LogInfo.begin_track('[S3-4] Ordinal constraint: ')
        oc_set = extract_ordinal_predicates()   # extract all predicate with int/float/DT as range.
        rank_set = self.rank_extraction(tml_result, tokens, vb=vb)
        if vb >= 1:
            LogInfo.end_track()

        tc_size, tmc_size, tmv_size, oc_size, rank_size = \
            [len(x) for x in (tc_dict, tmc_set, tmv_dict, oc_set, rank_set)]
        constr_size_info = ec_size, tc_size, tmc_size * tmv_size, oc_size * rank_size * 2
        LogInfo.logs('Summary: Entity = %d, Type = %d, ' +
                     'Time_Value = %d*%d = %d,' +
                     'Ordinal_Rank = %d*%d*2 = %d, Total = %d.',
                     ec_size, tc_size,
                     tmc_size, tmv_size, tmc_size * tmv_size,
                     oc_size, rank_size, oc_size * rank_size * 2,
                     sum(constr_size_info))
        LogInfo.end_track()     # End of constraint extraction

        schema_list = []
        LogInfo.begin_track('[S4]: Constraint filtering (use_ext_sk=%s):', use_ext_sk)
        constraint_filtering = \
            constraint_filter_extended.constraint_filtering if use_ext_sk \
            else constraint_filter_basic.constraint_filtering
        # Set the constraint filtering fuction that we use in the following steps.

        LogInfo.logs('%d skeletons and %d constraints in total.', sk_size, sum(constr_size_info))
        for sk_idx in range(sk_size):
            if sk_idx % 20 == 0:
                LogInfo.logs('skeleton %d / %d scanned.', sk_idx, sk_size)
            sk = skeleton_list[sk_idx]
            if vb >= 1:
                LogInfo.begin_track(
                    'Constr-filt on sk %d / %d [%s (%s), %s]: ',
                    sk_idx + 1, sk_size, sk.focus_item.entity.id.encode('utf-8'),
                    sk.focus_item.name.encode('utf-8'), ', '.join(sk.path).encode('utf-8'))

            if vb >= 1:
                LogInfo.begin_track('Extended skeleton generation [%s]: ', use_ext_sk)
            if use_ext_sk:
                ext_sk_list = skeleton_extension(sk, tc_dict, self.driver, vb=vb)
            else:
                ext_sk_list = [sk, ]   # just use the standard skeleton without any other constraints
            ext_size = len(ext_sk_list)
            if vb >= 1:
                LogInfo.end_track()

            if vb >= 1:
                LogInfo.logs('extened skeleton size = %d.', ext_size)
            for ext_sk_idx in range(ext_size):
                ext_sk = ext_sk_list[ext_sk_idx]
                if vb >= 1:
                    tp_str_list = []
                    for constr in ext_sk.constraints:
                        tp_str_list.append('<x%d, %s>' % (constr.x, constr.o.encode('utf-8')))
                    LogInfo.begin_track('ext_sk %d / %d %s: ',
                                        ext_sk_idx + 1, ext_size, tp_str_list)
                filt_constraint_list = constraint_filtering(
                    ext_sk, ec_dict, tc_dict,
                    tmc_set, tmv_dict,
                    oc_set, rank_set,
                    self.driver, min_ratio=min_ratio, vb=vb)
                schema_list += self.cand_construct.construct(
                    ext_sk, filt_constraint_list, driver=self.driver, vb=vb)
                if vb >= 1:
                    LogInfo.end_track('End of ext_sk %d / %d.', ext_sk_idx + 1, ext_size)
            if vb >= 1:
                LogInfo.end_track('End of sk %d / %d.', sk_idx + 1, sk_size)
        LogInfo.end_track()  # End of [S4]

#        self.schema_cache.put(q, schema_list)
        return schema_list


def check_el_quality(cand_gen, q_list):
    el_info_list = []
    for q in q_list:
        linking_result = cand_gen.linking(q, vb=-1)
        _, el_result, _, _ = linking_result
        for el_item in el_result:
            surface_score = el_item.surface_score
            pop = el_item.score
            mid = el_item.entity.id.encode('utf-8')
            perfect_match = el_item.perfect_match
            tokens = el_item.tokens
            wd_list = []
            for t in tokens:
                wd_list.append(t.token)
            token_surface = ' '.join(wd_list).encode('utf-8')
            el_info = (q, el_item.name.encode('utf-8'), token_surface, surface_score, pop, perfect_match, mid)
            el_info_list.append(el_info)
    el_info_list.sort(lambda x, y: cmp(x[3], y[3]))
    with open('check_el_quality.txt', 'w') as bw:
        bw.write('%-8s\t%-8s\t%-8s\t%-8s\t%-8s\t%-8s\t%-8s\n' % (
                'score', 'mid', 'perfect', 'popularity', 'e_name', 'interval', 'question'))
        for el_info in el_info_list:
            q, name, token_surface, surface_score, pop, perfect_match, mid = el_info
            bw.write('%8.6f\t%s\t%s\t%d\t"%s"\t"%s"\t%s\n' % (
                surface_score, mid, str(perfect_match), pop, name, token_surface, q))


def main():
    LogInfo.begin_track('[cand_gen] starts ... ')

    import sys
    from kangqi.util.config import load_configs

    root_path = '/home/kangqi/workspace/PythonProject'
    try_dir = sys.argv[1]

    config_fp = '%s/runnings/candgen/%s/param_config' % (root_path, try_dir)
    config_dict = load_configs(config_fp)

    parser_ip = config_dict['parser_ip']
    parser_port = int(config_dict['parser_port'])

    use_sparql_cache = True if config_dict['use_sparql_cache'] == 'True' else False
    cache_dir = config_dict['cache_dir']
    check_only = True if config_dict['check_only'] == 'True' else False

    use_ext_sk = True if config_dict['use_ext_sk'] == 'True' else False
    s_mart = True if config_dict['S-MART'] == 'True' else False
    min_ratio = float(config_dict['min_ratio'])
    min_surface_score = float(config_dict['min_surface_score'])
    min_pop = int(config_dict['min_pop'])

    # from ..data_prepare.u import load_complex_questions
    # train_qa_list, test_qa_list = load_complex_questions()
    # compq_list = train_qa_list + test_qa_list
    # LogInfo.logs('%d ComplexQuestions loaded.', len(compq_list))
    # surface_list = [qa.q for qa in compq_list]

    import json
    webq_fp = '/home/kangqi/Webquestions/Json/webquestions.examples.json'
    with open(webq_fp, 'r') as br:
        webq_data = json.load(br)
        surface_list = [webq['utterance'] for webq in webq_data]
    LogInfo.logs('%d WebQuesetions loaded.', len(webq_data))

#    cand_gen = CandidateGenerator(use_sparql_cache=False, check_only=True)
#    q_list = [qa.q for qa in train_qa_list]
#    check_el_quality(cand_gen, q_list)

    cand_gen = CandidateGenerator(use_sparql_cache=use_sparql_cache,
                                  cache_dir=cache_dir,
                                  check_only=check_only,
                                  parser_ip=parser_ip,
                                  parser_port=parser_port,
                                  k_hop=1,
                                  max_hops=2)
    for q_idx, q in enumerate(surface_list):
            LogInfo.begin_track('Checking Q %d / %d: ', q_idx + 1, len(surface_list))
            LogInfo.logs('Surface: %s', q)
            schema_list = cand_gen.run_candgen(q_idx=q_idx, q=q,
                                               min_surface_score=min_surface_score,
                                               min_pop=min_pop, use_ext_sk=use_ext_sk,
                                               min_ratio=min_ratio, s_mart=s_mart, vb=1)
            LogInfo.logs('Finally: generated %d schemas.', len(schema_list))

            sc_sz = len(schema_list)
            for idx in range(sc_sz):
                LogInfo.begin_track('Showing schema %d / %d: ', idx + 1, sc_sz)
                sc = schema_list[idx]
                sc.display()
    #            sc.display_sparql()
                LogInfo.end_track()

            LogInfo.end_track()

#    cand_gen = CandidateGenerator(use_sparql=False)
#    q = 'what language do most people speak in afghanistan?'
#    cand_gen.linking(q, vb=0)

    LogInfo.end_track()


if __name__ == '__main__':
    main()
