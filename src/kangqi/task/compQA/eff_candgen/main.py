# -*- coding:utf-8 -*-

import os
import re
import json
import shutil
import cPickle
import argparse
import xmlrpclib

from .combinator import make_combination, construct_gather_linkings
from .type_link_filter import type_filtering
from .translator import convert_combination
from .answer_query import kernel_querying, prepare_forbidden_mid_given_raw_paths

from ..dataset.u import load_webq, load_compq, load_simpq, load_simpq_sp_dict
from ..sparql.sparql import SparqlDriver

from ..linking.linking_wrapper import LinkingWrapper
from ..linking.lukov_ngram.new_lukov_linker import LukovLinker
from ..linking.entity_linker import DateValue
from ..candgen.smart import load_webq_linking_data, load_compq_linking_data

from ..eval.official_eval import compute_f1
from ..util.fb_helper import get_entity_name, load_entity_name

from kangqi.util.cache import DictCache
from kangqi.util.LogUtil import LogInfo


working_ip = '202.120.38.146'
parser_port_dict = {'Blackhole': 9601, 'Darkstar': 8601}
sparql_port_dict = {'Blackhole': 8999, 'Darkstar': 8699}

parser = argparse.ArgumentParser(description='Efficient Candidate Generation')

parser.add_argument('--machine', default='Blackhole', choices=['Blackhole', 'Darkstar'])
parser.add_argument('--use_sparql_cache', action='store_true')
parser.add_argument('--sparql_verbose', action='store_true')
parser.add_argument('--linking_cache_fp', default=None)
parser.add_argument('--linking_mode', choices=['Raw', 'S-MART', 'Lukov'], default='Raw')
parser.add_argument('--ground_truth', action='store_true')

parser.add_argument('--mode', choices=['link_only', 'default'], default='default')
parser.add_argument('--data_name', choices=['WebQ', 'CompQ', 'SimpQ'])
parser.add_argument('--fb_meta_dir', default='data/fb_metadata')
parser.add_argument('--q_start', type=int, default=0)
parser.add_argument('--q_end', type=int, default=1000000)
parser.add_argument('--output_dir')
parser.add_argument('--verbose', type=int, default=0)


def perform_linking(linking_wrapper, data_name,
                    q_idx, q, target_value, ground_truth):
    if not ground_truth:
        linking_result = linking_wrapper.link(q_idx=q_idx, sentence=q)
        # Each linking data is stored as IdentifiedEntity object
    else:       # SimpQ + ground_truth
        assert data_name == 'SimpQ'
        focus_mid = target_value[0]
        focus_name = get_entity_name(mid=focus_mid)
        linking_result = linking_wrapper.link_with_ground_truth(
            sentence=q,
            focus_name=focus_name,
            focus_mid=focus_mid
        )
    return linking_result


def process_single_question(data_name, q, q_idx, target_value,
                            linking_wrapper, linking_cache, ground_truth,
                            sparql_driver, simpq_sp_dict, vb):
    # Step 1: E/T/Tm Linking
    linking_result = linking_cache.get(q)
    if linking_result is None:
        linking_result = perform_linking(
            linking_wrapper=linking_wrapper,
            data_name=data_name, q_idx=q_idx, q=q,
            target_value=target_value, ground_truth=ground_truth
        )
        linking_cache.put(q, linking_result)
    tokens, el_result, tl_result, tml_result = linking_result
    tl_result = type_filtering(el_result=el_result,
                               tl_result=tl_result,
                               sparql_driver=sparql_driver,
                               vb=vb)
    tml_result = tml_filtering(tml_result=tml_result)
    tml_comp_result = build_tml_comp_result(tml_result=tml_result, tokens=tokens)
    # if in SimpQ, then tl/tml/tml_comp will always be empty, and there's only one EL
    gather_linkings = construct_gather_linkings(el_result=el_result,
                                                tl_result=tl_result,
                                                tml_result=tml_result,
                                                tml_comp_result=tml_comp_result)

    # ======== Now Collecting Schemas ======== #
    if data_name in ('WebQ', 'CompQ'):
        # Step 2: 1-hop or 2-hop query (E+T+Tm, no ordinal)
        ground_comb_list = make_combination(gather_linkings=gather_linkings,
                                            sparql_driver=sparql_driver, vb=vb)
        # TODO: Now we ignore those cases without any ENTITY result
        ground_row_sz = sum([len(query_ret) for _, _, query_ret in ground_comb_list])
        LogInfo.logs('In total %d predicate rows queried from %d combinations.',
                     ground_row_sz, len(ground_comb_list))

        # Step 3: collect all distinct basic schemas
        sc_surface_set = set([])
        basic_schema_list = []      # [(comb, path_len, ground_pred_list, path_list)]
        for comb, path_len, query_ret in ground_comb_list:
            # LogInfo.begin_track('path_len: %d, comb: %s', path_len, comb)
            for ground_pred_list in query_ret:      # enumerate each query row
                _, path_list, raw_paths = convert_combination(comb=comb, path_len=path_len,
                                                              gather_linkings=gather_linkings,
                                                              sparql_driver=sparql_driver,
                                                              ground_pred_list=ground_pred_list,
                                                              vb=vb)
                # path_surface_list = ['|'.join(path) for path in path_list]
                # path_surface_list.sort()
                # sc_surface = ', '.join(path_surface_list)
                # if sc_surface not in sc_surface_set:        # ignore duplicate schemas
                #     sc_surface_set.add(sc_surface)
                """
                2017-12-18: In the new schema representation (main path going forwards),
                            the searching results should have no duplicate schemas.
                """
                basic_schema_list.append((comb, path_len, ground_pred_list, path_list, raw_paths))
            # LogInfo.end_track()
        if vb >= 1:
            LogInfo.begin_track('%d distinct basic schemas collected.', len(basic_schema_list))
            for sc_surface in sc_surface_set:
                LogInfo.logs('[%s]', sc_surface)
            LogInfo.end_track()
        else:
            LogInfo.logs('%d distinct basic schemas collected.', len(basic_schema_list))

        # Step 4: Ordinal Expansion
        # TODO: ordinal value extraction: store as IdentifiedEntity
        # TODO: Find possible type --> relation, try F1 calculation
        # TODO: However, this step is not in a hurry.
        ext_schema_list = basic_schema_list     # save the schemas after applying ordinal constraint

    else:           # in SimpleQuestions, we are doing much easier task
        basic_schema_list = []
        for link_idx, link_data in enumerate(gather_linkings):
            if link_data.category != 'Entity':
                continue
            focus_mid = link_data.detail.entity.id
            comb = [(0, link_idx)]
            pred_set = simpq_sp_dict.get(focus_mid, set([]))
            for pred in pred_set:
                path_list = [[pred]]
                """
                Note: there's only one path, which is just one-hop.
                In WebQ/SimpQ, each path looks like [!p2, !p1, subj] (2017-12-18: except the main path)
                But we are doing another style in SimpleQuestions
                """
                raw_paths = [('Main', link_idx, focus_mid, [pred])]     # only one path in the raw_paths
                basic_schema_list.append((comb, 1, None, path_list, raw_paths))
        ext_schema_list = basic_schema_list
    # ======== End of collecting schemas ======== #

    # Step 5: F1 calculation and output the schema data
    schema_json_list = []
    schema_ans_list = []
    LogInfo.begin_track('Now calculating F1 for %d schemas:', len(ext_schema_list))
    for schema_idx, schema_info in enumerate(ext_schema_list):
        if schema_idx % 50 == 0:
            LogInfo.logs('Current schema: %d / %d', schema_idx, len(ext_schema_list))
        comb, path_len, ground_pred_list, path_list, raw_paths = schema_info
        if data_name in ('WebQ', 'CompQ'):
            sparql_str, _, _ = convert_combination(comb=comb, path_len=path_len,
                                                   gather_linkings=gather_linkings,
                                                   sparql_driver=sparql_driver,
                                                   ground_pred_list=ground_pred_list,
                                                   vb=vb)
            forbidden_mid_set = prepare_forbidden_mid_given_raw_paths(raw_paths=raw_paths,
                                                                      gather_linkings=gather_linkings)
            predict_value = kernel_querying(sparql_str=sparql_str,
                                            sparql_driver=sparql_driver,
                                            forbidden_mid_set=forbidden_mid_set)
            if len(predict_value) == 0:
                continue        # ignore schemas which have no output answer
            gold_list = target_value
            predict_list = list(predict_value)  # change from set to list
            if data_name == 'WebQ':
                r, p, f1 = compute_f1(gold_list, predict_list)
            else:       # CompQ
                """
                1. force lowercase, both gold and predict
                2. hyphen normalize: -, \u2013, \u2212 
                Won't the affect the values to be stored in the file.
                """
                eval_gold_list = [compq_answer_normalize(x) for x in gold_list]
                eval_predict_list = [compq_answer_normalize(x) for x in predict_list]
                r, p, f1 = compute_f1(eval_gold_list, eval_predict_list)
            # LogInfo.logs('predict size = %d, P/R/F1=%.6f/%.6f/%.6f', len(predict_value), p, r, f1)
        else:           # SimpleQuestions, no need to query SPARQL
            link_idx = comb[0][1]
            focus_mid = gather_linkings[link_idx].detail.entity.id
            pred = path_list[0][0]
            if focus_mid == target_value[0] and pred == target_value[1]:
                p = r = f1 = 1.
            else:
                p = r = f1 = 0.
            predict_list = []

        # schema_json = {
        #     'path_list': path_list,
        #     'p': p, 'r': r, 'f1': f1,
        #     'path_len': path_len,
        #     'comb': comb
        # }
        """ 2017-12-18: We use the following schema_json, path_len is deprecated """
        schema_json = {
            'raw_paths': raw_paths,
            'ans_size': len(predict_list),
            'p': p, 'r': r, 'f1': f1
        }
        schema_json_list.append(schema_json)
        schema_ans_list.append(predict_list)
    LogInfo.logs('Finally: collected %d schemas with valid output.', len(schema_json_list))
    LogInfo.end_track()
    return gather_linkings, schema_json_list, schema_ans_list


def tml_filtering(tml_result):
    """
    Remove illegal Time likings like "90210", which brought SPARQL stucking bugs (WebQ-71)
    This function works mainly for S-MART linking results.
    :param tml_result: the time linking result before filtering
    :return: the filtered Time linkings
    """
    year_re = re.compile(r'^[1-2][0-9][0-9][0-9]$')     # 1000-2999 available
    filt_tml_result = []
    for tml in tml_result:
        assert isinstance(tml.entity, DateValue)
        tml_name = tml.entity.sparql_name()[:4]
        if re.match(year_re, tml_name):
            tml.entity.value = tml_name     # 1800s --> 1800, TODO: Currently we won't add more rules
            filt_tml_result.append(tml)
    return filt_tml_result


def build_tml_comp_result(tml_result, tokens):
    tml_comp_result = []  # store the comparison (==, <, >, >= ...)
    for tml_item in tml_result:
        st_pos = tml_item.tokens[0].index
        last_lemma = tokens[st_pos - 1].lemma if st_pos > 0 else ''
        comp = '=='
        if last_lemma == 'before':
            comp = '<'
        elif last_lemma == 'after':
            comp = '>'
        elif last_lemma == 'since':
            comp = '>='
        tml_comp_result.append(comp)
    return tml_comp_result


def compq_answer_normalize(ans):
    """
    Change hyphen and en-dash into '-', and then lower case.
    """
    return re.sub(u'[\u2013\u2212]', '-', ans).lower()


# Deprecated, use eval/official_eval.compute_f1 as instead
# def f1_calc(target_value, predict_value):
#     p = r = f1 = 0.0
#     if len(predict_value) != 0:
#         joint = predict_value & target_value
#         p = 1.0 * len(joint) / len(predict_value)
#         r = 1.0 * len(joint) / len(target_value)
#         f1 = 2.0 * p * r / (p + r) if p > 0.0 else 0.0
#     return p, r, f1


def main_linking_only(qa_list, linking_wrapper, linking_cache, args):
    q_size = len(qa_list)
    ground_truth = args.ground_truth        # only work for SimpleQuestions
    LogInfo.logs('linking_mode = %s, ground_truth = %s', args.linking_mode, ground_truth)

    if args.linking_mode == 'S-MART':
        assert args.data_name in ('WebQ', 'CompQ')
    elif args.linking_mode == 'Lukov':
        assert linking_wrapper.lukov_linker is not None
    if ground_truth:
        assert args.data_name == 'SimpQ'        # these switches only works for particular datasets
        e_set = set([])
        for qa in qa_list:
            e_set.add(qa['targetValue'][0])     # collect all subjects
        LogInfo.logs('%d distinct focus entity collected.', len(e_set))
        # LogInfo.logs(list(e_set)[: 10])
        load_entity_name(concern_e_set=e_set)

    for q_idx, qa in enumerate(qa_list):
        if q_idx < args.q_start or q_idx >= args.q_end:
            continue
        LogInfo.begin_track('Entering Q %d / %d [%s]:', q_idx, q_size, qa['utterance'].encode('utf-8'))
        q = qa['utterance']
        linking_result = linking_cache.get(q)
        if linking_result is None:
            linking_result = perform_linking(
                linking_wrapper=linking_wrapper,
                data_name=args.data_name, q_idx=q_idx,
                q=q, target_value=qa['targetValue'],
                ground_truth=args.ground_truth
            )
            linking_cache.put(q, linking_result)
        tokens, el_result, tl_result, tml_result = linking_result
        LogInfo.logs('%d E + %d T + %d Tm linked.',
                     len(el_result), len(tl_result), len(tml_result))
        for el in el_result:
            disp = 'E: %s (%s) %.6f' % (el.entity.id.encode('utf-8'),
                                        el.name.encode('utf-8'),
                                        el.surface_score)
            LogInfo.logs(disp)
        for tl in tl_result:
            disp = 'T: %s (%s) %.6f' % (tl.entity.id.encode('utf-8'),
                                        tl.name.encode('utf-8'),
                                        tl.surface_score)
            LogInfo.logs(disp)
        for tml in tml_result:
            disp = 'Tm: %s %.6f' % (tml.entity.sparql_name().encode('utf-8'),
                                    tml.surface_score)
            LogInfo.logs(disp)
        LogInfo.end_track()


def main_default(qa_list, linking_wrapper, linking_cache, sparql_driver, simpq_sp_dict, args):
    q_size = len(qa_list)
    for q_idx, qa in enumerate(qa_list):
        if q_idx < args.q_start or q_idx >= args.q_end:
            continue
        LogInfo.begin_track('Entering Q %d / %d [%s]:', q_idx, q_size, qa['utterance'].encode('utf-8'))
        sub_idx = q_idx / 100 * 100
        sub_dir = '%s/data/%d-%d' % (args.output_dir, sub_idx, sub_idx + 99)
        if not os.path.exists(sub_dir):
            os.makedirs(sub_dir)
        save_schema_fp = '%s/%d_schema' % (sub_dir, q_idx)
        save_ans_fp = '%s/%d_ans' % (sub_dir, q_idx)
        save_link_fp = '%s/%d_links' % (sub_dir, q_idx)
        if os.path.isfile(save_schema_fp) \
                and os.path.isfile(save_ans_fp) \
                and os.path.isfile(save_link_fp):
            LogInfo.end_track('Skip this question, already saved.')
            continue

        target_value = qa['targetValue']
        gather_linkings, schema_json_list, schema_ans_list = \
            process_single_question(data_name=args.data_name,
                                    q=qa['utterance'],
                                    q_idx=q_idx,
                                    target_value=target_value,
                                    linking_wrapper=linking_wrapper,
                                    linking_cache=linking_cache,
                                    ground_truth=args.ground_truth,
                                    sparql_driver=sparql_driver,
                                    simpq_sp_dict=simpq_sp_dict,
                                    vb=args.verbose)

        with open(save_link_fp + '.tmp', 'w') as bw_link:
            cPickle.dump(gather_linkings, bw_link)

        bw_sc = open(save_schema_fp + '.tmp', 'w')
        bw_ans = open(save_ans_fp + '.tmp', 'w')
        for schema_json, predict_list in zip(schema_json_list, schema_ans_list):
            json.dump(predict_list, bw_ans)
            bw_ans.write('\n')
            json.dump(schema_json, bw_sc)
            bw_sc.write('\n')
        bw_sc.close()
        bw_ans.close()
        LogInfo.logs('Schema/Answer/Link saving complete.')

        shutil.move(save_schema_fp + '.tmp', save_schema_fp)
        shutil.move(save_link_fp + '.tmp', save_link_fp)
        shutil.move(save_ans_fp + '.tmp', save_ans_fp)

        del gather_linkings
        del schema_ans_list
        del schema_json_list

        LogInfo.end_track()     # End of Q


def main(args):
    parser_port = parser_port_dict[args.machine]
    sparql_port = sparql_port_dict[args.machine]
    assert args.data_name in ('WebQ', 'CompQ', 'SimpQ')

    qa_list = []
    if args.data_name == 'WebQ':
        qa_list = load_webq()
    elif args.data_name == 'CompQ':
        qa_list = load_compq()
    elif args.data_name == 'SimpQ':
        qa_list = load_simpq()

    lukov_lexicon = xmlrpclib.ServerProxy('http://202.120.38.146:9602')
    LogInfo.logs('Lukov Lexicon Proxy binded.')
    lukov_linker = LukovLinker(lexicon=lukov_lexicon)

    q_links_dict = None
    if args.linking_mode == 'S-MART':
        if args.data_name == 'WebQ':
            q_links_dict = load_webq_linking_data()
        elif args.data_name == 'CompQ':
            q_links_dict = load_compq_linking_data()
    linking_wrapper = LinkingWrapper(parser_ip=working_ip,
                                     parser_port=parser_port,
                                     linking_mode=args.linking_mode,
                                     q_links_dict=q_links_dict,
                                     lukov_linker=lukov_linker)
    linking_cache = DictCache(cache_fp=args.linking_cache_fp)

    if args.mode == 'link_only':
        main_linking_only(qa_list=qa_list, linking_wrapper=linking_wrapper,
                          linking_cache=linking_cache, args=args)
    else:
        simpq_sp_dict = None
        if args.linking_mode == 'S-MART':
            assert args.data_name in ('WebQ', 'CompQ')
        if args.ground_truth:
            assert args.data_name == 'SimpQ'
            simpq_sp_dict = load_simpq_sp_dict()
        sparql_driver = SparqlDriver(sparql_ip=working_ip,
                                     sparql_port=sparql_port,
                                     use_cache=args.use_sparql_cache,
                                     verbose=args.verbose)
        main_default(qa_list=qa_list, linking_wrapper=linking_wrapper,
                     linking_cache=linking_cache, sparql_driver=sparql_driver,
                     simpq_sp_dict=simpq_sp_dict, args=args)


if __name__ == '__main__':
    _args = parser.parse_args()
    LogInfo.begin_track('[compQA.eff_candgen.main] Running start ...')
    LogInfo.logs('Mode = %s', _args.mode)
    main(_args)
    LogInfo.end_track('All Done.')
