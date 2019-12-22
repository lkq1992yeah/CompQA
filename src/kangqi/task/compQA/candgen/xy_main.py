# -*- coding:utf-8 -*-

# ===================================
# Author: Kangqi Luo
# Date: 170926
# Goal: The main function for the whole candidate generation step
# Output: Json-ed schema file, one question one output file.
# ===================================

import os
import json
import shutil
import argparse

from .cand_gen import CandidateGenerator
from ..data_prepare.loss_calc import LossCalculator
from ..dataset.schema import build_sc_from_line
from ..dataset.u import load_webq
from ..util.fb_helper import FreebaseHelper

from kangqi.util.LogUtil import LogInfo


parser = argparse.ArgumentParser(description='Candidate Generation')

parser.add_argument('--parser_ip', default='202.120.38.146')
parser.add_argument('--parser_port', type=int, default=9601)
parser.add_argument('--sparql_ip', default='202.120.38.146')
parser.add_argument('--sparql_port', type=int, default=8999)
parser.add_argument('--fb_meta_dir', default='data/fb_metadata')

parser.add_argument('--cache_dir', default='runnings/candgen/cache')
parser.add_argument('--use_sparql_cache', action='store_true')
parser.add_argument('--sparql_verbose', action='store_true')
parser.add_argument('--check_only', action='store_true')

parser.add_argument('--SMART', action='store_true')
parser.add_argument('--min_ratio', type=float, default=0.1)
parser.add_argument('--min_surface_score', type=float, default=0.0)
parser.add_argument('--min_pop', type=int, default=0)

parser.add_argument('--k_hop', type=int, default=1)
parser.add_argument('--max_hops', type=int, default=2)

parser.add_argument('--data_name', default='WebQuestions',
                    choices=['WebQuestions', 'ComplexQuestions'])
parser.add_argument('--q_start', type=int, default=0)
parser.add_argument('--q_end', type=int, default=1000000)
parser.add_argument('--output_dir')
parser.add_argument('--verbose', type=int, default=0)


def main(args):
    LogInfo.begin_track('Candidate Generation Starts ...')

    qa_list = load_webq()
    q_size = len(qa_list)

    cand_gen = CandidateGenerator(cache_dir=args.cache_dir,
                                  use_sparql_cache=args.use_sparql_cache,
                                  check_only=args.check_only,
                                  parser_ip=args.parser_ip,
                                  parser_port=args.parser_port,
                                  sparql_port=args.sparql_port,
                                  sparql_verbose=args.sparql_verbose,
                                  k_hop=args.k_hop,
                                  max_hops=args.max_hops)
    loss_calc = LossCalculator(driver=cand_gen.driver)
    fb_helper = FreebaseHelper(data_dir=args.fb_meta_dir)

    for q_idx, qa in enumerate(qa_list):
        if q_idx < args.q_start or q_idx >= args.q_end:
            continue
        LogInfo.begin_track('Entering Q %d / %d [%s]:', q_idx, q_size, qa['utterance'])
        sub_idx = q_idx / 100 * 100
        sub_dir = '%s/data/%d-%d' % (args.output_dir, sub_idx, sub_idx + 99)
        if not os.path.exists(sub_dir):
            os.makedirs(sub_dir)
        save_schema_fp = '%s/%d_schema' % (sub_dir, q_idx)
        save_ans_fp = '%s/%d_ans' % (sub_dir, q_idx)
        if os.path.isfile(save_schema_fp) and os.path.isfile(save_ans_fp):
            LogInfo.end_track('Skip this question, already saved.')
            continue

        target_value = set(qa['targetValue'])
        schema_list = cand_gen.run_candgen(
            q_idx=q_idx, q=qa['utterance'],
            min_surface_score=args.min_surface_score,
            min_pop=args.min_pop,
            use_ext_sk=True,
            min_ratio=args.min_ratio,
            s_mart=args.SMART,
            vb=args.verbose
        )
        sc_sz = len(schema_list)

        LogInfo.begin_track('%d candidate schemas generated, now calculating F1 ...', sc_sz)

        bw_sc = open(save_schema_fp + '.tmp', 'w')
        bw_ans = open(save_ans_fp + '.tmp', 'w')

        for sc_idx, sc in enumerate(schema_list):
            LogInfo.begin_track('Showing schema %d / %d: ', sc_idx+1, sc_sz)
            sc.display()
            # sc.display_sparql()
            predict_set, score_dict = loss_calc.calculate(schema=sc,
                                                          answer_set=target_value,
                                                          vb=args.verbose)
            if len(predict_set) > 0:        # only keep the schema having valid output (ignoring all '' as outputs)
                predict_list = list(predict_set)
                json.dump(predict_list, bw_ans)
                bw_ans.write('\n')

                p = score_dict['p']
                r = score_dict['r']
                f1 = score_dict['f1']
                LogInfo.logs('p = %.6f, r = %.6f, f1 = %.6f', p, r, f1)

                sc_line = sc.to_xy_schema_line()
                xy_schema, path_list_str = build_sc_from_line(schema_line=sc_line,
                                                              fb_helper=fb_helper)
                xy_schema_json = {
                    'path_list': xy_schema.path_list,
                    'p': p, 'r': r, 'f1': f1
                }
                LogInfo.logs('Converted: [%s]', path_list_str)
                json.dump(xy_schema_json, bw_sc)
                bw_sc.write('\n')
            else:
                LogInfo.logs('Will not keep due to zero outputs.')

            LogInfo.end_track()     # End of one schema

        bw_sc.close()
        bw_ans.close()
        shutil.move(save_schema_fp + '.tmp', save_schema_fp)
        shutil.move(save_ans_fp + '.tmp', save_ans_fp)
        LogInfo.end_track('%d schemas saved.', sc_sz)

        LogInfo.end_track()     # End of Q

    LogInfo.end_track('All Done.')


if __name__ == '__main__':
    _args = parser.parse_args()
    main(_args)
