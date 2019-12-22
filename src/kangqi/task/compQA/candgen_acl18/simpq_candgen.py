"""
Author: Kangqi Luo
Date: 180505
Goal: candgen for SimpQ
"""


import os
import math
import json
import codecs
import shutil
import argparse

from .global_linker import GlobalLinker, LinkData

from ..dataset.kq_schema import CompqSchema
from ..dataset.u import load_simpq
from ..util.word_emb import WordEmbeddingUtil
from ..util.fb_helper import inverse_predicate

from kangqi.util.LogUtil import LogInfo
from kangqi.util.time_track import TimeTracker as Tt


working_ip = '202.120.38.146'
parser_port_dict = {'Blackhole': 9601, 'Darkstar': 8601}
sparql_port_dict = {'Blackhole': 8999, 'Darkstar': 8699}

parser = argparse.ArgumentParser(description='Efficient Candidate Generation')
parser.add_argument('--machine', default='Blackhole', choices=['Blackhole', 'Darkstar'])
parser.add_argument('--word_emb', default='glove')
parser.add_argument('--dim_emb', type=int, default=300, help='word/predicate embedding dimension')
parser.add_argument('--lex_name', default='s-mart')
parser.add_argument('--fb_subset', choices=['FB2M', 'FB5M'])
parser.add_argument('--output_dir')
parser.add_argument('--verbose', type=int, default=0)


class SimpleQCandidateGenerator:

    def __init__(self, fb_subset, lex_name, wd_emb_util, vb=1):
        simpq_dir = '/home/data/SimpleQuestions/SimpleQuestions_v2'
        q_links_fp = '%s/ACL18/simpQ.all.%s.q_links' % (simpq_dir, lex_name)
        fb_fp = '%s/freebase-subsets/freebase-%s.txt' % (simpq_dir, fb_subset)

        self.global_linker = GlobalLinker(wd_emb_util=wd_emb_util, q_links_fp=q_links_fp)
        self.subj_pred_dict = self.load_fb_subset(fb_fp=fb_fp)

        """ 180505: manually add exp(.) to score of each entity linking,
            as Fengli forgot to do so in his code. """
        for q_idx, link_data_list in self.global_linker.q_links_dict.items():
            for gl_data in link_data_list:
                gl_data.link_feat['score'] = math.exp(gl_data.link_feat['score'])

        self.vb = vb
        # vb = 0: show basic flow of the process
        # vb = 1: show detail linking information

    @staticmethod
    def load_fb_subset(fb_fp):
        LogInfo.begin_track('Loading freebase subset from [%s] ...', fb_fp)
        prefix = 'www.freebase.com/'
        pref_len = len(prefix)
        subj_pred_dict = {}
        with codecs.open(fb_fp, 'r', 'utf-8') as br:
            lines = br.readlines()
        LogInfo.logs('%d lines loaded.', len(lines))
        for line_idx, line in enumerate(lines):
            if line_idx % 500000 == 0:
                LogInfo.logs('Current: %d / %d', line_idx, len(lines))
            s, p, _ = line.strip().split('\t')
            s = s[pref_len:].replace('/', '.')
            p = p[pref_len:].replace('/', '.')
            subj_pred_dict.setdefault(s, set([])).add(p)
        LogInfo.logs('%d related entities and %d <S, P> pairs saved.',
                     len(subj_pred_dict), sum([len(v) for v in subj_pred_dict.values()]))
        LogInfo.end_track()
        return subj_pred_dict

    def single_question_candgen(self, q_idx, qa, link_fp, opt_sc_fp):
        # =================== Linking first ==================== #
        Tt.start('linking')
        if os.path.isfile(link_fp):
            gather_linkings = []
            with codecs.open(link_fp, 'r', 'utf-8') as br:
                for line in br.readlines():
                    tup_list = json.loads(line.strip())
                    ld_dict = {k: v for k, v in tup_list}
                    gather_linkings.append(LinkData(**ld_dict))
            LogInfo.logs('Read %d links from file.', len(gather_linkings))
        else:
            tok_list = [tok.token for tok in qa['tokens']]
            gather_linkings = self.global_linker.perform_linking(
                q_idx=q_idx,
                tok_list=tok_list,
                entity_only=True
            )
        LogInfo.begin_track('Show %d E links :', len(gather_linkings))
        if self.vb >= 1:
            for gl in gather_linkings:
                LogInfo.logs(gl.display().encode('utf-8'))
        LogInfo.end_track()
        Tt.record('linking')

        # ==================== Save linking results ================ #
        if not os.path.isfile(link_fp):
            with codecs.open(link_fp + '.tmp', 'w', 'utf-8') as bw:
                for gl in gather_linkings:
                    bw.write(json.dumps(gl.serialize()) + '\n')
            shutil.move(link_fp + '.tmp', link_fp)
            LogInfo.logs('%d link data save to file.', len(gather_linkings))

        # ===================== simple predicate finding ===================== #
        gold_entity, gold_pred, _ = qa['targetValue']
        sc_list = []
        for gl_data in gather_linkings:
            entity = gl_data.value
            pred_set = self.subj_pred_dict.get(entity, set([]))
            for pred in pred_set:
                sc = CompqSchema()
                sc.hops = 1
                sc.aggregate = False
                sc.main_pred_seq = [pred]
                sc.inv_main_pred_seq = [inverse_predicate(pred) for pred in sc.main_pred_seq]  # [!p1, !p2]
                sc.inv_main_pred_seq.reverse()  # [!p2, !p1]
                sc.raw_paths = [('Main', gl_data, [pred])]
                sc.ans_size = 1
                if entity == gold_entity and pred == gold_pred:
                    sc.f1 = sc.p = sc.r = 1.
                else:
                    sc.f1 = sc.p = sc.r = 0.
                sc_list.append(sc)

        # ==================== Save schema results ================ #
        # p, r, f1, ans_size, hops, raw_paths, (agg)
        # raw_paths: (category, gl_pos, gl_mid, pred_seq)
        with codecs.open(opt_sc_fp + '.tmp', 'w', 'utf-8') as bw:
            for sc in sc_list:
                sc_info_dict = {k: getattr(sc, k) for k in ('p', 'r', 'f1', 'ans_size', 'hops')}
                if sc.aggregate is not None:
                    sc_info_dict['agg'] = sc.aggregate
                opt_raw_paths = []
                for cate, gl, pred_seq in sc.raw_paths:
                    opt_raw_paths.append((cate, gl.gl_pos, gl.value, pred_seq))
                sc_info_dict['raw_paths'] = opt_raw_paths
                bw.write(json.dumps(sc_info_dict) + '\n')
        shutil.move(opt_sc_fp + '.tmp', opt_sc_fp)
        LogInfo.logs('%d schemas successfully saved into [%s].', len(sc_list), opt_sc_fp)


def main(args):
    qa_list = load_simpq()
    wd_emb_util = WordEmbeddingUtil(wd_emb=args.word_emb, dim_emb=args.dim_emb)
    lex_name = args.lex_name
    # lex_name = 'filter-%s-xs-lex-score' % args.fb_subset
    cand_gen = SimpleQCandidateGenerator(fb_subset=args.fb_subset,
                                         lex_name=lex_name,
                                         wd_emb_util=wd_emb_util,
                                         vb=args.verbose)

    for q_idx, qa in enumerate(qa_list):
        LogInfo.begin_track('Entering Q %d / %d [%s]:',
                            q_idx, len(qa_list), qa['utterance'].encode('utf-8'))
        sub_idx = q_idx / 1000 * 1000
        sub_dir = '%s/data/%d-%d' % (args.output_dir, sub_idx, sub_idx + 999)
        if not os.path.exists(sub_dir):
            os.makedirs(sub_dir)
        opt_sc_fp = '%s/%d_schema' % (sub_dir, q_idx)
        link_fp = '%s/%d_links' % (sub_dir, q_idx)
        if os.path.isfile(opt_sc_fp):
            LogInfo.end_track('Skip this question, already saved.')
            continue
        Tt.start('single_q')
        cand_gen.single_question_candgen(q_idx=q_idx, qa=qa,
                                         link_fp=link_fp, opt_sc_fp=opt_sc_fp)
        Tt.record('single_q')
        LogInfo.end_track()     # End of Q


if __name__ == '__main__':
    LogInfo.begin_track('[kangqi.task.compQA.candgen_acl18.simpq_candgen] ... ')
    _args = parser.parse_args()
    main(_args)
    LogInfo.end_track('All Done.')
    Tt.display()
