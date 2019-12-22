"""
Goal:
1. add S-MART information into links
2. add ans_size into schema
"""
import os
import json
import math
import codecs
import cPickle
import shutil


from ...candgen.smart import load_webq_linking_data
from ...eff_candgen.combinator import LinkData

from kangqi.util.LogUtil import LogInfo
from kangqi.util.discretizer import Discretizer


log_score_disc = Discretizer(split_list=[0, 2, 4, 6, 8, 10, 12])    # 7+1
ratio_disc = Discretizer(split_list=[0.001, 0.01, 0.1, 0.2, 0.5])   # 5+1
feat_len = log_score_disc.len + ratio_disc.len


def build_feature_vector(score, max_score):
    log_score = math.log(score)
    ratio = 1. * score / max_score
    log_score_vec = log_score_disc.convert(score=log_score).tolist()
    ratio_vec = ratio_disc.convert(score=ratio).tolist()
    return log_score_vec + ratio_vec


def single_question(schema_fp, ans_fp, links_fp, smart_item_list):
    if os.path.isfile(schema_fp + '.ori') and os.path.isfile(links_fp + '.ori'):
        LogInfo.logs('Skip, already done.')
        return

    new_sc_info_list = []
    with codecs.open(schema_fp, 'r', 'utf-8') as br_sc, codecs.open(ans_fp, 'r', 'utf-8') as br_ans:
        sc_lines = br_sc.readlines()
        LogInfo.logs('sc_lines = %d', len(sc_lines))
        ans_lines = br_ans.readlines()
        LogInfo.logs('ans_lines = %d', len(ans_lines))
        assert len(sc_lines) == len(ans_lines)
        for sc_line, ans_line in zip(sc_lines, ans_lines):
            sc_info_dict = json.loads(sc_line.strip())
            ans_list = json.loads(ans_line.strip())
            sc_info_dict['ans_size'] = len(ans_list)        # add answer_size
            new_sc_info_list.append(sc_info_dict)

    smart_dict = {}     # Collect SMART information
    for item in smart_item_list:
        surface = item.surface_form.replace(' ', '').lower()
        smart_dict.setdefault(surface, {})
        mid = item.mid
        score = item.score
        smart_dict[surface][mid] = score

    with open(links_fp, 'rb') as br:
        gather_linkings = cPickle.load(br)
    new_gather_linkings = []
    for link_data in gather_linkings:
        detail, category, comp, disp = link_data
        link_feat = []
        if category == 'Entity':    # only consider focus / constraint entity
            surface = ''.join(t.token for t in detail.tokens).replace(' ', '').lower()
            mid = detail.entity.id
            if surface in smart_dict:
                score = smart_dict[surface][mid]
                max_score = max(smart_dict[surface].values())
                link_feat = build_feature_vector(score=score, max_score=max_score)
                LogInfo.logs('%s (%s): %s (score = %d, ratio = %.6f)',
                             mid.encode('utf-8'), detail.entity.name.encode('utf-8'),
                             link_feat, score, 1. * score / max_score)
            else:
                LogInfo.logs('Warning: [%s] not found.', surface.encode('utf-8'))
                link_feat = [0] * feat_len
        new_gather_linkings.append(LinkData(detail, category, comp, disp, link_feat))

    shutil.move(schema_fp, schema_fp + '.ori')
    shutil.move(links_fp, links_fp + '.ori')
    with open(links_fp, 'wb') as bw:
        cPickle.dump(new_gather_linkings, bw)
        LogInfo.logs('New links saved.')
    with codecs.open(schema_fp, 'w', 'utf-8') as bw:
        for sc_info_dict in new_sc_info_list:
            json.dump(sc_info_dict, bw)
            bw.write('\n')
        LogInfo.logs('%d new schema saved.', len(new_sc_info_list))


def main():
    LogInfo.begin_track('Patch start working ... ')
    webq_el_dict = load_webq_linking_data()
    data_dir = 'runnings/candgen_WebQ/180115_siva/data'

    q_st = 0
    q_ed = 5810
    for q_idx in range(q_st, q_ed):
        smart_item_list = webq_el_dict.get(q_idx, [])
        div = q_idx / 100
        sub_dir = '%d-%d' % (div*100, div*100+99)
        ans_fp = '%s/%s/%d_ans' % (data_dir, sub_dir, q_idx)
        schema_fp = '%s/%s/%d_schema' % (data_dir, sub_dir, q_idx)
        links_fp = '%s/%s/%d_links' % (data_dir, sub_dir, q_idx)
        LogInfo.begin_track('Entering Q-%d: ', q_idx)
        single_question(schema_fp, ans_fp, links_fp, smart_item_list)
        LogInfo.end_track()
    LogInfo.end_track()


if __name__ == '__main__':
    main()
