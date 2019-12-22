"""
Compare the upperbound between Ours and Bao
Currently, we just focus on strict level.
Display: cand_size, max_f1, delta, Bao's best row
"""

import codecs


our_meta_fp = 'runnings/candgen_CompQ/180123_Lukov_Lexicon/log.schema_check'
bao_meta_fp = 'codalab/CompQ/Bao2016/meta/compQ.all.meta'
ouput_comp_fp = 'runnings/candgen_CompQ/180123_Lukov_Lexicon/compare_with_bao.txt'

meta_info_list = []


with codecs.open(bao_meta_fp, 'r', 'utf-8') as br:
    lines = br.readlines()
    for idx, line in enumerate(lines):
        spt = line.strip().split('\t')
        q = spt[0]
        cand_size = int(spt[1])
        max_f1 = float(spt[2])
        meta_info_list.append({'idx': idx, 'q': q, 'bao_cand_size': cand_size,
                               'bao_max_f1': max_f1, 'bao_best_lines': spt[3]})

with codecs.open(our_meta_fp, 'r', 'utf-8') as br:
    lines = br.readlines()
    for line in lines:
        spt = line.strip().split('\t')
        if len(spt) != 11:
            continue
        if spt[1] == 'Q_idx':
            continue
        idx = int(spt[1])
        cand_size = int(spt[2])     # strict
        max_f1 = float(spt[6])
        diff = max_f1 - meta_info_list[idx]['bao_max_f1']
        meta_info_list[idx].update({'our_cand_size': cand_size, 'our_max_f1': max_f1, 'diff': diff})


srt_list = sorted(meta_info_list, key=lambda d: d['diff'])


with codecs.open(ouput_comp_fp, 'w', 'utf-8') as bw:
    bw.write('%4s\t%8s\t%8s\t%8s\t%8s\t%8s\n\n' % ('Idx', 'Our_Size', 'Bao_Size', 'Our_F1', 'Bao_F1', 'Diff.'))
    for info in srt_list:
        bw.write('%s\t%s\n' % (info['q'], info['bao_best_lines']))
        bw.write('%4d\t%8d\t%8d\t%8.6f\t%8.6f\t%8.6f\n' % (
            info['idx'], info['our_cand_size'], info['bao_cand_size'],
            info['our_max_f1'], info['bao_max_f1'], info['diff']
        ))
        bw.write('\n')
