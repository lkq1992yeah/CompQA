# -*- coding: utf-8 -*-

#==============================================================================
# Author: Kangqi Luo
# Goal: Pick a set of focus questions from ComplexQ.
# Will do manual filtering afterwards.
#==============================================================================

from ..candgen.cand_gen import CandidateGenerator

from kangqi.util.LogUtil import LogInfo


if __name__ == '__main__':
    LogInfo.begin_track('[question_picker] starts ... ')

    cand_gen = CandidateGenerator()
    linker = cand_gen.linker
    cq_dir = '/home/data/ComplexQuestions'
    for mark in ['train', 'test']:
        fp = '%s/compQ.%s.release' %(cq_dir, mark)
        save_fp = '%s/ordinal_picking/raw.%s' %(cq_dir, mark)
        bw = open(save_fp, 'w')
        br = open(fp, 'r'); lines = br.readlines(); br.close()
        q_sz = len(lines)
        LogInfo.begin_track('Dealing with %d %s questions.', q_sz, mark)
        for idx in range(q_sz):
            if idx % 50 == 0: LogInfo.logs('%d / %d scanned.', idx, q_sz)

            q = lines[idx].strip().split('\t')[0]
            tokens = linker.parse(q).tokens
            tml_result = linker.time_identify_with_parse(tokens)
            rank_set = cand_gen.rank_extraction(q, tml_result, tokens)
            if len(rank_set) > 0:
                LogInfo.logs('%s', q)
                bw.write('%d\t%s' %(idx + 1, q))
                for rank_item in rank_set:
                    bw.write('\t%s' %rank_item.to_string())
                bw.write('\n')
        LogInfo.end_track()
        bw.close()

    LogInfo.end_track('Done.')