import codecs
from kangqi.util.LogUtil import LogInfo


compq_raw_fp = '/home/data/ComplexQuestions/compQ.test.release'
official_eval_detail_fp = 'log.compq_check'
predict_fp = 'runnings/CompQ/180203_strict_relation_only/all_Compact/GRU_pCAtt_pwCAtt_mgSum_k1.0_H0.5_Neg-20-20_b32/result/test_schema_004.txt'



q_idx_dict = {}
idx_ret_dict = {}


with codecs.open(compq_raw_fp, 'r', 'utf-8') as br:
    lines = br.readlines()
    for idx, line in enumerate(lines):
        spt = line.strip().split('\t')
        q_idx_dict[spt[0]] = idx
        idx_ret_dict[idx] = {'utterance': spt[0]}
LogInfo.logs('%d CompQ-Test loaded.', len(lines))


with codecs.open(official_eval_detail_fp, 'r', 'utf-8') as br:
    lines = br.readlines()
    for line in lines:
        spt = line.strip().split('\t')
        idx = q_idx_dict[spt[0]]
        idx_ret_dict[idx]['official'] = float(spt[1])
LogInfo.logs('%d Official loaded.', len(lines))

with codecs.open(predict_fp, 'r', 'utf-8') as br:
    lines = br.readlines()
    for line in lines:
        spt = line.strip().split('\t')
        idx = int(spt[0]) - 1300
        idx_ret_dict[idx]['predict'] = float(spt[2])
LogInfo.logs('%d Predict loaded.', len(lines))

for idx in range(800):
    ret = idx_ret_dict[idx]
    official, predict= [ret.get(x, 0.) for x in ['official', 'predict']]
    utterance = ret['utterance']
    delta = abs(official - predict)
    flag = '    ' if delta < 1e-6 else '****'
    LogInfo.logs('%s\tTest-%03d\t%.6f\t%.6f\t%.6f\t%s',
                 flag, idx, official, predict, delta, utterance.encode('utf-8'))
