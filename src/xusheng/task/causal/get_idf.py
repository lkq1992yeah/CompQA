# Calculate IDF scores for each words in split_full.txt

from xusheng.util.tf_idf import TfIdf
from kangqi.util.LogUtil import LogInfo

if __name__=="__main__":
    rootFp = "/home/xusheng/Causal"
    with open(rootFp + "/split_full.txt", 'rb') as fin:
        lines = fin.readlines()
    cause = TfIdf()
    effect = TfIdf()
    words = dict()
    for idx, line in enumerate(lines):
        spt = line.strip().split('\t')
        if len(spt) < 2:
            LogInfo.logs("line %d error.", idx)
            continue
        for word in spt[0].split(' '):
            words[word] = words.get(word, 0) + 1
        for word in spt[1].split(' '):
            words[word] = words.get(word, 0) + 1
        cause.add_document(str(idx), spt[0].split(' '))
        effect.add_document(str(idx), spt[1].split(' '))
        # LogInfo.logs("%d docs done.", idx)

    LogInfo.logs("size of vocab: %d", len(words))
    cnt = 0
    for word in words:
        if words[word] < 10:
            continue
        cnt += 1
    LogInfo.logs("size of effective vocab: %d", cnt)

    with open(rootFp + "/idf_cause.txt", 'w') as fout:
        idf = sorted(cause.get_idf(threshold=1).items(), key=lambda x: x[1], reverse=True)
        for (word, score) in idf:
            if words[word] < 10:
                continue
            fout.write(word + '\t' + str(score) + '\n')

    with open(rootFp + "/idf_effect.txt", 'w') as fout:
        idf = sorted(effect.get_idf(threshold=1).items(), key=lambda x: x[1], reverse=True)
        for (word, score) in idf:
            if words[word] < 10:
                continue
            fout.write(word + '\t' + str(score) + '\n')