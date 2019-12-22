import json
import random
import re
import difflib


def foo():
    with open('webqsp_sklt.txt', 'w') as w:
        with open('/home/xianyang/workspace/WebQSP/data/WebQSP.train.json', 'r') as f:
            webqsp = json.load(f)
            webqsp = webqsp['Questions']
            # print len(webqsp)
            count0, count1, count2, count3 = 0, 0, 0, 0
            for q in webqsp:
                id = q['QuestionId']
                sp = q['Parses']
                for p in sp:
                    conf = p['AnnotatorComment']['Confidence']
                    # if conf != 'Normal':
                    #     continue
                    focus_name = p['TopicEntityName']
                    focus_mid = p['TopicEntityMid']
                    sklt = p['InferentialChain']
                    if sklt == None:
                        sklt = 'none'
                    # print conf, focus_name, focus_mid, sklt
                    w.write('%s\t%s\t%s\t%s\n' % (id, focus_name, focus_mid, '--'.join(sklt)))

TOTAL_SKELETONS = 200
BEST = 80
WORST = 30
RANDOM = 90
# assert: BEST + WORST + RANDOM == TOTAL_SKELETONS

def gen():
    gold = {}
    with open('webqsp_sklt.txt', 'r') as golden:
        for line in golden:
            qid, focus, mid, sklt = line[:-1].split('\t')
            if sklt == 'n--o--n--e':
                continue
            qid = int(qid[qid.find('-') + 1 : ])
            if qid in gold:
                gold[qid].append((mid, sklt))
            else:
                gold[qid] = [(mid, sklt)]
    cand = {}
    with open('basic.graph.train.tsv', 'r') as candidate:
        for line in candidate:
            qid, focus, mid, sklt = line[:-1].split('\t')
            qid = int(qid)
            if qid in gold:
                print qid
                sim = compute_similarity(gold[qid], (mid, sklt))
                if qid in cand:
                    cand[qid].append((sim, mid, sklt))
                else:
                    cand[qid] = [(sim, mid, sklt)]

    with open('filtered_skeleton.txt', 'w') as f:
        print len(cand)
        for i in range(3778):
            if i in cand:
                cand_list = cand[i]
                if len(cand_list) <= TOTAL_SKELETONS:
                    out = cand_list
                else:
                    cand_list.sort(key=lambda x: x[0])
                    best = cand_list[-BEST:]
                    worst = cand_list[:WORST]
                    rand = random.sample(cand_list[WORST:-BEST], RANDOM)
                    out = best + worst + rand
                for cand_ in out:
                    f.write('%d\t%f\t%s\t%s\n' % (i, cand_[0], cand_[1], cand_[2]))
                

def compute_similarity(goldList, s):
    # print goldList, s
    max_sim = 0
    for gold in goldList:
        if gold[0] != s[0]:
            # different focus
            continue
        else:
            sim_score = difflib.SequenceMatcher(None, gold[1], s[1]).ratio()
            # print gold[1], s[1], sim_score
            if sim_score > max_sim:
                max_sim = sim_score
    # bp = raw_input('bp:')
    return max_sim


if __name__ == '__main__':
    gen()