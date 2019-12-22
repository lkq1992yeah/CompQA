import numpy as np
from scipy import dot, linalg

from xusheng.util.log_util import LogInfo

def readcopa():
    copa = []
    worddic = {}
    fin = open("/home/yuchen/data/copa_phr.txt", "r")
    for i in range(1000):
        hyp = map(lambda x: x.split(":")[1], fin.readline().strip().split())
        alt1 = map(lambda x: x.split(":")[1], fin.readline().strip().split())
        alt2 = map(lambda x: x.split(":")[1], fin.readline().strip().split())
        for word in hyp:
            if word not in worddic:
                worddic[word] = ""
        for word in alt1:
            if word not in worddic:
                worddic[word] = ""
        for word in alt2:
            if word not in worddic:
                worddic[word] = ""

        copa.append([hyp, alt1, alt2])
    fin.close()
    
    return copa, worddic

def readlabel():
    label = []
    with open("/home/yuchen/data/copa_label.txt") as fin:
        for line in fin:
            _, qus, labl = line.strip().split()
            if qus == "cause":
                label.append([0, int(labl)])
            elif qus == "effect":
                label.append([1, int(labl)])
            else:
                print qus
    return label

def readvec(worddic, filename):
    vecdic = {}
    with open(filename) as fin:
        for line in fin:
            spt = line.strip().split()
            word = spt[0]
            if word in worddic:
                vec = spt[1:201]
                vecdic[word] = np.array(map(lambda x: float(x), vec))
    return vecdic
     
def get_similar(vec1, vec2, norm):
    if linalg.norm(vec1) == 0 or linalg.norm(vec2) == 0:
        return 0.0
    #print vec1, type(vec1), len(vec1)
    if norm:
        return dot(vec1, vec2.T)/linalg.norm(vec1)/linalg.norm(vec2)
    else:
        return dot(vec1, vec2.T)

def cal_score(cause, effect, cdic, enegdic, cnegdic, edic, lamd, setting, norm, ratio, verbose=False):
    score = 0
    num = 0
    rcause = []
    reffect = []
    for word in cause:
        if word in cdic and word in cnegdic:
            rcause.append(word)
    for word in effect:
        if word in edic and word in enegdic:
            reffect.append(word)

    sort_map = dict()
    for wordc in rcause:
        for worde in reffect:
            if wordc == worde:
               continue
            score_suf = get_similar(cdic[wordc], enegdic[worde], norm)
            score_nec = get_similar(cnegdic[wordc], enegdic[worde], norm)
            tmp = lamd * score_suf + (1-lamd) * score_nec
            # check reverse
            score_reverse = get_similar(cdic[worde], enegdic[wordc], norm)
            if abs(score_suf-score_reverse) / min(abs(score_suf), abs(score_reverse)) < ratio:
                continue
            score += tmp
            num += 1
            tmp_str = "[%s]-[%s] ==> %.1f*[%.2f]+%.1f*[%.2f]=[%.4f]" % \
                      (wordc, worde, lamd, score_suf, 1-lamd, score_nec, tmp)
            sort_map[tmp] = tmp_str

    if verbose:
        for line in [sort_map[k] for k in sorted(sort_map.keys(), reverse=True)]:
            LogInfo.logs(line)

    if setting == 1:
        return score
    elif setting == 2:
        if verbose:
            LogInfo.logs("%.4f / (%d+%d=%d) = %.4f", score,
                         len(rcause), len(reffect), len(rcause)+len(reffect),
                         score / (len(rcause) + len(reffect)))
        return score/(len(rcause)+len(reffect))
    elif setting == 3:
        return score/(len(rcause)*len(reffect))
    elif num == 0:
        if verbose:
            LogInfo.logs("%.4f / %d = %.4f", score, 0, 0.0)
        return 0.0
    else:
        if verbose:
            LogInfo.logs("%.4f / %d = %.4f", score, num, score/num)
        return score/num

def word_word1(copa, label, cdic, enegdic, cnegdic, edic, lamd, num, setting, norm, ratio, verbose=False):
    acc = 0
    wrong = 0
    for i in range(num, 1000):
        hyp, alt1, alt2 = copa[i]
        ask, labl = label[i]
        if verbose:
            LogInfo.begin_track("step into copa #%d", i+1)
            LogInfo.logs("q: %s", hyp)
            LogInfo.logs("o1: %s", alt1)
            LogInfo.logs("o2: %s", alt2)
            LogInfo.logs("answer: o%d", labl)
        # ask for cause
        if ask == 0:
            if verbose:
                LogInfo.begin_track("[ask for cause] o1/o2 -> q")
            cause, effect = alt1, hyp
            if verbose:
                LogInfo.begin_track("o1->q: [%s]->[%s]", cause, effect)
            score1 = cal_score(cause, effect, cdic, enegdic, cnegdic, edic, lamd, setting, norm, ratio, verbose)
            if verbose:
                LogInfo.logs("final score: %.4f", score1)
                LogInfo.end_track()

            cause, effect = alt2, hyp
            if verbose:
                LogInfo.begin_track("o2->q: [%s]->[%s]", cause, effect)
            score2 = cal_score(cause, effect, cdic, enegdic, cnegdic, edic, lamd, setting, norm, ratio, verbose)
            if verbose:
                LogInfo.logs("final score: %.4f", score2)
                LogInfo.end_track()

            if score1 > score2 and labl == 1:
                acc += 1
                if verbose:
                    LogInfo.logs("[[correct]]")
            if score1 < score2 and labl == 2:
                acc += 1
                if verbose:
                    LogInfo.logs("[[correct]]")
            if verbose:
                LogInfo.end_track()

        # ask for effect
        elif ask == 1:
            if verbose:
                LogInfo.begin_track("[ask for effect] q -> o1/o2")
            cause, effect = hyp, alt1
            if verbose:
                LogInfo.begin_track("q->o1: [%s]->[%s]", cause, effect)
            score1 = cal_score(cause, effect, cdic, enegdic, cnegdic, edic, lamd, setting, norm, ratio, verbose)
            if verbose:
                LogInfo.logs("final score: %.4f", score1)
                LogInfo.end_track()
            cause, effect = hyp, alt2
            if verbose:
                LogInfo.begin_track("q->o2: [%s]->[%s]", cause, effect)
            score2 = cal_score(cause, effect, cdic, enegdic, cnegdic, edic, lamd, setting, norm, ratio, verbose)
            if verbose:
                LogInfo.logs("final score: %.4f", score2)
                LogInfo.end_track()
            if score1 > score2 and labl == 1:
                acc += 1
                if verbose:
                    LogInfo.logs(">>correct<<")
            elif score1 < score2 and labl == 2:
                acc += 1
                if verbose:
                    LogInfo.logs(">>correct<<")
            else:
                wrong += 1
                if verbose:
                    LogInfo.logs(">>wrong<<")
            if verbose:
                LogInfo.end_track()
        else:
            print ask
            if verbose:
                LogInfo.logs("[error] ask=%d", ask)

        if verbose:
            LogInfo.end_track("end for #%d", i+1)
            LogInfo.logs("===========")

    if verbose:
        LogInfo.logs("status: %dY-%dW/%d", acc, wrong, 1000-num)
    return acc*1.0/(1000-num)
    
def senvec(words, vecdic):
    cnt = 0
    vec_res = np.zeros(200)
    for word in words:
        if word in vecdic:
            cnt += 1
            vec_res += vecdic[word]
    return vec_res/cnt

def sen_sen(copa, label, cdic, enegdic, cnegdic, edic, lamd, num, norm):
    acc = 0
    for i in range(num, 1000):
        hyp, alt1, alt2 = copa[i]
        ask, labl = label[i]
        # ask for cause
        if ask == 0:
            cause, effect = alt1, hyp
            suf1 = get_similar(senvec(cause, cdic), senvec(effect, enegdic), norm)
            nec1 = get_similar(senvec(cause, cnegdic), senvec(effect, edic), norm)
            score1 = lamd * suf1 + (1-lamd) * nec1
            cause, effect = alt2, hyp
            suf2 = get_similar(senvec(cause, cdic), senvec(effect, enegdic), norm)
            nec2 = get_similar(senvec(cause, cnegdic), senvec(effect, edic), norm)
            score2 = lamd * suf2 + (1-lamd) * nec2
            if score1 > score2 and labl == 1:
                acc += 1
            if score1 < score2 and labl == 2:
                acc += 1
        #ask for effect
        elif ask == 1:
            cause, effect = hyp, alt1
            suf1 = get_similar(senvec(cause, cdic), senvec(effect, enegdic), norm)
            nec1 = get_similar(senvec(cause, cnegdic), senvec(effect, edic), norm)
            score1 = lamd * suf1 + (1-lamd) * nec1
            cause, effect = hyp, alt2
            suf2 = get_similar(senvec(cause, cdic), senvec(effect, enegdic), norm)
            nec2 = get_similar(senvec(cause, cnegdic), senvec(effect, edic), norm)
            score2 = lamd * suf2 + (1-lamd) * nec2
            if score1 > score2 and labl == 1:
                acc += 1
            if score1 < score2 and labl == 2:
                acc += 1
        else:
            print ask
        #print i, cause, effect, suf1, nec1, suf2, nec2, score1, score2, labl, acc
    return acc*1.0/(1000-num)

def main():
    copa, worddic = readcopa()
    label = readlabel()
    cdic = readvec(worddic, "/home/yuchen/CppFiles/Causal/sync_wdFin_iter200.txt")
    enegdic = readvec(worddic, "/home/yuchen/CppFiles/Causal/syneneg_wdFin_iter200.txt")
    cnegdic = readvec(worddic, "/home/yuchen/CppFiles/Causal/syne_wdFin_iter200.txt")
    edic = readvec(worddic, "/home/yuchen/CppFiles/Causal/syncneg_wdFin_iter200.txt")

    verbose = False

    import sys
    mode = sys.argv[1]
    if mode == 'full':
        for ratio in range(21):
            for lamd in range(11):
                acc = word_word1(copa, label, cdic, enegdic, cnegdic, edic, lamd*0.1, 500, 4, True, ratio*0.1, verbose)
                print ratio*0.1, lamd*0.1, acc
        # print "word pair with norm:"
        # for setting in range(3):
        #     for lamd in range(11):
        #         acc = word_word1(copa, label, cdic, enegdic, cnegdic, edic, lamd*0.1, 500, setting, True, verbose)
        #         print lamd*0.1, setting, acc

        # print "\nword pair without norm:"
        # for setting in range(3):
        #     for lamd in range(11):
        #         acc = word_word1(copa, label, cdic, enegdic, cnegdic, edic, lamd*0.1, 500, setting, False, verbose)
        #         print lamd*0.1, setting, acc
        #
        # print "\nsentence level with norm:"
        # for lamd in range(11):
        #     acc = sen_sen(copa, label, cdic, enegdic, cnegdic, edic, lamd*0.1, 500, True)
        #     print lamd*0.1,  acc
        #
        # print "\nsentence level without norm:"
        # for lamd in range(11):
        #     acc = sen_sen(copa, label, cdic, enegdic, cnegdic, edic, lamd*0.1, 500, False)
        #     print lamd*0.1, acc
    elif mode == 'case':
        para1 = float(sys.argv[2])
        para2 = int(sys.argv[3])
        LogInfo.begin_track("case tracing for word-pair & lambda=%.1f, setting=%d:", para1, para2)
        verbose = True
        acc = word_word1(copa, label, cdic, enegdic, cnegdic, edic, para1, 500, para2, True, verbose)
        LogInfo.logs("[Accuracy] %.4f", acc)
        LogInfo.end_track()


if __name__ == '__main__':
    main()

