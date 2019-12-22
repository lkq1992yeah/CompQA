"""
Design features for entity linking of Web Q.
currently 5 feats:
1. lexicon level similarity
2. entity pop
3. entity pop ratio
4. entity type
5. entity coherence
"""

from xusheng.util.log_util import LogInfo

def is_legal(ch):
    if 'A' <= ch <= 'Z'  or 'a' <= ch <= 'z':
        return True
    return False

def jaccard_similarity(query, document):
    intersection = set(query).intersection(set(document))
    union = set(query).union(set(document))
    return float(len(intersection))/len(union)

def ngram_similarity(query, document):
    if len(query) == 1 or len(document) == 1:
        return 0.0
    if len(query) == 2:
        query = query + ' '
    if len(document) == 2:
        document = document + ' '
    set_q = set()
    set_d = set()
    for i in range(0, len(query)-2):
        set_q.add(query[i:i+3])
    for i in range(0, len(document)-2):
        set_d.add(document[i:i+3])
    intersection = set_q.intersection(set_d)
    union = set_q.union(set_d)
    return float(len(intersection))/len(union)


if __name__ == '__main__':
    wiki_path = "/home/xusheng/wikipedia/en-extracted"
    fb_path = "/home/kangqi/Freebase/Transform"

    LogInfo.begin_track("Loading wiki-fb entity map...")
    wiki_fb_map = dict()
    cnt = 0
    with open(fb_path + "/GS-cleanWiki-triple.txt") as fin:
        for line in fin:
            spt = line.strip().split('\t')
            if len(spt) < 3:
                continue
            fb_ent = spt[0]
            wiki_ent = spt[2].split('/wiki/')[1][:-1]
            wiki_ent = wiki_ent.lower().replace('_', ' ')
            wiki_fb_map[wiki_ent] = fb_ent
            cnt += 1
            LogInfo.show_line(cnt, 500000)
    LogInfo.end_track("%d pairs in total", cnt)

    LogInfo.begin_track("Loading fb entity pop...")
    fb_ent_pop_map = dict()
    cnt = 0
    with open("/home/xusheng/freebase/top5m.mid") as fin:
        for line in fin:
            spt = line.strip().split('\t')
            if len(spt) < 2:
                continue
            ent = spt[0]
            pop = int(spt[1])
            fb_ent_pop_map[ent] = pop
            cnt += 1
            LogInfo.show_line(cnt, 500000)
    LogInfo.end_track("%d entities in total", cnt)

    LogInfo.begin_track("Loading wiki entity pop...")
    wiki_ent_pop_map = dict()
    cnt = 0
    with open(wiki_path + "/entity.pop") as fin:
        for line in fin:
            spt = line.strip().split('\t')
            if len(spt) < 2:
                continue
            ent = spt[0]
            pop = int(spt[1])
            wiki_ent_pop_map[ent] = pop
            cnt += 1
            LogInfo.show_line(cnt, 500000)
    LogInfo.end_track("%d entities in total", cnt)

    # ===================== processing redirects & anchor text =======================

    ret = dict()
    LogInfo.begin_track("Processing wiki redirects...")
    mention_cnt = 0
    pair_cnt = 0
    with open("/home/xusheng/wikipedia/redirect/en-redirect.txt") as fin:
        for line in fin:
            spt = line.strip().split('\t')
            if len(spt) < 2:
                continue
            mention = spt[0].lower().replace('_', ' ').replace('\\', '')
            wiki_ent = spt[1].lower().replace('_', ' ').replace('\\', '')
            if wiki_ent in wiki_fb_map:
                # fb entity
                fb_ent = wiki_fb_map[wiki_ent]
                if mention + '\t' + fb_ent in ret:
                    # redirect already done
                    continue
                mention_cnt += 1
                if mention_cnt % 1000000 == 0:
                    LogInfo.logs("[log] %dm mention processed.", mention_cnt / 1000000)
                # fb entity popularity
                if fb_ent in fb_ent_pop_map:
                    fb_pop = fb_ent_pop_map[fb_ent]
                else:
                    # LogInfo.logs("[warning] fb pop not found for %s/%s", wiki_ent, fb_ent)
                    fb_pop = -1
                # wiki entity popularity
                if wiki_ent in wiki_ent_pop_map:
                    wiki_pop = wiki_ent_pop_map[wiki_ent]
                else:
                    # LogInfo.logs("[warning] wiki pop not found for %s/%s", wiki_ent, fb_ent)
                    wiki_pop = -1
                # similarity between mention & entity
                tmp_mention = mention
                word_simi = jaccard_similarity(mention.replace(',', '').replace('\'s', '').replace('(', '').replace(
                                                   ')', '').split(' '),
                                               wiki_ent.replace(',', '').replace('\'s', '').replace('(', '').replace(
                                                   ')', '').split(' '))
                ngram_simi = ngram_similarity(mention, wiki_ent)
                ret_line = "%s\t%s\t%s\t%.4f\t%.4f\t%.4f\t%d\t%d" % \
                           (mention, wiki_ent, fb_ent, 1.0, word_simi, ngram_simi, wiki_pop, fb_pop)
                # fout.write(ret_line + '\n')
                # fout.flush()
                ret[mention + '\t' + fb_ent] = ret_line
                pair_cnt += 1
                if pair_cnt % 1000000 == 0:
                    LogInfo.logs("[log] %dm m-e pairs generated.", pair_cnt / 1000000)
            else:
                # LogInfo.logs("[warning] fb entity not found for [%s]", wiki_ent)
                continue
    LogInfo.end_track("%d mentions and %d m-e pairs generated in redirects.", mention_cnt, pair_cnt)

    LogInfo.begin_track("Processing anchor text map...")
    # fout = open(wiki_path + "/m_e_lexicon.txt", 'w')
    with open(wiki_path + "/prior.txt", 'r') as fin:
        for line in fin:
            spt = line.strip().split("\t\t")
            mention = spt[0]
            if len(mention) == 1 and not is_legal(mention):
                continue
            mention_cnt += 1
            if mention_cnt % 1000000 == 0:
                LogInfo.logs("[log] %dm mention processed.", mention_cnt/1000000)
            wiki_ents = [x.split('\t')[0] for x in spt[1:]]
            link_prob = [float(x.split('\t')[1]) for x in spt[1:]]
            for wiki_ent, prob in zip(wiki_ents, link_prob):
                if wiki_ent in wiki_fb_map:
                    # fb entity
                    fb_ent = wiki_fb_map[wiki_ent]
                    if mention + '\t' + fb_ent in ret:
                        # redirect already done
                        continue
                    # fb entity popularity
                    if fb_ent in fb_ent_pop_map:
                        fb_pop = fb_ent_pop_map[fb_ent]
                    else:
                        # LogInfo.logs("[warning] fb pop not found for %s/%s", wiki_ent, fb_ent)
                        fb_pop = -1
                    # wiki entity popularity
                    if wiki_ent in wiki_ent_pop_map:
                        wiki_pop = wiki_ent_pop_map[wiki_ent]
                    else:
                        # LogInfo.logs("[warning] wiki pop not found for %s/%s", wiki_ent, fb_ent)
                        wiki_pop = -1
                    # similarity between mention & entity
                    word_simi = jaccard_similarity(mention.replace(',', '').replace('\'s', '').replace('(', '').replace(
                                                   ')', '').split(' '),
                                                   wiki_ent.replace(',', '').replace('\'s', '').replace('(', '').replace(')', '').split(' '))
                    ngram_simi = ngram_similarity(mention, wiki_ent)
                    ret_line = "%s\t%s\t%s\t%.4f\t%.4f\t%.4f\t%d\t%d" % \
                               (mention, wiki_ent, fb_ent, prob, word_simi, ngram_simi, wiki_pop, fb_pop)
                    # fout.write(ret_line + '\n')
                    # fout.flush()
                    ret[mention + '\t' + fb_ent] = ret_line
                    pair_cnt += 1
                    if pair_cnt % 1000000 == 0:
                        LogInfo.logs("[log] %dm m-e pairs generated.", pair_cnt/1000000)
                else:
                    # LogInfo.logs("[warning] fb entity not found for [%s]", wiki_ent)
                    continue
    LogInfo.end_track("%d mentions and %d m-e pairs generated in total.", mention_cnt, pair_cnt)

    LogInfo.begin_track("Writing to file...")
    cnt = 0
    with open(wiki_path + "/m_e_lexicon_v2.txt", 'w') as fout:
        for key, val in ret.items():
            fout.write(val + '\n')
            cnt += 1
            LogInfo.show_line(cnt, 1000000)
    LogInfo.end_track("%d lines written.", cnt)

