"""
Goal: Convert S-MART into the format used for candgen_acl18/global_linker.py
"""
import json
import codecs

from ...dataset.u import load_compq, load_webq
from kangqi.util.LogUtil import LogInfo


def process_single_file(in_fp, out_fp, qa_list):
    LogInfo.begin_track('Deal with [%s] --> [%s]', in_fp, out_fp)
    with codecs.open(in_fp, 'r', 'utf-8') as br, codecs.open(out_fp, 'w', 'utf-8') as bw:
        for line_idx, line in enumerate(br.readlines()):
            spt = line.strip().split('\t')
            q_id, mention, st, span_len, mid, wiki_name, score = spt
            st = int(st)
            score = float(score)
            pref, idx = q_id.split('-')
            q_idx = int(idx)
            if pref == 'CompQTest':
                q_idx += 1300
            elif pref == 'WebQTest':
                q_idx += 3778
            lower_tok_list = [tok.token.lower() for tok in qa_list[q_idx]['tokens']]
            lower_q = qa_list[q_idx]['utterance'].lower()
            if pref.startswith('WebQ'):
                lower_q = lower_q.replace("'s", " 's")  # a trick to correctly parse the starting position
            span_before = lower_q[: st]
            before_word_len = find_span_word_len(tok_list=lower_tok_list, span=span_before)
            if before_word_len == -1:
                LogInfo.logs('Line %d: span_before error.', line_idx + 1)
                continue
            inside_word_len = find_span_word_len(tok_list=lower_tok_list[before_word_len:], span=mention)
            if inside_word_len == -1:
                LogInfo.logs('Line %d: mention error.', line_idx + 1)
            feat_dict = {'score': score}
            st_wd_lvl = before_word_len
            ed_wd_lvl = before_word_len + inside_word_len
            if mid[0] == '/':
                mid = mid.replace('/', '.')[1:]
            bw.write('%04d\t%d\t%d\t%s\t%s\t%s\t%s\n' % (
                q_idx, st_wd_lvl, ed_wd_lvl, mention, mid, wiki_name, json.dumps(feat_dict)
            ))
    LogInfo.end_track()


def find_span_word_len(tok_list, span):
    cur_span_pos = 0
    cur_tok_pos = 0
    span = span.replace(' ', '')
    while cur_span_pos != len(span):
        cur_tok = tok_list[cur_tok_pos]
        if span[cur_span_pos:].startswith(cur_tok):
            cur_span_pos += len(cur_tok)
            cur_tok_pos += 1
        else:
            LogInfo.logs('Cannnot find [%s] in tokens [%s].',
                         span.encode('utf-8'), ' '.join(tok_list).encode('utf-8'))
            return -1
    return cur_tok_pos


def main():
    qa_list = load_webq()
    process_single_file('/home/data/Webquestions/S-MART/webquestions.examples.all.e2e.top10.filter.tsv',
                        '/home/data/Webquestions/ACL18/webq.all.s-mart.q_links',
                        qa_list)


if __name__ == '__main__':
    main()
