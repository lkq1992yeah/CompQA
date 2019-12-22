"""
Goal: extract gold mentions of SimpQ from old 171122 dataset.
"""

import cPickle
import codecs

from ...dataset.u import load_simpq

from kangqi.util.LogUtil import LogInfo


def extract_mention(link_fp, qa, q_idx, df_dict):
    with open(link_fp, 'r') as br:
        link_list = cPickle.load(br)
    link_data = link_list[0]
    detail = link_data[0]
    mid = detail.entity.id
    name = detail.name
    qa_tokens = qa['tokens']
    detail_tokens = detail.tokens

    if len(detail_tokens) > 0:
        start = detail.tokens[0].index
        end = detail.tokens[-1].index + 1
        mention_span = [tok.token for tok in qa_tokens][start: end]
        mention_str = ' '.join(mention_span)
    else:           # don't know the token, pick the most infrequent token as instead
        start = -1
        min_freq = 10000000
        lower_tok_list = [tok.token.lower() for tok in qa_tokens]
        for idx, lower_tok in enumerate(lower_tok_list):
            if df_dict[lower_tok] < min_freq:
                min_freq = df_dict[lower_tok]
                start = idx
        end = start + 1
        # LogInfo.logs('[Q-%06d]')
        mention_str = qa_tokens[start].token
    # LogInfo.logs('[Q-%06d] [%d, %d): %s --> %s (%s)', q_idx, start, end, mention_str, mid, name.encode('utf-8'))
    save_str = '%06d\t%d\t%d\t%s\t%s\t%s\t{"score": 1.0}' % (q_idx, start, end, mention_str, mid, name)

    q_len = len(qa['tokens'])
    assert end <= q_len
    return save_str


def main():
    qa_list = load_simpq()
    df_dict = {}
    for qa in qa_list:
        token_set = set([tok.token.lower() for tok in qa['tokens']])
        for lower_token in token_set:
            df_dict[lower_token] = 1 + df_dict.get(lower_token, 0)
    LogInfo.logs('df_dict with %d lower entries collected.', len(df_dict))

    data_dir = 'runnings/candgen_SimpQ/171122/data'
    save_fp = '/home/data/SimpleQuestions/SimpleQuestions_v2/ACL18/simpQ.all.gold_171122.q_links'
    with codecs.open(save_fp, 'w', 'utf-8') as bw:
        for q_idx in range(len(qa_list)):
            if q_idx % 100 == 0:
                LogInfo.logs('Current: %d / %d', q_idx, len(qa_list))
            div = q_idx / 100
            sub_dir = '%d-%d' % (div*100, div*100+99)
            link_fp = '%s/%s/%d_links' % (data_dir, sub_dir, q_idx)
            save_str = extract_mention(link_fp=link_fp, qa=qa_list[q_idx], q_idx=q_idx, df_dict=df_dict)
            bw.write(save_str + '\n')


if __name__ == '__main__':
    LogInfo.begin_track('patch_180514 start ...')
    main()
    LogInfo.end_track()
