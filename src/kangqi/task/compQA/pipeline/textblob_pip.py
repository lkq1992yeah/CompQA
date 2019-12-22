"""
Author: Kangqi Luo
Goal: Tokenize / NP Chunking throught textblob
"""

from textblob import TextBlob

from kangqi.util.LogUtil import LogInfo


wh_set = {'what', 'who', 'which', 'when', 'where'}


def process_single_question_simple(q):
    blob = TextBlob(q)
    for wd in blob.noun_phrases:
        LogInfo.logs('Chunk: %s', str(wd).encode('utf-8'))


def process_single_question(q):
    """
    Check EverNote 180127 for detail
    :param q: input question
    :return: token_list, a list of raw words
             chunk_pos_list, [(st, ed)] indicating the position [st, ed)
    """
    blob = TextBlob(q)
    shallow_parse = blob.parse().replace('\n', ' ').split(' ')
    LogInfo.logs(shallow_parse)
    chunk_tup_list = []
    for item in shallow_parse:
        spt = item.split('/')
        token = spt[0]
        chunk_tag = spt[2]
        chunk_tup_list.append([token, chunk_tag])

    while True:                     # deal with .
        tups_len = len(chunk_tup_list)
        dot_idx = -1
        for idx, tup in enumerate(chunk_tup_list):
            if tup[0] == u".":
                dot_idx = idx  # capture the index of '
                break
        if dot_idx == -1:
            break
        assert dot_idx > 0
        chunk_tup_list[dot_idx-1][0] += chunk_tup_list[dot_idx][0]
        del chunk_tup_list[dot_idx]

    while True:                     # deal with '
        tups_len = len(chunk_tup_list)
        quote_idx = -1
        for idx, tup in enumerate(chunk_tup_list):
            if tup[0] == u"'":
                quote_idx = idx     # capture the index of '
                break
        if quote_idx == -1:
            break
        assert quote_idx > 0
        if quote_idx < tups_len - 1 and chunk_tup_list[quote_idx+1][0] == u"s":
            chunk_tup_list[quote_idx][0] += chunk_tup_list[quote_idx+1][0]
            del chunk_tup_list[quote_idx+1]
        else:
            chunk_tup_list[quote_idx-1][0] += chunk_tup_list[quote_idx][0]
            chunk_tup_list[quote_idx-1][0] += chunk_tup_list[quote_idx+1][0]
            del chunk_tup_list[quote_idx+1]
            del chunk_tup_list[quote_idx]

    token_list = [tup[0] for tup in chunk_tup_list]
    chunk_pos_list = []
    st = -1
    for idx in range(len(chunk_tup_list)):
        tag = chunk_tup_list[idx][1]
        if tag in ('B-NP', 'I-NP'):
            if st != -1:
                continue
            else:
                st = idx
        else:
            if st != -1 and token_list[st].lower() not in wh_set:
                chunk_pos_list.append((st, idx))
            st = -1
    #     if tag  'I-NP':
    #         continue
    #     else:
    #         if st != -1 and token_list[st].lower() not in wh_set:
    #             chunk_pos_list.append((st, idx))
    #         st = -1
    #         if tag == 'B-NP':
    #             st = idx

    if st != -1 and token_list[st].lower() not in wh_set:
        chunk_pos_list.append((st, len(chunk_tup_list)))
    return token_list, chunk_pos_list





def qa_preprocess(qa_list):
    for qa_idx, qa in enumerate(qa_list[:100]):
        q = qa['utterance']
        LogInfo.begin_track('Entering Q-%d [%s]:', qa_idx, q.encode('utf-8'))
        token_list, chunk_pos_list = process_single_question(q=q)
        LogInfo.end_track()


if __name__ == '__main__':
    from ..dataset.u import load_webq
    _qa_list = load_webq()
    q_list = [qa['utterance'] for qa in _qa_list[:100]]

    # q_list = [
    #     'who plays ken barlow in coronation street?',
    #     'where did c.s. lewis go to college?',
    #     'which wife did king henry behead?',
    #     'what did st. matthew do?',
    #     "who was ishmael's mom?",
    #     "what is the second longest river in china?",
    #     "who did georgia o'keeffe inspired?",
    #     "where are you if you're in khartoum?",
    #     "where was toussaint l'ouverture born?",
    #     "what book did w.e.b. dubois wrote?",
    #     "what atom did j.j thomson discover?",
    #     "what organization did dr. carter g. woodson found?"
    # ]
    for q in q_list:
        LogInfo.begin_track('Entering [%s]:', q.encode('utf-8'))
        # token_list, chunk_pos_list = process_single_question(q=q)
        # LogInfo.logs('token_list: %s', ' | '.join(token_list).encode('utf-8'))
        # for st, ed in chunk_pos_list:
        #     LogInfo.logs('Chunk:%s', ' '.join(token_list[st: ed]).encode('utf-8'))
        process_single_question_simple(q=q)
        LogInfo.end_track()
