"""
Author: Kangqi Luo
Date: 180118
Goal: Generate Embedding & Relation Matching data
"""

import numpy as np

from ..kq_schema import CompqSchema
from ...util.fb_helper import load_type_name, load_pred_name, get_domain, get_range, get_item_name
from kangqi.util.LogUtil import LogInfo
# from kangqi.util.time_track import TimeTracker as Tt


def load_necessary_entity_predicate_dict(all_cands_tup_list):
    """
    Scan FB E/T/P names, just keeping <mid, index> pairs which occur in the candidate pool.
    :return: <mid, index> dictionary for both entities (including types) and predicates
    """
    type_set = set([])
    pred_set = set([])
    for data_idx, q_idx, cand in all_cands_tup_list:
        update_mid_set(schema=cand, type_set=type_set, pred_set=pred_set)
    LogInfo.logs('%d T + %d P collected.', len(type_set), len(pred_set))
    load_pred_name()
    load_type_name()
    mid_dict = {'': 0}      # give index 0 to represent empty entity(for padding)
    for item_set in (type_set, pred_set):
        for item in item_set:
            mid_dict[item] = len(mid_dict)
    return mid_dict


def update_mid_set(schema, type_set, pred_set):
    """
    (Copied from CompqSchema)
    Got the concerned type / predicates (NO ENTITIES) into the mid dict
    :param schema: CompqSchema
    :param type_set: Keeping all types observed in schemas
    :param pred_set: Keeping all predicates observed in schemas
    """
    if schema.raw_paths is None:
        return
    if schema.path_list is None:
        schema.construct_path_list()      # build the path list first
    for raw_path, mid_seq in zip(schema.raw_paths, schema.path_list):
        path_cate, _, _ = raw_path
        pred_seq = mid_seq
        if path_cate == 'Main':         # Retrieve the acting type of the focus
            out_pred = pred_seq[0]          # predicate direction: Focus --> Answer
            act_type = get_domain(pred=out_pred)
            if act_type == '':
                act_type = '#unknown#'      # note: unknown is not the same as blank ''
            type_set.add(act_type)
        elif path_cate == 'Entity':     # Retrieve the acting type of the constraint entity
            in_pred = pred_seq[-1]          # predicate direction: Answer --> Constraint
            act_type = get_range(pred=in_pred)
            if act_type == '':
                act_type = '#unknown#'      # note: unknown is not the same as blank ''
            type_set.add(act_type)
        elif path_cate == 'Type':       # Retrieve the constraint type
            pred_seq = mid_seq[: -1]        # the last mid in the sequence is a type
            type_set.add(mid_seq[-1])

        for pred in pred_seq:
            pred_set.add(pred)


def add_placeholder_to_q(q_word_seq, replace_links):
    """
    Given the word sequence and the entity linking detail,
    replace the mention words by <e>
    :param q_word_seq: The word sequence of the question (tokenized)
    :param replace_links: A list of links to be replaced.
                          Each: namedtuple (category, detail, comparison, display)
    :return: The word sequence after replacement
    """
    target_word_seq = list(q_word_seq)
    for link_data in replace_links:
        link_cate = link_data.category
        ph = {'Entity': '<e>', 'Type': '<t>', 'Time': '<tm>'}[link_cate]    # get the corresponding placeholder
        st = link_data.start
        ed = link_data.end              # [st, ed) interval
        if ed == st:        # fail to find the linking position
            continue
        if st == -1 or ed == -1:        # error mention position
            continue
        target_word_seq[st] = ph
        for idx in range(st+1, ed):
            target_word_seq[idx] = ''   # clear remaining words
    ret_word_list = filter(lambda x: x != '', target_word_seq)  # remove all empty words
    return ret_word_list


def get_word_idx_list_from_string(word_seq, w_dict):
    """
    Given the word sequence (may contains placeholder <e>),
    we gather into the list and update the dictionary
    :param word_seq: the word sequence
    :param w_dict: the word dictionary to be updated
    :return: a list of word indices
    """
    wd_idx_list = []
    for wd in word_seq:
        # if wd not in wd_emb_util.wd_idx_dict:       # ignore rare words
        #     continue
        wd_idx = w_dict.setdefault(wd, len(w_dict))
        wd_idx_list.append(wd_idx)
    return wd_idx_list


def create_wep_init_emb(wd_emb_util, word_dict, mid_dict, dim_emb):
    """
    Provided with <word/mid, index> dictionary in use, we check the global embedding matrix,
    and generate the small w/e/p_init_emb, which is used in the learning model
    :return: word/mid_init_emb
             three numpy matrix with (actual_size, emb_dim) shape
    """
    word_idx_dict = wd_emb_util.load_word_indices()
    word_emb_matrix = wd_emb_util.load_word_embeddings()
    w_init_emb = init_emb(
        name='word', actual_dict=word_dict, dim_emb=dim_emb,
        full_dict=word_idx_dict, full_mat=word_emb_matrix
    )
    m_init_emb = init_emb(name='mid', actual_dict=mid_dict, dim_emb=dim_emb)
    return w_init_emb, m_init_emb


def init_emb(name, actual_dict, dim_emb, full_dict=None, full_mat=None):
    """
    Given the actual entries and the full embedding info, construct the actual initial embedding matrix
    :param name: word/entity/predicate
    :param actual_dict: the dict storing actual entries <item, idx>
    :param full_dict: the full dict of entries <item, idx>
    :param full_mat: the full embedding matrix in numpy format
    :param dim_emb: embedding dimension
    :return: the actual initial embedding matrix in numpy format
    """
    if full_mat is not None:
        assert dim_emb == full_mat.shape[1]
    actual_size = len(actual_dict)
    ret_emb_matrix = np.random.uniform(
        low=-0.1, high=0.1, size=(actual_size, dim_emb)).astype('float32')
    # [-0.1, 0.1] as random initialize.
    if full_dict is None or full_mat is None:
        LogInfo.logs('%s: build %s actual init embedding matrix by random.', name, ret_emb_matrix.shape)
        return ret_emb_matrix                   # all random initialize

    for item, target_row_idx in actual_dict.items():
        if item in full_dict:
            # full_mat is None: we don't use TransE as initial embedding
            original_row_idx = full_dict[item]
            ret_emb_matrix[target_row_idx] = full_mat[original_row_idx]
    LogInfo.logs('%s: build %s actual init embedding matrix from full matrix with shape %s.',
                 name, ret_emb_matrix.shape, full_mat.shape if full_mat is not None else '[None]')
    return ret_emb_matrix


""" Above: Utility; Below: Main Function """


def build_relation_matching_data(all_cands_tup_list, qa_list, wd_emb_util,
                                 q_max_len, sc_max_len, path_max_len, pword_max_len):
    """1
    Goal: given all the candidate schemas, build:
          1. numpy data for the use of relation matching
          2. initial embedding of words / mids observed in all candidates
    ** Used in WebQ and SimpQ **
    ** Used after 01/18/2018 **
    :return: A list of following information
    1. np_data, all the numpy data for relation detection
        qwords, qwords_len, sc_len, preds, preds_len, pwords, pwords_len
    2. word/mid_dict, the actual <word/mid, index> dictionary (ignoring unused items)
    3. word/mid_init_emb, the corresponding initial embedding parameter
    """
    cand_size = len(all_cands_tup_list)
    LogInfo.begin_track('Build Relation Matching Data for %d candidates:', cand_size)

    """ Step 1: Collect Necessary Mids in Candidates """

    mid_dict = load_necessary_entity_predicate_dict(all_cands_tup_list=all_cands_tup_list)
    # with padding item --> index = 0
    word_dict = {'': 0}    # <word, idx>: used for constructing the actual initial embedding matrix
    # with padding word --> index = 0

    """ Step 2: Allocate Memory of Relation Matching """

    np_data = []
    for shape in [(cand_size, q_max_len),  # qwords
                  (cand_size,),  # qwords_len
                  (cand_size,),  # sc_len
                  (cand_size, sc_max_len, path_max_len),  # preds
                  (cand_size, sc_max_len),  # preds_len
                  (cand_size, sc_max_len, pword_max_len),  # pwords
                  (cand_size, sc_max_len)]:  # pwords_len
        LogInfo.logs('Adding tensor with shape %s ... ', shape)
        np_data.append(np.zeros(shape=shape, dtype='int32'))
    [qwords, qwords_len, sc_len, preds, preds_len, pwords, pwords_len] = np_data

    """ Step 3: Scan each candidate, get ready to fill in with their numpy data """

    for data_idx, q_idx, sc in all_cands_tup_list:
        if data_idx % 50000 == 0:
            LogInfo.logs('%d / %d schemas scanned.', data_idx, cand_size)
        qword_seq_lower = [tok.token.lower() for tok in qa_list[q_idx]['tokens']]

        assert isinstance(sc, CompqSchema)  # not Schema, but CompqSchema
        sc.use_idx = data_idx  # set the global index of the schema
        sc.path_words_list = []

        # First: adding placeholders to the question, save into qwords/qwords_len
        qword_seq_with_ph = add_placeholder_to_q(q_word_seq=qword_seq_lower, replace_links=sc.replace_linkings)
        qword_indices = get_word_idx_list_from_string(word_seq=qword_seq_with_ph, w_dict=word_dict)
        use_qword_len = min(len(qword_indices), q_max_len)
        qwords_len[data_idx] = use_qword_len
        qwords[data_idx, :use_qword_len] = qword_indices[:use_qword_len]

        # Then: deal with each mid sequence
        sc_len[data_idx] = len(sc.path_list)
        for path_idx, mid_seq in enumerate(sc.path_list):  # enumerate each mid sequence
            local_words = []
            preds_len[data_idx, path_idx] = len(mid_seq)
            for pos, mid in enumerate(mid_seq):
                mid_idx = mid_dict[mid]  # note: all E/T/P information are stored together
                mid_name = get_item_name(mid)
                local_words.append(mid_name)
                preds[data_idx, path_idx, pos] = mid_idx
            sc.path_words_list.append(local_words)
            pword_indices = []  # saving all the indices of words in one mid sequence
            for mid_name in local_words:
                mid_word_seq = mid_name.split(' ')      # simply split by blanks
                pword_indices += get_word_idx_list_from_string(word_seq=mid_word_seq, w_dict=word_dict)
            use_pword_len = min(len(pword_indices), pword_max_len)
            pwords_len[data_idx, path_idx] = use_pword_len
            pwords[data_idx, path_idx, :use_pword_len] = pword_indices[:use_pword_len]

    LogInfo.end_track('Relation Matching Build Complete.')

    """ Step 4: Create initial embedding matrix for word/mid """
    LogInfo.begin_track('now creating actual initial embedding matrix ...')
    word_init_emb, mid_init_emb = create_wep_init_emb(wd_emb_util=wd_emb_util, word_dict=word_dict,
                                                      mid_dict=mid_dict, dim_emb=wd_emb_util.dim_emb)
    LogInfo.end_track()
    return np_data, word_dict, mid_dict, word_init_emb, mid_init_emb
