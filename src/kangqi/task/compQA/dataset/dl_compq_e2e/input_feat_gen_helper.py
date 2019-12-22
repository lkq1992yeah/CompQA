"""
Goal: Given the whole dataset and one schema, generate different part of inputs
"""

import numpy as np

from ...util.dependency_util import DependencyUtil
from ...util.fb_helper import get_item_name, is_mediator_as_expect

from kangqi.util.discretizer import Discretizer

# from kangqi.util.LogUtil import LogInfo


class InputFeatureGenHelper:

    path_cate_dict = {
        'Main': 0,
        'Entity': 1,
        'Type': 2,
        'Time': 3,
        'Ordinal': 4
    }

    stop_word_set = {
        'be', 'am', 'is', 'are', 'was', 'were',
        'do', 'does', 'did',
        'the', 'a', 'an'
    }

    def __init__(self, schema_dataset, parser_ip='202.120.38.146', parser_port=9601):
        self.schema_dataset = schema_dataset
        self.path_max_size = schema_dataset.path_max_size
        self.qw_max_len = schema_dataset.qw_max_len
        self.pw_max_len = schema_dataset.pw_max_len
        self.pseq_max_len = schema_dataset.pseq_max_len
        self.el_feat_size = schema_dataset.el_feat_size
        self.extra_feat_size = schema_dataset.extra_feat_size
        self.dep_util = DependencyUtil(e_t_dict=schema_dataset.e_t_dict,
                                       parser_ip=parser_ip,
                                       parser_port=parser_port)
        self.word_idx_dict = schema_dataset.active_dicts['word']
        self.path_idx_dict = schema_dataset.active_dicts['path']
        self.mid_idx_dict = schema_dataset.active_dicts['mid']

    def generate_qw_feat__same(self, sc, is_in_train):
        q_idx = sc.q_idx
        qa = self.schema_dataset.qa_list[q_idx]

        """ 180516: try lemma instead of tokens """
        # lower_tok_list = [tok.lemma.lower() for tok in qa['tokens']]
        lower_tok_list = [tok.token.lower() for tok in qa['tokens']]

        for raw_path in sc.raw_paths:
            category, gl_data, _ = raw_path
            if gl_data.category == 'Entity':
                for idx in range(gl_data.start, gl_data.end-1):
                    lower_tok_list[idx] = ''
                lower_tok_list[gl_data.end-1] = '<E>'
            elif gl_data.category == 'Tm':
                for idx in range(gl_data.start, gl_data.end-1):
                    lower_tok_list[idx] = ''
                lower_tok_list[gl_data.end-1] = '<Tm>'

        """ 180516: try randomly remove stop words in the training sentence """
        if is_in_train:
            pass
            # for tok_idx in range(len(lower_tok_list)):
            #     if lower_tok_list[tok_idx] in self.stop_word_set:
            #         rdm = random.random()
            #         if rdm < 0.2:
            #             lower_tok_list[tok_idx] = ''

        ph_lower_tok_list = filter(lambda x: x != '', lower_tok_list)
        ph_qw_idx_seq = [self.word_idx_dict.get(token, 2) for token in ph_lower_tok_list]

        ph_len = min(self.qw_max_len, len(ph_lower_tok_list))
        qw_input = np.zeros((self.path_max_size, self.qw_max_len), dtype='int32')
        qw_len = np.zeros((self.path_max_size,), dtype='int32')
        for path_idx in range(self.path_max_size):
            qw_input[path_idx, :ph_len] = ph_qw_idx_seq[:ph_len]
            qw_len[path_idx] = ph_len
        return qw_input, qw_len

    def generate_whole_path_feat(self, sc):
        path_size = len(sc.raw_paths)
        path_cates = np.zeros((self.path_max_size, 5), dtype='float32')  # 5 = Main + E + T + Tm + Ord
        path_ids = np.zeros((self.path_max_size,), dtype='int32')
        pw_input = np.zeros((self.path_max_size, self.pw_max_len), dtype='int32')
        pw_len = np.zeros((self.path_max_size,), dtype='int32')
        pseq_ids = np.zeros((self.path_max_size, self.pseq_max_len), dtype='int32')
        pseq_len = np.zeros((self.path_max_size,), dtype='int32')
        sc.path_words_list = []

        for path_idx, (raw_path, mid_seq) in enumerate(zip(sc.raw_paths, sc.path_list)):
            path_cate = raw_path[0]
            path_cate_pos = self.path_cate_dict[path_cate]
            path_cates[path_idx, path_cate_pos] = 1.

            path_str = '%s|%s' % (path_cate, '\t'.join(mid_seq))
            path_ids[path_idx] = self.path_idx_dict.get(path_str, 2)

            local_words = []  # collect path words
            pseq_len[path_idx] = len(mid_seq)
            for mid_pos, mid in enumerate(mid_seq):
                pseq_ids[path_idx, mid_pos] = self.mid_idx_dict.get(mid, 2)
                p_name = get_item_name(mid)
                if p_name != '':
                    spt = p_name.split(' ')
                    local_words += spt
            sc.path_words_list.append(local_words)
            local_pw_idx_seq = [self.word_idx_dict.get(token, 2) for token in local_words]
            local_pw_len = min(self.pw_max_len, len(local_pw_idx_seq))
            pw_input[path_idx, :local_pw_len] = local_pw_idx_seq[:local_pw_len]
            pw_len[path_idx] = local_pw_len

        return path_size, path_cates, path_ids, pw_input, pw_len, pseq_ids, pseq_len

    def generate_dep_feat(self, sc, dep_or_cp):
        q_idx = sc.q_idx
        qa = self.schema_dataset.qa_list[q_idx]
        lower_tok_list = [tok.token.lower() for tok in qa['tokens']]
        linkings = [raw_path[1] for raw_path in sc.raw_paths]

        if dep_or_cp == 'dep':
            dep_path_tok_lists = self.dep_util.dep_path_seq(tok_list=lower_tok_list, linkings=linkings)
        else:
            dep_path_tok_lists = self.dep_util.context_pattern(tok_list=lower_tok_list, linkings=linkings)

        # LogInfo.begin_track('Q-%04d: [%s]', sc.q_idx, qa['utterance'].encode('utf-8'))
        # for gl_data, dep_path_tok_list in zip(linkings, dep_path_tok_lists):
        #     LogInfo.logs('Linking [%d, %d) "%s" ---> %s',
        #                  gl_data.start, gl_data.end, gl_data.mention.encode('utf-8'),
        #                  ' '.join(dep_path_tok_list).encode('utf-8'))
        # LogInfo.end_track()

        dep_input = np.zeros((self.path_max_size, self.qw_max_len), dtype='int32')
        dep_len = np.zeros((self.path_max_size,), dtype='int32')
        for path_idx, local_dep_seq in enumerate(dep_path_tok_lists):
            local_len = min(self.qw_max_len, len(local_dep_seq))
            local_dep_idx_seq = [self.word_idx_dict.get(token, 2) for token in local_dep_seq]
            dep_input[path_idx, :local_len] = local_dep_idx_seq[:local_len]
            dep_len[path_idx] = local_len
        return dep_input, dep_len

    def generate_el_feat(self, sc):
        el_indv_feats = np.zeros((self.path_max_size, self.el_feat_size), dtype='float32')
        el_comb_feats = np.zeros((1,), dtype='float32')
        # TODO: el_comb_feats need implementation
        el_mask = np.zeros((self.path_max_size,), dtype='float32')
        for path_idx, raw_path in enumerate(sc.raw_paths):
            category, gl_data, _ = raw_path
            if gl_data.category == 'Entity':
                """ Naive normalization: roughly log_score \in [0, 10] """
                score = gl_data.link_feat['score']
                log_score = np.log(score)
                norm_log_score = log_score / 10
                """ Source information, 0.0: from both / 1.0: from SMART / 2.0: from xs """
                source = gl_data.link_feat.get('source', -1.0)
                if source < 0.:
                    from_smart = from_wiki = 0.
                else:
                    from_smart = 1. if source != 2.0 else 0.
                    from_wiki = 1. if source != 1.0 else 0.
                el_indv_feats[path_idx] = [norm_log_score, from_smart, from_wiki]
                el_mask[path_idx] = 1.
        return el_indv_feats, el_comb_feats, el_mask

    def generate_extra_feat(self, sc):
        ans_size_disc = Discretizer([2, 3, 5, 10, 50], output_mode='list')  # 5+1
        extra_feat_list = []  # E/T/Tm/Ord size, T/T/Tm/Ord indicator, 2_hop, with_med, ans_size_discrete
        # 4 + 4 + 2 + 6 = 16
        constr_size_dict = {}
        main_pred_seq = []
        for category, _, pred_seq in sc.raw_paths:
            if category == 'Main':
                main_pred_seq = pred_seq
            constr_size_dict[category] = 1 + constr_size_dict.get(category, 0)
        for cate in ('Entity', 'Type', 'Time', 'Ordinal'):
            extra_feat_list.append(constr_size_dict.get(cate, 0))  # how many constraint paths
        for cate in ('Entity', 'Type', 'Time', 'Ordinal'):
            extra_feat_list.append(min(1, constr_size_dict.get(cate, 0)))  # whether have such constraints
        is_two_hop = 1 if sc.hops == 2 else 0
        with_med = 1 if is_mediator_as_expect(main_pred_seq[0]) else 0
        extra_feat_list += [is_two_hop, with_med]
        extra_feat_list += ans_size_disc.convert(sc.ans_size)
        assert len(extra_feat_list) == self.extra_feat_size

        return np.array(extra_feat_list, dtype='float32')
