"""
Author: Kangqi Luo
Date: 180410
Goal: Generate input features for the experiments of EMNLP18
"""

import math
import numpy as np

from ..dataset_emnlp18 import SchemaDatasetEMNLP18
from ..kq_schema import CompqSchema

# from kangqi.util.LogUtil import LogInfo


class InputFeatureGenerator:

    def __init__(self, schema_dataset):
        self.schema_dataset = schema_dataset
        assert isinstance(schema_dataset, SchemaDatasetEMNLP18)

        self.path_max_size = schema_dataset.path_max_size
        self.el_max_size = schema_dataset.el_max_size
        self.qw_max_len = schema_dataset.qw_max_len
        self.sup_max_size = schema_dataset.sup_max_size
        self.mem_size = len(schema_dataset.path_idx_dict)

        self.word_idx_dict = schema_dataset.wd_emb_util.load_word_indices()
        self.path_idx_dict = schema_dataset.path_idx_dict
        self.entity_idx_dict = schema_dataset.entity_idx_dict

        self.support_dict = {}      # <'q_idx:start_end', set([support path list])>

    def input_gen(self, q_idx, cand):
        assert isinstance(cand, CompqSchema)
        if cand.input_np_dict is not None:
            return cand.input_np_dict     # already generated

        target_dict = {}
        self.gen_rm_input(q_idx=q_idx, cand=cand, target_dict=target_dict)
        self.gen_el_input(q_idx=q_idx, cand=cand, target_dict=target_dict)
        # TODO: structural part
        cand.input_np_dict = target_dict
        return target_dict

    def gen_rm_input(self, q_idx, cand, target_dict):
        # rm_qw_input, rm_qw_len, path_size, path_ids
        v_idx_list = self.schema_dataset.v_input_mat[q_idx].tolist()
        path_size = len(cand.path_list)
        path_ids = np.zeros((self.path_max_size,), dtype='int32')
        rm_qw_input = np.zeros((self.path_max_size, self.qw_max_len), dtype='int32')
        rm_qw_len = np.zeros((self.path_max_size,), dtype='int32')
        for idx, (raw_path, path_list) in enumerate(zip(cand.raw_paths, cand.path_list)):
            path_cate = raw_path[0]
            gl_data = raw_path[1]
            path_str = '%s|%s' % (path_cate, '\t'.join(path_list))
            path_id = self.path_idx_dict[path_str]
            path_ids[idx] = path_id
            qw_idx_list = self.get_local_qw(path_cate=path_cate,
                                            gl_data=gl_data,
                                            v_idx_list=v_idx_list)
            qw_seq_len = len(qw_idx_list)
            rm_qw_input[idx, :qw_seq_len] = qw_idx_list
            rm_qw_len[idx] = qw_seq_len
        target_dict['rm_qw_input'] = rm_qw_input
        target_dict['rm_qw_len'] = rm_qw_len
        target_dict['path_size'] = path_size
        target_dict['path_ids'] = path_ids

    def gen_el_input(self, q_idx, cand, target_dict):
        # el_qw_input, el_qw_len, el_size, el_ids,
        # el_indv_feats, el_comb_feats,
        # path_sup_ids, path_sup_size
        v_idx_list = self.schema_dataset.v_input_mat[q_idx].tolist()
        ent_gl_list = []
        for raw_path in cand.raw_paths:
            gl_data = raw_path[1]
            if gl_data.category == 'Entity':
                ent_gl_list.append(gl_data)
        el_size = len(ent_gl_list)
        el_ids = np.zeros((self.el_max_size,), dtype='int32')
        el_qw_input = np.zeros((self.el_max_size, self.qw_max_len), dtype='int32')
        el_qw_len = np.zeros((self.el_max_size,), dtype='int32')
        el_indv_feats = np.zeros((self.el_max_size, 3), dtype='float32')
        # TODO: Let's make it simpler: log-score, with simple normalization
        el_comb_feats = np.zeros((1,), dtype='float32')     # vector with 1-dim
        # TODO: connectivity information
        # path_sup_ids = np.zeros((self.el_max_size, self.sup_max_size), dtype='int32')
        # path_sup_size = np.zeros((self.el_max_size,), dtype='int32')
        el_sup_mask = np.zeros((self.el_max_size, self.mem_size), dtype='float32')
        local_sup_path_table = [[] for _ in range(self.el_max_size)]
        for idx, gl_data in enumerate(ent_gl_list):
            mid = gl_data.value
            ent_id = self.entity_idx_dict[mid]
            """ Naive normalization: roughly log_score \in [0, 10] """
            score = gl_data.link_feat['score']
            log_score = math.log(score)
            norm_log_score = log_score / 10
            """ Source information, 0.0: from both / 1.0: from SMART / 2.0: from xs """
            source = gl_data.link_feat.get('source', -1.0)
            if source < 0.:
                from_smart = from_wiki = 0.
            else:
                from_smart = 1. if source != 2.0 else 0.
                from_wiki = 1. if source != 1.0 else 0.
            el_ids[idx] = ent_id
            el_indv_feats[idx] = [norm_log_score, from_smart, from_wiki]
            qw_idx_list = self.get_local_qw(path_cate='Entity',
                                            gl_data=gl_data,
                                            v_idx_list=v_idx_list)
            qw_seq_len = len(qw_idx_list)
            el_qw_input[idx, :qw_seq_len] = qw_idx_list
            el_qw_len[idx] = qw_seq_len
            local_sup_path_list = self.get_support_paths(q_idx=q_idx,
                                                         el_start=gl_data.start,
                                                         el_end=gl_data.end)
            local_sup_path_table[idx] = local_sup_path_list
            for path_idx in local_sup_path_list:
                el_sup_mask[idx, path_idx] = 1.
            # local_sup_size = len(local_sup_path_list)
            # path_sup_ids[idx, :local_sup_size] = local_sup_path_list
            # path_sup_size[idx] = local_sup_size
        target_dict['el_qw_input'] = el_qw_input
        target_dict['el_qw_len'] = el_qw_len
        target_dict['el_size'] = el_size
        target_dict['el_ids'] = el_ids
        target_dict['el_indv_feats'] = el_indv_feats
        target_dict['el_comb_feats'] = el_comb_feats
        # target_dict['path_sup_ids'] = path_sup_ids
        # target_dict['path_sup_size'] = path_sup_size
        target_dict['el_sup_mask'] = el_sup_mask
        target_dict['local_sup_path_table'] = local_sup_path_table      # additional information

    def get_local_qw(self, path_cate, gl_data, v_idx_list):
        """
        Given a center of focus, return placeholder-ed word sequence.
        Apply different placeholder strategy here.
        """
        # TODO: Be careful with several entities
        start = gl_data.start  # TODO: Position Encoding
        end = gl_data.end
        if path_cate in ('Main', 'Entity', 'Time'):  # need placeholder
            if path_cate == 'Time':
                ph_idx = self.word_idx_dict['<Tm>']
            else:
                ph_idx = self.word_idx_dict['<E>']
            qw_idx_list = v_idx_list[:start] + [ph_idx] + v_idx_list[end:]
        else:  # no need placeholder
            qw_idx_list = v_idx_list
        return qw_idx_list

    def get_support_paths(self, q_idx, el_start, el_end):
        """
        Given the position (q_idx, start, end) of the entity mention,
        return all related paths
        """
        sup_key = '%d:%d_%d' % (q_idx, el_start, el_end)
        if sup_key not in self.support_dict:
            """ scan all related schemas of the current question """
            cand_list = self.schema_dataset.smart_q_cand_dict[q_idx]
            for cand in cand_list:      # enumerate each candidate
                for idx, (raw_path, path) in enumerate(zip(cand.raw_paths, cand.path_list)):
                    path_cate = raw_path[0]
                    gl_data = raw_path[1]
                    if gl_data.category != 'Entity':
                        continue    # doesn't care about types, times, ordinal constraints
                    local_sup_key = '%d:%d_%d' % (q_idx, gl_data.start, gl_data.end)
                    path_str = '%s|%s' % (path_cate, '\t'.join(path))
                    path_id = self.path_idx_dict[path_str]
                    self.support_dict.setdefault(local_sup_key, set([])).add(path_id)
            # LogInfo.logs('[%s] --> %d supports.', sup_key, len(self.support_dict[sup_key]))

        """ now, sup_key should be in the dictionary """
        sup_path_set = self.support_dict[sup_key]
        sup_path_list = list(sup_path_set)
        sup_path_list.sort()
        return sup_path_list
