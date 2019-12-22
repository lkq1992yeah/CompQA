# -*- coding: utf-8 -*-

import os
import json
import cPickle
import numpy as np

from kangqi.util.LogUtil import LogInfo


# This is a dataset, not a DataLoader
# The goal is to just extract information from Xianyang's output format
# We'd better load all the train/valid/test information here, and then split them through DataLoader


def get_webq_range_by_mode(mode):
    q_idx_list = []
    if mode == 'train':
        q_idx_list = range(3023)
    elif mode == 'valid':
        q_idx_list = range(3023, 3778)
    elif mode == 'test':
        q_idx_list = range(3778, 5810)
    return q_idx_list


class Schema(object):

    def __init__(self, path_list=None, path_list_str=None, usable_flag=False,
                 p=0., r=0., f1=0., verbose=0):
        self.path_list = path_list                  # all converted to indices
        self.path_list_str = path_list_str          # readable version
        self.usable_flag = usable_flag

        self.embedding_index_tup = None

        self.p = p
        self.r = r
        self.f1 = f1

        if self.usable_flag and verbose >= 3:
            LogInfo.logs('schema: [%s]', self.path_list_str)

    def is_schema_ok(self, sc_max_len, path_max_len):
        if not self.usable_flag:  # remove not usable
            return False
        if len(self.path_list) > sc_max_len:    # sc_max_len exceeds the limit
            return False
        len_exceed = False
        for path in self.path_list:
            if len(path) - 1 > path_max_len:    # path length exceeds the limit
                len_exceed = True
                break
        if len_exceed:
            return False
        return True

    def build_embedding_index(self, sc_max_len, path_max_len, item_max_len):
        if self.embedding_index_tup is not None:
            return self.embedding_index_tup

        focus_kb = np.zeros((sc_max_len,), dtype='int32')
        focus_item = np.zeros((sc_max_len, item_max_len), dtype='int32')
        focus_item_len = np.zeros((sc_max_len,), dtype='int32')

        path_len = np.zeros((sc_max_len,), dtype='int32')
        path_kb = np.zeros((sc_max_len, path_max_len), dtype='int32')
        path_item = np.zeros((sc_max_len, path_max_len, item_max_len), dtype='int32')
        path_item_len = np.zeros((sc_max_len, path_max_len), dtype='int32')

        assert len(self.path_list) <= sc_max_len
        sc_len = self.path_list

        for path_idx, path in enumerate(self.path_list):    # enumerate each path
            focus = path[0]                                 # saved as int
            item_info = []  # TODO
            item_len = min(len(item_info), item_max_len)
            focus_kb[path_idx] = focus
            focus_item_len[path_idx] = item_len
            focus_item[path_idx, :item_len] = item_info

            pred_list = path[1:]
            assert 0 < len(pred_list) <= path_max_len
            path_len[path_idx] = len(pred_list)
            for p_idx, pred in enumerate(pred_list):        # enumerate each predicate
                item_info = []  # TODO
                item_len = min(len(item_info), item_max_len)
                path_kb[path_idx, p_idx] = pred
                path_item_len[path_idx, p_idx] = item_len
                path_item[path_idx, p_idx, :item_len] = item_info

        self.embedding_index_tup = (
            sc_len, focus_kb, focus_item, focus_item_len,
            path_len, path_kb, path_item, path_item_len
        )
        return self.embedding_index_tup

    def display_embedding_info(self):
        pass
        # if self.embedding_index_tup is None:
        #     LogInfo.logs('Embedding info is missing.')
        # else:
        #     for item, item_name in zip(self.embedding_index_tup, ['focus_input', 'path_input', 'path_len_input']):
        #         LogInfo.logs('%s %s: %s', item_name, item.shape, item)


class XianyangQScDataset(object):

    def __init__(self, data_dir, file_list_name, allow_sk_only,
                 webq_list, wd_emb_util, kb_emb_util, fb_helper, verbose=0):
        self.webq_list = webq_list
        self.wd_emb_util = wd_emb_util
        self.kb_emb_util = kb_emb_util
        self.fb_helper = fb_helper

        self.q_cand_dict = {}       # save the candidates of each question
        self.q_words_dict = {}      # save the word index of each question

        self.q_max_len = 0
        self.sk_num = 0
        self.sk_max_len = 0
        self.max_cands = 0

        dump_fp = '%s/%s.cPickle' % (data_dir, file_list_name)
        if os.path.isfile(dump_fp):
            LogInfo.begin_track('Loading XY-schemas from dump [%s] ...', dump_fp)
            with open(dump_fp, 'rb') as br:
                self.q_cand_dict = cPickle.load(br)
                LogInfo.logs('%d q_cand_dict loaded.', len(self.q_cand_dict))
                # self.q_emb_dict = cPickle.load(br)
                # LogInfo.logs('%d q_emb_dict loaded.', len(self.q_emb_dict))
                self.q_words_dict = cPickle.load(br)
                LogInfo.logs('%d q_words_dict loaded.', len(self.q_words_dict))
            LogInfo.end_track()
        else:
            LogInfo.begin_track('Loading XY-schemas from [%s] ... ', data_dir)

            total_cand_size = total_usable_cand_size = 0
            list_fp = '%s/%s' % (data_dir, file_list_name)
            with open(list_fp, 'r') as br:
                schema_fp_list = map(lambda line: '%s/%s' % (data_dir, line.strip()),
                                     br.readlines())
            LogInfo.logs('%d schema files found in [%s].', len(schema_fp_list), file_list_name)

            for scan_idx, schema_fp in enumerate(schema_fp_list):
                if scan_idx % 100 == 0:
                    LogInfo.logs('%d / %d scanned.', scan_idx, len(schema_fp_list))
                q_idx = int(schema_fp.split('/')[-1].split('_')[0])     # data/compQA/xy_q_schema/train/0-99/1_2 --> 1
                q = webq_list[q_idx]
                q_words = np.array(self.wd_emb_util.q2idx(q), dtype='int32')
                # q_emb_matrix = self.wd_emb_util.q2emb(q)
                self.q_max_len = max(self.q_max_len, len(q_words))

                candidate_list = []         # store candidates (dedup, and no matter usable or not)
                if verbose >= 1:
                    LogInfo.begin_track('scan_idx = %d, q_idx = %d:', scan_idx, q_idx)
                    LogInfo.logs('Q: %s', q.encode('utf-8'))
                    # LogInfo.logs('q_emb_matrix: %s', q_emb_matrix.shape)
                    LogInfo.logs('Q words: %s', q_words)

                with open(schema_fp, 'r') as br:
                    schema_data = json.load(br)
                    path_list_str_set = set([])            # store all schemas/skeletons in this data, just for dedup
                    for data_item in schema_data:
                        sk_p, sk_r, sk_f1 = [float(data_item[x]) for x in ['P', 'R', 'F1']]
                        sk_line = data_item['basicgraph']
                        sk_path_list, sk_path_list_str, sk_flag = self.build_sc_from_line(sk_line)
                        if sk_path_list_str not in path_list_str_set:
                            path_list_str_set.add(sk_path_list_str)
                            candidate_list.append(Schema(
                                sk_path_list, sk_path_list_str, sk_flag, sk_p, sk_r, sk_f1, verbose))
                        if allow_sk_only:
                            continue            # ignore detail schema information

                        local_schema_list = data_item['schemas']
                        for local_schema in local_schema_list:
                            sc_line = local_schema['schema']
                            sc_path_list, sc_path_list_str, sc_flag = self.build_sc_from_line(sc_line)
                            if 'P' in local_schema:
                                sc_p, sc_r, sc_f1 = [float(local_schema[x]) for x in ['P', 'R', 'F1']]
                            else:
                                sc_p, sc_r, sc_f1 = sk_p, sk_r, sk_f1       # derived from the skeleton
                            if sc_path_list_str not in path_list_str_set:
                                path_list_str_set.add(sc_path_list_str)
                                candidate_list.append(Schema(
                                    sc_path_list, sc_path_list_str, sc_flag, sc_p, sc_r, sc_f1, verbose))

                candidate_list.sort(key=lambda _sc: _sc.f1, reverse=True)   # sort by F1 DESC
                cand_size = len(candidate_list)
                usable_cand_size = len(filter(lambda _sc: _sc.usable_flag, candidate_list))
                total_cand_size += cand_size
                total_usable_cand_size += usable_cand_size
                for sc in candidate_list:
                    self.sk_num = max(self.sk_num, len(sc.path_list))
                    for path in sc.path_list:
                        self.sk_max_len = max(self.sk_max_len, len(path) - 1)

                if verbose >= 1:
                    for rank_idx, sc in enumerate(candidate_list[:5]):
                        LogInfo.logs('#%d: F1 = %.6f, schema = %s', rank_idx+1, sc.f1, sc.path_list_str)
                    LogInfo.logs('Usable candidates = %d / %d (%.3f%%)',
                                 usable_cand_size, cand_size,
                                 100. * usable_cand_size / cand_size if cand_size > 0 else 0.)
                    LogInfo.end_track()
                self.q_cand_dict[q_idx] = candidate_list
                # self.q_emb_dict[q_idx] = q_emb_matrix
                self.q_words_dict[q_idx] = q_words
                self.max_cands = max(self.max_cands, len(candidate_list))

                # if scan_idx >= 1000:
                #     break

            q_size = len(self.q_cand_dict)
            LogInfo.begin_track('[STAT] %d questions scanned:', q_size)
            LogInfo.logs('Total cand = %d (%.3f avg)',
                         total_cand_size,
                         1. * total_cand_size / q_size)
            LogInfo.logs('Total usable cand = %d (%.3f avg)',
                         total_usable_cand_size,
                         1. * total_usable_cand_size / q_size)
            LogInfo.logs('Keep Ratio = %.3f%%', 100. * total_usable_cand_size / total_cand_size)
            LogInfo.logs('q_max_len = %d', self.q_max_len)
            LogInfo.logs('sk_num = %d', self.sk_num)
            LogInfo.logs('sk_max_len = %d', self.sk_max_len)
            LogInfo.logs('max_cands = %d', self.max_cands)
            cand_dist_vec = np.array([len(cands) for cands in self.q_cand_dict.values()])
            for pos in range(0, 100, 5):
                LogInfo.logs('Percentile = %d%%: %.6f', pos, np.percentile(cand_dist_vec, pos))
            LogInfo.end_track()

            LogInfo.begin_track('Saving XY-schemas into [%s] ...', dump_fp)
            with open(dump_fp, 'wb') as bw:
                LogInfo.logs('Saving q_cand_dict ... ')
                cPickle.dump(self.q_cand_dict, bw)
                # LogInfo.logs('Saving q_emb_dict ... ')
                # cPickle.dump(self.q_emb_dict, bw)
                LogInfo.logs('Saving q_words_dict ... ')
                cPickle.dump(self.q_words_dict, bw)
            LogInfo.end_track()

            LogInfo.end_track()

    # Currently won't use XS's schema definition,
    # but we try to split the schema into multiple skeletons
    def build_sc_from_line(self, schema_line):

        # Step 1: Extract and prepare all subj --> (pred, obj) information
        buf = schema_line.split('\t')
        path_tup_list = []          # [(path, path_str)]
        edge_dict = {}
        for line in buf:
            subj, pred, obj = line.split('--')  # subj, pred, obj
            # Now checking whether to change the order
            reverse = False
            if '.' in obj:          # the object is not x or y*
                reverse = True
            elif subj == 'x':
                reverse = True
            elif subj.startswith('y') and obj.startswith('y') and obj > subj:       # y1 v.s. y0
                reverse = True
            if reverse:
                subj, obj = obj, subj
                pred = self.fb_helper.inverse_predicate(pred)
            edge_dict[subj] = (pred, obj)

        # Step 2: Search for start entity, and traverse the schema.
        flag = True                     # whether some component (entity or predicate) is missing
        for start_subj in edge_dict:
            if '.' not in start_subj:   # either final obj, or nodes on the skeleton
                continue
            start_subj_idx = self.kb_emb_util.get_entity_idx(start_subj)
            if start_subj_idx == -1:
                flag = False
            # now traverse the tree
            path = [start_subj_idx]             # store the [e, p1, p2, ...]
            name_list = [start_subj]            # store the [e_name, p1_name, ...]
            cur_subj = start_subj
            while cur_subj != 'x':
                pred, obj = edge_dict[cur_subj]
                pred_idx = self.kb_emb_util.get_predicate_idx(pred)
                if pred_idx == -1:
                    flag = False
                path.append(pred_idx)
                name_list.append(pred)
                cur_subj = obj  # go to next node in the path
            path_str = '-->'.join(name_list)
            path_tup_list.append((path, path_str))

        path_tup_list.sort(key=lambda x: x[1])            # sort by alphabet order of the path name
        path_list = [tup[0] for tup in path_tup_list]
        path_list_str = '\t'.join([tup[1] for tup in path_tup_list])
        return path_list, path_list_str, flag
