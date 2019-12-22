"""
Author: Kangqi Luo
Date: 180118
Goal: Create the dataset for the OVERALL MODEL
"""

import os
import json
import codecs
import cPickle
import numpy as np

from ..candgen_acl18.global_linker import LinkData

from .u import load_simpq, load_webq, load_compq
from .ds_helper.dataset_schema_reader import load_schema_by_kqnew_protocol
from .ds_helper.dataset_relation_matching_helper import build_relation_matching_data
from .ds_helper.dataset_entity_linking_helper import build_entity_linking_data
from .ds_helper.dataset_structural_helper import build_structural_data

from kangqi.util.LogUtil import LogInfo


class QScDataset:

    def __init__(self, data_name, data_dir, file_list_name,
                 q_max_len, sc_max_len, path_max_len, item_max_len,
                 schema_level, wd_emb_util, verbose=0):
        self.data_name = data_name
        self.data_dir = data_dir
        self.file_list_name = file_list_name

        if self.data_name == 'WebQ':
            self.qa_list = load_webq()
        elif self.data_name == 'CompQ':
            self.qa_list = load_compq()
        else:
            self.qa_list = load_simpq()

        """ Following: Need Save to File """
        self.q_cand_dict = None
        self.np_data_list = []      # several numpy arrays for all schemas

        self.word_dict = None
        self.mid_dict = None        # the actual <item, index> dictionary

        self.word_init_emb = None
        self.mid_init_emb = None    # the actual initial embedding data

        self.word_size = self.mid_size = 0
        self.e_feat_len = self.extra_len = self.array_num = 0
        """ ============================ """

        self.wd_emb_util = wd_emb_util
        self.q_link_dict = None

        self.q_max_len = q_max_len
        self.sc_max_len = sc_max_len
        self.path_max_len = path_max_len
        self.item_max_len = item_max_len
        self.e_max_size = self.sc_max_len   # at most one entity per path
        self.schema_level = schema_level
        self.verbose = verbose

        self.save_dir = '%s/%s_useful-%d-%d-%d-%d-%s' % (
            data_dir, file_list_name, q_max_len, sc_max_len, path_max_len, item_max_len, schema_level)
        self.size_fp = self.save_dir + '/element_size'
        # saving the size of actual vocabularies and number of arrays in the np_data
        self.dict_fp = self.save_dir + '/word_mid_dict.cPickle'
        # <word, index> and <mid, index>
        self.init_mat_fp = self.save_dir + '/init_emb.npz'
        # saving the initial embedding of words and mids
        self.dump_fp = self.save_dir + '/cand.cPickle'
        # saving each candidate schema
        self.np_data_fp = self.save_dir + '/np_data.npz'
        # saving np data for relation matching / entity linking / structural information

    def get_schema_tensor_inputs(self, schema):
        use_idx = schema.use_idx
        tensor_inputs = [np_data[use_idx] for np_data in self.np_data_list]
        return tensor_inputs

    def prepare_all_data(self):
        """ Generate ALL the data we need for further training """
        LogInfo.begin_track('Loading schema dataset from [%s] ...', self.data_dir)

        """ Step 1: Load Auxiliary Information """

        verbose = self.verbose
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.wd_emb_util.load_word_indices()
        list_fp = '%s/%s' % (self.data_dir, self.file_list_name)
        with open(list_fp, 'r') as br:
            schema_fp_list = map(lambda line: '%s/%s' % (self.data_dir, line.strip()), br.readlines())
        LogInfo.logs('%d schema files found in [%s].', len(schema_fp_list), self.file_list_name)

        """ Step 2: Traverse & Make Statistics """

        self.q_cand_dict = {}
        self.q_link_dict = {}
        sc_len_dist = []    # distribution of number of paths in a schema
        path_len_dist = []  # distribution of length of each path
        q_len_dist = []     # distribution of question word length
        ans_size_dist = []  # distribution of answer size
        total_cand_size = useful_cand_size = 0
        for scan_idx, schema_fp in enumerate(schema_fp_list):
            link_fp = schema_fp[0: schema_fp.rfind('_')] + '_links'

            if scan_idx % 100 == 0:
                LogInfo.logs('%d / %d scanned.', scan_idx, len(schema_fp_list))
            q_idx = int(schema_fp.split('/')[-1].split('_')[0])
            q = self.qa_list[q_idx]['utterance']
            q_len_dist.append(len(q.split(' ')))        # just a rough count of question length

            # with open(link_fp, 'rb') as br:
            #     self.q_link_dict[q_idx] = cPickle.load(br)

            gather_linkings = []
            with codecs.open(link_fp, 'r', 'utf-8') as br:
                for gl_line in br.readlines():
                    tup_list = json.loads(gl_line.strip())
                    ld_dict = {k: v for k, v in tup_list}
                    gather_linkings.append(LinkData(**ld_dict))
            self.q_link_dict[q_idx] = gather_linkings

            if verbose >= 1:
                LogInfo.begin_track('scan_idx = %d, q_idx = %d:', scan_idx, q_idx)
                LogInfo.logs('Q: %s', q.encode('utf-8'))

            candidate_list, total_lines = load_schema_by_kqnew_protocol(
                q_idx=q_idx, schema_fp=schema_fp, gather_linkings=self.q_link_dict[q_idx],
                sc_max_len=self.sc_max_len, schema_level=self.schema_level,
                sc_len_dist=sc_len_dist, path_len_dist=path_len_dist, ans_size_dist=ans_size_dist
            )
            total_cand_size += total_lines
            useful_cand_size += len(candidate_list)

            if verbose >= 1:
                srt_cand_list = sorted(candidate_list, key=lambda _sc: _sc.f1, reverse=True)
                for rank_idx, sc in enumerate(srt_cand_list[:5]):
                    LogInfo.logs('#%d: F1 = %.6f, schema = [%s]', rank_idx + 1, sc.f1, sc.disp())
                LogInfo.logs('Candidate size = %d', len(candidate_list))
                LogInfo.end_track()
            self.q_cand_dict[q_idx] = candidate_list

        """ Step 3: Show Statistics """

        q_size = len(self.q_cand_dict)
        LogInfo.begin_track('[STAT] %d questions scanned:', q_size)
        LogInfo.logs('Total schemas = %d', total_cand_size)
        LogInfo.logs('Useful schemas = %d (%.3f%%)', useful_cand_size, 100. * useful_cand_size / total_cand_size)
        LogInfo.logs('Avg candidates = %.3f', 1. * useful_cand_size / q_size)
        cand_size_dist = [len(v) for v in self.q_cand_dict.values()]
        for show_name, show_dist in zip(['q_len', 'cand_size', 'sc_len', 'path_len', 'ans_size'],
                                        [q_len_dist, cand_size_dist, sc_len_dist, path_len_dist, ans_size_dist]):
            dist_arr = np.array(show_dist)
            LogInfo.begin_track('Show %s distribution:', show_name)
            for pos in (25, 50, 75, 90, 95, 99, 99.9, 100):
                LogInfo.logs('Percentile = %.1f%%: %.6f', pos, np.percentile(dist_arr, pos))
            LogInfo.end_track()
        LogInfo.end_track()  # end of showing stat.
        # Displaying candidate size (DESC) and save into file

        cand_disp_list = []
        for q_idx in self.q_cand_dict.keys():
            q = self.qa_list[q_idx]['utterance']
            cand_sz = len(self.q_cand_dict[q_idx])
            cand_disp_list.append((q_idx, q, cand_sz))
        cand_disp_list.sort(key=lambda tup: tup[2], reverse=True)
        cand_disp_fp = '%s/%s_cand_size' % (self.data_dir, self.file_list_name)
        with codecs.open(cand_disp_fp, 'w', 'utf-8') as bw:
            for q_idx, q, cand_sz in cand_disp_list:
                bw.write('%d\t#%d\t%s\n' % (cand_sz, q_idx, q))

        """ Step 4: Prepare Embedding & Relation Matching & Structural Data """

        all_cands_tup_list = []     # (data_idx, q_idx, cand)
        data_idx = 0
        qidx_list = sorted(self.q_cand_dict.keys())
        for q_idx in qidx_list:
            for sc in self.q_cand_dict[q_idx]:
                all_cands_tup_list.append((data_idx, q_idx, sc))
                data_idx += 1
        assert data_idx == len(all_cands_tup_list) == sum([len(v) for v in self.q_cand_dict.values()])

        [rel_np_data, self.word_dict, self.mid_dict,
         self.word_init_emb, self.mid_init_emb] = build_relation_matching_data(
            all_cands_tup_list=all_cands_tup_list, qa_list=self.qa_list,
            wd_emb_util=self.wd_emb_util,
            q_max_len=self.q_max_len, sc_max_len=self.sc_max_len,
            path_max_len=self.path_max_len, pword_max_len=self.path_max_len*self.item_max_len
        )
        self.e_feat_len, link_np_data = build_entity_linking_data(
            all_cands_tup_list=all_cands_tup_list,
            e_max_size=self.e_max_size, mid_dict=self.mid_dict
        )
        self.extra_len, struct_np_data = build_structural_data(all_cands_tup_list=all_cands_tup_list)
        self.np_data_list = rel_np_data + link_np_data + struct_np_data

        """ Step 5: Save Information """
        self.array_num = len(self.np_data_list)
        self.word_size = self.word_init_emb.shape[0]
        self.mid_size = self.mid_init_emb.shape[0]

        self.save_size()
        self.save_dicts()
        self.save_init_emb()
        self.save_cands()

        LogInfo.end_track('Dataset Build Complete.')     # end of build dataset

    # ========================== Utility Function ========================== #

    """ 1. Word / Mid / ArrNum """

    def save_size(self):
        with open(self.size_fp, 'w') as bw:
            bw.write('words\t%d\n' % self.word_size)
            bw.write('mids\t%d\n' % self.mid_size)
            bw.write('e_feat_len\t%d\n' % self.e_feat_len)
            bw.write('extra_len\t%d\n' % self.extra_len)
            bw.write('array_num\t%d\n' % self.array_num)
        LogInfo.logs('Word/Mid/EntityFeatLen/ExtraLen/ArrNum size saved.')

    def load_size(self):
        if not os.path.isfile(self.size_fp):
            self.prepare_all_data()
            return
        with open(self.size_fp, 'r') as br:
            self.word_size = int(br.readline().strip().split('\t')[1])
            self.mid_size = int(br.readline().strip().split('\t')[1])
            self.e_feat_len = int(br.readline().strip().split('\t')[1])
            self.extra_len = int(br.readline().strip().split('\t')[1])
            self.array_num = int(br.readline().strip().split('\t')[1])
        LogInfo.logs('Word size = %d, Mid size = %d, EntityFeatLen = %d, ExtraLen = %d, ArrNum size = %d.',
                     self.word_size, self.mid_size, self.e_feat_len, self.extra_len, self.array_num)

    """ 2. <Word, Index> & <Mid, Index> """

    def save_dicts(self):
        LogInfo.begin_track('Saving actual word/mid dict into [%s] ...', self.dict_fp)
        with open(self.dict_fp, 'w') as bw:
            cPickle.dump(self.word_dict, bw)
            LogInfo.logs('%d w_dict saved.', len(self.word_dict))
            cPickle.dump(self.mid_dict, bw)
            LogInfo.logs('%d e_dict saved.', len(self.mid_dict))
        LogInfo.end_track()

    def load_dicts(self):
        if self.word_dict is not None and self.mid_dict is not None:
            return
        if not os.path.isfile(self.dict_fp):
            self.prepare_all_data()
            return
        LogInfo.begin_track('Loading w/e/p dictionaries from [%s] ...', self.dict_fp)
        with open(self.dict_fp, 'rb') as br:
            self.word_dict = cPickle.load(br)
            LogInfo.logs('word_dict: %d loaded.', len(self.word_dict))
            self.mid_dict = cPickle.load(br)
            LogInfo.logs('mid_dict: %d loaded.', len(self.mid_dict))
        LogInfo.end_track()

    """ 3. Word & Mid Initial Embedding """

    def save_init_emb(self):
        LogInfo.begin_track('Saving initial embedding matrix into [%s] ...', self.init_mat_fp)
        np.savez(self.init_mat_fp,
                 word_init_emb=self.word_init_emb,
                 mid_init_emb=self.mid_init_emb)
        LogInfo.logs('Word embedding: %s saved.', self.word_init_emb.shape)
        LogInfo.logs('Mid embedding: %s saved.', self.mid_init_emb.shape)
        LogInfo.end_track()

    def load_init_emb(self):
        if self.word_init_emb is not None and self.mid_init_emb is not None:
            return
        if not os.path.isfile(self.dump_fp):
            self.prepare_all_data()
            return
        LogInfo.begin_track('Loading initial embedding matrix from [%s] ...', self.init_mat_fp)
        npz = np.load(self.init_mat_fp)
        self.word_init_emb = npz['word_init_emb']
        self.mid_init_emb = npz['mid_init_emb']
        LogInfo.logs('Word embedding: %s loaded.', self.word_init_emb.shape)
        LogInfo.logs('Mid embedding: %s loaded.', self.mid_init_emb.shape)
        LogInfo.end_track()

    """ 4. Candidate schemas & np_data """

    def save_cands(self):
        assert len(self.np_data_list) == self.array_num
        LogInfo.begin_track('Saving candidates into [%s] ...', self.dump_fp)
        with open(self.dump_fp, 'wb') as bw:
            cPickle.dump(self.q_cand_dict, bw)
            LogInfo.logs('%d q_cand_dict saved.', len(self.q_cand_dict))
        LogInfo.end_track()
        np.savez(self.np_data_fp, *self.np_data_list)
        LogInfo.logs('%d np_data saved.', len(self.np_data_list))

    def load_cands(self):
        if len(self.np_data_list) > 0 and self.q_cand_dict is not None:
            return
        if not os.path.isfile(self.dump_fp):
            self.prepare_all_data()
            return
        LogInfo.begin_track('Loading candidates from [%s] ...', self.dump_fp)
        with open(self.dump_fp, 'rb') as br:
            self.q_cand_dict = cPickle.load(br)
            LogInfo.logs('q_cand_dict for %d questions loaded.', len(self.q_cand_dict))
            cand_size_dist = np.array([len(v) for v in self.q_cand_dict.values()])
            for pos in (25, 50, 75, 90, 95, 99, 99.9, 100):
                LogInfo.logs('Percentile = %.1f%%: %.6f', pos, np.percentile(cand_size_dist, pos))
        LogInfo.end_track()
        LogInfo.begin_track('Loading np_data from [%s] ...', self.np_data_fp)
        npz = np.load(self.np_data_fp)
        self.np_data_list = []
        for idx in range(self.array_num):
            key = 'arr_%d' % idx
            np_data = npz[key]
            self.np_data_list.append(np_data)
            LogInfo.logs('np-data-%d loaded: %s', idx, np_data.shape)
        LogInfo.end_track()
