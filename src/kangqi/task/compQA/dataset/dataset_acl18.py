import os
import json
import codecs
import cPickle
import numpy as np

from ..candgen_acl18.global_linker import LinkData
from .ds_helper.dataset_schema_reader import load_schema_by_kqnew_protocol

from .u import load_simpq, load_webq, load_compq

from kangqi.util.LogUtil import LogInfo


class SchemaDatasetACL18:

    def __init__(self, data_name, data_dir, file_list_name,
                 schema_level, wd_emb_util,
                 use_ans_type_dist=False, placeholder_policy='ActiveOnly',
                 full_constr=True, fix_dir=True,
                 q_max_len=20, sc_max_len=3, path_max_len=3, item_max_len=5,
                 type_dist_len=20, verbose=0):
        self.data_name = data_name
        self.data_dir = data_dir
        self.file_list_name = file_list_name
        self.schema_level = schema_level
        self.wd_emb_util = wd_emb_util
        self.verbose = verbose

        self.q_max_len = q_max_len
        self.sc_max_len = sc_max_len
        self.path_max_len = path_max_len
        self.pword_max_len = path_max_len * item_max_len
        self.type_dist_len = type_dist_len

        self.use_ans_type_dist = use_ans_type_dist
        self.placeholder_policy = placeholder_policy
        self.full_constr = full_constr
        self.fix_dir = fix_dir
        """
        180326: whether to use full constraint (from constraint node to answer node),
                or using a shorter version (from constraint node to main path)
        """

        if self.use_ans_type_dist:
            self.sc_max_len += 1        # increase the size of components from 3 to 4

        if self.data_name == 'WebQ':
            self.qa_list = load_webq()
        elif self.data_name == 'CompQ':
            self.qa_list = load_compq()
        else:
            self.qa_list = load_simpq()

        """ Generate v_input and v_len for all questions """
        qa_size = len(self.qa_list)
        word_idx_dict = wd_emb_util.load_word_indices()
        self.v_input_mat = np.zeros((qa_size, q_max_len), dtype='int32')
        self.v_len_vec = np.zeros((qa_size,), dtype='int32')
        self.clause_input_mat = np.zeros((qa_size, q_max_len), dtype='int32')   # default: 0
        # 0: NO-CLAUSE; 1: OUT-CLAUSE; 2: IN-CLAUSE
        for q_idx, qa in enumerate(self.qa_list):
            lower_tok_list = [tok.token.lower() for tok in qa['tokens']]
            v_idx_list = [word_idx_dict.get(wd, 2) for wd in lower_tok_list]        # 2 = <UNK>
            v_idx_list = v_idx_list[:q_max_len]
            v_len = len(v_idx_list)
            self.v_len_vec[q_idx] = v_len
            self.v_input_mat[q_idx, :v_len] = v_idx_list
            clause_begin_pos = -1
            for wd_idx, wd in enumerate(lower_tok_list):
                if wd in ('when', 'during') and wd_idx >= 1:        # found when/during in the middle of the sentence
                    clause_begin_pos = wd_idx
                    break
            if clause_begin_pos != -1:
                for pos in range(clause_begin_pos):
                    self.clause_input_mat[q_idx, pos] = 1   # OUT-CLAUSE
                for pos in range(clause_begin_pos, v_len):
                    self.clause_input_mat[q_idx, pos] = 2   # IN-CLAUSE
        LogInfo.logs('v_input, v_len and clause_input loaded for %d questions.', len(self.v_len_vec))

        self.smart_q_cand_dict = None
        self.smart_q_link_dict = None
        self.dynamic_q_cand_dict = None
        self.dynamic_q_link_dict = None

        self.q_idx_list = None      # indicating all active questions

        assert not self.use_ans_type_dist
        assert self.placeholder_policy == 'ActiveOnly'      # if asserted, then won't occur in the file names

        self.save_dir = '%s/%s-%d-%d-%d-%d-%s-fcons%s-fix%s' % (
            data_dir, file_list_name,
            q_max_len, sc_max_len, path_max_len, item_max_len,
            schema_level, full_constr, fix_dir)
        self.dump_fp = self.save_dir + '/cand.cPickle'
        # saving each candidate schema
        # self.np_data_fp = self.save_dir + '/np_data.npz'
        # # saving np data for relation matching / entity linking / structural information

    def load_smart_schemas_from_txt(self):
        LogInfo.begin_track('Loading S-MART schemas from [%s] ...', self.data_dir)

        """ Step 1: Load Auxiliary Information """
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        list_fp = '%s/%s' % (self.data_dir, self.file_list_name)
        with open(list_fp, 'r') as br:
            schema_fp_list = map(lambda line: '%s/%s' % (self.data_dir, line.strip()), br.readlines())
        LogInfo.logs('%d schema files found in [%s].', len(schema_fp_list), self.file_list_name)

        """ Step 2: Traverse & Make Statistics """
        self.smart_q_cand_dict = {}
        self.smart_q_link_dict = {}
        self.q_idx_list = []
        sc_len_dist = []  # distribution of number of paths in a schema
        path_len_dist = []  # distribution of length of each path
        ans_size_dist = []  # distribution of answer size
        q_len_dist = self.v_len_vec     # distribution of question word length
        total_cand_size = useful_cand_size = 0

        for scan_idx, schema_fp in enumerate(schema_fp_list):
            if scan_idx % 100 == 0:
                LogInfo.logs('%d / %d scanned.', scan_idx, len(schema_fp_list))
            link_fp = schema_fp[0: schema_fp.rfind('_')] + '_links'
            q_idx = int(schema_fp.split('/')[-1].split('_')[0])
            q = self.qa_list[q_idx]['utterance']
            self.q_idx_list.append(q_idx)

            gather_linkings = []
            with codecs.open(link_fp, 'r', 'utf-8') as br:
                for gl_line in br.readlines():
                    tup_list = json.loads(gl_line.strip())
                    ld_dict = {k: v for k, v in tup_list}
                    gather_linkings.append(LinkData(**ld_dict))
            self.smart_q_link_dict[q_idx] = gather_linkings

            if self.verbose >= 1:
                LogInfo.begin_track('scan_idx = %d, q_idx = %d:', scan_idx, q_idx)
                LogInfo.logs('Q: %s', q.encode('utf-8'))

            candidate_list, total_lines = load_schema_by_kqnew_protocol(
                q_idx=q_idx, schema_fp=schema_fp, gather_linkings=self.smart_q_link_dict[q_idx],
                sc_max_len=self.sc_max_len, schema_level=self.schema_level,
                sc_len_dist=sc_len_dist, path_len_dist=path_len_dist, ans_size_dist=ans_size_dist,
                use_ans_type_dist=self.use_ans_type_dist, placeholder_policy=self.placeholder_policy,
                full_constr=self.full_constr, fix_dir=self.fix_dir
            )
            total_cand_size += total_lines
            useful_cand_size += len(candidate_list)

            if self.verbose >= 1:
                srt_cand_list = sorted(candidate_list, key=lambda _sc: _sc.f1, reverse=True)
                for rank_idx, sc in enumerate(srt_cand_list[:5]):
                    LogInfo.logs('#%d: F1 = %.6f, schema = [%s]', rank_idx + 1, sc.f1, sc.disp())
                LogInfo.logs('Candidate size = %d', len(candidate_list))
                LogInfo.end_track()
            self.smart_q_cand_dict[q_idx] = candidate_list
        self.q_idx_list.sort()

        """ Step 3: Show Statistics """
        q_size = len(self.smart_q_cand_dict)
        LogInfo.begin_track('[STAT] %d questions scanned:', q_size)
        LogInfo.logs('Total schemas = %d', total_cand_size)
        LogInfo.logs('Useful schemas = %d (%.3f%%)', useful_cand_size, 100. * useful_cand_size / total_cand_size)
        LogInfo.logs('Avg candidates = %.3f', 1. * useful_cand_size / q_size)
        cand_size_dist = [len(v) for v in self.smart_q_cand_dict.values()]
        for show_name, show_dist in zip(['q_len', 'cand_size', 'sc_len', 'path_len', 'ans_size'],
                                        [q_len_dist, cand_size_dist, sc_len_dist, path_len_dist, ans_size_dist]):
            dist_arr = np.array(show_dist)
            LogInfo.begin_track('Show %s distribution:', show_name)
            for pos in (25, 50, 75, 90, 95, 99, 99.9, 100):
                LogInfo.logs('Percentile = %.1f%%: %.6f', pos, np.percentile(dist_arr, pos))
            LogInfo.logs('Average: %.6f', np.mean(dist_arr))
            LogInfo.end_track()
        LogInfo.end_track()  # end of showing stat.

        self.save_smart_cands()     # won't save np_input information, just calculate on-the-fly

        # """ Step 4: Generate the input np data for each schema """
        # LogInfo.begin_track('Prepare input np data for %d schemas: ', useful_cand_size)
        # word_idx_dict = self.wd_emb_util.load_word_indices()
        # mid_idx_dict = self.wd_emb_util.load_mid_indices()
        # scan_size = 0
        # for q_idx, cand_list in self.smart_q_cand_dict.items():
        #     for sc in cand_list:
        #         if scan_size % 10000 == 0:
        #             LogInfo.logs('Current: %d / %d', scan_size, useful_cand_size)
        #         scan_size += 1
        #         sc.create_input_np_dict(
        #             qw_max_len=self.q_max_len, sc_max_len=self.sc_max_len,
        #             p_max_len=self.path_max_len, pw_max_len=self.pword_max_len,
        #             q_len=len(self.qa_list[q_idx]['tokens']),
        #             word_idx_dict=word_idx_dict, mid_idx_dict=mid_idx_dict
        #         )
        # LogInfo.end_track()

        LogInfo.end_track('S-MART schemas load complete.')     # end of dataset generation

    def save_smart_cands(self):
        LogInfo.begin_track('Saving candidates into [%s] ...', self.dump_fp)
        with open(self.dump_fp, 'wb') as bw:
            cPickle.dump(self.smart_q_cand_dict, bw)
            LogInfo.logs('%d smart_q_cand_dict saved.', len(self.smart_q_cand_dict))
        LogInfo.end_track()

    def load_smart_cands(self):
        if self.smart_q_cand_dict is not None:      # already loaded
            return
        if not os.path.isfile(self.dump_fp):        # no dump, read from txt
            self.load_smart_schemas_from_txt()
            return
        LogInfo.begin_track('Loading smart_candidates from [%s] ...', self.dump_fp)
        with open(self.dump_fp, 'rb') as br:
            self.smart_q_cand_dict = cPickle.load(br)
            LogInfo.logs('smart_q_cand_dict for %d questions loaded.', len(self.smart_q_cand_dict))
            cand_size_dist = np.array([len(v) for v in self.smart_q_cand_dict.values()])
            LogInfo.logs('Total schemas = %d, avg = %.6f.', np.sum(cand_size_dist), np.mean(cand_size_dist))
            for pos in (25, 50, 75, 90, 95, 99, 99.9, 100):
                LogInfo.logs('Percentile = %.1f%%: %.6f', pos, np.percentile(cand_size_dist, pos))
        self.q_idx_list = sorted(self.smart_q_cand_dict.keys())
        LogInfo.end_track()

        """ 180330: count distinct paths """
        path_set = set([])
        cand_size = 0
        support_stat_list = []
        for q_idx, cand_list in self.smart_q_cand_dict.items():
            qph_paths_dict = {}
            for cand in cand_list:
                cand_size += 1
                for (cate, gl_data, mid_seq), path in zip(cand.raw_paths, cand.path_list):
                    path_str = '\t'.join(path)
                    path_set.add(path_str)
                    if cate in ('Main', 'Entity'):
                        key = '%d-%d' % (gl_data.start, gl_data.end)
                        qph_paths_dict.setdefault(key, set([])).add(path_str)
            for key in qph_paths_dict:
                support_stat_list.append(len(qph_paths_dict[key]))
        LogInfo.logs('%d distinct path from %d candidates.', len(path_set), cand_size)
        support_dist = np.array(support_stat_list)
        LogInfo.logs('%d distinct Q-ph found, averaging %d supports per Q-ph.',
                     len(support_dist), np.mean(support_dist))
        for pos in (25, 50, 75, 90, 95, 99, 99.9, 100):
            LogInfo.logs('Percentile = %.1f%%: %.6f', pos, np.percentile(support_dist, pos))
