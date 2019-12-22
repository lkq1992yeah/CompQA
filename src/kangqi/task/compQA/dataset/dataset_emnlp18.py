import os
import json
import codecs
import cPickle
import numpy as np

from ..candgen_acl18.global_linker import LinkData
from .ds_helper.dataset_schema_reader import load_schema_by_kqnew_protocol

from .u import load_simpq, load_webq, load_compq
from ..util.fb_helper import get_item_name, get_domain, get_range

from kangqi.util.LogUtil import LogInfo


class SchemaDatasetEMNLP18:

    def __init__(self, data_name, data_dir, file_list_name,
                 schema_level, wd_emb_util,
                 use_ans_type_dist=False, placeholder_policy='ActiveOnly',
                 full_constr=False, fix_dir=False,
                 qw_max_len=20, path_max_size=3, pw_cutoff=15, verbose=0):
        self.data_name = data_name
        self.data_dir = data_dir
        self.file_list_name = file_list_name
        self.schema_level = schema_level
        self.wd_emb_util = wd_emb_util
        self.verbose = verbose

        self.qw_max_len = qw_max_len
        self.path_max_size = self.el_max_size = path_max_size
        self.pw_cutoff = pw_cutoff
        self.sup_max_size = None

        self.use_ans_type_dist = use_ans_type_dist
        self.placeholder_policy = placeholder_policy
        self.full_constr = full_constr
        self.fix_dir = fix_dir
        """
        180326: whether to use full constraint (from constraint node to answer node),
                or using a shorter version (from constraint node to main path)
        """

        # Always first load QA-related data (not related to candidates)
        self.qa_list = self.v_input_mat = self.v_len_vec = self.clause_input_mat = None
        self.load_qa_data(wd_emb_util=wd_emb_util, qw_max_len=qw_max_len)

        self.q_idx_list = None      # indicating all active questions
        self.smart_q_cand_dict = None

        self.path_idx_dict = None
        self.entity_idx_dict = None
        self.type_idx_dict = None
        self.word_idx_dict = self.wd_emb_util.word_idx_dict     # TODO: will be replaced by active words
        """ 180409: All active element --> index mappings, Leaving <PAD>=0, <START>=1, <UNK>=2 """

        self.pw_max_len = None
        self.pw_voc_inputs = None   # (path_voc, pw_max_len)
        self.pw_voc_length = None   # (path_voc,)
        self.pw_voc_domain = None   # (path_voc,)
        self.entity_type_matrix = None      # (entity_voc, type_voc)

        if self.use_ans_type_dist:
            self.path_max_size += 1        # increase the size of components from 3 to 4
        assert not self.use_ans_type_dist
        assert self.placeholder_policy == 'ActiveOnly'
        # if asserted, then won't occur in the file names
        self.save_dir = '%s/%s-%d-%d-%d-%s-fcons%s-fix%s-EMNLP18' % (
            data_dir, file_list_name,
            qw_max_len, path_max_size, pw_cutoff,
            schema_level, full_constr, fix_dir)
        self.dump_fp = self.save_dir + '/cand.cPickle'
        # saving each candidate schema
        # self.np_data_fp = self.save_dir + '/np_data.npz'
        # # saving np data for relation matching / entity linking / structural information

    def load_qa_data(self, wd_emb_util, qw_max_len):
        if self.data_name == 'WebQ':
            self.qa_list = load_webq()
        elif self.data_name == 'CompQ':
            self.qa_list = load_compq()
        else:
            self.qa_list = load_simpq()

        """ Generate v_input and v_len for all questions """
        qa_size = len(self.qa_list)
        word_idx_dict = wd_emb_util.load_word_indices()
        self.v_input_mat = np.zeros((qa_size, qw_max_len), dtype='int32')
        self.v_len_vec = np.zeros((qa_size,), dtype='int32')
        self.clause_input_mat = np.zeros((qa_size, qw_max_len), dtype='int32')   # default: 0
        # 0: NO-CLAUSE; 1: OUT-CLAUSE; 2: IN-CLAUSE
        for q_idx, qa in enumerate(self.qa_list):
            lower_tok_list = [tok.token.lower() for tok in qa['tokens']]
            v_idx_list = [word_idx_dict.get(wd, 2) for wd in lower_tok_list]        # 2 = <UNK>
            v_idx_list = v_idx_list[:qw_max_len]
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
        self.path_idx_dict = {'<PAD>': 0, '<START>': 1, '<UNK>': 2}
        self.entity_idx_dict = {'<PAD>': 0, '<START>': 1, '<UNK>': 2}
        self.type_idx_dict = {'<PAD>': 0, '<START>': 1, '<UNK>': 2}
        path_domain_dict = {}       # a temporary dict recording domain type of each path

        self.q_idx_list = []
        self.smart_q_cand_dict = {}
        smart_q_link_dict = {}

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
            for gl_data in gather_linkings:
                """ Collect active entities / types """
                mid = gl_data.value
                if gl_data.category == 'Entity':
                    self.entity_idx_dict.setdefault(mid, len(self.entity_idx_dict))
                elif gl_data.category == 'Type':
                    self.type_idx_dict.setdefault(mid, len(self.type_idx_dict))
            smart_q_link_dict[q_idx] = gather_linkings

            if self.verbose >= 1:
                LogInfo.begin_track('scan_idx = %d, q_idx = %d:', scan_idx, q_idx)
                LogInfo.logs('Q: %s', q.encode('utf-8'))

            candidate_list, total_lines = load_schema_by_kqnew_protocol(
                q_idx=q_idx, schema_fp=schema_fp, gather_linkings=smart_q_link_dict[q_idx],
                sc_max_len=self.path_max_size, schema_level=self.schema_level,
                sc_len_dist=sc_len_dist, path_len_dist=path_len_dist, ans_size_dist=ans_size_dist,
                use_ans_type_dist=self.use_ans_type_dist, placeholder_policy=self.placeholder_policy,
                full_constr=self.full_constr, fix_dir=self.fix_dir
            )
            total_cand_size += total_lines
            useful_cand_size += len(candidate_list)
            """ Collect active paths and the corresponding domain type """
            for cand in candidate_list:
                for raw_path, path_list in zip(cand.raw_paths, cand.path_list):
                    path_cate = raw_path[0]
                    pred_seq = raw_path[-1]     # the raw pred seq (not affected by fix-dir)
                    path_str = '%s|%s' % (path_cate, '\t'.join(path_list))
                    self.path_idx_dict.setdefault(path_str, len(self.path_idx_dict))
                    domain_type = ''
                    if path_cate == 'Main':
                        domain_type = get_domain(pred_seq[0])   # pred_seq from focus to answer
                    elif path_cate == 'Entity':
                        domain_type = get_range(pred_seq[-1])   # pred_seq from answer to constraint
                    path_domain_dict[path_str] = domain_type
                    if domain_type != '':   # collect domain types
                        self.type_idx_dict.setdefault(domain_type, len(self.type_idx_dict))

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

        """ Step 4: Build path word vocabulary """
        LogInfo.begin_track('Building path word & entity-type vocabulary ... ')
        self.build_active_voc(wd_emb_util=self.wd_emb_util, path_domain_dict=path_domain_dict)
        LogInfo.end_track()

        """ Step 5: Build active entity-type matrix """
        LogInfo.begin_track('Building active entity-type matrix ... ')
        self.build_active_entity_type_matrix()
        LogInfo.end_track()

        self.save_smart_cands()     # won't save np_input information, just calculate on-the-fly
        # LogInfo.logs('Active E / T / Path = %d / %d / %d (with PAD, START, UNK)',
        #              len(self.entity_idx_dict), len(self.type_idx_dict), len(self.path_idx_dict))
        LogInfo.end_track('S-MART schemas load complete.')     # end of dataset generation

    def build_active_voc(self, wd_emb_util, path_domain_dict):
        # LogInfo.begin_track('Showing path_domain samples:')
        # for k, v in path_domain_dict.items()[:50]:
        #     LogInfo.logs('[%s] --> %s', k, v)
        # LogInfo.end_track()
        word_idx_dict = wd_emb_util.load_word_indices()
        path_size = len(self.path_idx_dict)
        self.pw_max_len = 0
        self.pw_voc_length = np.zeros(shape=(path_size, ), dtype='int32')
        self.pw_voc_domain = np.zeros(shape=(path_size, ), dtype='int32')
        pw_voc_dict = {}    # dict of path word sequence (each word is represented by word index)

        for path_str, idx in self.path_idx_dict.items():
            if idx <= 2:        # PAD, START, UNK
                pw_idx_seq = []
            else:
                path_cate, mid_str = path_str.split('|')
                mid_seq = mid_str.split('\t')
                pw_idx_seq = []
                for mid in mid_seq:
                    p_name = get_item_name(mid)
                    if p_name != '':
                        spt = p_name.split(' ')
                        for wd in spt:
                            wd_idx = word_idx_dict.get(wd, 2)       # UNK if needed
                            pw_idx_seq.append(wd_idx)
                # pw_idx_seq = pw_idx_seq[:self.pw_cutoff]  # truncate if exceeding length limit
            self.pw_voc_length[idx] = len(pw_idx_seq)
            domain_type = path_domain_dict.get(path_str, '')
            if domain_type == '':
                domain_type_idx = 0         # PAD
            else:
                domain_type_idx = self.type_idx_dict.get(domain_type, 2)    # UNK
            self.pw_voc_domain[idx] = domain_type_idx
            pw_voc_dict[idx] = pw_idx_seq
        LogInfo.logs('IN_USE: %s pw_voc_domain constructed.', self.pw_voc_domain.shape)
        LogInfo.logs('IN_USE: %s pw_voc_length constructed.', self.pw_voc_length.shape)
        for pos in (25, 50, 75, 90, 95, 99, 99.9, 100):
            LogInfo.logs('Percentile = %.1f%%: %.6f', pos, np.percentile(self.pw_voc_length, pos))
        self.pw_max_len = np.max(self.pw_voc_length)
        LogInfo.logs('IN_USE: pw_max_len = %d.', self.pw_max_len)

        # for path_str, idx in self.path_idx_dict.items():
        #     local_len = self.pw_voc_length[idx]
        #     if local_len > 7:
        #         LogInfo.logs('Length = %d [%s] --> %s', local_len, path_str, pw_voc_dict[idx])

        assert len(pw_voc_dict) == path_size        # ensure no paths sharing the same index
        self.pw_voc_inputs = np.zeros(shape=(path_size, self.pw_max_len), dtype='int32')
        for idx, pw_idx_seq in pw_voc_dict.items():
            local_len = len(pw_idx_seq)
            self.pw_voc_inputs[idx, :local_len] = pw_idx_seq
        LogInfo.logs('IN_USE: %s pw_voc_inputs constructed.', self.pw_voc_inputs.shape)

    def build_active_entity_type_matrix(self):
        entity_size = len(self.entity_idx_dict)
        type_size = len(self.type_idx_dict)
        LogInfo.logs('E size = %d, T size = %d.', entity_size, type_size)
        self.entity_type_matrix = np.zeros(shape=(entity_size, type_size), dtype='float32')
        local_et_fp = '%s/%s_entity_type.txt' % (self.data_dir, self.file_list_name)
        global_et_fp = 'data/fb_metadata/entity_type_siva.txt'
        if not os.path.isfile(local_et_fp):
            LogInfo.logs('Scanning global ET data from [%s] ...', global_et_fp)
            cnt = 0
            with open(global_et_fp, 'r') as br, open(local_et_fp, 'w') as bw:
                lines = br.readlines()
                LogInfo.logs('%d global ET line loaded.', len(lines))
                for line_idx, line in enumerate(lines):
                    if line_idx % 5000000 == 0:
                        LogInfo.logs('Current: %d / %d', line_idx, len(lines))
                    mid, tp = line.strip().split('\t')
                    e_idx = self.entity_idx_dict.get(mid, -1)
                    t_idx = self.type_idx_dict.get(tp, -1)  # doesn't care OOV entities / types
                    if e_idx == -1 or t_idx == -1:
                        continue
                    cnt += 1
                    self.entity_type_matrix[e_idx, t_idx] = 1.
                    bw.write(line)
            LogInfo.logs('%d local ET data extracted and filled.', cnt)
        else:
            LogInfo.logs('Scanning local ET data from [%s] ...', local_et_fp)
            with open(local_et_fp, 'r') as br:
                lines = br.readlines()
                LogInfo.logs('%d local ET line loaded.', len(lines))
                for line in lines:
                    mid, tp = line.strip().split('\t')
                    e_idx = self.entity_idx_dict.get(mid, -1)
                    t_idx = self.type_idx_dict.get(tp, -1)  # doesn't care OOV entities / types
                    if e_idx == -1 or t_idx == -1:
                        continue
                    self.entity_type_matrix[e_idx, t_idx] = 1.
        LogInfo.logs('IN_USE: %s entity_type_matrix constructed.', self.entity_type_matrix.shape)

    """ ======== Save / Load / Meta-Display ======== """

    def save_smart_cands(self):
        LogInfo.begin_track('Saving candidates into [%s] ...', self.dump_fp)
        with open(self.dump_fp, 'wb') as bw:
            cPickle.dump(self.smart_q_cand_dict, bw)
            LogInfo.logs('%d smart_q_cand_dict saved.', len(self.smart_q_cand_dict))
            cPickle.dump(self.path_idx_dict, bw)
            cPickle.dump(self.entity_idx_dict, bw)
            cPickle.dump(self.type_idx_dict, bw)
            LogInfo.logs('Active E/T/Path dict dumped.')
            cPickle.dump(self.pw_voc_inputs, bw)    # (path_voc, pw_max_len)
            cPickle.dump(self.pw_voc_length, bw)    # (path_voc,)
            cPickle.dump(self.pw_voc_domain, bw)    # (path_voc,)
            cPickle.dump(self.entity_type_matrix, bw)    # (entity_voc, type_voc)
            LogInfo.logs('path word & entity_type lookup tables dumped.')
        LogInfo.end_track()

    def load_smart_cands(self):
        if self.smart_q_cand_dict is not None:      # already loaded
            return
        if not os.path.isfile(self.dump_fp):        # no dump, read from txt
            self.load_smart_schemas_from_txt()
        else:
            LogInfo.begin_track('Loading smart_candidates from [%s] ...', self.dump_fp)
            with open(self.dump_fp, 'rb') as br:
                LogInfo.begin_track('Loading smart_q_cand_dict ... ')
                self.smart_q_cand_dict = cPickle.load(br)
                LogInfo.logs('Candidates for %d questions loaded.', len(self.smart_q_cand_dict))
                cand_size_dist = np.array([len(v) for v in self.smart_q_cand_dict.values()])
                LogInfo.logs('Total schemas = %d, avg = %.6f.', np.sum(cand_size_dist), np.mean(cand_size_dist))
                for pos in (25, 50, 75, 90, 95, 99, 99.9, 100):
                    LogInfo.logs('Percentile = %.1f%%: %.6f', pos, np.percentile(cand_size_dist, pos))
                LogInfo.end_track()
                self.path_idx_dict = cPickle.load(br)
                self.entity_idx_dict = cPickle.load(br)
                self.type_idx_dict = cPickle.load(br)
                LogInfo.logs('Active E/T/Path dict loaded.')
                self.pw_voc_inputs = cPickle.load(br)  # (path_voc, pw_max_len)
                self.pw_voc_length = cPickle.load(br)  # (path_voc,)
                self.pw_voc_domain = cPickle.load(br)  # (path_voc,)
                self.entity_type_matrix = cPickle.load(br)    # (entity_voc, type_voc)
                self.pw_max_len = self.pw_voc_inputs.shape[1]
                LogInfo.logs('path word & entity_type lookup tables loaded.')
            self.q_idx_list = sorted(self.smart_q_cand_dict.keys())
            LogInfo.end_track()  # end of loading
        self.meta_stat()    # show meta statistics

    def meta_stat(self):
        LogInfo.begin_track('Meta statistics:')
        LogInfo.logs('Active E / T / Path = %d / %d / %d (with PAD, START, UNK)',
                     len(self.entity_idx_dict), len(self.type_idx_dict), len(self.path_idx_dict))
        LogInfo.logs('path_max_size = %d, qw_max_len = %d, pw_max_len = %d.',
                     self.path_max_size, self.qw_max_len, self.pw_max_len)
        """ 180330: count distinct paths """
        path_set = set([])
        cand_size = 0
        support_stat_list = []
        for q_idx, cand_list in self.smart_q_cand_dict.items():
            qph_paths_dict = {}
            for cand in cand_list:
                cand_size += 1
                for (cate, gl_data, mid_seq), path in zip(cand.raw_paths, cand.path_list):
                    path_str = '%s|%s' % (cate, '\t'.join(path))
                    path_set.add(path_str)
                    if cate in ('Main', 'Entity'):      # only concern supports of entity constraints
                        key = '%d:%d_%d' % (q_idx, gl_data.start, gl_data.end)
                        qph_paths_dict.setdefault(key, set([])).add(path_str)
            for key in qph_paths_dict:
                support_stat_list.append((key, len(qph_paths_dict[key])))
        LogInfo.logs('%d distinct path from %d candidates.', len(path_set), cand_size)
        support_stat_list.sort(key=lambda _tup: _tup[-1], reverse=True)
        support_dist = np.array([tup[1] for tup in support_stat_list])
        self.sup_max_size = np.max(support_dist)
        LogInfo.logs('%d distinct Q-ph found, averaging %d supports per Q-ph, with maximum support = %d.',
                     len(support_dist), np.mean(support_dist), self.sup_max_size)
        for pos in (25, 50, 75, 90, 95, 99, 99.9, 100):
            LogInfo.logs('Percentile = %.1f%%: %.6f', pos, np.percentile(support_dist, pos))
        LogInfo.end_track()

    """ ======== (Possible: Uhash function for displaying) ======== """
