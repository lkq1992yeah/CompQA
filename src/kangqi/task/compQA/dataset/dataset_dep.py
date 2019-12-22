import os
import json
import codecs
import cPickle
import numpy as np

from ..candgen_acl18.global_linker import LinkData
from .ds_helper.dataset_schema_reader import load_schema_by_kqnew_protocol

from .u import load_simpq, load_webq, load_compq
from ..util.fb_helper import get_item_name

from kangqi.util.LogUtil import LogInfo


class SchemaDatasetDep:

    def __init__(self, data_name, data_dir, file_list_name,
                 schema_level, wd_emb_util,
                 use_ans_type_dist=False, placeholder_policy='ActiveOnly',
                 full_constr=False, fix_dir=False,
                 el_feat_size=3, extra_feat_size=16,
                 qw_max_len=20, path_max_size=3, pseq_max_len=3, pw_cutoff=15, verbose=0):
        self.data_name = data_name
        self.data_dir = data_dir
        self.file_list_name = file_list_name
        self.schema_level = schema_level
        self.wd_emb_util = wd_emb_util
        self.verbose = verbose

        self.qw_max_len = qw_max_len
        self.path_max_size = path_max_size
        self.pseq_max_len = pseq_max_len
        self.pw_cutoff = pw_cutoff
        self.el_feat_size = el_feat_size
        self.extra_feat_size = extra_feat_size

        self.use_ans_type_dist = use_ans_type_dist
        self.placeholder_policy = placeholder_policy
        self.full_constr = full_constr
        self.fix_dir = fix_dir

        if self.data_name == 'WebQ':
            self.qa_list = load_webq()
        elif self.data_name == 'CompQ':
            self.qa_list = load_compq()
        else:
            self.qa_list = load_simpq()

        """ Need to save: Tvt split information & detail schema information """
        self.q_idx_list = []          # indicating all active questions
        self.spt_q_idx_lists = []     # T/v/t list
        self.smart_q_cand_dict = {}

        """ Need to save: static memory or global information """
        self.e_t_dict = {}
        self.pw_max_len = 0

        """ Need to save: active dict / uhash """
        self.active_dicts = {'word': {}, 'mid': {}, 'path': {}}
        self.active_uhashs = {'word': [], 'mid': [], 'path': []}
        # Leaving <PAD>=0, <START>=1, <UNK>=2
        # mid: for type and predicate only (won't embed entities)

        if self.use_ans_type_dist:
            self.path_max_size += 1        # increase the size of components from 3 to 4
        assert not self.use_ans_type_dist
        assert self.placeholder_policy == 'ActiveOnly'
        # if asserted, then won't occur in the file names
        self.save_dir = '%s/%s-%d-%d-%d-%s-fcons%s-fix%s-DEP' % (data_dir, file_list_name,
                                                                 qw_max_len, path_max_size, pw_cutoff,
                                                                 schema_level, full_constr, fix_dir)
        self.dump_fp = self.save_dir + '/cand.cPickle'

    def load_all_data(self):
        if len(self.smart_q_cand_dict) > 0:         # already loaded
            return
        if not os.path.isfile(self.dump_fp):        # no dump, read from txt
            self.load_smart_schemas_from_txt()
        else:
            self.load_smart_schemas_from_pickle()

        LogInfo.begin_track('Meta statistics:')

        LogInfo.logs('Total questions = %d', len(self.q_idx_list))
        LogInfo.logs('T / v / t questions = %s', [len(lst) for lst in self.spt_q_idx_lists])
        LogInfo.logs('Active Word / Mid / Path = %d / %d / %d (with PAD, START, UNK)',
                     len(self.active_dicts['word']), len(self.active_dicts['mid']), len(self.active_dicts['path']))
        LogInfo.logs('path_max_size = %d, qw_max_len = %d, pw_max_len = %d.',
                     self.path_max_size, self.qw_max_len, self.pw_max_len)

        cand_size_dist = np.array([len(v) for v in self.smart_q_cand_dict.values()])
        LogInfo.logs('Total schemas = %d, avg = %.6f.', np.sum(cand_size_dist), np.mean(cand_size_dist))
        for pos in (25, 50, 75, 90, 95, 99, 99.9, 100):
            LogInfo.logs('cand_size @ %.1f%%: %.6f', pos, np.percentile(cand_size_dist, pos))

        qlen_dist = np.array([len(qa['tokens']) for qa in self.qa_list])
        LogInfo.logs('Avg question length = %.6f.', np.mean(qlen_dist))
        for pos in (25, 50, 75, 90, 95, 99, 99.9, 100):
            LogInfo.logs('question_len @ %.1f%%: %.6f', pos, np.percentile(qlen_dist, pos))

        LogInfo.end_track()

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
        q_len_dist = [len(qa['tokens']) for qa in self.qa_list]
        sc_len_dist = []  # distribution of number of paths in a schema
        path_len_dist = []  # distribution of length of each path
        ans_size_dist = []  # distribution of answer size
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

            if self.verbose >= 1:
                LogInfo.begin_track('scan_idx = %d, q_idx = %d:', scan_idx, q_idx)
                LogInfo.logs('Q: %s', q.encode('utf-8'))

            candidate_list, total_lines = load_schema_by_kqnew_protocol(
                q_idx=q_idx, schema_fp=schema_fp, gather_linkings=gather_linkings,
                sc_max_len=self.path_max_size, schema_level=self.schema_level,
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

        # T/v/t split
        self.q_idx_list.sort()
        train_pos = {'WebQ': 3023, 'CompQ': 1000, 'SimpQ': 75910}[self.data_name]
        valid_pos = {'WebQ': 3778, 'CompQ': 1300, 'SimpQ': 86755}[self.data_name]
        self.spt_q_idx_lists.append(filter(lambda x: x < train_pos, self.q_idx_list))
        self.spt_q_idx_lists.append(filter(lambda x: train_pos <= x < valid_pos, self.q_idx_list))
        self.spt_q_idx_lists.append(filter(lambda x: x >= valid_pos, self.q_idx_list))

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
        LogInfo.begin_track('Building active word / mid / path vocabulary ... ')
        self.build_active_voc()
        LogInfo.end_track()

        """ Step 5: Build global entity-type matrix & pw_max_len """
        LogInfo.begin_track('Building global information ... ')
        self.build_global_information()
        LogInfo.end_track()

        self.save_smart_cands()     # won't save np_input information, just calculate on-the-fly
        # LogInfo.logs('Active E / T / Path = %d / %d / %d (with PAD, START, UNK)',
        #              len(self.entity_idx_dict), len(self.type_idx_dict), len(self.path_idx_dict))
        LogInfo.end_track('S-MART schemas load complete.')     # end of dataset generation

    def add_item_into_active_voc(self, category, value):
        if value not in self.active_dicts[category]:
            self.active_dicts[category][value] = len(self.active_dicts[category])
            self.active_uhashs[category].append(value)

    def build_active_voc(self):
        for category in ('word', 'mid', 'path'):
            for mask in ('<PAD>', '<START>', '<UNK>'):
                self.add_item_into_active_voc(category=category, value=mask)
        for mask in ('<E>', '<T>', '<Tm>', '<Ord>'):
            self.add_item_into_active_voc(category='word', value=mask)

        """ 180423: Try collect word from all data, while mid / path from train only """
        for mode, q_idx_list in zip(['train', 'valid', 'test'], self.spt_q_idx_lists):
            for q_idx in q_idx_list:
                """ Scanning questions """
                qa = self.qa_list[q_idx]
                lower_tok_list = [tok.token.lower() for tok in qa['tokens']]
                # LogInfo.logs(lower_tok_list)
                for tok in lower_tok_list:
                    self.add_item_into_active_voc(category='word', value=tok)
                for rel, head, dep in qa['parse'].dependency_parse.dependencies:
                    # LogInfo.logs('%s --%s--> %s', head, rel, dep)
                    self.add_item_into_active_voc(category='word', value=rel)
                    self.add_item_into_active_voc(category='word', value='!'+rel)

                """ Scanning schemas """
                for cand in self.smart_q_cand_dict[q_idx]:
                    for raw_path, mid_seq in zip(cand.raw_paths, cand.path_list):
                        path_cate, gl_data, _ = raw_path
                        path_str = '%s|%s' % (path_cate, '\t'.join(mid_seq))
                        if mode == 'train':
                            self.add_item_into_active_voc(category='path', value=path_str)
                            # the whole path, added in train only
                        for mid in mid_seq:
                            if mode == 'train':
                                self.add_item_into_active_voc(category='mid', value=mid)
                                # separated mid in the path, also added in train only
                            p_name = get_item_name(mid)
                            if p_name != '':
                                spt = p_name.split(' ')
                                for tok in spt:
                                    self.add_item_into_active_voc(category='word', value=tok)   # type / pred name

        for category in ('word', 'mid', 'path'):
            LogInfo.logs('Active %s size = %d', category, len(self.active_dicts[category]))

    def build_global_information(self):
        """ e_t_dict, and pw_max_len """
        """ Part 1: e_t_dict """
        local_et_fp = '%s/%s_entity_type.txt' % (self.data_dir, self.file_list_name)
        global_et_fp = 'data/fb_metadata/entity_type_siva.txt'

        if self.data_name == 'SimpQ':
            LogInfo.logs('Skip Entity-Type collection.')
        else:
            if not os.path.isfile(local_et_fp):
                LogInfo.logs('Scanning global ET data from [%s] ...', global_et_fp)
                global_e_set = set([])
                for q_idx in self.q_idx_list:
                    for cand in self.smart_q_cand_dict[q_idx]:
                        for _, gl_data, _ in cand.raw_paths:
                            if gl_data.category == 'Entity':
                                global_e_set.add(gl_data.value)
                LogInfo.logs('%d distinct entities concerned among all candidates.', len(global_e_set))
                cnt = 0
                with open(global_et_fp, 'r') as br, open(local_et_fp, 'w') as bw:
                    lines = br.readlines()
                    LogInfo.logs('%d global ET line loaded.', len(lines))
                    for line_idx, line in enumerate(lines):
                        if line_idx % 5000000 == 0:
                            LogInfo.logs('Current: %d / %d', line_idx, len(lines))
                        mid, tp = line.strip().split('\t')
                        if mid not in global_e_set:
                            continue
                        cnt += 1
                        self.e_t_dict.setdefault(mid, set([])).add(tp)
                        bw.write(line)
                LogInfo.logs('%d global concerned ET data extracted and filled.', cnt)
            else:
                LogInfo.logs('Scanning local ET data from [%s] ...', local_et_fp)
                with open(local_et_fp, 'r') as br:
                    lines = br.readlines()
                    for line in lines:
                        mid, tp = line.strip().split('\t')
                        self.e_t_dict.setdefault(mid, set([])).add(tp)
                LogInfo.logs('%d global concerned ET data filled.', len(lines))

        """ Part 2: pw_max_len, remember to collect paths from valid / test, making it global """
        global_path_set = set(self.active_uhashs['path'])
        vt_q_idx_list = self.spt_q_idx_lists[1] + self.spt_q_idx_lists[2]
        for q_idx in vt_q_idx_list:
            for cand in self.smart_q_cand_dict[q_idx]:
                for raw_path, mid_seq in zip(cand.raw_paths, cand.path_list):
                    path_cate, gl_data, _ = raw_path
                    path_str = '%s|%s' % (path_cate, '\t'.join(mid_seq))
                    global_path_set.add(path_str)
        pw_len_list = []
        for path_str in global_path_set:
            spt = path_str.split('|')
            if len(spt) != 2:
                continue
            path_cate, mid_str = spt
            mid_seq = mid_str.split('\t')
            pw_seq = []
            for mid in mid_seq:
                p_name = get_item_name(mid)
                if p_name != '':
                    pw_seq += p_name.split(' ')
            pw_len_list.append(len(pw_seq))
        for pos in (50, 75, 90, 99, 99.9, 100):
            LogInfo.logs('pw_len @ %.1f%%: %.6f', pos, np.percentile(pw_len_list, pos))
        self.pw_max_len = np.max(pw_len_list)
        LogInfo.logs('pw_max_len = %d, extracted from %d global paths.', self.pw_max_len, len(global_path_set))

    def save_smart_cands(self):
        LogInfo.begin_track('Saving candidates into [%s] ...', self.dump_fp)
        with open(self.dump_fp, 'wb') as bw:
            cPickle.dump(self.q_idx_list, bw)
            cPickle.dump(self.spt_q_idx_lists, bw)
            cPickle.dump(self.smart_q_cand_dict, bw)
            LogInfo.logs('%d smart_q_cand_dict saved.', len(self.smart_q_cand_dict))
            cPickle.dump(self.active_dicts, bw)
            cPickle.dump(self.active_uhashs, bw)
            LogInfo.logs('Active Word/Mid/Path dict/uhash dumped.')
            cPickle.dump(self.e_t_dict, bw)
            cPickle.dump(self.pw_max_len, bw)
            LogInfo.logs('entity_type lookup tables dumped.')
        LogInfo.end_track()

    def load_smart_schemas_from_pickle(self):
        LogInfo.begin_track('Loading smart_candidates from [%s] ...', self.dump_fp)
        with open(self.dump_fp, 'rb') as br:
            LogInfo.logs('Loading smart_q_cand_dict ... ')
            self.q_idx_list = cPickle.load(br)
            self.spt_q_idx_lists = cPickle.load(br)
            self.smart_q_cand_dict = cPickle.load(br)
            LogInfo.logs('Candidates for %d questions loaded.', len(self.smart_q_cand_dict))
            self.active_dicts = cPickle.load(br)
            self.active_uhashs = cPickle.load(br)
            LogInfo.logs('Active Word/Mid/Path dict/uhash loaded.')
            self.e_t_dict = cPickle.load(br)
            self.pw_max_len = cPickle.load(br)
            LogInfo.logs('entity_type lookup tables loaded.')
        LogInfo.end_track()  # end of loading
