"""
Author: Kangqi Luo
Goal: define the new schema structure for running after Nov 2017
"""
import json
import numpy as np
from collections import namedtuple

from ..candgen_acl18.global_linker import LinkData
# from ..eff_candgen.translator import build_time_filter
from ..util.fb_helper import get_domain, get_range, get_item_name,\
    get_dt_preds_given_type, get_ord_preds_given_type, \
    is_mediator_as_expect, inverse_predicate, get_end_dt_pred, load_sup_sub_types

from kangqi.util.discretizer import Discretizer
# from kangqi.util.LogUtil import LogInfo


tml_comp_dict = {
    '==': u'm.__in__',
    '<': u'm.__before__',
    '>': u'm.__after__',
    '>=': u'm.__since__'
}       # convert time comparison into a virtual mid


ordinal_dict = {
    'max': u'm.__max__',
    'min': u'm.__min__'
}


ans_size_disc = Discretizer([2, 3, 5, 10, 50], output_mode='list')      # 5+1
# ans < 2
# 2 <= ans < 3
# 3 <= ans < 5
# 5 <= ans < 10
# 10 <= ans < 50
# ans >= 50


RawPath = namedtuple('RawPath', ['path_cate', 'focus', 'pred_seq'])


class CompqSchema(object):

    def __init__(self):
        self.q_idx = None
        self.gather_linkings = None     # all related linkings of this question (either used or not used)

        self.use_idx = None     # UNIQUE schema index mainly used in dataset & dataloader
        self.ori_idx = None     # original index, equals to the row number of the schema in the file

        self.p = self.r = self.f1 = None
        self.rm_f1 = self.el_f1 = None      # the adjusted F1 for different tasks
        self.ans_size = None
        self.hops = None
        self.aggregate = None           # whether using COUNT(*) as aggregation operator or not.

        self.placeholder_policy = None  # How to replace entities / times for each schema
        self.use_ans_type_dist = None   # determine how to generate input_np_dict
        self.full_constr = None         # whether to use full length of constraints (or ignore predicates at main path)
        self.fix_dir = None             # whether to fix the direction of constraints (ANS-->CONSTR ---> CONSTR-->ANS)

        self.induced_type_weight_tups = None   # [(type, weight)]

        """ The following attributes are used in candgen_acl18.cand_searcher.py """
        self.main_pred_seq = None       # saving [p1, p2]
        self.inv_main_pred_seq = None   # saving [!p2, !p1]
        self.var_types = None           # list of sets, saving the explicit types of o1 and o2
        self.var_time_preds = None      # list of sets, saving all possible time predicates derived from o1 and o2
        self.var_ord_preds = None       # list of sets, saving all ordinal predicates derived from o1 and o2

        self.raw_paths = None
        # [ (path category, linking result, predicate sequence) ]
        # path category: 'Main', 'Entity', 'Type', 'Ordinal', 'Time'
        # linking result: LinkData
        #                   old: eff_candgen/combinator.py: LinkData,
        #                   new: candgen_acl18/global_linker.py: LinkData
        # predicate sequence: sequences of predicates only
        #   main path: focus --> answer
        #   constraint path: answer --> constraint node

        self.path_list = None   # a list of PURE mid sequence
        # mid sequence: may have some virtual mids,
        # like "max" "min" "before" "after"

        self.path_words_list = None
        # a list of human-readable words of each mid sequence
        # used in xy_dataset & compq_eval_model

        self.replace_linkings = None
        # If one linking result is not used in the path list,
        # we save it here, and replace the corresponding mention by placeholders.

        self.gl_group_info = None
        # used for generating EL-part input data

        self.input_np_dict = None
        # the dictionary storing various input tensors

        self.run_info = None
        # additional attribute to save runtime results

    @staticmethod
    def read_schema_from_json(q_idx, json_line, gather_linkings,
                              use_ans_type_dist, placeholder_policy, full_constr, fix_dir):
        """
        Read the schema from json files. (provided with detail linking results)
        We will read the following information from the json file:
            1. p / r / f1
            2. raw_paths: [ (category, focus_idx, focus_mid, pred_seq) ]
        :param q_idx: question index
        :param json_line: a line of schema (json format, usually a dict)
        :param gather_linkings: [LinkData]
        :param use_ans_type_dist: whether to use the answer type distribution
        :param placeholder_policy: which kind of policy to be used for making the placeholder.
        :param full_constr: whether to use full length constraint, or the shorter length one.
        :param fix_dir: whether to fix direction of constraints or not.
        :return: A schema instance
        """
        info_dict = json.loads(json_line.strip())
        ret_sc = CompqSchema()
        for k in ('p', 'r', 'f1', 'ans_size'):
            setattr(ret_sc, k, info_dict[k])
        ret_sc.q_idx = q_idx

        ret_sc.gather_linkings = gather_linkings
        ret_sc.use_ans_type_dist = use_ans_type_dist
        ret_sc.placeholder_policy = placeholder_policy
        ret_sc.full_constr = full_constr
        ret_sc.fix_dir = fix_dir

        ret_sc.aggregate = info_dict.get('agg', False)
        ret_sc.hops = info_dict.get('hops')
        ret_sc.raw_paths = []
        for raw_path in info_dict['raw_paths']:
            category, focus_idx, focus_mid, pred_seq = raw_path
            if category == 'Main':
                ret_sc.hops = len(pred_seq)
                ret_sc.main_pred_seq = pred_seq
            elif not full_constr:
                """ 180326: shorten constraints, be careful with the constraint direction, still ANS-->CONSTR """
                assert 1 <= len(pred_seq) <= 2
                if len(pred_seq) == 2:
                    pred_seq = pred_seq[1:]     # ignore the overlapping predicate at the main path
            focus = gather_linkings[focus_idx]
            ret_sc.raw_paths.append(RawPath(path_cate=category, focus=focus, pred_seq=pred_seq))
        assert 1 <= ret_sc.hops <= 2
        return ret_sc

    """ ================ Functions used for creating input np data ================ """

    """ Used in ACL18-related experiments only """
    def create_input_np_dict(self, qw_max_len, sc_max_len, p_max_len, pw_max_len, type_dist_len,
                             q_len, word_idx_dict, mid_idx_dict):
        if self.input_np_dict is None:
            self.input_np_dict = {}
            self.construct_path_list()
            self.path_words_list = []
            assert len(self.path_list) <= sc_max_len

            """ Part 1: Connection """
            e_mask = np.zeros((qw_max_len,), dtype='int32')     # will cast to float in dataloader
            tm_mask = np.zeros((qw_max_len,), dtype='int32')
            gather_pos = np.zeros((qw_max_len,), dtype='int32')
            qw_len = q_len
            bio_list = ['O'] * q_len
            for gl in self.replace_linkings:
                assert gl.end <= q_len
                bio_list[gl.start] = 'E' if gl.category == 'Entity' else 'Tm'
                for pos in range(gl.start+1, gl.end):
                    bio_list[pos] = 'I'
            gp_idx = 0
            for idx in range(q_len):
                if bio_list[idx] == 'I':    # inside a placeholder, act as skipping word
                    qw_len -= 1
                else:           # T/Tm/O, all words are used in RM task
                    gather_pos[gp_idx] = idx
                    gp_idx += 1
                    if bio_list[idx] == 'E':        # starting of entity
                        e_mask[idx] = 1
                    elif bio_list[idx] == 'Tm':     # starting of time
                        tm_mask[idx] = 1
            self.input_np_dict['e_mask'] = e_mask
            self.input_np_dict['tm_mask'] = tm_mask
            self.input_np_dict['gather_pos'] = gather_pos
            self.input_np_dict['qw_len'] = qw_len

            """ Part 2: EL """
            # feat length could be dynamic across different exp settings
            # we will not specify explicitly
            gl_group_dict = self.generate_linking_group_info()
            el_max_len = sc_max_len
            el_len = 0
            el_type_input = np.zeros((el_max_len,), dtype='int32')
            el_feat_list = []       # [el_max_size, el_feat_size (dyn) ]
            for category, gl_data, pred_seq in self.raw_paths:
                if category not in ('Main', 'Entity'):
                    continue
                if category == 'Main':
                    tp = get_domain(pred_seq[0])
                else:
                    tp = get_range(pred_seq[-1])
                tp_idx = mid_idx_dict.get(tp, 2)
                link_feat = gl_data.link_feat
                assert 'score' in link_feat     # TODO: Current S-MART only
                pos = '%d_%d' % (gl_data.start, gl_data.end)
                sum_score = gl_group_dict[pos]
                ratio = 1. * link_feat['score'] / sum_score
                log_ratio = np.log(ratio)
                """ 180414: add norm_log_score, and from_smart / from_wiki """
                log_score = np.log(link_feat['score'])
                norm_log_score = log_score / 10
                # Roughly assume log_score \in [0, 10]
                # source = link_feat.get('source', -1.0)
                # if source < 0.:
                #     from_smart = from_wiki = 0.
                # else:
                #     from_smart = 1. if source != 2.0 else 0.
                #     from_wiki = 1. if source != 1.0 else 0.
                # # 0.0: from both / 1.0: from SMART / 2.0: from xs
                from_smart = from_wiki = 0.
                el_feat_list.append([ratio, log_ratio, norm_log_score, from_smart, from_wiki])
                el_type_input[el_len] = tp_idx
                el_len += 1
            # convert list storage into numpy.
            el_feat_size = len(el_feat_list[0])
            el_feats = np.zeros((el_max_len, el_feat_size), dtype='float32')
            for el_idx, feat in enumerate(el_feat_list):
                el_feats[el_idx] = feat
            self.input_np_dict['el_len'] = el_len
            self.input_np_dict['el_type_input'] = el_type_input
            self.input_np_dict['el_feats'] = el_feats

            """ Part 3: RM """
            p_input = np.zeros((sc_max_len, p_max_len), dtype='int32')
            pw_input = np.zeros((sc_max_len, pw_max_len), dtype='int32')
            p_len = np.zeros((sc_max_len,), dtype='int32')
            pw_len = np.zeros((sc_max_len,), dtype='int32')

            path_idx = 0
            for raw_path, mid_seq in zip(self.raw_paths, self.path_list):
                category = raw_path[0]
                if self.use_ans_type_dist and category == 'Type':
                    continue        # omit the explicit type constraint, if we apply answer type inducing
                p_len[path_idx] = len(mid_seq)
                pw_idx_seq = []         # collect path words --> index
                local_words = []        # collect path words
                for p_pos, mid in enumerate(mid_seq):
                    p_idx = mid_idx_dict.get(mid, 2)        # set OOV mid a 2 = <UNK>
                    p_input[path_idx, p_pos] = p_idx
                    p_name = get_item_name(mid)
                    if p_name != '':
                        spt = p_name.split(' ')
                        local_words += spt
                        for wd in spt:
                            pw_idx_seq.append(word_idx_dict[wd])
                pw_idx_seq = pw_idx_seq[:pw_max_len]        # truncate if exceeding length limit
                use_pw_len = len(pw_idx_seq)
                pw_len[path_idx] = use_pw_len
                pw_input[path_idx, :use_pw_len] = pw_idx_seq
                self.path_words_list.append(local_words)
                path_idx += 1

            self.input_np_dict['sc_len'] = path_idx
            self.input_np_dict['p_input'] = p_input
            self.input_np_dict['pw_input'] = pw_input
            self.input_np_dict['p_len'] = p_len
            self.input_np_dict['pw_len'] = pw_len

            type_dist = np.zeros((type_dist_len, ), dtype='int32')
            type_dist_weight = np.zeros((type_dist_len, ), dtype='float32')
            type_weight_tups = self.induce_answer_types(type_dist_len=type_dist_len,
                                                        mid_idx_dict=mid_idx_dict)
            for idx, (tp, weight) in enumerate(type_weight_tups):
                type_dist[idx] = mid_idx_dict[tp]
                type_dist_weight[idx] = weight
            self.input_np_dict['type_dist'] = type_dist
            self.input_np_dict['type_dist_weight'] = type_dist_weight

            """ Part 4: Extra """
            extra_feat_list = []    # E/T/Tm/Ord size, T/T/Tm/Ord indicator, 2_hop, with_med, ans_size_discrete
            # 4 + 4 + 2 + 6 = 16
            constr_size_dict = {}
            main_pred_seq = []
            for category, _, pred_seq in self.raw_paths:
                if category == 'Main':
                    main_pred_seq = pred_seq
                constr_size_dict[category] = 1 + constr_size_dict.get(category, 0)
            for cate in ('Entity', 'Type', 'Time', 'Ordinal'):
                extra_feat_list.append(constr_size_dict.get(cate, 0))          # how many constraint paths
            for cate in ('Entity', 'Type', 'Time', 'Ordinal'):
                extra_feat_list.append(min(1, constr_size_dict.get(cate, 0)))  # whether have such constraints
            is_two_hop = 1 if self.hops == 2 else 0
            with_med = 1 if is_mediator_as_expect(main_pred_seq[0]) else 0
            extra_feat_list += [is_two_hop, with_med]
            extra_feat_list += ans_size_disc.convert(self.ans_size)
            self.input_np_dict['extra_feats'] = np.array(extra_feat_list, dtype='float32')

        return self.input_np_dict

    def construct_path_list(self):
        assert self.placeholder_policy in ('ActiveOnly', 'ActiveDT')

        """ Convert raw path into formal mid sequence """
        if self.path_list is not None and self.replace_linkings is not None:
            return
        assert self.raw_paths is not None
        self.path_list = []
        self.replace_linkings = []
        if self.placeholder_policy == 'ActiveDT':
            """ 
            all datetime mentions will be put into replace_linkings, no matter used or not.
            Duplicate gl_data in self.replace_linkings doesn't matter.
            """
            for gl_data in self.gather_linkings:
                if gl_data.category == 'Time':
                    self.replace_linkings.append(gl_data)

        for raw_path in self.raw_paths:
            path_cate, link_data, pred_seq = raw_path
            assert isinstance(link_data, LinkData)
            if path_cate == 'Main':
                self.path_list.append(pred_seq)
                self.replace_linkings.append(link_data)
                # Skip entities in main path / entity constraint
            elif path_cate == 'Type':
                type_mid = link_data.value
                use_mid_seq = list(pred_seq)    # type.object.type
                use_mid_seq.append(type_mid)
                self.path_list.append(use_mid_seq)
                # Add type mid into the mid sequence
                # Type is special: always related to the target entity, it's no need to fix its direction
            else:
                """ 180221: Prepare to fix the direction, if possible """
                use_mid_seq = []
                if path_cate == 'Entity':
                    use_mid_seq = list(pred_seq)
                    # self.path_list.append(pred_seq)
                    self.replace_linkings.append(link_data)
                elif path_cate == 'Time':
                    comp = link_data.comp
                    assert comp in tml_comp_dict
                    use_mid_seq = list(pred_seq)
                    use_mid_seq.append(tml_comp_dict[comp])    # virtual mid for time comparison
                    # self.path_list.append(use_mid_seq)
                    self.replace_linkings.append(link_data)
                    # do not care about the detail time
                    """
                    180308: though time interval is considered, 
                            it doesn't affect the repr. of the constraint.
                            Just change the way of generating SPARQL.
                    """
                elif path_cate == 'Ordinal':
                    comp = link_data.comp
                    assert comp in ordinal_dict
                    use_mid_seq = list(pred_seq)
                    use_mid_seq.append(ordinal_dict[comp])
                    # 180209: I think we don't need to encode the detail rank number.
                    #         Because less ambiguity was brought in rank number generation.
                    # rank_num = int(focus.detail.entity)
                    # rank_mid = 'm.__%d__' % rank_num
                    # use_mid_seq.append(rank_mid)
                    # self.path_list.append(use_mid_seq)

                # LogInfo.logs('Fix_direction = %s', FIX_DIRECTION)
                if self.fix_dir:
                    # LogInfo.begin_track('Raw_path: %s', pred_seq)
                    # LogInfo.logs('Ready to fix: %s', use_mid_seq)
                    use_mid_seq = [inverse_predicate(pred) for pred in use_mid_seq]
                    use_mid_seq.reverse()       # mirror each predicate, then reverse the whole sequence
                    # LogInfo.end_track('Sequence reversed: %s', use_mid_seq)
                self.path_list.append(use_mid_seq)

    def generate_linking_group_info(self):
        """
        Current: S-MART only. Collect sum scores of each possible [st, ed) mention.
        """
        if self.gl_group_info is None:
            self.gl_group_info = {}
            for gl_data in self.gather_linkings:
                if gl_data.category != 'Entity':
                    continue
                pos = '%d_%d' % (gl_data.start, gl_data.end)
                assert 'score' in gl_data.link_feat
                self.gl_group_info[pos] = self.gl_group_info.get(pos, 0.) + gl_data.link_feat['score']
        return self.gl_group_info

    def induce_answer_types(self, type_dist_len, mid_idx_dict):
        ans_types = set([])         # explicit type + induced main range
        for category, gl_data, pred_seq in self.raw_paths:
            if category == 'Type':
                ans_types.add(gl_data.value)        # explicit type
        ans_types.add(get_range(self.main_pred_seq[-1]))    # induce the range type of the main path

        sup_type_dict, _ = load_sup_sub_types()
        ext_ans_types = set([])
        for tp in ans_types:
            ext_ans_types |= sup_type_dict.get(tp, set([]))
            ext_ans_types.add(tp)

        ext_ans_types = set(filter(lambda _tp: _tp in mid_idx_dict, ext_ans_types))     # remove OOV types
        assert len(ext_ans_types) <= type_dist_len
        # In practice, setting type_dist_len >= 30 will be enough.

        weight = 1. / len(ext_ans_types) if len(ext_ans_types) > 0 else 0.        # simple: on average
        self.induced_type_weight_tups = []
        for tp in sorted(ext_ans_types & ans_types):                    # explicit / induced types
            self.induced_type_weight_tups.append((tp, weight))
        for tp in sorted(ext_ans_types - ans_types):    # super types
            self.induced_type_weight_tups.append((tp, weight))
        return self.induced_type_weight_tups

    """ ================ Functions used in candidate generation ================ """

    def infer_var_types(self):      # given the main predicate, infer the explicit types of o1 and o2
        assert self.main_pred_seq is not None
        assert 1 <= self.hops <= 2
        if self.hops == 1:
            o1_type_set = set([])
            p1_range = get_range(self.main_pred_seq[0])
            if p1_range != '':
                o1_type_set.add(p1_range)
            self.var_types = [o1_type_set]
        else:
            o1_type_set = set([])
            o2_type_set = set([])
            p1_range = get_range(self.main_pred_seq[0])
            p2_domain = get_domain(self.main_pred_seq[1])
            p2_range = get_range(self.main_pred_seq[1])
            if p1_range != '':              # for o1
                o1_type_set.add(p1_range)
            if p2_domain != '':             # for o1
                o1_type_set.add(p2_domain)
            if p2_range != '':              # for o2
                o2_type_set.add(p2_range)
            self.var_types = [o1_type_set, o2_type_set]

    def infer_time_preds(self):     # gien the explicit types of each variable, produce its time-related predicates
        assert self.var_types is not None
        self.var_time_preds = []
        for type_set in self.var_types:
            time_pred_set = set([])
            for tp in type_set:
                time_pred_set |= get_dt_preds_given_type(tp)
            self.var_time_preds.append(time_pred_set)

    def infer_ordinal_preds(self):
        assert self.var_types is not None
        self.var_ord_preds = []
        for type_set in self.var_types:
            ord_pred_set = set([])
            for tp in type_set:
                ord_pred_set |= get_ord_preds_given_type(tp)
            self.var_ord_preds.append(ord_pred_set)

    """ ================ Functions used for display & multi-task dedup ================ """

    def disp(self):
        names = []
        self.construct_path_list()
        for raw_path, path in zip(self.raw_paths, self.path_list):
            path_cate, _, _ = raw_path
            names.append('%s: %s' % (path_cate, '-->'.join(path)))
        return '\t'.join(names)

    def disp_raw_path(self):
        show_tups = []
        for cate, gl, pred_seq in self.raw_paths:
            show_tups.append((cate, gl.comp, gl.value, pred_seq))
        return str(show_tups)

    def get_rm_key(self):
        """
        Given the schema, return a simple representation of raw_paths
        Just ignore the detail focus and create the string representation,
        then merge them together.
        """
        rep_list = []
        for raw_path, using_pred_seq in zip(self.raw_paths, self.path_list):
            category, gl_data, _ = raw_path
            """ force convert Main to Entity, because we fixed the direction """
            use_category = category if category != 'Main' else 'Entity'
            """ entity position matters ! """
            local_rep = '%s:%s:%s:%s' % (use_category, gl_data.start, gl_data.end, '|'.join(using_pred_seq))
            rep_list.append(local_rep)
        rep_list.sort()
        return '\t'.join(rep_list)

    def get_el_key(self, el_use_type):
        rep_list = []
        for raw_path in self.raw_paths:
            category, gl_data, pred_seq = raw_path
            if category not in ('Main', 'Entity'):
                continue
            start = gl_data.start
            end = gl_data.end
            mid = gl_data.value
            mid_pos_repr = '%d_%d_%s' % (start, end, mid)
            if not el_use_type:
                rep_list.append(mid_pos_repr)
            else:
                tp = get_domain(pred_seq[0]) if category == 'Main' else get_range(pred_seq[-1])
                rep_list.append('%s:%s' % (mid_pos_repr, tp))
        rep_list.sort()
        return '\t'.join(rep_list)

    def get_full_key(self, el_use_type):
        # Just the combination of EL and RM keys.
        return self.get_el_key(el_use_type) + '\t' + self.get_rm_key()

    """ SPARQL query converter, which is copied from eff_candgen/translator.py, but I changed a lot """

    def build_sparql(self, simple_time_match):
        """
        raw_paths: [(category, link_data, pred_seq)]
        comparison: ==, >, <, >=, max, min
        simple_time_match == True: won't use time interval
        """
        where_lines = []        # for constraints + OPTIONAL + FILTER
        order_line = ''         # for ordinal constraint
        # add_o_idx = self.hops       # additional object index representing time object and ordinal object
        var_symbol_list = []

        for cate, link_data, pred_seq in self.raw_paths:
            mid = link_data.value
            attach_idx = self.hops if len(pred_seq) == 1 else 1
            if cate == 'Main':
                self.main_pred_seq = pred_seq
                where_lines.append('fb:%s fb:%s ?o1 .' % (mid, pred_seq[0]))
                if self.hops == 2:
                    where_lines.append('?o1 fb:%s ?o2 .' % pred_seq[-1])
            elif cate == 'Entity':
                where_lines.append('?o%d fb:%s fb:%s .' % (attach_idx, pred_seq[-1], mid))
                # if len(pred_seq) == 1:      # constraint at the answer node (?o_hops)
                #     where_lines.append('?o%d fb:%s fb:%s .' % (self.hops, pred_seq[0], mid))
                # else:                       # constraint at mediator node (must be ?o1)
                #     where_lines.append('?o1 fb:%s fb:%s .' % (pred_seq[-1], mid))
            elif cate == 'Type':        # must be at the answer node
                assert len(pred_seq) == 1 and pred_seq[0] == 'type.object.type'
                where_lines.append('?o%d fb:type.object.type fb:%s .' % (self.hops, mid))
            elif cate == 'Time':
                tm_begin_pred = pred_seq[-1]
                if simple_time_match:
                    tm_end_pred = ''        # won't consider time interval
                else:
                    tm_end_pred = get_end_dt_pred(tm_begin_pred)        # try finding a time interval
                attach_idx = self.hops if len(pred_seq) == 1 else 1
                # where_lines: similar to 'Entity' operation
                for tm_var, tm_pred in (['?tm1', tm_begin_pred], ['?tm2', tm_end_pred]):
                    if tm_pred != '':
                        where_lines.append('OPTIONAL { ?o%s fb:%s %s . } .' % (attach_idx, tm_pred, tm_var))
                        var_symbol_list.append(tm_var)
                """ 180308: update sparql for time constraints (return all times, and then filter out of SPARQL) """
                # comp = link_data.comp  # ==, >, <, >=  (in fact, >= never occurs)
                # year = link_data.value
                # if len(pred_seq) == 1:  # constraint at the answer node (?o_hops)
                #     where_lines.append('?o%d fb:%s ?o%d .' % (self.hops, pred_seq[0], add_o_idx))
                # else:  # constraint at mediator node (must be ?o1)
                #     where_lines.append('?o1 fb:%s ?o%d .' % (pred_seq[-1], add_o_idx))
                # build_time_filter(year=year, comp=comp, add_o_idx=add_o_idx, filter_list=where_lines)
            elif cate == 'Ordinal':
                where_lines.append('?o%d fb:%s ?ord .' % (attach_idx, pred_seq[-1]))
                var_symbol_list.append('?ord')
                """ 180308: also update ordinal constraints, sort results out of SPARQL """
                # add_o_idx += 1
                # comp = link_data.comp           # max / min
                # num = int(link_data.value)      # rank value
                # # where_lines: similar to 'Entity' & 'Time' operation
                # if len(pred_seq) == 1:  # constraint at the answer node (?o_hops)
                #     where_lines.append('?o%d fb:%s ?o%d .' % (self.hops, pred_seq[0], add_o_idx))
                # else:  # constraint at mediator node (must be ?o1)
                #     where_lines.append('?o1 fb:%s ?o%d .' % (pred_seq[-1], add_o_idx))
                # order_line = 'ORDER BY %s(?o%d) LIMIT 1 OFFSET %d' % (
                #     'DESC' if comp == 'max' else 'ASC', add_o_idx, num-1)

        last_main_pred = self.main_pred_seq[-1]
        range_type = get_range(last_main_pred)
        if range_type in ('type.int', 'type.float', 'type.datetime'):
            # no need to output answer names
            var_symbol_list = ['?o%d' % self.hops] + var_symbol_list
        else:
            # we must have answer names, therefore OPTIONAL can be removed
            var_symbol_list = ['?o%d' % self.hops] + ['?n%d' % self.hops] + var_symbol_list
            where_lines.append('?o%d fb:type.object.name ?n%d .' % (self.hops, self.hops))

        var_symbol_str = ' '.join(var_symbol_list)
        select_line = 'SELECT DISTINCT %s WHERE {' % var_symbol_str
        where_lines.append('}')
        sparql_lines = [select_line] + where_lines
        if order_line != '':
            sparql_lines.append(order_line)
        ret = ' '.join(sparql_lines)
        return ret

    def build_aux_for_sparql(self):
        tm_comp = ord_comp = 'None'
        tm_value = ord_rank = ''
        agg = 'Agg' if self.aggregate else 'None'
        for category, gl_data, pred_seq in self.raw_paths:
            if category == 'Time':
                tm_comp = gl_data.comp
                tm_value = '%s_%s' % (gl_data.value, gl_data.value)     # interval as single point
            elif category == 'Ordinal':
                ord_comp = gl_data.comp
                ord_rank = gl_data.value
        return tm_comp, tm_value, ord_comp, ord_rank, agg
