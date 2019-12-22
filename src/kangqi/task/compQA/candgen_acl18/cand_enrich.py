import os
import codecs
import json
import copy
import cPickle

from .global_linker import LinkData
from .smart_candgen import construct_conflict_matrix
from ..dataset.kq_schema import CompqSchema
from ..dataset.ds_helper.dataset_schema_reader import schema_classification

from ..util.fb_helper import get_domain

from kangqi.util.LogUtil import LogInfo

vb = 1
q_size = 2100
train_q_size = 1000
perm_list_dict = {
    2: [[0, 1], [1, 0]],
    3: [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]]
}

e_t_fp = '/home/data/ComplexQuestions/S-MART/compQ.ET.cPickle'
with open(e_t_fp, 'r') as br_et:
    e_t_dict = cPickle.load(br_et)
LogInfo.logs('%d E-->T loaded.', len(e_t_dict))


def get_entity_type(mid):
    return e_t_dict.get(mid, set([]))


"""
tag: the signature which represents the limitation to the entity/type/time/ord combination.
     Entity: E[type]; Type: t[Type]; Time: Tm; Ordinal: Ord
     Example: Epeople.person_Tlocation.location_Tm_Ord
     This tag stands for the schemas describing the location information of a person, with time & ordinal constraints.
label: the string representation of a schema (RM only mode)
       Usually the concatenation of path_lists.
Note: 
1. One schema may have several representations (we may switch the order of entity constraints).
   For the existing schemas, there are duplicates (different entities as focus),
   but all constraint entities are sorted by mid order.
   Therefore we need to permute these constraint entities, so that we won't miss anything.
2. All schemas linking the same label will have exactly the same structure. 
"""


def main(old_data_fp, new_data_fp):
    """
    :param old_data_fp: Old schema dataset
    :param new_data_fp: schema dataset after enrichment
    """
    assert new_data_fp != ''

    LogInfo.begin_track('Collecting all occurred schemas:')
    q_links_dict, q_schema_dict = collect_data(old_data_fp=old_data_fp)
    q_label_dict = {}           # q_idx --> [labels], recording all available schemas

    LogInfo.begin_track('Generating labels & tags for each schema ...')
    tag_label_dict = {}      # tag --> set([label])
    label_sc_dict = {}       # label --> unique_schema
    for q_idx, sc_list in q_schema_dict.items():
        q_label_dict[q_idx] = set([])
        for sc in sc_list:
            legal_sc_list = schema_rotating(sc)
            for legal_sc in legal_sc_list:
                label, tag = get_schema_label(legal_sc)
                q_label_dict[q_idx].add(label)
                if q_idx >= train_q_size:
                    """ ignore schemas at validation / test part. """
                    continue
                if sc.f1 < 1e-6:
                    """
                    For reducing the number of NA schemas,
                    we ignore a schema, if it CANNOT hit ANY gold answer in ANY questions. 
                    """
                    continue
                if label not in label_sc_dict:
                    label_sc_dict[label] = sc     # each repr maps to a unique schema
                    tag_label_dict.setdefault(tag, set([])).add(label)
    LogInfo.end_track()
    LogInfo.logs('In total collect %d schemas from %d questions.',
                 sum([len(v) for v in q_schema_dict.values()]), len(q_schema_dict))
    LogInfo.begin_track('Pick F1 > 0 schemas from T-%d: ', train_q_size)
    LogInfo.logs('%d tag --> [label] collected.', len(tag_label_dict))
    LogInfo.logs('%d unique label --> schema collected.', len(label_sc_dict))
    LogInfo.end_track()
    LogInfo.end_track()

    tag_set = set(tag_label_dict.keys())
    LogInfo.begin_track('Now start to traverse all constraint combinations for %d questions:', q_size)
    for q_idx in range(q_size):
        # if q_idx > 1:
        #     break
        LogInfo.begin_track('Entering Q-%04d: ', q_idx)
        gather_linkings = q_links_dict[q_idx]
        conflict_matrix = construct_conflict_matrix(gather_linkings)
        entity_linkings = filter(lambda x: x.category == 'Entity', gather_linkings)
        type_linkings = filter(lambda x: x.category == 'Type', gather_linkings)
        time_linkings = filter(lambda x: x.category == 'Time', gather_linkings)
        ordinal_linkings = filter(lambda x: x.category == 'Ordinal', gather_linkings)
        entity_linkings.sort(key=lambda _el: _el.value)

        """ Following: [(gl_data indices, corresponding tag elements, visit_arr)] """
        entity_available_combs = start_entity_search(
            entity_linkings=entity_linkings, conflict_matrix=conflict_matrix, tag_set=tag_set
        )
        type_available_combs = pick_one_search(
            spec_linkings=type_linkings, conflict_matrix=conflict_matrix,
            tag_set=tag_set, av_combs=entity_available_combs, spec='T'
        )
        time_available_combs = pick_one_search(
            spec_linkings=time_linkings, conflict_matrix=conflict_matrix, tag_set=tag_set,
            av_combs=entity_available_combs+type_available_combs, spec='Tm'
        )
        ordinal_available_combs = pick_one_search(
            spec_linkings=ordinal_linkings, conflict_matrix=conflict_matrix, tag_set=tag_set,
            av_combs=entity_available_combs+type_available_combs+time_available_combs, spec='Ord'
        )
        av_combs_groups = [entity_available_combs, type_available_combs,
                           time_available_combs, ordinal_available_combs]

        """
        Final: build NA schemas based on these available combinations.
        Do not output schemas which are already existed! 
        """
        LogInfo.logs('Writing for %d possible tag entries ...', len(entity_available_combs))
        build_na_schemas(q_idx=q_idx, old_data_fp=old_data_fp, new_data_fp=new_data_fp,
                         gather_linkings=gather_linkings, local_label_set=q_label_dict[q_idx],
                         av_combs_groups=av_combs_groups, tag_label_dict=tag_label_dict, label_sc_dict=label_sc_dict)

        LogInfo.end_track()

    LogInfo.end_track()


def collect_data(old_data_fp):
    q_links_dict = {}
    q_schema_dict = {}
    for q_idx in range(q_size):
        if q_idx % 100 == 0:
            LogInfo.logs('Current: %d / %d', q_idx, q_size)
        # if q_idx >= 100:
        #     break
        div = q_idx / 100
        sub_dir = '%d-%d' % (div*100, div*100+99)
        schema_fp = '%s/%s/%d_schema' % (old_data_fp, sub_dir, q_idx)
        link_fp = '%s/%s/%d_links' % (old_data_fp, sub_dir, q_idx)
        gather_linkings = []
        with codecs.open(link_fp, 'r', 'utf-8') as br:
            for line in br.readlines():
                tup_list = json.loads(line.strip())
                ld_dict = {k: v for k, v in tup_list}
                gather_linkings.append(LinkData(**ld_dict))
        strict_sc_list = []
        with codecs.open(schema_fp, 'r', 'utf-8') as br:
            lines = br.readlines()
            for ori_idx, line in enumerate(lines):
                sc = CompqSchema.read_schema_from_json(q_idx, json_line=line,
                                                       gather_linkings=gather_linkings,
                                                       use_ans_type_dist=False,
                                                       placeholder_policy='ActiveOnly')
                sc.ori_idx = ori_idx
                if schema_classification(sc) == 0:      # only pick strict schemas
                    strict_sc_list.append(sc)
        q_links_dict[q_idx] = gather_linkings
        q_schema_dict[q_idx] = strict_sc_list
    return q_links_dict, q_schema_dict


def schema_rotating(sc):
    """
    Given a schema, consider changing the order of main/constraint entities.
    In such case, one schema could have several legal representations.
    """
    assert isinstance(sc, CompqSchema)
    ret_sc_list = []
    """ 
    The rotation only works between entity constraints.
    Therefore, there's no need to worry about the direction adjustment of predicate sequences.
    """
    raw_paths = sc.raw_paths    # predicate paths follow the order: Main > Entity > Type > Time > Ordinal
    main_raw_paths = []
    ec_raw_paths = []
    rem_raw_paths = []
    for tup in raw_paths:
        if tup[0] == 'Main':
            main_raw_paths.append(tup)
        elif tup[0] == 'Entity':
            ec_raw_paths.append(tup)
        else:
            rem_raw_paths.append(tup)
    ec_len = len(ec_raw_paths)
    assert len(main_raw_paths) == 1
    assert 0 <= ec_len <= 3

    if ec_len <= 1:     # there's no need to rotate constraint entities.
        ret_sc_list.append(sc)
        return ret_sc_list

    """ Now enumerating each possible permutation """
    # LogInfo.logs('Need permutate for Q-%04d, line-%d.', sc.q_idx, sc.ori_idx+1)
    # if ec_len == 3:
    #     LogInfo.logs('Special 3 entities!!!')
    for perm in perm_list_dict[ec_len]:
        new_ec_raw_paths = []
        """ Permutation """
        for idx in perm:
            new_ec_raw_paths.append(ec_raw_paths[idx])
        new_raw_paths = main_raw_paths + new_ec_raw_paths + rem_raw_paths
        new_sc = copy.deepcopy(sc)
        new_sc.raw_paths = new_raw_paths
        ret_sc_list.append(new_sc)
    return ret_sc_list


def get_schema_label(sc):
    sc.construct_path_list()
    lb_list = []
    tag_list = []
    for raw_path, mid_seq in zip(sc.raw_paths, sc.path_list):
        lb_list.append('_'.join(mid_seq))
        path_cate, gl_data, _ = raw_path
        if path_cate in ('Main', 'Entity'):
            tp = get_domain(mid_seq[0])
            if path_cate == 'Main':
                tag_list.append('M:%s' % tp)
            else:
                tag_list.append('E:%s' % tp)
        elif path_cate == 'Type':
            tag_list.append('T:%s' % gl_data.value)
        elif path_cate == 'Time':
            tag_list.append('Tm')
        else:
            tag_list.append('Ord')
    label = '|'.join(lb_list)
    tag = '|'.join(tag_list)
    return label, tag


def start_entity_search(entity_linkings, conflict_matrix, tag_set):
    LogInfo.begin_track('Searching at M/E level ...')
    entity_available_combs = []     # the return value
    el_size = len(entity_linkings)
    gl_size = len(conflict_matrix)
    for mf_idx, main_focus in enumerate(entity_linkings):
        gl_pos = main_focus.gl_pos
        visit_arr = [0] * gl_size
        for conf_idx in conflict_matrix[gl_pos]:
            visit_arr[conf_idx] += 1
        gl_data_indices = [gl_pos]
        tag_elements = []       # create the initial state of search
        mid = main_focus.value
        type_list = get_entity_type(mid)
        for tp_idx, tp in enumerate(type_list):
            state_marker = ['M%d/%d-(t%d/%d)' % (mf_idx+1, el_size, tp_idx+1, len(type_list))]
            tag_elements.append('M:%s' % tp)
            entity_search_dfs(entity_linkings=entity_linkings,
                              conflict_matrix=conflict_matrix,
                              tag_set=tag_set,
                              cur_el_idx=-1,
                              gl_data_indices=gl_data_indices,
                              tag_elements=tag_elements,
                              visit_arr=visit_arr,
                              entity_available_combs=entity_available_combs,
                              state_marker=state_marker)
            del tag_elements[-1]
    LogInfo.end_track()
    return entity_available_combs


def entity_search_dfs(entity_linkings, conflict_matrix, tag_set,
                      cur_el_idx, gl_data_indices, tag_elements, visit_arr,
                      entity_available_combs, state_marker):
    if vb >= 2:
        LogInfo.begin_track('[%s]', '||'.join(state_marker))
    tag = '|'.join(tag_elements)
    if tag not in tag_set:      # no need to search further
        if vb >= 2:
            LogInfo.end_track()
        return

    if vb >= 1:
        LogInfo.logs('Available tag: [%s]', tag)
    av_comb = [list(gl_data_indices), list(tag_elements), list(visit_arr)]
    entity_available_combs.append(av_comb)      # keep a new combination

    el_size = len(entity_linkings)
    for nxt_el_idx in range(cur_el_idx+1, el_size):
        gl_pos = entity_linkings[nxt_el_idx].gl_pos
        mid = entity_linkings[nxt_el_idx].value
        if visit_arr[gl_pos] != 0:  # cannot be visited due to conflict
            continue
        for conf_idx in conflict_matrix[gl_pos]:  # ready to enter the next state
            visit_arr[conf_idx] += 1
        gl_data_indices.append(gl_pos)
        type_list = get_entity_type(mid)
        for tp_idx, tp in enumerate(type_list):
            state_marker.append('E%d/%d-(t%d/%d)' % (nxt_el_idx+1, el_size, tp_idx+1, len(type_list)))
            tag_elements.append('E:%s' % tp)
            entity_search_dfs(entity_linkings=entity_linkings,
                              conflict_matrix=conflict_matrix,
                              tag_set=tag_set,
                              cur_el_idx=nxt_el_idx,
                              gl_data_indices=gl_data_indices,
                              tag_elements=tag_elements,
                              visit_arr=visit_arr,
                              entity_available_combs=entity_available_combs,
                              state_marker=state_marker)
            del state_marker[-1]
            del tag_elements[-1]
        del gl_data_indices[-1]
        for conf_idx in conflict_matrix[gl_pos]:  # return back
            visit_arr[conf_idx] -= 1
    if vb >= 2:
        LogInfo.end_track()


def pick_one_search(spec_linkings, conflict_matrix, tag_set, av_combs, spec):
    """
    Work for T/Tm/Ord, since only one of them can be selected, no need for DFS.
    """
    assert spec in ('T', 'Tm', 'Ord')
    LogInfo.begin_track('Searching at %s level ...', spec)
    spec_available_combs = []
    for gl_data_indices, tag_elements, visit_arr in av_combs:
        for gl_data in spec_linkings:
            gl_pos = gl_data.gl_pos
            if visit_arr[gl_pos] != 0:  # cannot be visited due to conflict
                continue
            new_visit_arr = list(visit_arr)  # new state after applying types
            for conf_idx in conflict_matrix[gl_pos]:
                new_visit_arr[conf_idx] += 1
            if spec in ('Tm', 'Ord'):
                tag_elem = spec
            else:
                tag_elem = 'T:%s' % gl_data.value
            new_gl_data_indices = list(gl_data_indices) + [gl_pos]
            new_tag_elements = list(tag_elements) + [tag_elem]
            tag = '|'.join(new_tag_elements)
            if tag in tag_set:
                if vb >= 1:
                    LogInfo.logs(tag)
                spec_available_combs.append((new_gl_data_indices, new_tag_elements, new_visit_arr))
    LogInfo.end_track()
    return spec_available_combs


def build_na_schemas(q_idx, old_data_fp, new_data_fp,
                     gather_linkings, local_label_set,
                     av_combs_groups, tag_label_dict, label_sc_dict):
    div = q_idx / 100
    sub_dir = '%d-%d' % (div * 100, div * 100 + 99)
    old_schema_fp = '%s/%s/%d_schema' % (old_data_fp, sub_dir, q_idx)
    new_schema_fp = '%s/%s/%d_schema' % (new_data_fp, sub_dir, q_idx)
    os.system('cp %s %s' % (old_schema_fp, new_schema_fp))      # copy from old to new

    mark_list = ['Entity', 'Type', 'Time', 'Ordinal']
    with codecs.open(new_schema_fp, 'a', 'utf-8') as bw:
        for av_combs, mark in zip(av_combs_groups, mark_list):
            na_count = 0
            for gl_pos_indices, tag_elements, _ in av_combs:
                tag = '|'.join(tag_elements)
                for label in tag_label_dict[tag]:
                    if label in local_label_set:
                        continue        # won't add existing schemas
                    sc = label_sc_dict[label]
                    assert len(sc.raw_paths) == len(gl_pos_indices)
                    sc_info_dict = {'ans_size': 0, 'p': 0., 'r': 0., 'f1': 0.,
                                    'agg': False, 'hops': len(sc.raw_paths[0][-1])}
                    raw_path_list = []
                    for gl_pos, tag_elem, raw_path in zip(gl_pos_indices, tag_elements, sc.raw_paths):
                        category = 'Main'
                        if tag_elem.startswith('E:'):
                            category = 'Entity'
                        elif tag_elem.startswith('T:'):
                            category = 'Type'
                        elif tag_elem.startswith('Tm'):
                            category = 'Time'
                        elif tag_elem.startswith('Ord'):
                            category = 'Ordinal'
                        raw_path_list.append([category, gl_pos, gather_linkings[gl_pos].value, raw_path[-1]])
                    sc_info_dict['raw_paths'] = raw_path_list
                    bw.write(json.dumps(sc_info_dict) + '\n')
                    na_count += 1
            LogInfo.logs('[%7s]: Enrich [%4d] NA schemas from [%4d] tags.', mark, na_count, len(av_combs))


if __name__ == '__main__':
    LogInfo.begin_track('compQA.candgen_acl18.cand_enrich starts ... ')
    main(old_data_fp='runnings/candgen_CompQ/180308_SMART_Fhalf/data',
         new_data_fp='runnings/candgen_CompQ/180314_SMART_Fhalf_enrich/data')
    LogInfo.end_track()
