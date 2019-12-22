"""
Goal: For the 800 testing questions, pick one of EL configuration, and pair it with all possible main predicates.
We are just want to find out some regularities.
"""
import codecs
import copy
import json
from ..candgen_acl18.global_linker import LinkData
from ..dataset.kq_schema import CompqSchema
from ..dataset.ds_helper.dataset_schema_reader import schema_classification
from kangqi.util.LogUtil import LogInfo


def pick_template_and_main_path(q_idx, sc_fp, gather_linkings):
    strict_sc_list = []
    with codecs.open(sc_fp, 'r', 'utf-8') as br:
        lines = br.readlines()
        for ori_idx, line in enumerate(lines):
            sc = CompqSchema.read_schema_from_json(q_idx, json_line=line,
                                                   gather_linkings=gather_linkings,
                                                   use_ans_type_dist=False,
                                                   placeholder_policy='ActiveOnly')
            sc.ori_idx = ori_idx
            if schema_classification(sc) == 0:
                strict_sc_list.append(sc)

    main_path_set = set([])
    best_sc = None
    best_f1 = -1.
    best_edges = -1
    for sc in strict_sc_list:
        main_path_set.add('\t'.join(sc.main_pred_seq))
        if sc.f1 > best_f1 or (sc.f1 == best_f1 and len(sc.raw_paths) > best_edges):
            best_edges = len(sc.raw_paths)
            best_f1 = sc.f1
            best_sc = sc
    return best_sc, main_path_set


def save_fake_schemas(temp_sc, main_path_list, save_fp):
    if temp_sc is None:
        bw = codecs.open(save_fp, 'w', 'utf-8')
        bw.close()
        return

    gl_data = None
    for cate, gl, pred_seq in temp_sc.raw_paths:
        if cate == 'Main':
            gl_data = gl
            break
    assert isinstance(gl_data, LinkData)
    tmp_dict = {'ans_size': 1, 'p': 0., 'r': 0., 'f1': 0., 'agg': False}

    with codecs.open(save_fp, 'w', 'utf-8') as bw:
        for main_path in main_path_list:
            main_pred_seq = main_path.split('\t')
            use_dict = copy.deepcopy(tmp_dict)
            use_dict['raw_paths'] = [['Main', gl_data.gl_pos, gl_data.value, main_pred_seq]]
            bw.write(json.dumps(use_dict) + '\n')


def main():
    old_data_fp = 'runnings/candgen_CompQ/180314_SMART_Fhalf_enrich/data'
    new_data_fp = 'runnings/candgen_CompQ/180315_SMART_FAKE/data'
    main_path_dict = {}
    temp_sc_list = []

    LogInfo.begin_track('Scanning standard schemas from [%s] ...', old_data_fp)
    for q_idx in range(2100):
        if q_idx % 100 == 0:
            LogInfo.logs('Current: %d, collected %d predicates.', q_idx, len(main_path_dict))
        div = q_idx / 100
        sub_dir = '%d-%d' % (div * 100, div * 100 + 99)
        old_sc_fp = '%s/%s/%d_schema' % (old_data_fp, sub_dir, q_idx)
        link_fp = '%s/%s/%d_links' % (old_data_fp, sub_dir, q_idx)
        gather_linkings = []
        with codecs.open(link_fp, 'r', 'utf-8') as br:
            for gl_line in br.readlines():
                tup_list = json.loads(gl_line.strip())
                ld_dict = {k: v for k, v in tup_list}
                gather_linkings.append(LinkData(**ld_dict))
        temp_sc, local_main_path_set = pick_template_and_main_path(q_idx, old_sc_fp, gather_linkings)
        temp_sc_list.append(temp_sc)
        for path in local_main_path_set:
            main_path_dict[path] = main_path_dict.get(path, 0) + 1
    LogInfo.end_track()
    LogInfo.logs('%d main predicates collected.', len(main_path_dict))

    main_path_tups = sorted(main_path_dict.items(), key=lambda _tup: _tup[-1], reverse=True)
    with codecs.open(new_data_fp + '/main_predicates.txt', 'w', 'utf-8') as bw:
        for path, freq in main_path_tups:
            bw.write('%4d\t%s\n' % (freq, path))

    pick_path_list = []
    for path, freq in main_path_dict.items():
        if freq >= 20 or not (path.startswith('user.') or path.startswith('base.')):
            pick_path_list.append(path)
    LogInfo.logs('%d paths retained with frequency >= 20, or non-trivial predicates.', len(pick_path_list))

    LogInfo.begin_track('Saving fake data ...')
    for q_idx in range(2100):
        if q_idx % 100 == 0:
            LogInfo.logs('Current: %d.', q_idx)
        div = q_idx / 100
        sub_dir = '%d-%d' % (div * 100, div * 100 + 99)
        new_sc_fp = '%s/%s/%d_schema' % (new_data_fp, sub_dir, q_idx)
        save_fake_schemas(temp_sc_list[q_idx], pick_path_list, new_sc_fp)
    LogInfo.end_track()


if __name__ == '__main__':
    main()
