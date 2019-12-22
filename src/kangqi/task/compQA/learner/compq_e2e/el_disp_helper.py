from kangqi.util.LogUtil import LogInfo


def show_el_detail_without_type(sc, qa, schema_dataset):
    # el_final_feats, el_raw_score = [
    #     sc.run_info[k].tolist() for k in ('el_final_feats', 'el_score')
    # ]
    assert qa is not None
    assert schema_dataset is not None

    el_final_feats = sc.run_info['el_final_feats'].tolist()
    LogInfo.logs('el_final_feats = [%s]', ' '.join(['%6.3f' % x for x in el_final_feats]))

    el_mask = sc.input_np_dict['el_mask']
    path_size = sc.input_np_dict['path_size']
    el_indv_feats = sc.input_np_dict['el_indv_feats']
    gl_list = []
    for category, gl_data, pred_seq in sc.raw_paths:
        gl_list.append(gl_data)
    assert path_size == len(gl_list)

    for el_idx in range(path_size):
        msk = el_mask[el_idx]
        gl_data = gl_list[el_idx]
        LogInfo.begin_track('Entity %d / %d:', el_idx + 1, path_size)

        LogInfo.logs(gl_data.display())
        if msk == 0.:
            LogInfo.logs('[Mask = 0, IGNORED.]')
        else:
            local_feats = el_indv_feats[el_idx]
            LogInfo.logs('local_feats = [%s]', '  '.join(['%6.3f' % x for x in local_feats]))
        LogInfo.end_track()
