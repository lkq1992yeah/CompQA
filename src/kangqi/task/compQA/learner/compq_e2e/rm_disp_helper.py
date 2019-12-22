from kangqi.util.LogUtil import LogInfo


def show_basic_rm_info(sc, qa, schema_dataset):
    assert qa is not None
    assert schema_dataset is not None

    rm_final_feats = sc.run_info['rm_final_feats'].tolist()
    LogInfo.logs('rm_final_feats = [%s]', ' '.join(['%6.3f' % x for x in rm_final_feats]))

    show_path_list = sc.path_list
    show_path_words_list = sc.path_words_list       # generated in input_feat_gen_helper
    show_path_size = len(show_path_list)

    """ show the detail of each path one by one """
    for path_idx in range(show_path_size):
        LogInfo.begin_track('Showing path-%d / %d:', path_idx + 1, show_path_size)
        LogInfo.logs('Path: [%s]', '-->'.join(show_path_list[path_idx]).encode('utf-8'))
        LogInfo.logs('Path-Word: [%s]', ' | '.join(show_path_words_list[path_idx]).encode('utf-8'))
        LogInfo.end_track()
