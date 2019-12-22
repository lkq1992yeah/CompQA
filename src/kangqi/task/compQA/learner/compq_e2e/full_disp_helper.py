from kangqi.util.LogUtil import LogInfo


def show_basic_full_info(sc, qa, schema_dataset):
    assert qa is not None
    assert schema_dataset is not None

    full_final_feats = sc.run_info['full_final_feats'].tolist()
    for category, gl_data, pred_seq in sc.raw_paths:
        LogInfo.logs('%s: link = [(#-%d) %s %s], pred_seq = %s',
                     category, gl_data.gl_pos, gl_data.comp, gl_data.value, pred_seq)
    show_str = '  '.join(['%6.3f' % x for x in full_final_feats])
    LogInfo.logs('rich_feats_concat = [%s]', show_str)
