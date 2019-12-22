import pyltr
import numpy as np

from ..compq_e2e_old.full_evaluator import show_overall_detail

from kangqi.util.LogUtil import LogInfo


def ltr_whole_process(pyltr_data_list, eval_dl_list, output_dir):
    """
    :param pyltr_data_list: a list of T/v/t q_cands_dict
    :param eval_dl_list: T/v/t data loaders
    :param output_dir: the working dir
    """
    """ Step 1: Convert <q, [cand]> into sorted tuples """
    q_cands_tup_lists = []
    for q_cands_dict in pyltr_data_list:
        q_cands_tup_list = q_cands_dict.items()
        q_cands_tup_list.sort(key=lambda _tup: _tup[0])  # sort by q_idx ASC
        for _, cands in q_cands_tup_list:
            cands.sort(key=lambda _sc: _sc.f1)  # sort by f1 ASC
        q_cands_tup_lists.append(q_cands_tup_list)

    """ Step 2: Prepare pyltr format data """
    ltr_format_tups = []
    for q_cands_tup_list, mark in zip(q_cands_tup_lists, ['train', 'valid', 'test']):
        opt_fp = '%s/detail/%s.ltr.txt' % (output_dir, mark)
        schema_to_pyltr(q_cands_tup_list=q_cands_tup_list, opt_fp=opt_fp)
        with open(opt_fp, 'r') as br:
            ltr_format_tups.append(pyltr.data.letor.read_dataset(br))
        LogInfo.logs('[%s] ltr data formatted.', mark)

    """ Step 3: LTR running (optm & eval) """
    metric = pyltr.metrics.NDCG(k=10)
    train_x, train_y, train_qids, _ = ltr_format_tups[0]       # T/v/t
    valid_x, valid_y, valid_qids, _ = ltr_format_tups[1]
    test_x, test_y, test_qids, _ = ltr_format_tups[2]

    n_estimators = 100
    monitor = pyltr.models.monitors.ValidationMonitor(
        valid_x, valid_y, valid_qids, metric=metric, stop_after=int(0.2*n_estimators))
    model = pyltr.models.LambdaMART(
        metric=metric,
        n_estimators=n_estimators,
        learning_rate=0.02,
        max_features=0.5,
        query_subsample=0.5,
        max_leaf_nodes=10,
        min_samples_leaf=1,
        verbose=1,
    )
    model.fit(train_x, train_y, train_qids, monitor=monitor)
    LogInfo.logs('LTR: fit complete.')
    train_pred = model.predict(train_x)
    LogInfo.logs('LTR: predict %d score of train schemas.', len(train_pred))
    valid_pred = model.predict(valid_x)
    LogInfo.logs('LTR: predict %d score of valid schemas.', len(valid_pred))
    test_pred = model.predict(test_x)
    LogInfo.logs('LTR: predict %d score of test schemas.', len(test_pred))

    """ Step 4: get final F1 score """
    final_metric_list = []
    for q_cands_tup_list, predict_score_list, eval_dl, mark in zip(
        q_cands_tup_lists, [train_pred, valid_pred, test_pred], eval_dl_list, 'Tvt'
    ):
        detail_fp = '%s/detail/ltr.%s.tmp' % (output_dir, mark)
        eval_metric = post_process(q_cands_tup_list=q_cands_tup_list,
                                   predict_score_list=predict_score_list,
                                   eval_dl=eval_dl, detail_fp=detail_fp)
        final_metric_list.append(eval_metric)
    return final_metric_list


def schema_to_pyltr(q_cands_tup_list, opt_fp):
    """
    [(q, [cand])] into ltr-format file
    :param q_cands_tup_list: sort by q_idx ASC, and for each candidate list, sort by F1 ASC.
    :param opt_fp: target file
    """
    keep_size = 0
    with open(opt_fp, 'w') as bw:
        for q_idx, cand_list in q_cands_tup_list:
            lvl = 0
            cur_f1 = 0.
            for sc in cand_list:
                if sc.f1 > cur_f1 + 1e-6:
                    lvl += 1
                    cur_f1 = sc.f1
                rich_feats = sc.run_info['rich_feats_concat']
                write_str = '%d qid:%d' % (lvl, q_idx)
                for f_idx, feat_val in enumerate(rich_feats):
                    write_str += ' %d:%.6f' % (f_idx+1, feat_val)
                write_str += ' # ori_idx=%d F1=%.6f' % (sc.ori_idx, sc.f1)
                bw.write(write_str + '\n')
                keep_size += 1
    LogInfo.logs('%d schema-ltr info saved into file.', keep_size)


def post_process(q_cands_tup_list, predict_score_list, eval_dl, detail_fp):
    scan_idx = 0
    for q_idx, cand_list in q_cands_tup_list:
        for cand in cand_list:
            cand.run_info['ltr_score'] = predict_score_list[scan_idx]
            scan_idx += 1
    assert len(predict_score_list) == scan_idx

    f1_list = []
    for q_idx, cand_list in q_cands_tup_list:
        cand_list.sort(key=lambda x: x.run_info['ltr_score'], reverse=True)  # sort by score DESC
        if len(cand_list) == 0:
            f1_list.append(0.)
        else:
            f1_list.append(cand_list[0].f1)
    LogInfo.logs('[ltr-%s] Predict %d out of %d questions.', eval_dl.mode, len(f1_list), eval_dl.total_questions)
    ret_metric = np.sum(f1_list).astype('float32') / eval_dl.total_questions

    if detail_fp is not None:
        schema_dataset = eval_dl.schema_dataset
        bw = open(detail_fp, 'w')
        LogInfo.redirect(bw)
        np.set_printoptions(threshold=np.nan)
        LogInfo.logs('Avg_f1 = %.6f', ret_metric)

        for q_idx, cand_list in q_cands_tup_list:
            qa = schema_dataset.qa_list[q_idx]
            q = qa['utterance']
            LogInfo.begin_track('Q-%04d [%s]:', q_idx, q.encode('utf-8'))
            best_label_f1 = np.max([sc.f1 for sc in cand_list])
            best_label_f1 = max(best_label_f1, 0.000001)
            for rank, sc in enumerate(cand_list):
                if rank < 20 or sc.f1 == best_label_f1:
                    LogInfo.begin_track('#-%04d [F1 = %.6f] [row_in_file = %d]', rank + 1, sc.f1, sc.ori_idx)
                    LogInfo.logs('ltr_score: %.6f', sc.run_info['ltr_score'])
                    show_overall_detail(sc)
                    LogInfo.end_track()
            LogInfo.end_track()
        LogInfo.logs('Avg_f1 = %.6f', ret_metric)
        np.set_printoptions()  # reset output format
        LogInfo.stop_redirect()
        bw.close()

    LogInfo.logs('[ltr] %s_F1 = %.6f', eval_dl.mode, ret_metric)
    return ret_metric
