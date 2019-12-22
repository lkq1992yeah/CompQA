"""
Author: Kangqi Luo
Goal: A general evaluator, compatible for multiple models.
"""

import tensorflow as tf
import numpy as np

from .base_model import BaseModel
from ..dataset.base_dataloader import DataLoader

from kangqi.util.LogUtil import LogInfo


# The main code copies from q_sc_dyn_eval_model
def general_evaluate(eval_model, data_loader, epoch_idx, ob_batch_num=20,
                     detail_fp=None, result_fp=None, summary_writer=None):
    """
    First evaluate all batches, and then count the final score via out-of-TF codes
    """
    assert isinstance(eval_model, BaseModel)
    if data_loader is None or len(data_loader) == 0:  # empty eval data
        return 0.

    """ Step 1: Run each batch, calculate case-specific results """

    eval_name_list = [tup[0] for tup in eval_model.eval_output_tf_tup_list]
    eval_tensor_list = [tup[1] for tup in eval_model.eval_output_tf_tup_list]
    assert isinstance(data_loader, DataLoader)
    eval_model.prepare_data(data_loader=data_loader)
    run_options = run_metadata = None
    if summary_writer is not None:
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
    scan_size = 0
    ret_q_score_dict = {}  # <q, [(schema, score)]>
    for batch_idx in range(data_loader.n_batch):
        local_data_list, local_indices = data_loader.get_next_batch()
        local_size = len(local_data_list[0])  # the first dimension is always batch size
        fd = {input_tf: local_data for input_tf, local_data in zip(eval_model.eval_input_tf_list, local_data_list)}

        # Dynamically evaluate all the concerned tensors, no hard coding any more
        eval_result_list = eval_model.sess.run(eval_tensor_list, feed_dict=fd,
                                               options=run_options, run_metadata=run_metadata)
        scan_size += local_size
        if (batch_idx + 1) % ob_batch_num == 0:
            LogInfo.logs('[eval-%s-B%d/%d] scanned = %d/%d',
                         data_loader.mode, batch_idx + 1,
                         data_loader.n_batch, scan_size, len(data_loader))
        if summary_writer is not None and batch_idx == 0:
            summary_writer.add_run_metadata(run_metadata, 'epoch-%d' % epoch_idx)

        for pos in range(len(local_indices)):   # enumerate each data in the batch, and record corresponding scores
            local_idx = local_indices[pos]
            q_idx, cand = data_loader.cand_tup_list[local_idx]
            cand.run_info = {}
            for eval_name, eval_result in zip(eval_name_list, eval_result_list):
                cur_result = eval_result[pos]
                cand.run_info[eval_name] = cur_result
            ret_q_score_dict.setdefault(q_idx, []).append(cand)

    """ Step 2: After scanning all the batch, now count the final F1 result """

    f1_list = []
    for q_idx, score_list in ret_q_score_dict.items():
        score_list.sort(key=lambda x: x.run_info['score'], reverse=True)  # sort by score DESC
        if len(score_list) == 0:
            f1_list.append(0.)
        else:
            f1_list.append(score_list[0].f1)  # pick the f1 of the highest scored schema
    LogInfo.logs('Predict %d out of %d questions.', len(f1_list), data_loader.question_size)
    ret_metric = np.sum(f1_list) / data_loader.question_size

    """ Step 3: Got non-case-specific results, that are, parameters """

    param_name_list = [tup[0] for tup in eval_model.show_param_tf_tup_list]
    param_tensor_list = [tup[1] for tup in eval_model.show_param_tf_tup_list]
    param_result_list = eval_model.sess.run(param_tensor_list)  # don't need any feeds, since we focus on parameters

    """ Step 4: Save detail information: Schema Results & Parameters """

    if result_fp is not None:
        srt_q_idx_list = sorted(ret_q_score_dict.keys())
        with open(result_fp, 'w') as bw:        # write question --> selected schema
            for q_idx in srt_q_idx_list:
                srt_list = ret_q_score_dict[q_idx]
                ori_idx = -1
                f1 = 0.
                if len(srt_list) > 0:
                    best_sc = srt_list[0]
                    ori_idx = best_sc.ori_idx
                    f1 = best_sc.f1
                bw.write('%d\t%d\t%.6f\n' % (q_idx, ori_idx, f1))

    if detail_fp is not None:
        bw = open(detail_fp, 'w')
        LogInfo.redirect(bw)
        np.set_printoptions(threshold=np.nan)

        LogInfo.logs('Epoch-%d: avg_f1 = %.6f', epoch_idx, ret_metric)
        srt_q_idx_list = sorted(ret_q_score_dict.keys())
        for q_idx in srt_q_idx_list:
            q = data_loader.dataset.qa_list[q_idx]['utterance']
            LogInfo.begin_track('Q-%04d [%s]:', q_idx, q.encode('utf-8'))

            srt_list = ret_q_score_dict[q_idx]  # already sorted
            best_label_f1 = np.max([sc.f1 for sc in srt_list])
            best_label_f1 = max(best_label_f1, 0.000001)
            for rank, sc in enumerate(srt_list):
                if rank < 20 or sc.f1 == best_label_f1:
                    LogInfo.begin_track('#-%04d [F1 = %.6f] [row_in_file = %d]', rank+1, sc.f1, sc.ori_idx)
                    for eval_name in eval_name_list:
                        val = sc.run_info[eval_name]
                        if isinstance(val, float) or isinstance(val, np.float32):   # displaying a single float value
                            LogInfo.logs('%16s: %9.6f', eval_name, sc.run_info[eval_name])
                        elif isinstance(val, np.ndarray):
                            if 'att_mat' in eval_name:  # displaying several attention matrices
                                disp_mat_list = att_matrix_auto_crop(att_mat=val)
                                for idx, crop_mat in enumerate(disp_mat_list):
                                    if np.prod(crop_mat.shape) == 0:
                                        continue        # skip attention matrix of padding paths
                                    LogInfo.begin_track('%16s: shape = %s',
                                                        '%s-%d' % (eval_name, idx),
                                                        crop_mat.shape)
                                    LogInfo.logs(crop_mat)      # output attention matrix one-by-one
                                    LogInfo.end_track()
                            else:                       # displaying a single ndarray
                                LogInfo.begin_track('%16s: shape = %s', eval_name, val.shape)
                                LogInfo.logs(val)
                                LogInfo.end_track()
                        else:
                            LogInfo.logs('%16s: illegal value %s', eval_name, type(val))
                    for path_idx, path in enumerate(sc.path_list):
                        LogInfo.logs('Path-%d: [%s]', path_idx, '-->'.join(path))
                    for path_idx, words in enumerate(sc.path_words_list):
                        LogInfo.logs('Path-Word-%d: [%s]', path_idx, ' | '.join(words).encode('utf-8'))
                    LogInfo.end_track()
            LogInfo.end_track()
        LogInfo.logs('Epoch-%d: avg_f1 = %.6f', epoch_idx, ret_metric)

        LogInfo.logs('=================== Parameters ===================')
        for param_name, param_result in zip(param_name_list, param_result_list):
            LogInfo.begin_track('%s: shape = %s ', param_name, param_result.shape)
            LogInfo.logs(param_result)
            LogInfo.end_track()

        np.set_printoptions()       # reset output format
        LogInfo.stop_redirect()
        bw.close()
    return ret_metric


def att_matrix_auto_crop(att_mat):
    """
    Given an attention matrix, automatically crop it.
    :param att_mat: 3-dim ndarray (sc_max_len, q_max_len, item_max_len)
    :return: a list of cropped matrices
    """
    assert len(att_mat.shape) == 3
    sc_max_len, max_r, max_c = att_mat.shape
    ret_mat_list = []
    for idx in range(sc_max_len):
        mat = att_mat[idx]
        row_sum = np.sum(mat ** 2, axis=1)      # (max_r, )
        col_sum = np.sum(mat ** 2, axis=0)  # (max_c, )
        use_r = max_r
        use_c = max_c
        for r in range(max_r):
            if row_sum[r] < 1e-6:
                use_r = r
                break
        for c in range(max_c):
            if col_sum[c] < 1e-6:
                use_c = c
                break
        ret_mat_list.append(mat[:use_r, :use_c])
    return ret_mat_list
