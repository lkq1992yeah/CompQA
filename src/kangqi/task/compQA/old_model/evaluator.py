# -*- coding: utf-8 -*-

#==============================================================================
# Author: Kangqi Luo
# Goal: Evaluate End-to-End.
#==============================================================================

import tensorflow as tf
import numpy as np
import sys
import os
import shutil

from .model_ours import Model_Ours
from ..data_prepare.data_saving import load_numpy_input_with_names

from kangqi.util.tf.learn.default_learner import DefaultLearner

from kangqi.util.LogUtil import LogInfo
from kangqi.util.config import load_configs


# Convert the raw np_list (read from file) to the numpy arrays
# that we feed into tensors.
# Therefore we must ensure that the output np_list is in consistent
# with input_tf_list defined in the evaluator.
def load_data_and_reformulate(pydump_fp):
    np_list = load_numpy_input_with_names(pydump_fp)
    # ==== 140419: The np list contains the following items: ==== #
    q_tensor3, el_tensor3, path_tensor4, \
    score_tensor3, mask_matrix, \
    ord_x_matrix, ord_pred_tensor3, ord_op_tensor3, \
    ord_obj_tensor3, ord_mask_matrix = np_list
    # =========================================================== #
    size = q_tensor3.shape[0]
    LogInfo.logs('QA size = %d.', size)

    gold_matrix = score_tensor3[:, :, 2]    # just use F1
    best_matrix = np.zeros(shape=gold_matrix.shape, dtype='float32')
    best_matrix[:, 0] = 1.0
    # we've ranked all schemas, so the first candidate must be the best

    opt_np_list = []
    opt_np_list += [
            q_tensor3, path_tensor4, el_tensor3,
            gold_matrix, best_matrix, mask_matrix
    ]   # corresponding to basic_tf_list
    opt_np_list += [
            ord_x_matrix, ord_pred_tensor3,
            ord_op_tensor3, ord_obj_tensor3, ord_mask_matrix
    ]   # corresponding to ordinal_tf_list
    return opt_np_list

def build_Tvt_data(train_pydump_fp, test_pydump_fp, batch_size,
                   T_size, v_size, t_size):
    info_tup_list = []  # [(np_list, indices, n_batch)]

    train_np_list = load_data_and_reformulate(train_pydump_fp)
    test_np_list = load_data_and_reformulate(test_pydump_fp)
    LogInfo.logs('T/v/t data load & reformulate complete.')

    train_len = train_np_list[0].shape[0]
    test_len = test_np_list[0].shape[0]
    assert T_size + v_size == train_len
    assert t_size == test_len

    T_indices = range(T_size)
    v_indices = range(T_size, T_size + v_size)
    t_indices = range(t_size)
    for Tvt, np_list, indices in zip(
            ['T', 'v', 't'],
            [train_np_list, train_np_list, test_np_list],
            [T_indices, v_indices, t_indices]):
        n_batch = (len(indices) - 1) / batch_size + 1
        info_tup = (np_list, indices, n_batch)
        info_tup_list.append(info_tup)
    return info_tup_list


if __name__ == '__main__':
    LogInfo.begin_track('[kangqi.task.compQA.model.evaluator] starts ... ')
    root_path = '/home/kangqi/workspace/PythonProject'
    force_train = False
    try_dir = sys.argv[1]

    # ==== Preprocess: Retrieve Necessary Parameters from Config ==== #

    config_fp = '%s/runnings/compQA/input/%s/param_config' %(root_path, try_dir)
    config_dict = load_configs(config_fp)

    exp_dir = '%s/%s' %(root_path, config_dict['exp_dir'])
    if os.path.exists(exp_dir) and force_train == False:
        LogInfo.logs('Experiment results already produced in [%s].', exp_dir)
        LogInfo.end_track('Swith force_train = True if you really want to overwrite previous results.')
        exit()
    elif not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    os.system('cp %s %s' %(config_fp, exp_dir + '/param_config.tmp'))

    batch_size = int(config_dict['batch_size'])
    T_size = int(config_dict['T_size'])
    v_size = int(config_dict['v_size'])
    t_size = int(config_dict['t_size'])
    Tv_size = T_size + v_size

    n_el = int(config_dict['n_el'])
    sent_max_len = int(config_dict['sent_max_len'])
    path_max_len = int(config_dict['path_max_len'])
    n_word_emb = int(config_dict['n_word_emb'])
    n_path_emb = int(config_dict['n_path_emb'])
    n_window = int(config_dict['n_window'])
    n_cnn_hidden = int(config_dict['n_cnn_hidden'])
    n_rnn_hidden = int(config_dict['n_rnn_hidden'])
    n_feat_sk = int(config_dict['n_feat_sk'])
    n_compat_hidden = int(config_dict['n_compat_hidden'])

    loss_func = config_dict['loss_func']
    margin = float(config_dict['margin'])
    learning_rate = float(config_dict['learning_rate'])
    PN = int(config_dict['PN'])
    sc_mode = config_dict['sc_mode']

    activation = None
    act_str = config_dict['activation']
    if act_str == 'relu':
        activation = tf.nn.relu
    elif act_str == 'tanh':
        activation = tf.nn.tanh
    elif act_str == 'sigmoid':
        activation = tf.nn.sigmoid

    n_epoch = int(config_dict['n_epoch'])
    max_patience = int(config_dict['max_patience'])



    # ================ S1: Building the Model ================ #

    LogInfo.begin_track('======== S1: Building the model ========')
    q_tf = tf.placeholder(tf.float32, [None, sent_max_len, n_word_emb])
    sk_tf = tf.placeholder(tf.float32, [None, PN, path_max_len, n_path_emb])
    el_tf = tf.placeholder(tf.float32, [None, PN, n_el])
    gold_tf = tf.placeholder(tf.float32, [None, PN])
    best_tf = tf.placeholder(tf.float32, [None, PN])
    mask_tf = tf.placeholder(tf.float32, [None, PN])
    basic_tf_list = [q_tf, sk_tf, el_tf, gold_tf, best_tf, mask_tf]

    ord_x_tf = tf.placeholder(tf.int32, [None, PN])
    ord_pred_tf = tf.placeholder(tf.float32, [None, PN, n_path_emb])
    ord_op_tf = tf.placeholder(tf.float32, [None, PN, 2])
    ord_obj_tf = tf.placeholder(tf.float32, [None, PN, n_word_emb])
    ord_mask_tf = tf.placeholder(tf.float32, [None, PN])
    ordinal_tf_list = [ord_x_tf, ord_pred_tf, ord_op_tf, ord_obj_tf, ord_mask_tf]
    LogInfo.logs('* Tensor placeholder defined.')
    input_tf_list = basic_tf_list + ordinal_tf_list

    model = Model_Ours(n_el=n_el,
        sent_max_len=sent_max_len, n_word_emb=n_word_emb,
        n_window=n_window, n_cnn_hidden=n_cnn_hidden,
        path_max_len=path_max_len, n_path_emb=n_path_emb,
        n_rnn_hidden=n_rnn_hidden,
        n_feat_sk=n_feat_sk, n_compat_hidden=n_compat_hidden,
        activation=activation,
        loss_func=loss_func, margin=margin,
        learning_rate=learning_rate,
        PN=PN, batch_size=batch_size,
        sc_mode=sc_mode)
    train_step, final_loss_tf, score_tf, predict_result_tf = \
            model.build(q_tf, sk_tf, el_tf, gold_tf, best_tf, mask_tf, ordinal_tf_list)
    LogInfo.end_track()             # End of Model Building



    # ================ S2: Loading Real Data ================ #

    LogInfo.begin_track('======== S2: Loading T/v/t Data ========')
    train_pydump_fp = '%s/runnings/compQA/input/%s/T_input.pydump' %(root_path, try_dir)
    test_pydump_fp = '%s/runnings/compQA/input/%s/t_input.pydump' %(root_path, try_dir)
    info_tup_list = build_Tvt_data(
            train_pydump_fp, test_pydump_fp,
            batch_size, T_size, v_size, t_size)
    LogInfo.logs('T/v/t size = %s', [len(tup[1]) for tup in info_tup_list])

    train_np_list = info_tup_list[0][0]
    test_np_list = info_tup_list[2][0]
    assert len(input_tf_list) == len(train_np_list) == len(test_np_list)

    q_tensor3, path_tensor4, el_tensor3, \
    gold_matrix, best_matrix, mask_matrix = train_np_list[0 : 6]
    assert (Tv_size, sent_max_len, n_word_emb) == q_tensor3.shape
    assert (Tv_size, PN, n_el) == el_tensor3.shape
    assert (Tv_size, PN, path_max_len, n_path_emb) == path_tensor4.shape
    assert (Tv_size, PN) == mask_matrix.shape == best_matrix.shape == gold_matrix.shape
    LogInfo.logs('Passed assertion test.')
    LogInfo.end_track()



    # ================ S3: Train / Valid / Test ================ #

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
#    LogInfo.begin_track('Showing all parameters: ')
#    for v in tf.global_variables():
#        LogInfo.logs('%s: %s', v.name, v.get_shape().as_list())
#    LogInfo.end_track()

    log_fp = exp_dir + '/log.trend'
    weight_fp = exp_dir + '/weights'
    log_header = '%-8s\t%-8s\t%-8s\t%-8s\t%-8s\t%-8s' %(
        '#', 'T_loss', 'T_avg_F1', 'v_avg_F1', 't_avg_F1', 'Status')
    learner = DefaultLearner(
            n_epoch=n_epoch,
            max_patience=max_patience,
            batch_size=batch_size,
            info_tup_list=info_tup_list,
            input_tf_list=input_tf_list,
            log_fp=log_fp,
            sess=sess,
            train_step=train_step,
            predict_tf=predict_result_tf,
            final_loss_tf=final_loss_tf)
    learner.learn(log_header=log_header)
    # We just use default functions

    # ================ End of Learning Phase ================ #

    if os.path.isfile(exp_dir + '/param_config.tmp'):
        shutil.move(exp_dir + '/param_config.tmp', exp_dir + '/param_config')
    LogInfo.end_track()     # End of program