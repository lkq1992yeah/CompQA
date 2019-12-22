# -*- coding: utf-8 -*-

#==============================================================================
# Author: Kangqi Luo
# Goal: Get hidden features, build loss function, then perform training / testing
# We have mainly 2 loss functions. HingeLoss, or Rank-based (RankNet, LambdaRank)
#==============================================================================

import sys
import numpy as np
import tensorflow as tf

from .feature_ours import Hidden_Feature_Ours
from ..data_prepare.data_saving import load_numpy_input

from kangqi.util.tf.hinge_loss import HingeLoss
from kangqi.util.tf.ranknet.ranknet_improved import RankNet
from kangqi.util.config import load_configs
from kangqi.util.LogUtil import LogInfo


class Model_Ours(object):

    def __init__(self, n_el,
                 sent_max_len, n_word_emb, n_window, n_cnn_hidden,
                 path_max_len, n_path_emb, n_rnn_hidden,
                 n_feat_sk, n_compat_hidden,
                 activation, loss_func, margin, learning_rate,
                 PN, batch_size, sc_mode):
        self.loss_func = loss_func

        assert self.loss_func in ('HingeLoss', 'RankNet')

        self.n_el = n_el
        self.sent_max_len = sent_max_len
        self.n_word_emb = n_word_emb
        self.path_max_len = path_max_len
        self.n_path_emb = n_path_emb

        self.margin = margin
        self.PN = PN
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.feature_ours = Hidden_Feature_Ours(n_el=n_el,
                sent_max_len=sent_max_len, n_word_emb=n_word_emb,
                n_window=n_window, n_cnn_hidden=n_cnn_hidden,
                path_max_len=path_max_len, n_path_emb=n_path_emb,
                n_rnn_hidden=n_rnn_hidden,
                n_feat_sk=n_feat_sk, n_compat_hidden=n_compat_hidden,
                activation=activation, sc_mode=sc_mode)


    # Note: data_size = ds = batch * PN
    # q_tf: (batch, sent_max_len, n_word_emb)
    # sk_tf: (batch, PN, path_max_len, n_path_emb)
    # el_tf: (batch, PN, n_el)    # just an entity linking score (discrete version)
    # gold_tf: (batch, PN), store the gold loss (distance) of each candidate schema (We use F1 currently.)
    # best_tf: (batch, PN), store 0/1 indicating whether this candidate is the best one
    # mask_tf: (batch, PN), indicating the effective candidates in each QA data.
    # ** 170419: we changed lots of (ds, xxx) into (batch, PN, xxx) format,
    # because it simplifies the general codes in evaluator.py

    # Note: In the current setting, the best one is always the first schema in the training data.
    # That is, best_tf[:, 0] is always equal to 1 in the training data.
    def build(self, q_tf, sk_tf, el_tf, gold_tf, best_tf, mask_tf, ordinal_tf_list):
        # First: copy q_tf into (batch, PN, sent_max_len, n_word_emb),
        # and reshape into (batch * PN, sent_max_len, n_word_emb)
        q_tf = tf.concat([q_tf] * self.PN, axis=1)      # (batch, PN * sent_max_len, n_word_emb)
        q_tf = tf.reshape(q_tf, [-1, self.sent_max_len, self.n_word_emb]) # (ds, sent_max_len, n_word_emb)

        sk_tf = tf.reshape(sk_tf, [-1, self.path_max_len, self.n_path_emb]) # (ds, path_max_len, n_path_emb)
        el_tf = tf.reshape(el_tf, [-1, self.n_el]) # (ds, n_el)

        # Unfold, reshape and fold back.
        ord_x_tf, ord_pred_tf, ord_op_tf, ord_obj_tf, ord_mask_tf = ordinal_tf_list
        ord_x_tf = tf.reshape(ord_x_tf, [-1])
        ord_pred_tf = tf.reshape(ord_pred_tf, [-1, self.n_path_emb])
        ord_op_tf = tf.reshape(ord_op_tf, [-1, 2])
        ord_obj_tf = tf.reshape(ord_obj_tf, [-1, self.n_word_emb])
        ord_mask_tf = tf.reshape(ord_mask_tf, [-1])
        ordinal_tf_list = [ord_x_tf, ord_pred_tf, ord_op_tf, ord_obj_tf, ord_mask_tf]

        # Now all tensors has converted to (ds, xxx) style.
        score_tf = self.feature_ours.build(q_tf, sk_tf, el_tf, ordinal_tf_list)
        score_tf = tf.reshape(score_tf, [-1, self.PN])  # (batch, PN)

        if self.loss_func == 'HingeLoss':
            hinge_loss = HingeLoss(
                    margin=self.margin,
                    learning_rate=self.learning_rate)
            final_loss_tf, train_step = hinge_loss.build(score_tf, best_tf, mask_tf)
        elif self.loss_func == 'RankNet':
            ranknet = RankNet(
                    batch_size=self.batch_size,
                    list_len=self.PN,
                    learning_rate=self.learning_rate)
            final_loss_tf, train_step = ranknet.build(score_tf, gold_tf, mask_tf)

        useful_score_tf = score_tf - 10000.0 * (1.0 - mask_tf)
        # (batch, PN) gives very low score to padding schemas
        predict_row_tf = tf.cast(tf.range(tf.shape(mask_tf)[0]), tf.int64) # (batch, )
        predict_col_tf = tf.argmax(useful_score_tf, axis=1)   # (batch, ) of a int64 tensor
        # Get the co-ordination of the best candidate
        coo_tf = tf.stack(values=[predict_row_tf, predict_col_tf], axis=-1)
        predict_result_tf = tf.gather_nd(gold_tf, coo_tf)
        # (batch, ) returning the F1 result of predicted schemas
        LogInfo.logs('* Predicting function compiled.')

        return train_step, final_loss_tf, score_tf, predict_result_tf

    ''' ======== End of Model definition, the codes below focus on learning control ======== '''

    # Given the question index of the batch, return the fee dict.
    # For EL + Skeleton version.
    def prepare_feed_dict__el_sk(self, np_list, batch_indices, tf_list):
        q_tensor3, el_tensor3, path_tensor4, score_tensor3, mask_matrix = np_list
        q_input_raw = q_tensor3[batch_indices]
        q_input = np.stack([q_input_raw] * self.PN, axis=1).\
                    reshape([-1, self.sent_max_len, self.n_word_emb])
        # q_input: (batch*PN, sent_max_len, n_word_emb)
        sk_input = path_tensor4[batch_indices].\
                    reshape([-1, self.path_max_len, self.n_path_emb])
        # sk_input: (batch*PN, path_max_len, n_path_emb)
        el_input = el_tensor3[batch_indices].reshape([-1, self.n_el])
        # el_input: (batch*PN, n_el)
        gold_input = f1_matrix[batch_indices]   # (batch, PN)
        best_input = np.zeros(shape=gold_input.shape, dtype='float32')      # (batch, PN)
        best_input[:, 0] = 1.0      # we've ranked all schemas, so the first candidate must be the best
        mask_input = mask_matrix[batch_indices]

        batch_input_list = [q_input, sk_input, el_input, gold_input, best_input, mask_input]
        fd = {k: v for k, v in zip(tf_list, batch_input_list)}
        return fd



if __name__ == '__main__':
    LogInfo.begin_track('[model_ours] starts ... ')
    root_path = '/home/kangqi/workspace/PythonProject'
    try_dir = sys.argv[1]

    config_fp = '%s/runnings/compQA/input/%s/param_config' %(root_path, try_dir)
    config_dict = load_configs(config_fp)

    log_trend_fp = '%s/runnings/compQA/input/%s/log.trend' %(root_path, try_dir)

    LogInfo.begin_track('======== S1: Loading T/v/t Data ========')
    train_pydump_fp = '%s/runnings/compQA/input/%s/T_input.pydump' %(root_path, try_dir)
    test_pydump_fp = '%s/runnings/compQA/input/%s/t_input.pydump' %(root_path, try_dir)
    train_np_list = load_numpy_input(train_pydump_fp)
    test_np_list = load_numpy_input(test_pydump_fp)

    q_tensor3, el_tensor3, path_tensor4, score_tensor3, mask_matrix = train_np_list
    q_sz, sent_max_len, n_word_emb = q_tensor3.shape
    q_sz_1, PN, n_el = el_tensor3.shape
    q_sz_2, PN_1, path_max_len, n_path_emb = path_tensor4.shape
    q_sz_3, PN_2, categories = score_tensor3.shape
    q_sz_4, PN_3 = mask_matrix.shape
    T_size = int(config_dict['T_size'])
    v_size = int(config_dict['v_size'])
    t_size = int(config_dict['t_size'])


    # Just check the size of each numpy array in training data
    check_list = [sent_max_len, n_word_emb, path_max_len, n_path_emb, PN]
    check_str_list = ['sent_max_len', 'n_word_emb', 'path_max_len', 'n_path_emb', 'PN']
    assert check_list == [int(config_dict[check_str]) for check_str in check_str_list]

    assert q_sz == q_sz_1 == q_sz_2 == q_sz_3 == q_sz_4
    assert T_size + v_size == q_sz
    assert t_size == test_np_list[0].shape[0]
    assert PN == PN_1 == PN_2 == PN_3
    assert categories == 3
    LogInfo.logs('Passed assertion test.')
    LogInfo.end_track()

    f1_matrix = score_tensor3[:, :, 2]

    LogInfo.begin_track('======== S2: Building the model ========')
    q_tf = tf.placeholder(tf.float32, [None, sent_max_len, n_word_emb])
    sk_tf = tf.placeholder(tf.float32, [None, path_max_len, n_path_emb])
    el_tf = tf.placeholder(tf.float32, [None, n_el])
    gold_tf = tf.placeholder(tf.float32, [None, PN])
    best_tf = tf.placeholder(tf.float32, [None, PN])
    mask_tf = tf.placeholder(tf.float32, [None, PN])
    LogInfo.logs('* Tensor placeholder defined.')
    tf_list = [q_tf, sk_tf, el_tf, gold_tf, best_tf, mask_tf]

    activation = None
    act_str = config_dict['activation']
    if act_str == 'relu':
        activation = tf.nn.relu
    elif act_str == 'tanh':
        activation = tf.nn.tanh
    elif act_str == 'sigmoid':
        activation = tf.nn.sigmoid

    n_epoch = int(config_dict['n_epoch'])
    batch_size = int(config_dict['batch_size'])
    max_patience = int(config_dict['max_patience'])
    T_q_indices = range(T_size)
    v_q_indices = range(T_size, T_size + v_size)
    t_q_indices = range(t_size)
    n_T_batch, n_v_batch, n_t_batch = \
        [(sz - 1) / batch_size + 1 for sz in (T_size, v_size, t_size)]

    model = Model_Ours(n_el=n_el,
        sent_max_len=sent_max_len,
        n_word_emb=n_word_emb,
        n_window=int(config_dict['n_window']),
        n_cnn_hidden=int(config_dict['n_cnn_hidden']),
        path_max_len=path_max_len,
        n_path_emb=n_path_emb,
        n_rnn_hidden=int(config_dict['n_rnn_hidden']),
        n_feat_sk=int(config_dict['n_feat_sk']),
        n_compat_hidden=int(config_dict['n_compat_hidden']),
        activation=activation,
        loss_func=config_dict['loss_func'],
        margin=float(config_dict['margin']),
        learning_rate=float(config_dict['learning_rate']),
        PN=PN, batch_size=batch_size)
    train_step, final_loss_tf, score_tf, predict_result_tf = \
                    model.build(q_tf, sk_tf, el_tf, gold_tf, best_tf, mask_tf)
    LogInfo.end_track()

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    LogInfo.begin_track('Showing all parameters: ')
    for v in tf.global_variables():
        LogInfo.logs('%s: %s', v.name, v.get_shape().as_list())
    LogInfo.end_track()


    LogInfo.begin_track('======== S3: Training model ======== ' +
            '(n_epoch = %d, batch_size = %d, n_Tvt_batch = %d/%d/%d): ',
            n_epoch, batch_size, n_T_batch, n_v_batch, n_t_batch)
    bw_trend = open(log_trend_fp, 'w')
    bw_trend.write('%-8s\t%-8s\t%-8s\t%-8s\t%-8s\t%-8s\n'
                   %('#', 'T_loss', 'T_result', 'v_result', 't_result', 'Status'))
    bw_trend.flush()

    best_v_result = 0.0
    cur_patience = max_patience
    for epoch in range(n_epoch):
        if cur_patience == 0: break
        LogInfo.begin_track('Entering epoch %d / %d: ', epoch + 1, n_epoch)
        T_loss = 0.0
        T_predict_result_list = []   # record the predict F1 result of training data
        v_predict_result_list = []
        t_predict_result_list = []

        LogInfo.begin_track('Training [QA = %d, batch = %d]: ', T_size, n_T_batch)
        np.random.shuffle(T_q_indices)      # Random batch for training
        for batch_idx in range(n_T_batch):
            batch_indices = T_q_indices[batch_idx * batch_size : (batch_idx + 1) * batch_size]
            fd = model.prepare_feed_dict__el_sk(train_np_list, batch_indices, tf_list)
            _, batch_loss, batch_score_matrix, batch_predict_result_vec = \
                    sess.run([train_step, final_loss_tf, score_tf, predict_result_tf], feed_dict=fd)
            T_loss += (batch_loss * len(batch_indices))
            T_predict_result_list.append(batch_predict_result_vec)
            if (batch_idx + 1) % 5 == 0:
                LogInfo.logs('Batch %d / %d: T_loss = %g', batch_idx + 1, n_T_batch, batch_loss)
#                LogInfo.begin_track('Batch %d / %d: ', batch_idx + 1, n_batch)
#                LogInfo.logs('T_loss = %g', batch_loss)
#                concern_score_matrix = batch_score_matrix[0:3, 0:6]
#                LogInfo.logs('Score matrix: %s', concern_score_matrix)
#                concern_gold_matrix = gold_input[0:3, 0:6]
#                LogInfo.logs('Gold matrix: %s', concern_gold_matrix)
#                LogInfo.end_track()

        T_loss /= T_size
        T_result = np.concatenate(T_predict_result_list).mean()
        LogInfo.logs('Overall on %d T_QA: T_loss = %g, T_F1 = %g', T_size, T_loss, T_result)
        LogInfo.end_track()

        LogInfo.begin_track('Validating [QA = %d, batch = %d]: ', v_size, n_v_batch)
        for batch_idx in range(n_v_batch):
            batch_indices = v_q_indices[batch_idx * batch_size : (batch_idx + 1) * batch_size]
            fd = model.prepare_feed_dict__el_sk(train_np_list, batch_indices, tf_list)
            # Note: validate questions are still in train_np_list
            batch_score_matrix, batch_predict_result_vec = \
                sess.run([score_tf, predict_result_tf], feed_dict=fd)
            v_predict_result_list.append(batch_predict_result_vec)
            if (batch_idx + 1) % 5 == 0:
                LogInfo.logs('Batch %d / %d.', batch_idx + 1, n_v_batch)
        v_result = np.concatenate(v_predict_result_list).mean()
        LogInfo.logs('Overall on %d v_QA: v_F1 = %g', v_size, v_result)
        LogInfo.end_track()

        LogInfo.logs('[%s] cur_v_result = %g, best_v_result = %g, delta = %g',
             'UPDATE' if v_result > best_v_result else 'stay',
             v_result, best_v_result, v_result - best_v_result)
        if v_result > best_v_result:    # Update validation result, and perform testing
            best_v_result = v_result
            cur_patience = max_patience
            LogInfo.begin_track('Testing [QA = %d, batch = %d]: ', t_size, n_t_batch)
            for batch_idx in range(n_t_batch):
                batch_indices = t_q_indices[batch_idx * batch_size : (batch_idx + 1) * batch_size]
                fd = model.prepare_feed_dict__el_sk(test_np_list, batch_indices, tf_list)
                batch_score_matrix, batch_predict_result_vec = \
                    sess.run([score_tf, predict_result_tf], feed_dict=fd)
                t_predict_result_list.append(batch_predict_result_vec)
                if (batch_idx + 1) % 5 == 0:
                    LogInfo.logs('Batch %d / %d.', batch_idx + 1, n_t_batch)
            t_result = np.concatenate(t_predict_result_list).mean()
            LogInfo.logs('Overall on %d t_QA: t_F1 = %g', t_size, t_result)
            LogInfo.end_track()
        else:
            t_result = '--------'
            cur_patience -= 1

        bw_trend.write('%-8d\t%8.6f\t%8.6f\t%8.6f\t' %(epoch + 1, T_loss, T_result, v_result))
        if t_result == '--------':
            bw_trend.write('%s\t%-d\n' %(t_result, cur_patience))
        else:
            bw_trend.write('%8.6f\t[UPDATE]\n' %t_result)
        bw_trend.flush()

        LogInfo.end_track() # End of one iteration

    bw_trend.close()
    LogInfo.logs('Early stopping at %d epochs.', epoch + 1)
    LogInfo.end_track()     # End of learning

    LogInfo.end_track()     # End of program