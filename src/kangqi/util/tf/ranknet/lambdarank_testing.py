# -*- coding: utf-8 -*-

#==============================================================================
# Author: Kangqi Luo
# Goal: test LambdaRank model with multiple lists as a batch
#==============================================================================

import tensorflow as tf
import numpy as np
import time
import math
import sys

from .lambdarank_basic import LambdaRank
from ..tf_basics import weight_variable, bias_variable

from kangqi.toy.ranknet.u import create_data, NDCG, get_orders
from kangqi.util.LogUtil import LogInfo

train_size = 100
batch_size = 5
test_size = 20
list_len = 30
feat_num = 500
hidden_num = 500
rate = 0.005
max_epoch = 100
Activation = tf.nn.relu


# Input: row_tf (batch_size, list_len, feat_num) as one input list
# Output: score_tf (batch_size, list_len)
def get_score_tf(row_tf):
    W_fc1 = weight_variable([feat_num, hidden_num])
    b_fc1 = bias_variable([hidden_num])
    W_fc2 = weight_variable([hidden_num, 1])
    b_fc2 = bias_variable([1])

    use_row_tf = tf.reshape(row_tf, [-1, feat_num]) # (batch_size * list_len, feat_num)
    h_fc1 = Activation(tf.matmul(use_row_tf, W_fc1) + b_fc1)   # (batch_size * list_len, hidden)
    # h_fc2 = tf.nn.sigmoid(tf.matmul(h_fc1, W_fc2) + b_fc2)     # (batch_size * list_len, 1)

    h_fc2 = tf.matmul(h_fc1, W_fc2) + b_fc2     # (batch_size * list_len, 1)
    score_tf = tf.reshape(h_fc2, [-1, list_len])    # (batch_size, list_len)
    LogInfo.logs('score_tf compiled.')
    return score_tf


# given the gold vector, return the list of relevance (min as 0)
def get_relevance_vec(gold_vec):
    sz = len(gold_vec)
    srt_list = [(i, gold_vec[i]) for i in range(sz)]
    srt_list.sort(lambda x, y: -cmp(x[1], y[1]))

    relevance_vec = np.zeros((sz, ), dtype='float32')
    cur_level = -1
    cur_score = -2111222333
    for i in range(sz):
        pos = sz - i - 1
        idx, score = srt_list[pos]
        if score > cur_score:
            cur_score = score
            cur_level += 1
        relevance_vec[idx] = cur_level
    return relevance_vec

def get_order_vec(score_vec):
    sz = len(score_vec)
    srt_list = [(i, score_vec[i]) for i in range(sz)]
    srt_list.sort(lambda x, y: -cmp(x[1], y[1]))
    order_list = [ x[0] for x in srt_list ]
    return np.asarray(order_list, dtype='int64')


def NDCG_improved(relevance_vec, pred_vec,
          pred_order_vec=None, gold_order_vec=None, truncate=-1):
    sz = len(relevance_vec)
    if pred_order_vec is None:
        pred_order_vec = get_order_vec(pred_vec)
    if gold_order_vec is None:
        gold_order_vec = get_order_vec(relevance_vec)
#    LogInfo.logs('pred_order_vec: %s', pred_order_vec)
#    LogInfo.logs('gold_order_vec: %s', gold_order_vec)

#    LogInfo.begin_track('NDCG_Improved ... ')
    dcg = mdcg = 0.0
    topK = sz
    if truncate >= 0:
        topK = min(topK, truncate)
    for pos in range(topK):
        discount = 1.0 / math.log(1 + pos + 1, 2.0)  # pos starts from 0, while in NDCG, pos shall start from 1
        pred_idx = pred_order_vec[pos]
        gold_idx = gold_order_vec[pos]
        dcg += discount * (math.pow(2, relevance_vec[pred_idx]) - 1)
        mdcg += discount * (math.pow(2, relevance_vec[gold_idx]) - 1)
#        LogInfo.logs('pos = %d, disc = %g, pred_relev = %g, gold_relev = %g, dcg = %g, mdcg = %g',
#                     pos, discount, relevance_vec[pred_idx], relevance_vec[gold_idx], dcg, mdcg)
#    LogInfo.end_track()
    return dcg / mdcg


def manually_get_lambda(
        score_vec, gold_vec, mask_vec,
        relevance_vec, gold_order_vec,
        list_len, metric, gold_diff_method):
    assert score_vec.shape == gold_vec.shape == mask_vec.shape
    assert score_vec.shape[0] == list_len
    assert metric == 'NDCG'
    assert gold_diff_method in ('Sigmoid', 'Discrete')

    pred_order_vec = get_order_vec(score_vec)    # get the current predict order list in advance
    ndcg = NDCG_improved(relevance_vec=relevance_vec, pred_vec=None,
                         pred_order_vec=pred_order_vec, gold_order_vec=gold_order_vec)

    lambda_vec = np.zeros(shape=(list_len, ), dtype='float32')
    for i in range(list_len):
        if mask_vec[i] == 0.0:
            continue
        for j in range(list_len):
            if mask_vec[j] == 0.0 or gold_vec[i] <= gold_vec[j]:
                continue
            # Now we ensure that gold_vec[i] > gold_vec[j] and both i and j are effective data
            LogInfo.begin_track('Position %d vs %d: ', i, j)
            LogInfo.logs('relevance[i] = %.6f, gold[i] = %.6f, pred[i] = %.6f',
                         relevance_vec[i], gold_vec[i], score_vec[i])
            LogInfo.logs('relevance[j] = %.6f, gold[j] = %.6f, pred[j] = %.6f',
                         relevance_vec[j], gold_vec[j], score_vec[j])
            pred_diff = score_vec[i] - score_vec[j]
            LogInfo.logs('pred_diff = %.6f', pred_diff)
            pred_delta = 1.0 / (1.0 + math.exp(-1.0 * pred_diff))
            gold_delta = 1.0 # default value, since i is more relevant than j in the gold data
            if gold_diff_method == 'Sigmoid':
                gold_diff = gold_vec[i] - gold_vec[j]
                LogInfo.logs('gold_diff = %.6f', gold_diff)
                gold_delta = 1.0 / (1.0 + math.exp(-1.0 * gold_diff))
            # Now calculate the delta_ndcg
            metric_weight = 1.0     # store the weight introduced by the external evaluation metric
            if metric == 'NDCG':
                # first swap their orders in the predict rank
                pred_order_vec[i], pred_order_vec[j] = pred_order_vec[j], pred_order_vec[i]
                ndcg_swap = NDCG_improved(
                        relevance_vec=relevance_vec, pred_vec=None,
                        pred_order_vec=pred_order_vec, gold_order_vec=gold_order_vec)
                # swap back
                pred_order_vec[i], pred_order_vec[j] = pred_order_vec[j], pred_order_vec[i]
                metric_weight = abs(ndcg - ndcg_swap)
            LogInfo.logs('metric_weight = %.6f', metric_weight)
            lambda_ij = metric_weight * (pred_delta - gold_delta)
            lambda_vec[i] += lambda_ij
            lambda_vec[j] -= lambda_ij
            LogInfo.end_track()
    return lambda_vec


if __name__ == '__main__':
    LogInfo.begin_track('[ranknet_batch] starts ... ')

    save_idx = 0
    if len(sys.argv) >= 2:
        save_idx = int(sys.argv[1])

    metric = 'NDCG'
    gold_diff_method = 'Sigmoid'

    LogInfo.logs('rate: %g', rate)
    LogInfo.logs('metric: %s', metric)
    LogInfo.logs('gold_diff_method: %s', gold_diff_method)

    W_oracle = np.random.randn(feat_num)

    LogInfo.begin_track('Preparing data ... ')
    train_data, train_label = create_data(train_size, list_len, feat_num, W_oracle)
    test_data, test_label = create_data(test_size, list_len, feat_num, W_oracle)
    LogInfo.logs('Training data created, shape = %s.', train_data.shape)
    LogInfo.logs('Training label created, shape = %s.', train_label.shape)
    LogInfo.logs('Testing data created, shape = %s.', test_data.shape)
    LogInfo.logs('Testing label created, shape = %s.', test_label.shape)
    LogInfo.end_track()

    train_relevance_matrix = np.zeros((train_size, list_len), dtype='float32')
    test_relevance_matrix = np.zeros((test_size, list_len), dtype='float32')
    train_gold_order_matrix = np.zeros((train_size, list_len), dtype='int64')
    test_gold_order_matrix = np.zeros((test_size, list_len), dtype='int64')

    for label, relevance_matrix, gold_order_matrix in (
            [train_label, train_relevance_matrix, train_gold_order_matrix],
            [test_label, test_relevance_matrix, test_gold_order_matrix]):
        for idx in range(label.shape[0]):
            gold_vec = label[idx]
            relevance_vec = get_relevance_vec(gold_vec)
            relevance_matrix[idx] = relevance_vec
            gold_order_vec = get_order_vec(relevance_vec)
            gold_order_matrix[idx] = gold_order_vec



    LogInfo.begin_track('Compiling model ... ')
    row_tf = tf.placeholder(tf.float32, [batch_size, list_len, feat_num])
#    label_tf = tf.placeholder(tf.float32, [batch_size, list_len])
#    mask_tf = tf.placeholder(tf.float32, [batch_size, list_len])
    score_tf = get_score_tf(row_tf)

    merged_grad_tf_list = []            # tensor for storing merged updates
    for var in tf.global_variables():
        merged_grad_tf = tf.placeholder(tf.float32, var.get_shape().as_list())
        merged_grad_tf_list.append(merged_grad_tf)

    lambdarank = LambdaRank(
            metric=metric,
            gold_diff_method=gold_diff_method,
            batch_size=batch_size,
            list_len=list_len,
            learning_rate=rate)
    grad_tf_list = lambdarank.get_gradient_tf_list(score_tf)    # list of (batch_size, list_len, "var_shape")
    train_step = lambdarank.build_update(merged_grad_tf_list)
    LogInfo.logs('train_step: %s', train_step)
    LogInfo.end_track()


    bw = open('lambda_%s_%s_%g_%d.txt' %(metric, gold_diff_method, rate, save_idx), 'w')
    gold_orders = get_orders(test_label[0])
    bw.write('Example Gold Orders: %s\n' %gold_orders)
    bw.write('%-8s\t%-8s\t%-8s\t%-8s\t%-8s\t%-8s\n' %(
        '#-Iter',
        'T_NDCG', 'T_NDCG_new',
        't_NDCG', 't_NDCG_new',
        'Example output orders'
    ))
    bw.flush()
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    for v in tf.global_variables():
        LogInfo.logs('%s: %s', v.name, v)

    train_time = 0
    test_time = 0

    LogInfo.begin_track('Learning starts, epoch = %d: ', max_epoch)
    for epoch in xrange(max_epoch):
        LogInfo.begin_track('Epoch %d / %d: ', epoch + 1, max_epoch)
        ta = time.time()

        batch_num = (train_size - 1) / batch_size + 1
        for batch_idx in xrange(batch_num):    # handle each batch
            cur_indices = range(batch_idx * batch_size, (batch_idx + 1) * batch_size)
            # LogInfo.begin_track('Batch %d / %d: ', batch_idx + 1, train_size)
            # if (batch_idx + 1) % 10 == 0:
            LogInfo.logs('Batch %d / %d', batch_idx + 1, batch_num)
            row_input = train_data[cur_indices]    # (batch_size, list_len, feat_num)
            label_input = train_label[cur_indices] # (batch_size, list_len, )
            mask_input = np.ones(label_input.shape, dtype='float32')
            relevance_input = train_relevance_matrix[cur_indices]
            gold_order_input = train_gold_order_matrix[cur_indices]

            fd_score = {row_tf: row_input}
            score_matrix, grad_opt_list = sess.run([score_tf, grad_tf_list], fd_score)
            # score_matrix: (batch_size, list_len)
            # grad_opt_list: list of (batch_size, list_len, "var_shape")

            merged_grad_val_list = []
            for merged_grad_tf in merged_grad_tf_list:
                merged_grad_val_list.append(np.zeros(merged_grad_tf.get_shape().as_list(), 'float32'))
            for idx in range(len(cur_indices)):   # Manually traverse each case in the batch
                lambda_vec = manually_get_lambda(
                        score_vec=score_matrix[idx],
                        gold_vec=label_input[idx],
                        mask_vec=mask_input[idx],
                        relevance_vec=relevance_input[idx],
                        gold_order_vec=gold_order_input[idx],
                        list_len=list_len,
                        metric=metric,
                        gold_diff_method=gold_diff_method)
                for pos in range(list_len):
                    for grad_opt, merged_grad_val in zip(grad_opt_list, merged_grad_val_list):
                        # grad_opt: (batch_size, list_len, "var_shape")
                        # merged_grad_val: ("var_shape")
                        merged_grad_val += lambda_vec[pos] * grad_opt[idx, pos]

            fd_upd = {merged_grad_tf : merged_grad_val \
                      for merged_grad_tf, merged_grad_val in zip(
                              merged_grad_tf_list, merged_grad_val_list
                      )}
            sess.run(train_step, fd_upd)
        train_time += time.time() - ta

        ta = time.time()
        eval_dict = {}
        for eval_data, eval_label, eval_relevance_matrix, eval_gold_order_matrix, mark in [
                (train_data, train_label, train_relevance_matrix, train_gold_order_matrix, 'T'),
                (test_data, test_label, test_relevance_matrix, test_gold_order_matrix, 't')
        ]:
            LogInfo.begin_track('Eval on %s ... ', mark)
            cnt = 0; avg_ndcg = 0; avg_ndcg_new = 0
            batch_num = (eval_data.shape[0] - 1) / batch_size + 1
            for batch_idx in range(batch_num):
                cur_indices = range(batch_idx * batch_size, (batch_idx + 1) * batch_size)
                t0 = time.time()
                row_input = eval_data[cur_indices]
                label_input = eval_label[cur_indices]
                mask_input = np.ones(label_input.shape, dtype='float32')
                relevance_input = eval_relevance_matrix[cur_indices]
                gold_order_input = eval_gold_order_matrix[cur_indices]

                cases = row_input.shape[0]
                fd = {row_tf: row_input}
                opt_score = sess.run(score_tf, fd)
                cnt += cases
                t1 = time.time()

                if mark == 't' and batch_idx == 0:
                    output_orders = get_orders(opt_score[0])
                    eval_dict['output'] = output_orders

                for case_idx in range(cases):
                    ndcg = NDCG(opt_score[case_idx], label_input[case_idx])
                    ndcg_new = NDCG_improved(
                            relevance_vec=relevance_input[case_idx],
                            pred_vec=opt_score[case_idx],
                            gold_order_vec=gold_order_input[case_idx])
                    avg_ndcg += ndcg
                    avg_ndcg_new += ndcg_new
                t2 = time.time()

            avg_ndcg /= cnt
            avg_ndcg_new /= cnt
            LogInfo.logs('Avg NDCG on %s: %.6f', mark, avg_ndcg)
            eval_dict[mark + '_NDCG'] = avg_ndcg
            LogInfo.logs('Avg NDCG_new on %s: %.6f', mark, avg_ndcg_new)
            eval_dict[mark + '_NDCG_new'] = avg_ndcg_new
            LogInfo.end_track()

        test_time += time.time() - ta

        bw.write('%-8d\t%8.6f\t%8.6f\t%8.6f\t%8.6f\t%s\n' %(
            epoch + 1,
            eval_dict['T_NDCG'], eval_dict['T_NDCG_new'],
            eval_dict['t_NDCG'], eval_dict['t_NDCG_new'],
            eval_dict['output']
        ))
        bw.flush()

        LogInfo.end_track('End of epoch %d / %d.', epoch + 1, max_epoch)
    LogInfo.end_track('End of learning.')
    bw.close()


    for k, v in zip(
            ['train_time', 'test_time'],
            [train_time, test_time]
    ):
        LogInfo.logs('%s = %.3fs.', k, v)

    LogInfo.end_track('Done.')
