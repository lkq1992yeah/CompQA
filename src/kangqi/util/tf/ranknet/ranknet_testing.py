# -*- coding: utf-8 -*-

#==============================================================================
# Author: Kangqi Luo
# Goal: test RankNet model with multiple lists as a batch
# We call the function in task.compQA.model.ranknet_improved.py
#==============================================================================

import tensorflow as tf
import numpy as np
import time

from .ranknet_improved import RankNet
from ..tf_basics import weight_variable, bias_variable

from kangqi.toy.ranknet.u import create_data, NDCG, get_orders
from kangqi.util.LogUtil import LogInfo

train_size = 100
batch_size = 5
test_size = 20
list_len = 30
feat_num = 500
hidden_num = 500
rate = 0.0005
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



if __name__ == '__main__':
    LogInfo.begin_track('[ranknet_batch] starts ... ')

    W_oracle = np.random.randn(feat_num)

    LogInfo.begin_track('Preparing data ... ')
    train_data, train_label = create_data(train_size, list_len, feat_num, W_oracle)
    test_data, test_label = create_data(test_size, list_len, feat_num, W_oracle)
    LogInfo.logs('Training data created, shape = %s.', train_data.shape)
    LogInfo.logs('Training label created, shape = %s.', train_label.shape)
    LogInfo.logs('Testing data created, shape = %s.', test_data.shape)
    LogInfo.logs('Testing label created, shape = %s.', test_label.shape)
    LogInfo.end_track()

    LogInfo.begin_track('Compiling model ... ')
    row_tf = tf.placeholder(tf.float32, [batch_size, list_len, feat_num])
    label_tf = tf.placeholder(tf.float32, [batch_size, list_len])
    mask_tf = tf.placeholder(tf.float32, [batch_size, list_len])
    score_tf = get_score_tf(row_tf)
    ranknet = RankNet(batch_size, list_len, rate)
    # final_loss_tf, train_step = ranknet.build_improved(score_tf, label_tf, mask_tf)
    final_loss_tf, train_step = ranknet.build(score_tf, label_tf, mask_tf)
    LogInfo.end_track()


    bw = open('batch_result.txt', 'w')
    gold_orders = get_orders(test_label[0])
    bw.write('Example Gold Orders: %s\n' %gold_orders)
    bw.write('%-8s\t%-8s\t%-8s\t%-8s\t%-8s\t%-8s\n' %(
        '#-Iter',
        'T_NDCG', 'T_my_loss',
        't_NDCG', 't_my_loss',
        'Example output orders'
    ))
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    for v in tf.global_variables():
        LogInfo.logs('%s: %s', v.name, v)

    train_score_time = train_lambda_time = train_update_time = train_time = 0
    test_score_time = test_ndcg_time = test_myloss_time = test_time = 0

    LogInfo.begin_track('Learning starts, epoch = %d: ', max_epoch)
    for epoch in xrange(max_epoch):
        LogInfo.begin_track('Epoch %d / %d: ', epoch + 1, max_epoch)
        ta = time.time()

        batch_num = (train_size - 1) / batch_size + 1
        for batch_idx in xrange(batch_num):    # handle each batch
            cur_indices = range(batch_idx * batch_size, (batch_idx + 1) * batch_size)
            # LogInfo.begin_track('Batch %d / %d: ', batch_idx + 1, train_size)
            if (batch_idx + 1) % 10 == 0:
                LogInfo.logs('Batch %d / %d', batch_idx + 1, batch_num)
            row_input = train_data[cur_indices]    # (batch_size, list_len, feat_num)
            label_input = train_label[cur_indices] # (batch_size, list_len, )
            mask_input = np.ones(label_input.shape, dtype='float32')
            fd = {row_tf: row_input, label_tf: label_input, mask_tf: mask_input}
            sess.run(train_step, fd)   # update weights for each list
        train_time += time.time() - ta

        ta = time.time()
        eval_dict = {}
        for eval_data, eval_label, mark in [
                (train_data, train_label, 'T'),
                (test_data, test_label, 't')
        ]:
            LogInfo.begin_track('Eval on %s ... ', mark)
            batch_loss = 0.0; cnt = 0; avg_ndcg = 0
            batch_num = (eval_data.shape[0] - 1) / batch_size + 1
            for batch_idx in range(batch_num):
                cur_indices = range(batch_idx * batch_size, (batch_idx + 1) * batch_size)
                t0 = time.time()
                row_input = eval_data[cur_indices]
                label_input = eval_label[cur_indices]
                mask_input = np.ones(label_input.shape, dtype='float32')
                cases = row_input.shape[0]
                fd = {row_tf: row_input, label_tf: label_input, mask_tf: mask_input}
                opt_score, final_loss = sess.run([score_tf, final_loss_tf], fd)
                batch_loss += final_loss * cases
                cnt += cases
                t1 = time.time()
                test_score_time += t1 - t0

                if mark == 't' and batch_idx == 0:
                    output_orders = get_orders(opt_score[0])
                    eval_dict['output'] = output_orders

                for case_idx in range(cases):
                    ndcg = NDCG(opt_score[case_idx], label_input[case_idx])
                    avg_ndcg += ndcg
                t2 = time.time()
                test_ndcg_time += t2 - t1

            avg_ndcg /= cnt
            batch_loss /= cnt
            LogInfo.logs('Avg NDCG on %s: %.6f', mark, avg_ndcg)
            LogInfo.logs('Avg entropy on %s: %.6f', mark, batch_loss)
            eval_dict[mark + '_NDCG'] = avg_ndcg
            eval_dict[mark + '_my_loss'] = batch_loss
            LogInfo.end_track()

        test_time += time.time() - ta

        bw.write('%-8d\t%8.6f\t%8.6f\t%8.6f\t%8.6f\t%s\n' %(
            epoch + 1,
            eval_dict['T_NDCG'], eval_dict['T_my_loss'],
            eval_dict['t_NDCG'], eval_dict['t_my_loss'],
            eval_dict['output']
        ))
        bw.flush()

        LogInfo.end_track('End of epoch %d / %d.', epoch + 1, max_epoch)
    LogInfo.end_track('End of learning.')
    bw.close()


    for k, v in zip(
            ['train_time', 'test_score_time', 'test_ndcg_time', 'test_time'],
            [train_time, test_score_time, test_ndcg_time, test_time]
    ):
        LogInfo.logs('%s = %.3fs.', k, v)

    LogInfo.end_track('Done.')
