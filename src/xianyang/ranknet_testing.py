# -*- coding: utf-8 -*-

#==============================================================================
# Author: Kangqi Luo
# Goal: test RankNet model with multiple lists as a batch
# We call the function in task.compQA.model.ranknet_improved.py
#==============================================================================

import tensorflow as tf
import numpy as np
import time

from ranknet_improved import RankNet
# from ..tf_basics import weight_variable, bias_variable

# from kangqi.toy.ranknet.u import create_data, NDCG, get_orders
# from kangqi.util.LogUtil import LogInfo
from learn import load_training_data

train_size = 1974
batch_size = 1974
# test_size = 20
list_len = 50
feat_num = 17
# hidden_num = 500
rate = 0.05
max_epoch = 100
Activation = tf.nn.sigmoid

def weight_variable(shape):                                                                     
    initial = tf.truncated_normal(shape, stddev = 0.1)                                          
    return tf.Variable(initial)                                                                 

def bias_variable(shape):                                                                       
    initial = tf.constant(0.1, shape = shape)                                                   
    return tf.Variable(initial)

# Input: row_tf (batch_size, list_len, feat_num) as one input list
# Output: score_tf (batch_size, list_len)
def get_score_tf(row_tf):
    W_fc = weight_variable([feat_num, 1])
    b_fc = bias_variable([1])

    use_row_tf = tf.reshape(row_tf, [-1, feat_num]) # (batch_size * list_len, feat_num)
    h_fc = Activation(tf.matmul(use_row_tf, W_fc) + b_fc)   # (batch_size * list_len, hidden)
    score_tf = tf.reshape(h_fc, [-1, list_len])    # (batch_size, list_len)
    # LogInfo.logs('score_tf compiled.')
    return W_fc, b_fc, score_tf



if __name__ == '__main__':
    print('Loading data...')
    data, label, mask = load_training_data()
    # print 'label: %s' %label[0:10, 0:10]

    print('Compiling model...')
    row_tf = tf.placeholder(tf.float32, [batch_size, list_len, feat_num])
    label_tf = tf.placeholder(tf.float32, [batch_size, list_len])
    mask_tf = tf.placeholder(tf.float32, [batch_size, list_len])
    W_fc, b_fc, score_tf = get_score_tf(row_tf)
    ranknet = RankNet(batch_size, list_len, rate)
    # final_loss_tf, train_step = ranknet.build_improved(score_tf, label_tf, mask_tf)
    final_loss_tf, train_step = ranknet.build(score_tf, label_tf, mask_tf)

    # bw = open('batch_result.txt', 'w')
    # gold_orders = get_orders(test_label[0])
    # bw.write('Example Gold Orders: %s\n' %gold_orders)
    # bw.write('%-8s\t%-8s\t%-8s\t%-8s\t%-8s\t%-8s\n' %(
    #     '#-Iter',
    #     'T_NDCG', 'T_my_loss',
    #     't_NDCG', 't_my_loss',
    #     'Example output orders'
    # ))
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    for v in tf.global_variables():
        print '%s: %s' % (v.name, v)

    # train_score_time = train_lambda_time = train_update_time = train_time = 0
    # test_score_time = test_ndcg_time = test_myloss_time = test_time = 0

    print 'Learning starts, epoch = %d: ' % max_epoch
    for epoch in xrange(max_epoch):
        print 'Epoch %d / %d: ' % (epoch + 1, max_epoch)
        # ta = time.time()

        batch_num = (train_size - 1) / batch_size + 1
        for batch_idx in xrange(batch_num):    # handle each batch
            cur_indices = range(batch_idx * batch_size, (batch_idx + 1) * batch_size)
            # LogInfo.begin_track('Batch %d / %d: ', batch_idx + 1, train_size)
            # if (batch_idx + 1) % 10 == 0:
            #     LogInfo.logs('Batch %d / %d', batch_idx + 1, batch_num)
            row_input = data[cur_indices]    # (batch_size, list_len, feat_num)
            label_input = label[cur_indices] # (batch_size, list_len, )
            mask_input = mask[cur_indices]
            fd = {row_tf: row_input, label_tf: label_input, mask_tf: mask_input}
            _, batch_loss = sess.run([train_step, final_loss_tf], fd)   # update weights for each list
            print '  Batch %d: loss = %g' %(batch_idx + 1, batch_loss)

        # train_time += time.time() - ta
        '''
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
        '''

        # test ?
        if (epoch + 1) % 10 == 0:
            fd = {row_tf: data, label_tf: label, mask_tf: mask}
            opt_score, final_loss = sess.run([score_tf, final_loss_tf], fd)
            print opt_score.shape, final_loss
            print 'opt_score: %s' %opt_score[0:10, 0:10]

            first = []
            correct = 0
            for i in range(train_size):
                example = opt_score[i]
                max = 0
                max_at = 0
                #example = example[mask[i]]
                #print example
                for j in range(int(sum(mask[i]))):
                    if example[j] > max:
                        max = example[j]
                        max_at = j
                first.append(max_at)
                if max_at == 0:
                    correct += 1

            print correct, first

    '''
    for k, v in zip(
            ['train_time', 'test_score_time', 'test_ndcg_time', 'test_time'],
            [train_time, test_score_time, test_ndcg_time, test_time]
    ):
        LogInfo.logs('%s = %.3fs.', k, v)

    LogInfo.end_track('Done.')
    '''

    W_val, b_val = sess.run([W_fc, b_fc])
    with open('weights.pydump', 'wb') as bw:
        np.save(bw, W_val)
        np.save(bw, b_val)


class Ranker:
    def __init__(self):
        self.row_tf = tf.placeholder(tf.float32, [list_len, feat_num])
        # self.label_tf = tf.placeholder(tf.float32, [batch_size, list_len])
        # self.mask_tf = tf.placeholder(tf.float32, [batch_size, list_len])
        # self.W_fc, self.b_fc, self.score_tf = get_score_tf(self.row_tf)
        self.W_fc = weight_variable([feat_num, 1])
        self.b_fc = bias_variable([1])

        # use_row_tf = tf.reshape(row_tf, [-1, feat_num]) # (batch_size * list_len, feat_num)
        self.h_fc = Activation(tf.matmul(self.row_tf, self.W_fc) + self.b_fc)   # (batch_size * list_len, hidden)
        # score_tf = tf.reshape(h_fc, [-1, list_len])    # (batch_size, list_len)
        self.score_tf = self.h_fc
        self.sess = tf.InteractiveSession()

        with open('weights.pydump', 'rb') as br:
            W_val = np.load(br)
            b_val = np.load(br)
        assign_W = tf.assign(self.W_fc, W_val)
        assign_b = tf.assign(self.b_fc, b_val)
        self.sess.run([assign_W, assign_b])
    
    def rank(self, feats):
        # feats: (list_len, feat_num)
        padded = np.zeros((list_len, feat_num))
        origin_len = feats.shape[0]
        padded[:origin_len, :] = feats
        fd = {self.row_tf: padded}
        opt_score = self.sess.run(self.score_tf, fd)
        # print 'opt: ', opt_score[:origin_len]
        best = opt_score[:origin_len].argmax()
        return best