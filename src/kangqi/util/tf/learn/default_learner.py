# -*- coding: utf-8 -*-

#==============================================================================
# Author: Kangqi Luo
# Goal: Construct a framework for conventional learning process,
# that is, training->validation->testing procedure using tensorflow.
#==============================================================================

import numpy as np
import shutil

from kangqi.util.LogUtil import LogInfo



class DefaultLearner(object):

    def __init__(self, n_epoch, max_patience, batch_size,
                 info_tup_list, input_tf_list, log_fp,
                 sess, train_step, final_loss_tf, metric_tf=None,
                 predict_tf=None):
        self.Tvt_dict = {'T': 0, 'v': 1, 't': 2}
        self.n_epoch = n_epoch
        self.max_patience = max_patience
        self.batch_size = batch_size

        # info_tup: (np_list, indices, n_batch)
        self.info_tup_list = info_tup_list
        self.train_info_tup, self.valid_info_tup, self.test_info_tup = info_tup_list
        self.input_tf_list = input_tf_list
        self.log_fp = log_fp    # Showing T/v/t result trends

        self.sess = sess
        self.train_step = train_step

        self.metric_tf = None
        if predict_tf is not None and metric_tf is None:
            self.metric_tf = predict_tf
            LogInfo.logs('Warning: predict_tf is deprecated, use metric_tf as instead.')
        elif metric_tf is not None:
            self.metric_tf = metric_tf

        self.final_loss_tf = final_loss_tf
        self.disp_batch_interval = 5


    # This is the default feed_dict preparation code.
    # We allow custom functions, but the signature must be the same.
    def default_prepare_fd_func(self, info_tup, batch_idx):
        fd = {}
        np_list, indices, n_batch = info_tup
        # get the corresponding input numpy array and batch indices

        # Default mode: Just slice on the first dimension
        batch_indices = indices[
                batch_idx * self.batch_size :
                (batch_idx + 1) * self.batch_size
        ]
        for np_val, input_tf in zip(np_list, self.input_tf_list):
            batch_input = np_val[batch_indices]
            fd[input_tf] = batch_input
        return batch_indices, fd

    def default_train_func(self, fd):
            _, batch_metric_vec, batch_loss = self.sess.run(
                [self.train_step, self.metric_tf, self.final_loss_tf], fd)
            return batch_metric_vec, batch_loss

    def default_predict_func(self, fd):
        batch_metric_vec = self.sess.run(self.metric_tf, fd)
        return batch_metric_vec

    def default_save_weights_func(self):
        # Default: Nothing to do (since we don't know which arrays to save)
        pass

    # receive train / predict / save_weights callback.
    # skip_eval: how many epochs does it skip before the next v/t prediction.
    # larger_is_better: whether the evaluation metric is the-larger-the-better
    # loss_as_average: whether the returned training loss is the summation of cases in a batch,
    #              or the average in a batch. (Default: Average)
    def learn(self, log_header, skip_eval = 1, larger_is_better=True,
              prepare_fd_func=None, train_func=None,
              predict_func=None, save_weights_func=None,
              show_process=False, loss_as_average=True):
        # Register default functions
        if prepare_fd_func is None:
            prepare_fd_func = self.default_prepare_fd_func
        if train_func is None:
            train_func = self.default_train_func
        if predict_func is None:
            predict_func = self.default_predict_func
        if save_weights_func is None:
            save_weights_func = self.default_save_weights_func

        LogInfo.begin_track('[kangqi.util.tf.learner.DefaultLearner] starts ' +
                            '(epoch = %d, batch_size = %d): ', self.n_epoch, self.batch_size)

        LogInfo.logs('Save log into %s.', self.log_fp)
        bw_log = open(self.log_fp + '.tmp', 'w')
        bw_log.write('# %s\n' %self.log_fp)
        bw_log.write(log_header + '\n')
        bw_log.flush()

        best_row = ''       # write the best row information at the end of file
        best_v_result = -10000.0 if larger_is_better else 10000.0
        cur_patience = self.max_patience
        T_size = len(self.train_info_tup[1]); n_T_batch = self.train_info_tup[2]
        v_size = len(self.valid_info_tup[1]); n_v_batch = self.valid_info_tup[2]
        t_size = len(self.test_info_tup[1]);  n_t_batch = self.test_info_tup[2]
        LogInfo.logs('T/v/t batch numbers = %d / %d / %d', n_T_batch, n_v_batch, n_t_batch)

        for epoch in range(self.n_epoch):
            if cur_patience == 0: break
            LogInfo.begin_track('Entering epoch %d / %d: ', epoch + 1, self.n_epoch)
            T_result_list = []
            v_result_list = []
            t_result_list = []  # store the score of the specific evaluation metric

            T_loss = 0.0
            for batch_idx in range(n_T_batch):
                batch_indices, fd = prepare_fd_func(self.info_tup_list[0], batch_idx)
                batch_result_vec, batch_loss = train_func(fd=fd)
                if loss_as_average == True:  # batch_loss is the average loss
                    T_loss += batch_loss * len(batch_indices)
                else:                        # batch_loss is the sum loss
                    T_loss += batch_loss
                T_result_list.append(batch_result_vec)
                if show_process and (batch_idx + 1) % self.disp_batch_interval == 0:
                    LogInfo.logs('T-batch = %d / %d ... ', batch_idx + 1, n_T_batch)
            T_loss /= T_size
            T_result = np.concatenate(T_result_list).mean()
            LogInfo.logs('Overall on %d T_data: T_loss = %g, T_score = %g', T_size, T_loss, T_result)

            for batch_idx in range(n_v_batch):
                _, fd = prepare_fd_func(self.info_tup_list[1], batch_idx)
                batch_result_vec = predict_func(fd=fd)
                v_result_list.append(batch_result_vec)
                if show_process and (batch_idx + 1) % self.disp_batch_interval == 0:
                    LogInfo.logs('v-batch = %d / %d ... ', batch_idx + 1, n_v_batch)
            v_result = np.concatenate(v_result_list).mean()
            LogInfo.logs('Overall on %d v_data: v_score = %g', v_size, v_result)

            updated = False
            if ((larger_is_better and v_result > best_v_result) or
                ((not larger_is_better) and v_result < best_v_result)):
                updated = True

            LogInfo.logs('[%s] cur_v_result = %g, best_v_result = %g, delta = %g',
                 'UPDATE' if updated else 'stay=%d' %(cur_patience),
                 v_result, best_v_result, v_result - best_v_result)
            if updated: # Update validation result, and perform testing
                best_v_result = v_result
                cur_patience = self.max_patience
                save_weights_func()
                for batch_idx in range(n_t_batch):
                    _, fd = prepare_fd_func(self.info_tup_list[2], batch_idx)
                    batch_result_vec = predict_func(fd=fd)
                    t_result_list.append(batch_result_vec)
                    if show_process and (batch_idx + 1) % self.disp_batch_interval == 0:
                        LogInfo.logs('t-batch = %d / %d ... ', batch_idx + 1, n_t_batch)
                t_result = np.concatenate(t_result_list).mean()
                LogInfo.logs('Overall on %d t_data: t_score = %g', t_size, t_result)
                best_row = '# %-6d\t%8.6f\t%8.6f\t%8.6f\t%8.6f\t[BEST ITERATION]' %(
                                epoch + 1, T_loss, T_result, v_result, t_result)
            else:
                t_result = '--------'
                cur_patience -= 1

            bw_log.write('%-8d\t%8.6f\t%8.6f\t%8.6f\t' %(epoch + 1, T_loss, T_result, v_result))
            if t_result == '--------':
                bw_log.write('%s\t%-d\n' %(t_result, cur_patience))
            else:
                bw_log.write('%8.6f\t[UPDATE]\n' %t_result)
            bw_log.flush()

            LogInfo.end_track()

        # End of iteration, write best row information and close the file
        bw_log.write(best_row + '\n')
        bw_log.write('# %s\n' %self.log_fp)
        bw_log.close(); shutil.move(self.log_fp + '.tmp', self.log_fp)

        LogInfo.logs('Early stopping at %d epochs.', epoch + 1)
        LogInfo.logs('>>> Best result: %s', best_row)
        LogInfo.end_track()     # End of learning