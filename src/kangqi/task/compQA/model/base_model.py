# -*- coding:utf-8 -*-

import tensorflow as tf
from kangqi.util.LogUtil import LogInfo


class BaseModel(object):

    def __init__(self, sess, verbose=0):
        # Dropout control
        # 0: regular model;
        # 1: model with dropout
        # self.dropout_switch = tf.placeholder(dtype=tf.int32,
        #                                      shape=(),
        #                                      name='dropout_switch')
        self.sess = sess
        self.verbose = verbose

        # Receives different input for optimize and evluation task
        self.optm_input_tf_list = None
        self.eval_input_tf_list = None

        self.optm_step = None
        self.avg_loss = None
        self.eval_metric_val = None

        self.eval_output_tf_tup_list = []       # [(tensor_name, tensor)]
        self.show_param_tf_tup_list = []        # [(tensor_name, tensor)]

        self.optm_summary = None
        self.eval_summary = None

        # All elements above needs to be defined in the __init__() of children classes.

    def prepare_data(self, data_loader):
        if data_loader.dynamic or data_loader.np_data_list is None:     # create a brand new data for the model
            data_loader.renew_data_list()
        elif data_loader.shuffle:                                       # just change the order
            data_loader.update_statistics()
        if self.verbose > 0:
            LogInfo.logs('data size = %d, num of batch = %d.', len(data_loader), data_loader.n_batch)

    def evaluate(self, data_loader, epoch_idx, ob_batch_num=10, detail_fp=None, summary_writer=None):
        if data_loader is None:
            return 0.
        self.prepare_data(data_loader=data_loader)
        run_options = run_metadata = None
        if summary_writer is not None:
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()

        scan_size = 0
        ret_metric = 0.
        for batch_idx in range(data_loader.n_batch):
            point = (epoch_idx - 1) * data_loader.n_batch + batch_idx
            local_data_list, _ = data_loader.get_next_batch()
            local_size = len(local_data_list[0])    # the first dimension is always batch size
            fd = {input_tf: local_data for input_tf, local_data in zip(self.eval_input_tf_list, local_data_list)}

            local_metric, summary = self.sess.run(
                [self.eval_metric_val, self.eval_summary], feed_dict=fd,
                options=run_options,
                run_metadata=run_metadata
            )
            ret_metric = (ret_metric * scan_size + local_metric * local_size) / (scan_size + local_size)
            scan_size += local_size
            if (batch_idx+1) % ob_batch_num == 0:
                LogInfo.logs('[eval-%s-B%d/%d] metric = %.6f, scanned = %d/%d',
                             data_loader.mode,
                             batch_idx+1,
                             data_loader.n_batch,
                             ret_metric,
                             scan_size,
                             len(data_loader))
            if summary_writer is not None:
                summary_writer.add_summary(summary, point)
                if batch_idx == 0:
                    summary_writer.add_run_metadata(run_metadata, 'epoch-%d' % epoch_idx)
        return ret_metric

    def optimize(self, data_loader, epoch_idx, ob_batch_num=10, summary_writer=None):
        if data_loader is None:
            return -1.
        self.prepare_data(data_loader=data_loader)
        run_options = run_metadata = None
        if summary_writer is not None:
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()

        scan_size = 0
        ret_loss = 0.
        for batch_idx in range(data_loader.n_batch):
            point = (epoch_idx - 1) * data_loader.n_batch + batch_idx
            local_data_list, _ = data_loader.get_next_batch()
            local_size = len(local_data_list[0])    # the first dimension is always batch size
            fd = {input_tf: local_data for input_tf, local_data in zip(self.optm_input_tf_list, local_data_list)}
            # fd[self.dropout_switch] = 1
            _, local_loss, summary = self.sess.run(
                [self.optm_step, self.avg_loss, self.optm_summary], feed_dict=fd,
                options=run_options, run_metadata=run_metadata
            )
            ret_loss = (ret_loss * scan_size + local_loss * local_size) / (scan_size + local_size)
            scan_size += local_size
            if (batch_idx+1) % ob_batch_num == 0:
                LogInfo.logs('[optm-%s-B%d/%d] cur_batch_loss = %.6f, avg_loss = %.6f, scanned = %d/%d',
                             data_loader.mode,
                             batch_idx+1,
                             data_loader.n_batch,
                             local_loss,
                             ret_loss,
                             scan_size,
                             len(data_loader))
            if summary_writer is not None:
                summary_writer.add_summary(summary, point)
                if batch_idx == 0:
                    summary_writer.add_run_metadata(run_metadata, 'epoch-%d' % epoch_idx)
        return ret_loss
