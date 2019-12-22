"""
Author: Kangqi Luo
Date: 180206
Goal: the wrapper for training different kernels
"""

import tensorflow as tf

from kangqi.util.LogUtil import LogInfo


class BaseOptimizer:

    def __init__(self, name, input_tensor_list, optm_step, loss, extra_data, optm_summary,
                 sess, ob_batch_num=10, summary_writer=None):
        self.name = name
        self.input_tensor_list = input_tensor_list
        self.optm_summary = optm_summary
        self.optm_step = optm_step
        self.loss = loss
        self.sess = sess
        self.extra_data = extra_data

        self.ob_batch_num = ob_batch_num
        self.summary_writer = summary_writer
        self.run_options = self.run_metadata = None
        if self.summary_writer is not None:
            self.run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            self.run_metadata = tf.RunMetadata()
        self.scan_data = 0
        self.scan_batch = 0
        self.ret_loss = 0.
        self.tb_point = 0       # x-axis in tensorboard

    def reset_optm_info(self):
        self.scan_data = self.scan_batch = 0
        self.ret_loss = 0.

    def optimize(self, optm_dl, batch_idx):
        local_data_list, local_indices = optm_dl.get_batch(batch_idx=batch_idx)
        local_size = len(local_indices)
        fd = {input_tf: local_data for input_tf, local_data in
              zip(self.input_tensor_list, local_data_list)}

        _, local_loss, local_extra, summary = self.sess.run(
            [self.optm_step, self.loss, self.extra_data, self.optm_summary],
            feed_dict=fd, options=self.run_options,
            run_metadata=self.run_metadata
        )
        local_loss = float(local_loss)
        self.ret_loss = (self.ret_loss * self.scan_data + local_loss * local_size) / (self.scan_data + local_size)
        self.scan_data += local_size
        self.scan_batch += 1
        self.tb_point += 1
        if self.scan_batch % self.ob_batch_num == 0:
            LogInfo.logs('[%3s][optm-%s-B%d/%d] cur_batch_loss = %.6f, avg_loss = %.6f, scanned = %d/%d',
                         self.name,
                         optm_dl.mode,
                         self.scan_batch,
                         optm_dl.n_batch,
                         local_loss,
                         self.ret_loss,
                         self.scan_data,
                         len(optm_dl))

            # """ For batch=1 debug only!! """
            # q_idx, pos_sc, neg_sc, weight = optm_dl.optm_pair_tup_list[batch_idx]
            # LogInfo.logs('  q_idx = %4d, pos_sc: line = %4d, score = %.6f, rm_f1 = %.6f',
            #              q_idx, pos_sc.ori_idx, local_extra[0], pos_sc.rm_f1)
            # LogInfo.logs('  q_idx = %4d, neg_sc: line = %4d, score = %.6f, rm_f1 = %.6f',
            #              q_idx, neg_sc.ori_idx, local_extra[1], neg_sc.rm_f1)

        if self.summary_writer is not None:
            self.summary_writer.add_summary(summary, self.tb_point)
            # if batch_idx == 0:
            #     summary_writer.add_run_metadata(run_metadata, 'epoch-%d' % epoch_idx)

    def optimize_all(self, optm_dl):
        self.reset_optm_info()
        for batch_idx in range(optm_dl.n_batch):
            self.optimize(optm_dl=optm_dl, batch_idx=batch_idx)
