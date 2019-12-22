"""
Author: Kangqi Luo
Date: 180206
Goal: the wrapper for evaluating different kernels
"""

import tensorflow as tf

from kangqi.util.LogUtil import LogInfo


class BaseEvaluator:

    def __init__(self, name, sess,
                 input_name_list, output_name_list,
                 input_tensor_list, eval_output_tensor_list,
                 concern_name_list, metric_name_list,
                 ob_batch_num=10, summary_writer=None):
        self.name = name
        self.input_name_list = input_name_list      # the name of input tensors
        self.output_name_list = output_name_list    # the name of output tensors
        self.concern_name_list = concern_name_list  # all data name that we collect for calculating final results
        self.input_tensor_list = input_tensor_list
        self.output_tensor_list = eval_output_tensor_list
        self.sess = sess

        self.ob_batch_num = ob_batch_num
        self.metric_name_list = metric_name_list
        self.summary_writer = summary_writer
        self.run_options = self.run_metadata = None
        if self.summary_writer is not None:
            self.run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            self.run_metadata = tf.RunMetadata()
        self.scan_data = 0
        self.scan_batch = 0
        self.tb_point = 0       # x-axis in tensorboard
        self.eval_detail_dict = {}      # store evaluation detail of each data

    """ Reset informations for calculating the final result """
    def reset_eval_info(self):
        self.scan_data = self.scan_batch = 0
        self.eval_detail_dict = {}

    """ For each batch, save the key eval detail into the evluation pool """
    def update_local_eval_detail(self, local_eval_detail_dict):
        for name in self.concern_name_list:
            batch_val = local_eval_detail_dict[name]
            for val in batch_val:
                self.eval_detail_dict.setdefault(name, []).append(val)

    def evaluate(self, eval_dl, batch_idx):
        local_data_list, local_indices = eval_dl.get_batch(batch_idx=batch_idx)
        local_size = len(local_indices)
        fd = {input_tf: local_data for input_tf, local_data in zip(self.input_tensor_list, local_data_list)}
        local_output_list = self.sess.run(self.output_tensor_list, feed_dict=fd,
                                          options=self.run_options, run_metadata=self.run_metadata)

        local_eval_detail_dict = {k: v for k, v in zip(self.input_name_list, local_data_list)}
        local_eval_detail_dict.update({k: v for k, v in zip(self.output_name_list, local_output_list)})
        self.update_local_eval_detail(local_eval_detail_dict)
        # Collect all input / outputs of this batch,

        self.scan_data += local_size
        self.scan_batch += 1
        self.tb_point += 1
        if self.scan_batch % self.ob_batch_num == 0:
            LogInfo.logs('[%3s][eval-%s-B%d/%d] scanned = %d/%d',
                         self.name,
                         eval_dl.mode,
                         self.scan_batch,
                         eval_dl.n_batch,
                         self.scan_data,
                         len(eval_dl))
        # if self.summary_writer is not None:
        #     self.summary_writer.add_summary(summary, self.tb_point)
        #     # if batch_idx == 0:
        #     #     summary_writer.add_run_metadata(run_metadata, 'epoch-%d' % epoch_idx)

    def evaluate_all(self, eval_dl, detail_fp=None, result_fp=None):
        self.reset_eval_info()
        for batch_idx in range(eval_dl.n_batch):
            self.evaluate(eval_dl=eval_dl, batch_idx=batch_idx)
        metric_ret_list = self.post_process(eval_dl=eval_dl,
                                            detail_fp=detail_fp,
                                            result_fp=result_fp)
        assert len(self.metric_name_list) == len(metric_ret_list)
        for metric_name, val in zip(self.metric_name_list, metric_ret_list):
            LogInfo.logs('[%3s] %s_%s = %.6f', self.name, eval_dl.mode, metric_name, val)      # [ rm] train_F1 = xx
        if len(metric_ret_list) == 1:
            return metric_ret_list[0]       # unpack
        return metric_ret_list

    # ====================== Below: functions to override ====================== #

    # Given all the evaluation detail, calculate the final result.
    def post_process(self, eval_dl, detail_fp, result_fp):
        raise NotImplementedError
