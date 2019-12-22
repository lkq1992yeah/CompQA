"""
Copied from emnlp_dataloader
"""

import tensorflow as tf
import numpy as np

from kangqi.util.LogUtil import LogInfo


class BaseDataLoader(object):
    """
    Dataloader with simpler functions
    Try use dictionary in np_data_list
    """

    def __init__(self, batch_size):
        self.batch_size = batch_size                # how many data in each batch
        self.batch_np_data_list = None
        self.batch_real_size_list = None
        self.n_batch = None  # number of batches
        self.n_rows = None

    def __len__(self):
        return self.n_rows

    def prepare_np_input_list(self, global_input_dict, input_tensor_dict, n_rows):
        """
        :param global_input_dict:   <input_name, [np_value of each data point]>
        :param input_tensor_dict:   <input_name, tensor definition from model>
        :param n_rows: total number of data
        """
        self.batch_np_data_list = []
        self.batch_real_size_list = []
        self.n_rows = remain_rows = n_rows
        while remain_rows > 0:
            self.batch_np_data_list.append({})
            active_size = min(remain_rows, self.batch_size)
            self.batch_real_size_list.append(active_size)
            remain_rows -= active_size
        self.n_batch = len(self.batch_real_size_list)
        LogInfo.logs('n_rows = %d, batch_size = %d, n_batch = %d.', self.n_rows, self.batch_size, self.n_batch)

        """ Now enumerate each input name in global data """
        for key, input_data_list in global_input_dict.items():
            if key not in input_tensor_dict:        # the model doesn't receive such input
                continue
            if len(input_data_list) != n_rows:
                # len == n_rows: input values for all data points are present
                # len == 0: direct input is not available (generated in post-process)
                # Not allowed if input values are partially provided
                LogInfo.logs('Warning: len(%s) = %d, mismatch with n_row = %d.',
                             key, len(input_data_list), n_rows)
                continue
            dtype = input_tensor_dict[key].dtype
            np_type = 'float32' if dtype == tf.float32 else 'int32'
            for batch_idx in range(self.n_batch):
                st_idx = batch_idx * self.batch_size
                ed_idx = st_idx + self.batch_size
                local_batch_input = input_data_list[st_idx: ed_idx]
                np_arr = np.array(local_batch_input, dtype=np_type)
                self.batch_np_data_list[batch_idx][key] = np_arr

    def get_batch(self, batch_idx):
        local_data = self.batch_np_data_list[batch_idx]
        local_size = self.batch_real_size_list[batch_idx]
        return local_data, local_size
