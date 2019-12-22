# -*- coding:utf-8 -*-

import random

from kangqi.util.LogUtil import LogInfo


class DataLoader(object):
    """
    Goal: The dataloader has two main usages:
    1. save the input of train/valid/test data into np_data_list, we can access every single data through the index
    2. dynamically maintain the data that we use in each train/valid/test iteration.
    """

    def __init__(self, batch_size, proc_ob_num=500,
                 mode=None, dynamic=False, shuffle=False, np_data_list=None):
        self.n_batch = None                         # number of batches
        self.indices_list = None                    # maintain the indices of all data in the dataloader
        self.cur_batch_idx = 0                      # recording the current batch index
        # self.history_batch_dict = {}

        self.batch_size = batch_size                # how many data in each batch
        self.proc_ob_num = proc_ob_num              # number for logs when processing data
        self.mode = mode                            # T/v/t
        self.np_data_list = np_data_list            # the numpy-format input of each data
        self.dynamic = dynamic                      # whether the dataloader is dynamic updated
        self.shuffle = shuffle                      # whether the data we use is shuffled

        if self.np_data_list is not None:
            self.update_statistics()

    def __len__(self):
        return len(self.np_data_list[0])

    # Ready to produce new experimental data, and feed them into np_data_list
    # default: won't renew the data, just an empty func and update some statistics
    # Check subclass for detail.
    def renew_data_list(self):
        self.update_statistics()

    def update_statistics(self):
        # self.history_batch_dict = {}
        total_size = len(self.np_data_list[0])
        self.indices_list = range(total_size)
        if self.shuffle:
            LogInfo.logs('Data Shuffled.')
            random.shuffle(self.indices_list)
        self.n_batch = (total_size - 1) / self.batch_size + 1
        self.cur_batch_idx = 0

    def prepare_data(self):
        if self.dynamic or self.np_data_list is None or self.indices_list is None:
            # create a brand new data for the model
            self.renew_data_list()
        elif self.shuffle:
            # just change the order
            self.update_statistics()
        LogInfo.logs('data size = %d, num of batch = %d.', len(self), self.n_batch)

    def get_next_batch(self):
        local_data_list, local_indices = self.get_batch(batch_idx=self.cur_batch_idx)
        self.cur_batch_idx = (self.cur_batch_idx + 1) % self.n_batch
        return local_data_list, local_indices

    def get_batch(self, batch_idx):
        # if batch_idx in self.history_batch_dict:
        #     local_data_list, local_indices = self.history_batch_dict[batch_idx]
        # else:
        st_idx = batch_idx * self.batch_size
        ed_idx = st_idx + self.batch_size
        local_indices = self.indices_list[st_idx:ed_idx]
        local_data_list = [data[local_indices] for data in self.np_data_list]
        # self.history_batch_dict[batch_idx] = (local_data_list, local_indices)
        return local_data_list, local_indices
