# -*- coding:utf-8 -*-

import tensorflow as tf
import numpy as np

from kangqi.util.LogUtil import LogInfo


class EMNLPDataLoader(object):
    """
    Dataloader with simpler functions
    Try use dictionary in np_data_list
    """

    def __init__(self, batch_size, compq_mt_model, proc_ob_num=500, mode=None):
        self.batch_size = batch_size                # how many data in each batch
        self.compq_mt_model = compq_mt_model        # the active model
        self.proc_ob_num = proc_ob_num              # number for logs when processing data
        self.mode = mode                            # T/v/t, just for display

        self.np_data_list = None
        self.n_batch = None  # number of batches
        self.n_rows = None

    def __len__(self):
        return self.n_rows

    def prepare_np_input_list(self, global_input_dict, input_tensor_names, n_rows):
        self.np_data_list = []
        self.n_batch = (n_rows - 1) / self.batch_size + 1
        self.n_rows = n_rows
        for batch_idx in range(self.n_batch):
            self.np_data_list.append({})

        for k in input_tensor_names:
            input_data_list = global_input_dict.get(k, [])
            assert len(input_data_list) == 0 or len(input_data_list) == n_rows
            """
            len == n_rows: input values for all data points are present
            len == 0: direct input is not available (generated in post-process)
            Not allowed if input values are partially provided
            """
            if len(input_data_list) == 0:
                continue
            dtype = self.compq_mt_model.input_tensor_dict[k].dtype
            np_type = 'float32' if dtype == tf.float32 else 'int32'
            for batch_idx in range(self.n_batch):
                st_idx = batch_idx * self.batch_size
                ed_idx = st_idx + self.batch_size
                local_batch_input = input_data_list[st_idx: ed_idx]
                np_arr = np.array(local_batch_input, dtype=np_type)
                self.np_data_list[batch_idx][k] = np_arr

        """ Post-process, produce local_sup_lookup, update el_sup_mask """
        if 'el_sup_mask' in input_tensor_names:
            max_local_mem = 0
            # el_sup_mask: (batch, el_max_size, mem_size)
            for batch_idx, local_input_dict in enumerate(self.np_data_list):
                st = batch_idx * self.batch_size
                el_sup_mask = local_input_dict['el_sup_mask']
                ds, el_max_size, mem_size = el_sup_mask.shape
                # Step 1: Collect all <data_idx, el_idx, path_idx> triples
                hit_tup_list = []
                for i in range(ds):
                    local_sup_path_table = global_input_dict['local_sup_path_table'][st+i]
                    for j, local_sup_path_list in enumerate(local_sup_path_table):
                        for k in local_sup_path_list:
                            hit_tup_list.append((i, j, k))
                # Step 2: Create local_sup_lookup
                local_sup_lookup = []
                local_sup_dict = {}  # Collect all distinct support paths in this batch
                for tup in hit_tup_list:
                    path_idx = tup[-1]
                    if path_idx not in local_sup_dict:
                        local_sup_dict[path_idx] = len(local_sup_dict)
                        local_sup_lookup.append(path_idx)
                local_mem_size = len(local_sup_lookup)
                max_local_mem = max(local_mem_size, max_local_mem)
                # LogInfo.logs('hit_tup_list = %d, local_mem_size = %d', len(hit_tup_list), len(local_sup_lookup))
                # Step 3: Update el_sup_mask
                new_el_sup_mask = np.zeros((ds, el_max_size, local_mem_size), dtype='float32')
                for data_idx, el_idx, old_path_idx in hit_tup_list:
                    new_path_idx = local_sup_dict[old_path_idx]
                    new_el_sup_mask[data_idx, el_idx, new_path_idx] = 1.
                local_input_dict['el_sup_mask'] = new_el_sup_mask   # update to (ds, el_max_size, local_mem_size)
                local_input_dict['local_sup_lookup'] = np.array(local_sup_lookup, dtype='int32')
            LogInfo.logs('max_local_mem = %d', max_local_mem)

        # Finally: dict to list (just for compatibility)
        for batch_idx in range(self.n_batch):
            local_input_dict = self.np_data_list[batch_idx]
            local_input_list = [local_input_dict[k] for k in input_tensor_names]
            self.np_data_list[batch_idx] = local_input_list

    def get_batch(self, batch_idx):
        local_data_list = self.np_data_list[batch_idx]
        st_idx = batch_idx * self.batch_size
        ed_idx = min(st_idx + self.batch_size, self.n_rows)
        local_indices = range(st_idx, ed_idx)       # just for compatibility
        return local_data_list, local_indices
