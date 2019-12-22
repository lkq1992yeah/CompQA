# -*- coding:utf-8 -*-
import h5py
import numpy as np

from base_dataset import Dataset

from kangqi.util.LogUtil import LogInfo

class XY_Q_Sk_Dataset(Dataset):

    def __init__(self, q_hdf5_fp, sk_hdf5_fp, batch_size):
        super(XY_Q_Sk_Dataset, self).__init__(batch_size)

        LogInfo.logs('Loading [%s] ...', q_hdf5_fp)
        q_hdf5 = h5py.File(q_hdf5_fp, 'r')
        q_mat = np.array(q_hdf5['data'], dtype='int32')     # (data_size, qw_max_len)
        LogInfo.logs('q_mat: %s', np.shape(q_mat))

        LogInfo.logs('Loading [%s] ...', sk_hdf5_fp)
        sk_hdf5 = h5py.File(sk_hdf5_fp, 'r')
        sk_t3 = np.array(sk_hdf5['index'], dtype='int32')           # (data_size, PN, skw_max_len) 
        f1_mat = np.array(sk_hdf5['f1'][:,:,2], dtype='float32')    # (data_size, PN)
        # Note: sk_hdf5['f1'] contains all P/R/F1 information
        mask_mat = np.array(sk_hdf5['mask'], dtype='int32')         # (data_size, PN)
        LogInfo.logs('sk_t3: %s', np.shape(sk_t3))
        LogInfo.logs('f1_mat: %s', np.shape(f1_mat))
        LogInfo.logs('mask_mat: %s', np.shape(mask_mat))

        self.input_np_list = [q_mat, sk_t3, f1_mat, mask_mat]



if __name__ == '__main__':
    data_path = '/home/kangqi/workspace/PythonProject/data/compQA/xy_q_skeleton'
    q_hdf5_fp = '%s/webq.questions_index.hdf5' %data_path
    sk_hdf5_fp = '%s/webq.skeleton.word.hdf5' %data_path
    batch_size = 32
    dataset = XY_Q_Sk_Dataset(q_hdf5_fp, sk_hdf5_fp, batch_size)

    batch_np_list = dataset.get_next_batch()
    # for input_np in batch_np_list:
    #     LogInfo.logs('%s: %s', np.shape(input_np), input_np)
    batch_np_list = dataset.get_next_batch()
