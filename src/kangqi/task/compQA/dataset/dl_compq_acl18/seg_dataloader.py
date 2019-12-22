from ..base_dataloader import DataLoader

from kangqi.util.LogUtil import LogInfo


class SegmentDataLoader(DataLoader):

    def __init__(self, dataset, mode, shuffle, indices_list, batch_size):
        DataLoader.__init__(self, mode=mode,
                            batch_size=batch_size,
                            dynamic=False,
                            shuffle=shuffle)
        self.np_data_list = []
        for np_data in dataset.np_data_list:
            local_data = np_data[indices_list]
            self.np_data_list.append(local_data)
        self.update_statistics()
        LogInfo.logs('SegmentDataLoader [%s]: %d <q, tag> data collected.',
                     self.mode, self.np_data_list[0].shape[0])

        # for idx in range(10):
        #     LogInfo.begin_track('Show case-%d: ', idx)
        #     v = self.np_data_list[0][idx]
        #     v_len = self.np_data_list[1][idx]
        #     tag_indices = self.np_data_list[2][idx]
        #     LogInfo.logs('v: %s', v.tolist())
        #     LogInfo.logs('v_len: %d', v_len)
        #     LogInfo.logs('tag: %s', tag_indices.tolist())
        #     LogInfo.end_track()