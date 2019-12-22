"""
Author: Kangqi Luo
Date: 180206
Goal: Maintain seg dataset from:
      CoNLL 2003 (gold)
      WebQ/CompQ/SimpQ (silver)
"""

# TODO: Finish gold first

from seg_ds_helper.simpq_seg_helper import load_annotations_bio

from kangqi.util.LogUtil import LogInfo


class SegmentDataset:

    def __init__(self, seg_data_name, q_max_len, word_dict):
        self.q_max_len = q_max_len
        self.seg_data_name = seg_data_name
        self.np_data_list = []

        if seg_data_name == 'SimpQ':
            self.np_data_list = load_annotations_bio(word_dict=word_dict,
                                                     q_max_len=q_max_len)
