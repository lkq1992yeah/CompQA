# -*- coding:utf-8 -*-

import tensorflow as tf


class SiameseNetwork(object):

    def __init__(self, lf_module=None, rt_module=None, merge_func=None, share_weights=False):
        if not share_weights:
            assert lf_module is not None
            assert rt_module is not None
            self.lf_module = lf_module
            self.rt_module = rt_module
        else:
            assert lf_module is not None
            self.lf_module = self.rt_module = lf_module

        self.merge_func = merge_func
            

    def build(self, lf_input, rt_input):
        with tf.variable_scope('siamese'):
            lf_hidden = self.lf_module.build(lf_input)
            rt_hidden = self.rt_module.build(rt_input)
            merged_output = self.merge_func(lf_hidden, rt_hidden)
        return merged_output

