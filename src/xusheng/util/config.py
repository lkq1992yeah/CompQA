# -*- coding: utf-8 -*-
"""
add some modification
"""

import tensorflow as tf

from xusheng.util.log_util import LogInfo


def load_configs(fp):
    LogInfo.begin_track('Loading config from %s: ', fp)
    config_dict = {}
    with open(fp, 'r') as br:
        for line in br.readlines():
            line = line.strip()
            if line.startswith('#') or line == '':
                continue
            if line.find('\t') == -1:
                continue
            spt = line.split('\t')
            if len(spt) < 3:
                LogInfo.logs("[%s] is invalid, pls add type!", line)
                continue
            k = spt[0]
            v_str = spt[1]
            t = spt[2]
            if t == "d" or t == "int":
                config_dict[k] = int(v_str)
            elif t == "f" or t == "float" or t == "double":
                config_dict[k] = float(v_str)
            elif t == "b" or t == "bool":
                if v_str == "true" or v_str == "True" \
                        or v_str == "TRUE" or v_str == "1":
                    config_dict[k] = True
                else:
                    config_dict[k] = False
            elif t == "tf" or t == "tensorflow":
                if v_str == 'relu':
                    config_dict[k] = tf.nn.relu
                elif v_str == 'sigmoid':
                    config_dict[k] = tf.nn.sigmoid
                elif v_str == 'tanh':
                    config_dict[k] = tf.nn.tanh
            elif t == "None" or v_str == "None":
                config_dict[k] = None
            else:
                config_dict[k] = v_str
            LogInfo.logs('%s = %s', k, v_str)

    LogInfo.end_track()
    return config_dict


# used for tuning
def get_param_list(config_dict, name, re_type):
    return [re_type(s) for s in config_dict[name].split(',')]


class ConfigDict:

    def __init__(self, fp):
        self.config_dict = load_configs(fp)

    # given the name of a parameter, the value of which is a list,
    # return all the elements in the list
    def get_param_list(self, name, re_type=str):
        return [re_type(s) for s in self.config_dict[name].split(',')]

    # given a list of parameter name, return all the values of each parameter in the list
    def get_diff_params(self, name_list, re_type=str):
        return [re_type(self.config_dict[name]) for name in name_list]

    # get a single value
    def get(self, key):
        if key not in self.config_dict:
            LogInfo.logs("[warning] key [%s] not exists.")
        return self.config_dict.get(key, None)

    # add a key-value
    def add(self, key, value):
        if key in self.config_dict:
            LogInfo.logs("[warning] key already exists [%s: %s], now change to [%s].",
                         key, str(self.config_dict.get(key)), value)
        self.config_dict[key] = value
