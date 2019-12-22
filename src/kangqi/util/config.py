# -*- coding: utf-8 -*-

from .LogUtil import LogInfo

def load_configs(fp):
    LogInfo.begin_track('Loading config from %s: ', fp)
    config_dict = {}
    with open(fp, 'r') as br:
        for line in br.readlines():
            line = line.strip()
            if line.startswith('#') or line == '': continue
            if line.find('\t') == -1: continue
            spt = line.split('\t')
            if len(spt) < 2: continue
            k = spt[0]
            v_str = spt[1]
            config_dict[k] = v_str
            LogInfo.logs('%s = %s', k, v_str)
    LogInfo.end_track()
    return config_dict

# used for tuning
def get_param_list(config_dict, name, tp):
    return [tp(s) for s in config_dict[name].split(',')]



class ConfigDict:

    def __init__(self, fp):
        self.config_dict = load_configs(fp)

    # given the name of a parameter, the value of which is a list,
    # return all the elements in the list
    def get_param_list(self, name, tp=str):
        return [tp(s) for s in self.config_dict[name].split(',')]

    # given a list of parameter name, return all the values of each parameter in the list
    def get_diff_params(self, name_list, tp=str):
        return [tp(self.config_dict[name]) for name in name_list]

    # get a single value
    def get(self, name, tp=str):
        return tp(self.config_dict[name])

    def get_int(self, name):
        return int(self.config_dict[name])

    def get_float(self, name):
        return float(self.config_dict[name])

    def get_bool(self, name):
        return True if self.config_dict[name] == 'True' else False

    def __getattr__(self, item):
        if item not in self.config_dict:
            LogInfo.logs('Unknown attribute: %s', item)
            return None
        val = self.config_dict[item]
        if val == 'True':
            return True
        elif val == 'False':
            return False
        else:
            try:
                int_val = int(val)
                return int_val
            except ValueError:
                try:
                    float_val = float(val)
                    return float_val
                except ValueError:
                    return val
