# -*- coding: utf-8 -*-

#==============================================================================
# Author: Kangqi Luo
# Goal: Display results of different specifications
# Main advantage: can group by different parameters, find where's the tuning point.
# More detail:
# 1. the displayer collects all trend files with the path format like "root_path/spec_path/trend"
# 2. spec_path contains all the main parameters, and we split all trend files into several groups according to
#    some parameters specified by the user.
# 3. For each group, we show the rank of results (order by metric on the validation set),
#    mark the result with the hightest testing metric (no matter what validation set it has)
#    and also show the average result in this group.
#==============================================================================

import os

from kangqi.util.LogUtil import LogInfo

class Grid_Display(object):

    def __init__(self, root_path_list):
        self.result_tup_list = []

        for root_path in root_path_list:
            subdirs = os.listdir(root_path)
            for subdir in subdirs:
                param_list = subdir.split('_')
                fp_1 = '%s/%s/trend' %(root_path, subdir)
                fp_2 = '%s/%s/log.trend' %(root_path, subdir)
                if os.path.isfile(fp_1):
                    self.result_tup_list.append((fp_1, param_list))
                elif os.path.isfile(fp_2):
                    self.result_tup_list.append((fp_2, param_list))
        LogInfo.logs('%d result file collected.', len(self.result_tup_list))
#        for fp, param_list in result_tup_list:
#            LogInfo.logs('%s --> %s', fp, param_list)

    def show_results_split_by_param(self, param_name, out_fp, is_larger_better=True):
        group_dict = {}     # result files grouped by different param.
        if param_name == 'All':
            group_dict['All'] = self.result_tup_list
        else:
            for tup in self.result_tup_list:
                fp, param_list = tup
                param_val = '[Missing]'
                for param in param_list:        # find the value of the corresponding parameter (could be missing)
                    if param.startswith(param_name):
                        param_val = param
                        break
                if param_val not in group_dict:
                    group_dict[param_val] = []
                group_dict[param_val].append(tup)
        key_list = list(group_dict.keys())
        key_list.sort()

        ret_dict = {}       # {<param, avg_test_value>}
        bw = open(out_fp, 'w')
        for key in key_list:
            avg_test_val = self.show_results_in_a_group(key, group_dict[key], bw, is_larger_better)
            ret_dict[key] = avg_test_val
        srt_tup_list = sorted(ret_dict.items(),
              lambda x, y: cmp(x[1], y[1]) * (-1 if is_larger_better else 1))
        bw.write('# ======== Choice of parameters (show in Avg_t_result) ========\n')
        for key, avg_t_val in srt_tup_list:
            bw.write('%s\t%.6f\n' %(key, avg_t_val))
        bw.close()

    def show_results_in_a_group(self, key, result_tup_list, bw, is_larger_better):
        sum_values = [0.] * 4    # sum T_loss, T_result, v_result, t_result in this group
        info_tup_list = []       # [(fp, result_line, v_result, t_result)]
        for fp, _ in result_tup_list:
            with open(fp, 'r') as br:
                lines = br.readlines()
            result_line = lines[-2].strip()
            spt = result_line.split('\t')
            vals = T_loss, T_result, v_result, t_result = \
                [float(spt[idx + 1].strip()) for idx in range(4)]
            for idx in range(4):
                sum_values[idx] += vals[idx]
            info_tup_list.append((fp, result_line, v_result, t_result))
        for idx in range(4):
            sum_values[idx] /= len(info_tup_list)

        if is_larger_better:
            info_tup_list.sort(lambda x, y: -cmp(x[2], y[2]))
        else:
            info_tup_list.sort(lambda x, y: cmp(x[2], y[2]))

        best_t_val = -10000. if is_larger_better else 10000.
        best_t_idx = -1     # find the index with highest testing result (no matter what valid result it is)
        for idx, tup in enumerate(info_tup_list):
            t_val = tup[-1]
            flag = t_val > best_t_val if is_larger_better else t_val < best_t_val
            if flag:
                best_t_val = t_val
                best_t_idx = idx

        # Now starts displaying
        LogInfo.logs('Displaying %s ... ', key)
        bw.write('# ================ [%s] ================\n' %key)
        for idx, tup in enumerate(info_tup_list):
            fp, result_line, v_result, t_result = tup
            bw.write('%s%s\n' %(result_line, ' (Largest in Test)' if idx == best_t_idx else ''))
            bw.write('# %s\n' %fp)
        bw.write('%-8s\t%.6f\t%.6f\t%.6f\t%.6f\n' %(
                '# [Average]', sum_values[0], sum_values[1], sum_values[2], sum_values[3]))
        bw.write('# ================================\n\n')
        return sum_values[-1]           # return average test value


if __name__ == '__main__':
    LogInfo.begin_track('kangqi.util.grid_search.result_display starts ... ')
    root_path_list = [
        'runnings/tabel/e2e/noj_model_try_9'
    ]
    display = Grid_Display(root_path_list)
#    display.show_results_split_by_param('b', 'gs_b.txt')
#    display.show_results_split_by_param('lr', 'gs_lr.txt')
    display.show_results_split_by_param('pre', 'gs_pre.txt')
    display.show_results_split_by_param('m', 'gs_m.txt')
    display.show_results_split_by_param('reg', 'gs_reg.txt')
    display.show_results_split_by_param('Reg', 'gs_Reg.txt')
    display.show_results_split_by_param('All', 'gs_All.txt')
    LogInfo.end_track()
