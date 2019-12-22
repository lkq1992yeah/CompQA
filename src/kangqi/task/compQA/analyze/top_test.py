import re
import sys
import numpy as np

from kangqi.util.LogUtil import LogInfo


def coun_top_test(fp='status.csv', col_idx=5, tops=10):
    stat_list = []
    with open(fp, 'r') as br:
        for line in br.readlines():
            line = line.strip()
            line = re.sub(r' +', '\t', line)
            spt = line.split('\t')
            if len(spt) <= col_idx:
                continue
            val_str = spt[col_idx]
            try:
                val = float(val_str)
                stat_list.append(val)
            except ValueError:
                pass
    stat_list.sort(reverse=True)
    stat_list = stat_list[:tops]
    avg = np.mean(stat_list)
    LogInfo.logs('%s --> %.6f', stat_list, avg)


if __name__ == '__main__':
    if len(sys.argv) < 4:
        coun_top_test()
    else:
        argv = sys.argv
        coun_top_test(fp=argv[1], col_idx=int(argv[2]), tops=int(argv[3]))
