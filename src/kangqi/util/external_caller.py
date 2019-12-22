# -*- coding: utf-8 -*-

#==============================================================================
# Author: Kangqi Luo
# Goal: Call external function
#==============================================================================

import subprocess

from kangqi.util.LogUtil import LogInfo

def external_calling(proc_list, stdout=None):
    LogInfo.begin_track('Calling external program ... ')
    proc_str = ' '.join(proc_list)
    LogInfo.logs('%s', proc_str)
    LogInfo.logs('================================================================')
    if stdout is None:
        p1 = subprocess.Popen(proc_list)
        p1.wait()
    else:
        LogInfo.logs('Redirected to [%s]', stdout)
        with open(stdout, 'w') as bw:
            p1 = subprocess.Popen(proc_list, stdout=bw)
            p1.wait()
    LogInfo.end_track('================================================================')