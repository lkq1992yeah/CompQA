from kangqi.util.LogUtil import LogInfo

import time


class TimeTracker(object):

    start_dict = {}         # save the timestamp for diff. key
    time_dict = {}          # save the accumulate time
    freq_dict = {}          # save the frequency

    @staticmethod
    def start(key):
        tms = time.time()
        TimeTracker.start_dict[key] = tms

    @staticmethod
    def record(key):
        if key not in TimeTracker.start_dict:
            LogInfo.logs('[TimeTracker] Key "%s" not started.', key)
            return
        val = time.time() - TimeTracker.start_dict[key]
        if key not in TimeTracker.time_dict:
            TimeTracker.time_dict[key] = val
            TimeTracker.freq_dict[key] = 1
        else:
            TimeTracker.time_dict[key] += val
            TimeTracker.freq_dict[key] += 1
        TimeTracker.start_dict.pop(key, None)       # clear the last timestamp

    @staticmethod
    def manual_add(key, val):
        if key not in TimeTracker.time_dict:
            TimeTracker.time_dict[key] = val
            TimeTracker.freq_dict[key] = 1
        else:
            TimeTracker.time_dict[key] += val
            TimeTracker.freq_dict[key] += 1
        TimeTracker.start_dict.pop(key, None)       # clear the last timestamp

    @staticmethod
    def reset(key):
        if key in TimeTracker.time_dict:
            TimeTracker.time_dict.pop(key, None)
            TimeTracker.freq_dict.pop(key, None)
            TimeTracker.start_dict.pop(key, None)

    @staticmethod
    def reset_all():
        key_set = TimeTracker.time_dict.keys()
        for key in key_set:
            TimeTracker.reset(key)

    @staticmethod
    def display():
        LogInfo.begin_track('Time tracker display:')
        key_set = TimeTracker.time_dict.keys()
        item_list = []
        for key in key_set:
            tms = TimeTracker.time_dict[key]
            freq = TimeTracker.freq_dict[key]
            avg = tms / freq
            item_list.append((key, avg, freq, tms))
        item_list.sort(lambda x, y: -cmp(x[-1], y[-1]))     # sort by total time
        LogInfo.logs('%16s\t%8s\t%8s\t%8s', 'Name', 'Avg. (s)', 'Freq', 'Time (s)')
        for item in item_list:
            LogInfo.logs('%16s\t%8.4f\t%8d\t%8.4f', *item)
        LogInfo.end_track()
