import time


class LogInfo(object):

    lvl = 0
    time_list = []

    @staticmethod
    def get_blank():
        blank = ''
        for i in range(LogInfo.lvl):
            blank += '  '
        return blank

    @staticmethod
    def begin_track(fmt_string, *args):
        blank = LogInfo.get_blank()
        if len(args) == 0:
            print(blank + '%s' % fmt_string + ' {')
        else:
            fmt = blank + fmt_string + ' {'
            print(fmt %args)
        LogInfo.lvl += 1
        LogInfo.time_list.append(time.time())

    @staticmethod
    def logs(fmt_string, *args):
        blank = LogInfo.get_blank()
        if len(args) == 0:
            print(blank + '%s' %(fmt_string))
        else:
            fmt = blank + fmt_string
            print(fmt % args)

    @staticmethod
    def end_track(fmt_string='', *args):
        if fmt_string != '':
            LogInfo.logs(fmt_string, *args)
        LogInfo.lvl -= 1
        if LogInfo.lvl < 0:
            LogInfo.lvl = 0
        blank = LogInfo.get_blank()
        fmt = blank + '}'
        # if fmt_string != '':
        #     fmt += ' ' + fmt_string
        time_str = ''
        if len(LogInfo.time_list) >= 1:
            elapse = time.time() - LogInfo.time_list.pop()
            time_str = ' [%s]' %(LogInfo.show_time(elapse))
        fmt += time_str
        print(fmt)

    @staticmethod
    def show_time(elapse):
        ret = ''
        if elapse > 86400:
            d = elapse / 86400
            elapse %= 86400
            ret += '%dd' %(d)
        if elapse > 3600:
            h = elapse / 3600
            elapse %= 3600
            ret += '%dh' %(h)
        if elapse > 60:
            m = elapse / 60
            elapse %= 60
            ret += '%dm' % m
        ret += '%.3fs' % elapse
        return ret

    @staticmethod
    def show_line(cnt, num):
        if cnt % num == 0:
            LogInfo.logs("[log] %d lines loaded.", cnt)

if __name__ == '__main__':
    x = 10
    y = 45
    LogInfo.begin_track('Testing start ... %s %s', 'a', 'b')
    LogInfo.logs("x = %d, y = %d (%.3f%%)", x, y, 100.0 * x / y)
    LogInfo.end_track('Now finished')

