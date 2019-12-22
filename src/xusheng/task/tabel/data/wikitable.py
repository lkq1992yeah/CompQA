from wikitables import import_tables
from kangqi.util.LogUtil import LogInfo
import json
import sys
import io
import codecs
reload(sys)
sys.setdefaultencoding('utf-8')

UTF8Writer = codecs.getwriter('utf8')
sys.stdout = UTF8Writer(sys.stdout)

def get_list():
    name_list = list()
    fp = '/home/xusheng/wikipedia/zh-extracted/list.txt'
    with open(fp, 'rb') as fin:
        for line in fin.readlines():
            name_list.append(line.strip())
    return name_list

def valid(s):
    if len(s) < 6 or len(s) > 30:
        return False

    cnt = 0
    for ch in s:
        if '0'<=ch<='9':
            cnt += 1
    if float(cnt) / len(s) > 0.2:
        return False
    else:
        return True

if __name__=='__main__':
    name_list = get_list()
    bw = open("/home/xusheng/PythonProject/data/new_tables.txt", 'w')
    LogInfo.begin_track("Grab %d potential tables", len(name_list))
    cnt = 0
    for name in name_list:
        try:
            tables = import_tables(name, 'zh')
        except:
            LogInfo.logs("[error] %s", name)
            continue
        LogInfo.logs("Get %d table from %s", len(tables), name)
        if len(tables) == 0: continue
        LogInfo.begin_track("detail tables for %s", name)
        for idx, table in enumerate(tables):
            row_num = len(table.rows)
            if row_num == 0:
                continue
            col_num = len(table.rows[0].keys())
            if row_num < 6 or col_num < 3:
                continue
            LogInfo.logs("%d rows %d cols in #%d table", row_num, col_num, idx)
            tmp_row = table.rows[1]
            valid_list = list()
            LogInfo.logs("tmp: %s", str(tmp_row.values()))
            for x, v in enumerate(tmp_row.values()):
                if valid(str(v)):
                    valid_list.append(x)
                else:
                    LogInfo.logs("%s is not valid.", str(v))
            if len(valid_list) < 3:
                LogInfo.logs("too many non-zh in #%d table ", idx)
                continue
            LogInfo.logs("save #%d table", idx)
            for row in table.rows:
                s = ""
                for x, v in enumerate(row.items()):
                    if x not in valid_list:
                        continue
                    s += str(v) + "\t"
                LogInfo.logs(s.strip())
                bw.write(s.strip() + "\n")
            bw.write("\n")
        LogInfo.end_track()

    LogInfo.logs("%d tables grabbed.", cnt)
    LogInfo.end_track()

