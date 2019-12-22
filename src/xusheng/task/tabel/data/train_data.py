# -*- coding:utf-8 -*-
# import copy
import codecs
import json
import sys

# import pickle
import numpy as np

from kangqi.util.LogUtil import LogInfo
from xusheng.util.tf_util import get_variance


# OLD/unused version of training data generator
# Read from several .json files

def is_zero(vec):
    if vec[0] == 0 and vec[1] == 0 and vec[2] == 0:
        return True
    else:
        return False

def get_train_data():
    reload(sys)
    sys.setdefaultencoding('utf-8')

    UTF8Writer = codecs.getwriter('utf8')
    sys.stdout = UTF8Writer(sys.stdout)

    data_fp = "/home/xusheng/TabelProject/data"
    data =dict()

    # Get training data, shape: (batch_size, rows, columns, w2v_dim)
    data['cell'] = list()
    data['entity'] = list()
    data['coherence'] = list()
    data['context'] = list()
    PN = 50
    rows = 16
    cols = 6

    # step 1. cell data, filling zero-vecs and add duplicates
    # matching to negative data in entity data
    LogInfo.begin_track("Convert cell data...")
    table_fp =data_fp + "/tabM.vec.raw.train"
    effect = 0
    with open(table_fp, 'r') as fin:
        tables = json.loads(fin.readline())
        for table in tables:
            tlist = []
            row_num = 0
            for row in table:
                row_num += 1
                rlist = []
                col_num = 0
                for cell in row:
                    col_num += 1
                    try:
                        newCell = [float(val) for val in cell["vec"].strip().split(" ")]
                        rlist.append(newCell)
                    except ValueError:
                        LogInfo.logs("[error] %s", cell["vec"])
                    if not(is_zero(newCell)):
                        effect += 1
                # filling cell zero-vecs for each row
                for i in range(cols - col_num):
                    rlist.append([0.0 for val in range(100)])
                tlist.append(rlist)
            # filling row zero-vecs for each table
            for i in range(rows - row_num):
                zrow = []
                for j in range(cols):
                    zrow.append([0.0 for val in range(100)])
                tlist.append(zrow)
            # duplicate mention table for 20 times,
            # matching to positive+negative entity data
            for i in range(PN):
                data['cell'].append(tlist)

        LogInfo.logs("Cell data generated (%d). shape: %s", effect, np.array(data['cell']).shape)
    LogInfo.end_track()

    # step 2. entity data, merge positive data + negative data into one tensor
    LogInfo.begin_track("Convert entity data...")
    tabEP_fp = data_fp + "/tabEP.vec.raw.train"
    tabEN_fp = data_fp + "/tabEN.vec.raw.train." + str(PN-1)
    tablesP = json.loads(open(tabEP_fp, 'r').readline())
    tablesN = json.loads(open(tabEN_fp, 'r').readline())
    effect = 0
    for i in range(len(tablesP)):
        # 1 positive table
        tlist = []
        table = tablesP[i]
        row_num = 0
        for j in range(len(table)):
            rlist = []
            row = table[j]
            row_num += 1
            col_num = 0
            for k in range(len(row)):
                cell = row[k]
                col_num += 1
                try:
                    newCell = [float(val) for val in cell["vec"].strip().split(" ")]
                    rlist.append(newCell)
                except KeyError:
                    LogInfo.logs("[error@pos] %s", cell)
                if not(is_zero(newCell)):
                    effect += 1
            # filling cell zero-vecs for each row
            for l in range(cols - col_num):
                rlist.append([0.0 for val in range(100)])
            tlist.append(rlist)
        # filling row zero-vecs for each table
        for l in range(rows - row_num):
            zrow = []
            for m in range(cols):
                zrow.append([0.0 for val in range(100)])
            tlist.append(zrow)
        data['entity'].append(tlist)
        # LogInfo.logs("Positive table #%d converted.", i+1)

        ####################

        # 19 negative tables
        for r in range(i*(PN-1), i*(PN-1)+PN-1):
            tlist = []
            table = tablesN[r]
            row_num = 0
            for j in range(len(table)):
                rlist = []
                row = table[j]
                row_num += 1
                col_num = 0
                for k in range(len(row)):
                    cell = row[k]
                    col_num += 1
                    try:
                        rlist.append([float(val) for val in cell["vec"].strip().split(" ")])
                    except KeyError:
                        LogInfo.logs("[error@neg] %s", cell)
                # filling cell zero-vecs for each row
                for l in range(cols - col_num):
                    rlist.append([0.0 for val in range(100)])
                tlist.append(rlist)
            # filling row zero-vecs for each table
            for l in range(rows - row_num):
                zrow = []
                for m in range(cols):
                    zrow.append([0.0 for val in range(100)])
                tlist.append(zrow)
            data['entity'].append(tlist)
            # LogInfo.logs("Negative table #%d converted.", r+1)

    LogInfo.logs("Entity data generated (%d). shape: %s",
                 effect, np.array(data['entity']).shape)
    LogInfo.end_track()

    # step 3 coherence data, Kenny's version
    LogInfo.begin_track("Generate coherence data...")
    data['coherence'] = get_variance(data['entity'])
    LogInfo.logs("Coherence data generated. shape: %s",
                 np.array(data['coherence']).shape)
    LogInfo.end_track()

    # OLD step 3. Coherence data, basically rearrange entity data
    # LogInfo.begin_track("Generate coherence data...")
    # eTables = data['entity']
    # coTables = copy.deepcopy(eTables)
    # for i, eTable in enumerate(eTables):
    #     for k in range(cols):
    #         row_num = 0
    #         for j in range(rows):
    #             cell = eTable[j][k]
    #             if is_zero(cell):
    #                 continue
    #             else:
    #                 coTables[i][row_num][k] = copy.deepcopy(cell)
    #                 row_num += 1
    #         for j in range(row_num, rows):
    #             coTables[i][j][k] = [0.0 for val in range(100)]
    # data['coherence'] = coTables
    # LogInfo.logs("Coherence data generated. shape: %s",
    #              np.array(data['coherence']).shape)
    # LogInfo.end_track()

    # OLD context data for CNN layer, but removed in latest version
    # # step 4. Context data
    # LogInfo.begin_track("Generate context data...")
    # cTables = data['cell']
    # dim = rows + cols - 2
    # for i, cTable in enumerate(cTables):
    #     table = []
    #     for j, cRow in enumerate(cTable):
    #         row = []
    #         for k, cell in enumerate(cRow):
    #             # sub is a 3-dim tensor
    #             sub = []
    #             subRow = []
    #             if cell[0] == 0 and cell[1] == 0:
    #                 for l in range(dim):
    #                     subRow.append([0.0 for val in range(100)])
    #             else:
    #                 for l in range(rows):
    #                     if l != j:
    #                         subRow.append(cTables[i][l][k])
    #                 for l in range(cols):
    #                     if l != k:
    #                         subRow.append(cTables[i][j][l])
    #             sub.append(subRow)
    #             row.append(sub)
    #         table.append(row)
    #     data['context'].append(table)
    #     if (i+1) % 200 == 0:
    #         LogInfo.logs("Context table #%d converted.", (i+1)/20)
    #
    # LogInfo.logs("Context data generated. shape: %s", np.array(data['context']).shape)
    # LogInfo.end_track()

    # step 4. Context data
    # step 4 context data
    LogInfo.begin_track("Generate context data...")
    cTables = data['cell']
    dim = rows + cols - 2
    for i, cTable in enumerate(cTables):
        table = []
        for j, cRow in enumerate(cTable):
            row = []
            for k, cell in enumerate(cRow):
                # sub is a 3-dim tensor
                if is_zero(cell):
                    row.append(cell)  # all 0s
                else:
                    cnt = 0
                    ncell = np.array([0.0 for val in range(100)])
                    for l in range(rows):
                        if l != j and not (is_zero(cTables[i][l][k])):
                            ncell += np.array(cTables[i][l][k])
                            cnt += 1
                    for l in range(cols):
                        if l != k and not (is_zero(cTables[i][j][l])):
                            ncell += np.array(cTables[i][j][l])
                            cnt += 1
                    if cnt != 0:
                        ncell /= cnt
                        # LogInfo.logs(ncell.shape)
                    row.append(ncell)
            table.append(row)
        data['context'].append(table)
        if (i + 1) % 200 == 0:
            LogInfo.logs("Context table #%d converted.", (i + 1) / 20)
    LogInfo.logs("Context data generated. shape: %s", np.array(data['context']).shape)
    LogInfo.end_track()

    return data

def get_non_joint_train_data():
    data_fp = "/home/xusheng/TabelProject/data"
    data = dict()

    # Get training data, shape: (batch_size, rows, columns, w2v_dim)
    data['cell'] = list()
    data['entity'] = list()
    data['context'] = list()
    rows = 16
    cols = 6

    # step 1. cell data, select all non-zero vecs
    # matching to negative data in entity data
    LogInfo.begin_track("Convert cell data...")
    table_fp = data_fp + "/tabM.vec.raw.train"
    effect = 0
    with open(table_fp, 'r') as fin:
        tables = json.loads(fin.readline())
        for table in tables:
            tlist = []
            row_num = 0
            for row in table:
                row_num += 1
                rlist = []
                col_num = 0
                for cell in row:
                    col_num += 1
                    try:
                        newCell = [float(val) for val in cell["vec"].strip().split(" ")]
                        rlist.append(newCell)
                    except ValueError:
                        LogInfo.logs("[error] %s", cell["vec"])
                    if not (is_zero(newCell)):
                        effect += 1
                # filling cell zero-vecs for each row
                for i in range(cols - col_num):
                    rlist.append([0.0 for val in range(100)])
                tlist.append(rlist)
            # filling row zero-vecs for each table
            for i in range(rows - row_num):
                zrow = []
                for j in range(cols):
                    zrow.append([0.0 for val in range(100)])
                tlist.append(zrow)
            # duplicate mention table for 20 times,
            # matching to positive+negative entity data
            for i in range(20):
                data['cell'].append(tlist)

        LogInfo.logs("Cell data generated (%d). shape: %s", effect, np.array(data['cell']).shape)
    LogInfo.end_track()

    # step 2. entity data, merge positive data + negative data into one tensor
    LogInfo.begin_track("Convert entity data...")
    tabEP_fp = data_fp + "/tabEP.vec.raw.train"
    tabEN_fp = data_fp + "/tabEN.vec.raw.train"
    tablesP = json.loads(open(tabEP_fp, 'r').readline())
    tablesN = json.loads(open(tabEN_fp, 'r').readline())
    effect = 0
    for i in range(len(tablesP)):
        # 1 positive table
        tlist = []
        table = tablesP[i]
        row_num = 0
        for j in range(len(table)):
            rlist = []
            row = table[j]
            row_num += 1
            col_num = 0
            for k in range(len(row)):
                cell = row[k]
                col_num += 1
                try:
                    newCell = [float(val) for val in cell["vec"].strip().split(" ")]
                    rlist.append(newCell)
                except KeyError:
                    LogInfo.logs("[error@pos] %s", cell)
                if not (is_zero(newCell)):
                    effect += 1
            # filling cell zero-vecs for each row
            for l in range(cols - col_num):
                rlist.append([0.0 for val in range(100)])
            tlist.append(rlist)
        # filling row zero-vecs for each table
        for l in range(rows - row_num):
            zrow = []
            for m in range(cols):
                zrow.append([0.0 for val in range(100)])
            tlist.append(zrow)
        data['entity'].append(tlist)
        # LogInfo.logs("Positive table #%d converted.", i+1)

        ####################

        # 19 negative tables
        for r in range(i * 19, i * 19 + 19):
            tlist = []
            table = tablesN[r]
            row_num = 0
            for j in range(len(table)):
                rlist = []
                row = table[j]
                row_num += 1
                col_num = 0
                for k in range(len(row)):
                    cell = row[k]
                    col_num += 1
                    try:
                        rlist.append([float(val) for val in cell["vec"].strip().split(" ")])
                    except KeyError:
                        LogInfo.logs("[error@neg] %s", cell)
                # filling cell zero-vecs for each row
                for l in range(cols - col_num):
                    rlist.append([0.0 for val in range(100)])
                tlist.append(rlist)
            # filling row zero-vecs for each table
            for l in range(rows - row_num):
                zrow = []
                for m in range(cols):
                    zrow.append([0.0 for val in range(100)])
                tlist.append(zrow)
            data['entity'].append(tlist)
            # LogInfo.logs("Negative table #%d converted.", r+1)

    LogInfo.logs("Entity data generated (%d). shape: %s",
                 effect, np.array(data['entity']).shape)
    LogInfo.end_track()

    # step 3 coherence data, kenny's version
    LogInfo.begin_track("Generate coherence data...")
    data['coherence'] = get_variance(data['entity'])
    LogInfo.logs("Coherence data generated. shape: %s",
                 np.array(data['coherence']).shape)
    LogInfo.end_track()

    # step 4. Context data
    # step 4 context data
    LogInfo.begin_track("Generate context data...")
    cTables = data['cell']
    dim = rows + cols - 2
    for i, cTable in enumerate(cTables):
        table = []
        for j, cRow in enumerate(cTable):
            row = []
            for k, cell in enumerate(cRow):
                # sub is a 3-dim tensor
                if is_zero(cell):
                    row.append(cell)  # all 0s
                else:
                    cnt = 0
                    ncell = np.array([0.0 for val in range(100)])
                    for l in range(rows):
                        if l != j and not (is_zero(cTables[i][l][k])):
                            ncell += np.array(cTables[i][l][k])
                            cnt += 1
                    for l in range(cols):
                        if l != k and not (is_zero(cTables[i][j][l])):
                            ncell += np.array(cTables[i][j][l])
                            cnt += 1
                    if cnt != 0:
                        ncell /= cnt
                        # LogInfo.logs(ncell.shape)
                    row.append(ncell)
            table.append(row)
        data['context'].append(table)
        if (i + 1) % 200 == 0:
            LogInfo.logs("Context table #%d converted.", (i + 1) / 20)
    LogInfo.logs("Context data generated. shape: %s", np.array(data['context']).shape)
    LogInfo.end_track()

    return data

if __name__ == "__main__":
    # data_fp = "/home/xusheng/TabelProject/data"
    # with open(data_fp + "/train_data.pkl", 'wb') as fout:
    #     LogInfo.begin_track("Writing to train_data.pkl...")
    #     pickle.dump(get_train_data(), fout)
    #     LogInfo.end_track()
    data = get_train_data()
    fp = "/home/xusheng/TabelProject/data/m_e.vec"
    with open(fp, 'wb') as bw:
        mention = list()
        entity = list()
        cnt = 0
        for mTable, eTable in zip(data['cell'], data['entity']):
            if cnt % 20 != 0:
                cnt += 1
                continue
            cnt += 1
            for mRow, eRow in zip(mTable, eTable):
                for mCell, eCell in zip(mRow, eRow):
                    if is_zero(mCell) or is_zero(eCell):
                        continue
                    mention.append(mCell)
                    entity.append(eCell)
        LogInfo.logs("Shape: %s", np.array(mention).shape)
        LogInfo.logs("Shape: %s", np.array(entity).shape)
        np.save(bw, mention)
        np.save(bw, entity)
    LogInfo.logs("Saved into %s.", fp)
