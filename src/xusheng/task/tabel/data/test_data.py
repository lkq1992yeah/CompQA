# -*- coding:utf-8 -*-
# import copy
import codecs
import json
import pickle
import sys

import numpy as np

from kangqi.util.LogUtil import LogInfo
from xusheng.util.tf_util import get_variance


# OLD/unused version of validation/testing data generator
# Read from several .json files

def is_zero(vec):
    if vec[0] == 0 and vec[1] == 0 and vec[2] == 0:
        return True
    else:
        return False

def get_test_data(split, verbose):
    reload(sys)
    sys.setdefaultencoding('utf-8')

    UTF8Writer = codecs.getwriter('utf8')
    sys.stdout = UTF8Writer(sys.stdout)

    data_fp = "/home/xusheng/TabelProject/data"

    data = dict()

    # Get training data, shape: (batch_size, rows, columns, w2v_dim)
    data['cell'] = list()
    data['entity'] = list()
    data['coherence'] = list()
    data['context'] = list()
    data['candidate'] = list()
    data['truth'] = list()
    rows = 16
    cols = 6

    # step 1. cell data, filling zero-vecs and add duplicates
    table_fp =data_fp + "/tabM.vec.raw." + split
    if verbose:
        LogInfo.begin_track("Convert cell data from %s...", table_fp)
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
                        rlist.append([float(val) for val in cell["vec"].strip().split(" ")])
                    except ValueError:
                        if verbose:
                            LogInfo.logs("[error] %s", cell["vec"])
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

            data['cell'].append(tlist)

        if verbose:
            LogInfo.logs("Cell data converted. shape: %s", np.array(data['cell']).shape)
    if verbose:
        LogInfo.end_track()

    # step 2. initial entity data
    table_fp = data_fp + "/tabEI.vec.raw." + split
    if verbose:
        LogInfo.begin_track("Convert initial entity data from %s...", table_fp)
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
                        rlist.append([float(val) for val in cell["vec"].strip().split(" ")])
                    except ValueError:
                        if verbose:
                            LogInfo.logs("[error] %s", cell["vec"])
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

            data['entity'].append(tlist)

    if verbose:
        LogInfo.logs("Initial entity data converted. shape: %s.", np.array(data['entity']).shape)
    if verbose:
        LogInfo.end_track()

    # step 3. Coherence data, basically rearrange entity data
    if verbose:
        LogInfo.begin_track("Generate coherence data...")
    data['coherence'] = get_variance(data['entity'])
    if verbose:
        LogInfo.logs("Coherence data generated. shape: %s",
                 np.array(data['coherence']).shape)
    if verbose:
        LogInfo.end_track()

    # OLD coherence data
    # if verbose:
    #     LogInfo.begin_track("Generate coherence data...")
    # eTables = data['entity']
    # coTables = copy.deepcopy(eTables)
    # for i, eTable in enumerate(eTables):
    #     for k in range(cols):
    #         row_num = 0
    #         for j in range(rows):
    #             cell = eTable[j][k]
    #             if cell[0] == 0 and cell[1] == 0:
    #                 continue
    #             else:
    #                 coTables[i][row_num][k] = copy.deepcopy(cell)
    #                 row_num += 1
    #         for j in range(row_num, rows):
    #             coTables[i][j][k] = [0.0 for val in range(100)]
    # data['coherence'] = coTables
    # if verbose:
    #     LogInfo.logs("Coherence data generated. shape: %s",
    #              np.array(data['coherence']).shape)
    # if verbose:
    #     LogInfo.end_track()

    # step 4. Context data

    if verbose:
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
            if verbose:
                LogInfo.logs("Context table #%d converted.", (i + 1) / 20)
    if verbose:
        LogInfo.logs("Context data generated. shape: %s", np.array(data['context']).shape)
    if verbose:
        LogInfo.end_track()

    # LogInfo.begin_track("Generate context data from data[cell]...")
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
    #                     subRow.append([0 for val in range(100)])
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
    #     # LogInfo.logs("Context table #%d converted.", i + 1)
    #
    # LogInfo.logs("Context data generated. shape: %s.", np.array(data['context']).shape)
    # LogInfo.end_track()

    # step 5. candidate cells
    table_fp = data_fp + "/tabEC.vec.raw." + split
    if verbose:
        LogInfo.begin_track("Convert cell candidate data from %s...", table_fp)
    with open(table_fp, 'r') as fin:
        tables = json.loads(fin.readline())
        for table in tables:
            tlist = []
            for cell in table:
                cdict = dict()
                cdict['row'] = cell['row']
                cdict['col'] = cell['col']
                cdict['vec'] = []
                for vec in cell['vec']:
                    cdict['vec'].append([float(val) for val in vec.strip().split(" ")])
                tlist.append(cdict)
            data['candidate'].append(tlist)
    if verbose:
        LogInfo.logs("Cell candidate data converted. shape: %s.", np.array(data['candidate']).shape)
    if verbose:
        LogInfo.end_track()

    # step 6. ground truth
    table_fp = data_fp + "/tabEP.vec.raw." + split
    if verbose:
        LogInfo.begin_track("Convert ground truth entity data from %s...", table_fp)
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
                        rlist.append([float(val) for val in cell["vec"].strip().split(" ")])
                    except ValueError:
                        if verbose:
                            LogInfo.logs("[error] %s", cell["vec"])
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

            data['truth'].append(tlist)

    if verbose:
        LogInfo.logs("Ground truth entity data converted. shape: %s.", np.array(data['truth']).shape)
    if verbose:
        LogInfo.end_track()
    # data['cell'] = np.array(data['cell']).reshape((-1, 100))
    # data['entity'] = np.array(data['entity']).reshape((-1, 100))
    # data['context'] = np.array(data['context']).reshape((-1, 1, dim, 100))
    return data

if __name__ == "__main__":
    data_fp = "/home/xusheng/TabelProject/data"
    with open(data_fp + "/test_data.pkl", 'wb') as fout:
        LogInfo.begin_track("Writing to test_data.pkl...")
        pickle.dump(get_test_data("test"), fout)
        LogInfo.end_track()
