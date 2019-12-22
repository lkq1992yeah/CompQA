# -*- code: utf-8 -*-
from pipeline import *
from table import *
from feature import *
import numpy as np
from ranknet_testing import Ranker

def load_test_data(path='/home/xusheng/TabelProject/data/tabM.vec.raw.test'):
    f = open(path, 'r')
    data = json.load(f)
    tables = []
    for t in data:
        cells = []
        r_count = 0
        for r in t:
            c_count = 0
            for c in r:
                if 'translate' in c.keys() and 'en-entity' in c.keys():
                    surface = c['translate']
                    label = c['en-entity'].replace('_', ' ')
                    cell = Cell(surface, (r_count, c_count), ground=label)
                else:
                    cell = Cell(None, (r_count, c_count), None, True)
                cells.append(cell)
                c_count = c_count + 1
            r_count = r_count + 1
        table = Table((r_count, c_count), cells)
        tables.append(table)

    f.close()
    return tables

def load_test_data_new(path='/home/xusheng/PythonProject/data/tabel/monolingual/uniform_data.json', top=50):
    zh2en, zh2cand = load_dict()
    f = open(path, 'r')
    data = json.load(f)
    tables = []
    for t in data[-50:]:
        cells = []
        r_count = 0
        for r in t:
            c_count = 0
            for c in r:
                if 'mention' in c.keys() and 'en-entity' in c.keys():
                    zh_surface = c['mention']
                    label = c['en-entity'].replace('_', ' ')
                    surface = zh2en[zh_surface]
                    cell = Cell(surface, (r_count, c_count), ground=label)
                    # load candidate
                    cands = zh2cand[zh_surface]
                    for cand in cands[:top]:
                        m = Mention(surface, cand[0], cell, cell.pos)
                        m.foo = [cand[1]]
                        cell.mentions.append(m)
                else:
                    cell = Cell(None, (r_count, c_count), None, True)
                cells.append(cell)
                c_count = c_count + 1
            r_count = r_count + 1
        table = Table((r_count, c_count), cells)
        tables.append(table)

    f.close()
    return tables

def link_table(table, prior, feature, ranker, max_iter):
    # generate candidates
    # candidate_generation(table, prior)
    # initial assignment
    count_all = 0
    count_pos = 0
    for cell in table.cells:
        if cell.blank:
            continue
        if len(cell.mentions) == 0:
            cell.blank = True
            count_all += 1
            continue
        cell.label = cell.mentions[0].entity
        flag = 0
        for ment in cell.mentions:
            # print ment.entity, cell.ground
            if ment.entity == cell.ground:
                flag = 1
                break
        count_pos += flag
        count_all += 1
    # print 'covered:', count_pos, '/', count_all
    # iter
    for iter in range(max_iter):
        # compute feature
        for cell in table.cells:
            if cell.blank:
                continue
            for mention in cell.mentions:
                ft = feature.extract_features(table, cell.pos, mention, mention.entity)
                mention.feature = np.asarray(ft)
        # re-assign
        has_change = False
        for cell in table.cells:
            if cell.blank:
                continue
            feats = np.asarray([mention.feature for mention in cell.mentions])
            best = ranker.rank(feats)
            # print best, feats
            new_label = cell.mentions[best].entity
            if new_label != cell.label:
                has_change = True
                cell.label = new_label
        if not has_change:
            break             

def test():
    
    ranker = Ranker()
    prior, wikigraph, entity_context, weight, all_entities = preprocess()
    features = Features(prior, wikigraph, entity_context, weight, all_entities)
    max_iter = 100

    for top in range(1, 51):
        tables = load_test_data_new(top=top)
        accu_macro = []
        totol_correct = 0
        totol_count = 0
        for table in tables:
            count = 0
            for cell in table.cells:
                if not cell.blank:
                    count += 1
                    totol_count += 1
            link_table(table, prior, features, ranker, max_iter)
            correct = 0
            for cell in table.cells:
                if not cell.blank:
                    if cell.label == cell.ground:
                        correct += 1
                        totol_correct += 1
            print 'accu:', correct, '/', count
            accu_macro.append(correct / float(count))
    
        print 'accuracy_macro:', sum(accu_macro) / len(accu_macro)
        print 'accuracy_micro:', totol_correct / float(totol_count)


if __name__ == '__main__':
    test()