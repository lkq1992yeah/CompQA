# -*- code: utf-8 -*-
'''
use pair-wise ranking
for mention m and its candidates e_i where e_1 is the gold answer, let f(e_1) - f(e_j) be the positive examples and f(e_j) - f(e_1) be the negative examples
'''
import numpy as np
from sklearn.linear_model import LogisticRegression
from table import *
from feature import *
from pipeline import candidate_generation, preprocess, load_dict


def load_train_data(path='/home/xusheng/TabelProject/data/tabM.vec.raw.train'):
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
                    cell = Cell(surface, (r_count, c_count), label)
                else:
                    cell = Cell(None, (r_count, c_count), None, True)
                cells.append(cell)
                c_count = c_count + 1
            r_count = r_count + 1
        table = Table((r_count, c_count), cells)
        tables.append(table)

    f.close()

    ## also include validation set as training data
    f = open('/home/xusheng/TabelProject/data/tabM.vec.raw.valid', 'r')
    data = json.load(f)
    for t in data:
        cells = []
        r_count = 0
        for r in t:
            c_count = 0
            for c in r:
                if 'translate' in c.keys() and 'en-entity' in c.keys():
                    surface = c['translate']
                    label = c['en-entity'].replace('_', ' ')
                    cell = Cell(surface, (r_count, c_count), label)
                else:
                    cell = Cell(None, (r_count, c_count), None, True)
                cells.append(cell)
                c_count = c_count + 1
            r_count = r_count + 1
        table = Table((r_count, c_count), cells)
        tables.append(table)

    return tables

def load_train_data_new(path='/home/xusheng/PythonProject/data/tabel/monolingual/uniform_data.json'):
    zh2en, zh2cand = load_dict()
    f = open(path, 'r')
    data = json.load(f)
    tables = []
    for t in data[:-24]:
        cells = []
        r_count = 0
        for r in t:
            c_count = 0
            for c in r:
                if 'mention' in c.keys() and 'en-entity' in c.keys():
                    zh_surface = c['mention']
                    label = c['en-entity'].replace('_', ' ')
                    surface = zh2en[zh_surface]
                    cell = Cell(surface, (r_count, c_count), label)
                    # load candidate
                    cands = zh2cand[zh_surface]
                    for cand in cands:
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
    
    return tables

def gen_entity_context():
    tables = load_train_data_new()
    entity_context_word = dict()
    entity_context_entity = dict()
    for table in tables:
        for cell in table.cells:
            if cell.blank:
                continue
            entity = cell.label
            if not entity in entity_context_word:
                entity_context_word[entity] = dict()
            word_context = entity_context_word[entity]
            if not entity in entity_context_entity:
                entity_context_entity[entity] = dict()
            entity_context = entity_context_entity[entity]
            context = table.get_context(cell.pos)
            for c in context:
                tokens = c.surface.split(' ')
                for token in tokens:
                    if token in word_context:
                        word_context[token] = word_context[token] + 1
                    else:
                        word_context[token] = 1
                if c.label in entity_context:
                    entity_context[c.label] = entity_context[c.label] + 1
                else:
                    entity_context[c.label] = 1
    
    with open('entity_context_word.txt', 'w') as f:
        for e, context in entity_context_word.items():
            f.write(e.encode('utf8') + '\t\t')
            for word, frequency in context.items():
                f.write('\t\t' + word + '\t' + str(frequency))
            f.write('\n')
    with open('entity_context_entity.txt', 'w') as f:
        for e, context in entity_context_entity.items():
            f.write(e.encode('utf8') + '\t\t')
            for entity, frequency in context.items():
                f.write('\t\t' + entity.encode('utf8') + '\t' + str(frequency))
            f.write('\n')

def gen_training_data():
    tables = load_train_data_new()
    prior, wikigraph, entity_context, weight, all_entities = preprocess()
    features = Features(prior, wikigraph, entity_context, weight, all_entities)
    # positive = []
    # negative = []
    count = 0
    count_miss = 0
    f = open('feat.extracted.txt', 'w')
    for table in tables:
        # candidate_generation(table, prior)
        for cell in table.cells:
            if not cell.blank:
                count = count + 1
                golden_feature = None
                negative_features = []
                # print 'surface: ' + cell.surface + '  target: ' + cell.label
                for mention in cell.mentions:
                    ft = features.extract_features(table, cell.pos, mention, mention.entity)
                    print ft
                    mention.feature = np.asarray(ft)
                    # print mention.entity,
                    if mention.entity == cell.label:
                        golden_feature = mention.feature
                    else:
                        negative_features.append(mention.feature)
                # print '\n'
                if golden_feature is None:
                    count_miss = count_miss + 1
                    print 'not found'
                    continue
                f.write('\t'.join([str(s) for s in golden_feature]) + '\n')
                for neg in negative_features:
                    f.write('\t'.join([str(s) for s in neg]) + '\n')
                f.write('\n')
                    
    print count, count_miss
    f.close()

def load_training_data():
    dataset = []
    with open('feat.extracted.txt', 'r') as f:
        data = None
        new_flag = True
        for line in f:
            if line == '\n':
                dataset.append(data)
                new_flag = True
            else:
                feat = [float(num) for num in line[:-1].split('\t')]
                if new_flag:
                    data = [feat]
                    new_flag = False
                else:
                    data.append(feat)
    train_size = len(dataset)
    feat_num = len(dataset[0][0])
    list_len = max([len(data) for data in dataset])
    print train_size, feat_num, list_len
    npdata = np.zeros(shape=(train_size, list_len, feat_num))
    for i in range(train_size):
        for j in range(len(dataset[i])):
            npdata[i,j,:] = np.asarray(dataset[i][j])
    label = np.zeros(shape=(train_size, list_len))
    label[:, 0] = 1
    mask = np.zeros(label.shape)
    for i in range(train_size):
        mask[i, 0:len(dataset[i])] = 1
    # print npdata[0, 0, :]

    ## switch
    # switch = 5
    # npdata[:, 0, :], npdata[:, switch, :] = npdata[:, switch, :], npdata[:, 0, :]
    # npdata[:, [0, switch], :] = npdata[:, [switch, 0], :]
    # label[:, 0], label[:, switch] = label[:, switch], label[:, 0]
    # label[:, [0, switch]] = label[:, [switch, 0]]
    # mask[:, 0], mask[:, switch] = mask[:, switch], mask[:, 0]
    # mask[:, [0, switch]] = mask[:, [switch, 0]]
    # print 'label: %s' %label[0:10, 0:10]

    return npdata, label, mask


if __name__ == '__main__':
    ## load_train_data()
    gen_entity_context()
    gen_training_data()
    ## load_training_data()
