import pickle
import numpy as np
import json
import re
import gc
import h5py
from string import Template
import random


def load_dict(filename):
    dictionary = {}
    with open(filename, 'r') as f:
        for line in f:
            name, id = line[:-1].split('\t')
            dictionary[name] = int(id)
    return dictionary

def save_dict(filename, dictionary):
    with open(filename, 'w') as f:
        for name, id in dictionary.items():
            f.write('%s\t%d\n' % (name, id))

def look_up_dict(word, dictionary):
    if word in dictionary:
        return dictionary[word]
    dictionary[word] = len(dictionary) + 1
    return len(dictionary)

def tokenize(s):
    return s.replace('\'s', ' \'s').split()

def make_question_dump():
    with open('/home/xianyang/Webquestions/Json/webquestions.examples.train.json', 'r') as f:
        webq = json.load(f)
        dictionary = {}
        data = []
        for q in webq:
            uttr = q['utterance'][:-1]
            print(uttr)
            # tokens = uttr.split()
            tokens = tokenize(uttr)
            sentence = []
            for token in tokens:
                if not token in dictionary:
                    dictionary[token] = len(dictionary) + 1
                sentence.append(dictionary[token])
            data.append(sentence)

    max_len = 0
    for sentence in data:
        if len(sentence) > max_len:
            max_len = len(sentence)

    npdata = np.zeros((len(data), max_len))
    for i in range(len(data)):
        for j in range(len(data[i])):
            npdata[i, j] = data[i][j]

    # with open('webq.question.dump', 'w') as f:
    #     pickle.dump(npdata, f)
    f = h5py.File('webq.questions_index.hdf5', 'w')
    f.create_dataset('data', data=npdata)
    print f
    save_dict('dict.word', dictionary)

def convert_relation_to_words(relation):
    return re.sub('[._-]', ' ', relation).split()

def make_skeleton_surface_dump():
    dictionary = load_dict('dict.word')
    with open('sklt_f1.txt', 'r') as f:
        data = []
        for line in f:
            id, _, _, sklt, P, R, F1 = line[:-1].split('\t')
            id = int(id)
            P, R, F1 = map(float, (P, R, F1))
            while len(data) <= id:
                data.append([])
            data[id].append((list(map(lambda x: look_up_dict(x, dictionary), convert_relation_to_words(sklt))), (P, R, F1)))
    
    max_sklt = 1000
    sklt_max_len = 0
    for qi in range(len(data)):
        q = data[qi]
        # if len(q) > max_sklt:
        #     max_sklt = len(q)
        q.sort(key=lambda x: x[1][2], reverse=True)
        if len(q) > max_sklt:
            data[qi] = q[:10] + random.sample(q[10:], max_sklt - 10)
            print(len(data[qi]))
            if len(data[qi]) > max_sklt:
                pause = raw_input('pause')
        for sklt in q:
            if len(sklt[0]) > sklt_max_len:
                sklt_max_len = len(sklt[0])
    print(sklt_max_len)

    npdata = np.zeros((len(data), max_sklt, sklt_max_len))
    npdata_f1 = np.zeros((len(data), max_sklt, 3))
    npdata_mask = np.zeros((len(data), max_sklt))
    for i in range(len(data)):
        print(len(data[i]))
        for j in range(len(data[i])):
            npdata_mask[i, j] = 1
            npdata_f1[i, j, :] = data[i][j][1]
            for k in range(len(data[i][j][0])):
                npdata[i, j, k] = data[i][j][0][k]

    gc.collect()

    print(npdata.shape)
    '''
    with open('webq.skeleton.word.dump', 'w') as f:
        npdata.dump(f)
    with open('webq.skeleton.word.mask,dump', w) as f:
        npdata_mask.dump(f)
    save_dict('dict.word-2', dictionary)
    '''
    f = h5py.File('webq.skeleton.word.hdf5', 'w')
    f.create_dataset('index', data=npdata)
    f.create_dataset('mask', data=npdata_mask)
    f.create_dataset('f1', data=npdata_f1)
    print f
    save_dict('dict.word-2', dictionary)


def eval_skeleton():
    from eval import computeF1
    from sparql_backend import backend

    golden = []

    def parse_golden(s):
        result = []
        start = s.find('\"')
        while start > 0:
            end = s.find('\"', start + 1)
            result.append(s[start + 1 : end])
            start = s.find('\"', end + 1)
        return result

    with open('/home/data/Webquestions/Json/webquestions.examples.train.json', 'r') as f:
        webq = json.load(f)
        for q in webq:
            golden.append(parse_golden(q['targetValue']))

    temp1 = Template('''
    PREFIX fb: <http://rdf.freebase.com/ns/>
    SELECT DISTINCT ?ans
    WHERE {
        fb:${e} fb:${s} ?x .
        ?x fb:type.object.name ?ans .
    }
    ''')
    temp2 = Template('''
    PREFIX fb: <http://rdf.freebase.com/ns/>
    SELECT DISTINCT ?ans
    WHERE {
        fb:${e} fb:${s1} ?x .
        ?x fb:${s2} ?y .
        ?y fb:type.object.name ?ans .
    }
    ''')
    sparql = backend.SPARQLHTTPBackend('202.120.38.146', '8999', '/sparql')
    def gen_answer(focus, sklt):
        sklt = sklt.split('--')
        if len(sklt) == 1:
            # print temp1.substitute(e=focus, s=sklt[0])
            ret = sparql.query(temp1.substitute(e=focus, s=sklt[0]))
            # print ret
            return sum(ret, [])
        if len(sklt) == 2:
            ret = sparql.query(temp2.substitute(e=focus, s1=sklt[0], s2=sklt[1]))
            # print ret
            return sum(ret, [])
        return []

    result = []
    count = 0
    with open('basic.graph.train.tsv', 'r') as f:
        for line in f:
            count += 1
            if count == 1000:
                break
            id, e, mid, sklt = line[:-1].split('\t')
            id = int(id)
            answer = gen_answer(mid, sklt)
            (P, R, F1) = computeF1(golden[id], answer)
            print(golden[id], answer, (P, R, F1))
            result.append((id, P, R, F1))
            # pause = raw_input('pause')

    with open('f1.dump', 'w') as f:
        pickle.dump(result, f)

def eval_skeleton_new():
    from eval import computeF1
    golden = []

    def parse_golden(s):
        result = []
        start = s.find('(description ')
        while start > 0:
            end = s.find(')', start + 1)
            dscpt = s[start + 13 : end]
            if dscpt[0] == '\"':
                dscpt = dscpt[1:-1]
            result.append(dscpt)
            start = s.find('(description ', end + 1)
        return result

    with open('/home/xianyang/Webquestions/Json/webquestions.examples.train.json', 'r') as f:
        webq = json.load(f)
        for q in webq:
            golden.append(parse_golden(q['targetValue']))

    f1 = open('basic.graph.train.tsv', 'r')
    f2 = open('sklt.txt', 'r')
    outf = open('sklt_f1.txt', 'w')
    total_len = 4632438
    for i in range(total_len):
        s1 = f1.readline()
        s2 = f2.readline()
        if s2 == '\n':
            continue
        answer = s2[:-1].split('\t')
        qid, focus, mid, sklt = s1[:-1].split('\t')
        print(qid, golden[int(qid)], answer)
        (P, R, F1) = computeF1(golden[int(qid)], answer)
        outf.write('%s\t%f\t%f\t%f\n'% (s1[:-1], P, R, F1))

    f1.close()
    f2.close()
    outf.close()

def test_data():
    f = h5py.File('webq.questions_index.hdf5', 'r')
    print f
    question_index = f['data']
    print(question_index.shape)
    # sent = question_index[0, :]
    dictionary = load_dict('dict.word-2')
    inverse_dict = {}
    for k, v in dictionary.items():
        inverse_dict[v] = k
    inverse_dict[0] = ''

    def translate(s, dic):
        return ' '.join([dic[idx] for idx in s])

    for i in range(1):
        sent = question_index[i, :]
        print(translate(sent, inverse_dict))

    f2 = h5py.File('webq.skeleton.word.hdf5', 'r')
    index = f2['index']
    f1 = f2['f1']
    mask = f2['mask']
    print(index.shape, f1.shape, mask.shape)

    for i in range(10):
        sklt = index[0, i, :]
        (P, R, F) = f1[0, i, :]
        print(translate(sklt, inverse_dict), (P, R, F))
        

if __name__ == '__main__':
    # make_question_dump()
    # eval_skeleton()
    # eval_skeleton_new()
    # make_skeleton_surface_dump()
    # test_data()
    pass