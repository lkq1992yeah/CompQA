from query_graph import *
from basic_graph_gen import *
from constraints import *
import json
import itertools
import os


def basic_cand_gen():
    """
    generate skeletons 
    """
    link_file = '/home/kangqi/workspace/UniformProject/resources/linking/yih2015/webquestions.examples.train.e2e.top10.filter.tsv'
    linking_result = load_linking(link_file)
    worker = BasicQueryGraphGen()
    fout = open('basic.graph.txt', 'w')
    count = 0
    for question in linking_result:
        for entity, name in question:
            # entity = entity[1:].replace('/', '.')
            bgraphs = worker.gen_basic_query_graph(entity)
            for g in bgraphs:
                fout.write('%d\t%s\t%s\t%s\n' % (count, name, entity, '--'.join([e.ground for e in g.edges])))
        print count
        count += 1
    fout.close()

def load_data(filename):
    dataset = []
    with open(filename, 'r') as f:
        data = json.load(f)
        for q in data:
            dataset.append(q['utterance'])
    return dataset

def load_linking(filename):
    """
    load entity linking results of s-mart
    """
    linking = []
    with open(filename, 'r') as f:
        for line in f:
            splitted = line[:-1].split('\t')
            id = splitted[0]
            id = int(id[id.find('-')+1:])
            while len(linking) <= id:
                linking.append([])
            linking[id].append((splitted[4][1:].replace('/', '.'), splitted[1]))
    return linking

MAX_SCHEMA_PER_SKLT = 1000

class SchemaGen(object):
    def __init__(self):
        self.basic_gen = BasicQueryGraphGen()

    def gen(self, q, entities, f=None):
        """
        generate schemas of given question and entity linking result
        inputs: 
         q: question string
         entities: list of detected entities
         f: file to write results to 
        """
        # detect constraints. use entity, time and ordinal constraints so far
        cEntity = EntityConstraint.detect(None, entities, q)
        cTime = ExplicitTimeConstraint.detect(None, entities, q)
        cOrdinal = OrdinalConstraint.detect(None, entities, q)
        all_constraints = [cEntity, cTime, cOrdinal]
        all_nonempty_constraints = []
        for constraints in all_constraints:
            if constraints:
                all_nonempty_constraints.append(constraints)
        # get the combination of different types of constraints. here I decide that two same kind of constraints do not apply simultanoeusly.
        # for example, if I detected entity constraints [e1, e2] and time constraints [t1, t2], legal constraint combinations are [(e1, t1), (e1, t2), (e2, t1), (e2, t2)]
        all_combinations = list(itertools.product(*all_nonempty_constraints))
        # for a given constraint combination, apply the constraints in different order
        cPermute = sum([ list(itertools.permutations(comb)) for comb in all_combinations ], [])
        
        if f:
            f.write(q + '\n')
        else:
            print q

        for focus in entities:
            sklts = self.basic_gen.gen_basic_query_graph(focus)
            for sklt in sklts:
                if f:
                    f.write('sklt:\n' + str(sklt) + '\n')
                else:
                    print 'sklt:\n' + str(sklt)
                schemas = []
                for p in cPermute:
                    # choose a constraint order, then add them to skeleton one by one
                    current = [sklt]
                    for cstr in p:
                        current = sum([cstr.bind(s) for s in current], [])
                        if not current:
                            break
                        schemas += current
                        if len(schemas) >= MAX_SCHEMA_PER_SKLT:
                            break
                # tried to remove redundancy, didn't work actually because schema is not a primary type
                schemas = set(schemas)
           
                for schema in schemas:
                    if f:
                        f.write(str(schema) + '\n')
                    else:
                        print schema
                    

    def detect(self, q, entities):
        # for debug
        cTime = ExplicitTimeConstraint.detect(None, entities, q)
        if cTime:
            print q


if __name__ == '__main__':
    from sys import argv
    data = load_data('/home/xianyang/Webquestions/Json/webquestions.examples.train.json')
    linking = load_linking('/home/xianyang/webquestions.examples.train.e2e.top10.filter.tsv')
    test = SchemaGen()

    if not 'schema' in os.listdir('.'):
        os.mkdir('schema/')
    if not 'train' in os.listdir('schema/'):
        os.mkdir('schema/train/')

    # qid = 1 # 1, 289, 2513
    for qid in range(int(argv[1]), int(argv[2])):
        split = qid / 100 * 100
        split = str(split) + '-' + str(split + 99)
        if not split in os.listdir('schema/train/'):
            os.mkdir('schema/train/' + split)
        question = data[qid]
        entities = [x[0] for x in linking[qid]]
        with open('schema/train/' + split + '/%02d' % (qid), 'w') as f:
            test.gen(question, entities, f)
