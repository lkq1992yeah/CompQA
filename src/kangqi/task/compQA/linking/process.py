from entity_linker import entity_linker, surface_index_memory
from corenlp_parser import parser
import os
import time


class EL:
    def __init__(self):
        base = '/home/xianyang/aqqu/aqqu'
        print 'initiating parser...'
        self.parser = parser.CoreNLPParser('http://localhost:4000/parse')
        print 'initiating index...'
        self.surface_index = surface_index_memory.EntitySurfaceIndexMemory(base+'/data/entity-list', base+'/data/entity-surface-map', base+'/data/entity-index')
        print 'initiating entity_linker'
        self.entity_linker = entity_linker.EntityLinker(self.surface_index, 7)
        print 'initiation done.'

    def link(self, sentence):
        parse_result = self.parser.parse(sentence)
        identified = self.entity_linker.identify_entities_in_tokens(parse_result.tokens)
        return identified

    def parse(self, sentence):
        return self.parser.parse(sentence)

    def link_with_parse(self, tokens):
        return self.entity_linker.identify_entities_in_tokens(tokens)

def run_wiiki_filter(id):
    linker = EL()
    count = 0
    for root, dirs, files in os.walk('../wikidata/'):
        for filename in files:
                if len(filename) != 4:
                    continue
                print count
                t0 = time.time()
                f = open(os.path.join(root, filename), 'r')
                g = open(os.path.join(root, filename)+'-%d'%id, 'w')
                for line in f:
                    sentence = line[:-1].decode('utf-8')
                    if filter_1(sentence):
                        continue
                    parse_result = linker.parse(sentence)
                    tokens = parse_result.tokens
                    # check superlative
                    flag = False
                    for token in tokens:
                        if token.pos == 'JJS' or token.pos == 'RBS':
                            flag = True
                    if flag:
                        entities = linker.link_with_parse(tokens)
                        if len(entities) >= 2:
                            count += 1
                            g.write(line)
                            for e in entities:
                                try:
                                    # print (e.name.decode('utf-8'), e.surface_score, e.score, e.entity.id, e.perfect_match)
                                    g.write('\t{}\t{}\t{}\t{}\t{}\n'.format(e.name.decode('utf-8'), e.surface_score, e.score, e.entity.id, e.perfect_match))
                                except:
                                    pass
                f.close()
                g.close()
                duration = (time.time() - t0)
                print '%s took %.2f s' % (filename, duration)

def filter_1(sentence):
    sentence = sentence.lower()
    # filter complex sentence
    if sentence.find(', ') >= 0 or sentence.find('which') >= 0 or sentence.find('that') >= 0 or sentence.find('who') >= 0:
        return True 
    tokens = sentence.split(' ')
    for i in range(len(tokens)):
        token = tokens[i]
        if token == 'most' or token == 'least' or token == 'first' or token == 'last':
            if i + 1 < len(tokens) and tokens[i+1] == 'of':
                return True
            if i > 0 and tokens[i-1] != 'the':
                return True
            if i > 3 and tokens[i-3:i] == ['one', 'of', 'the'] or tokens[i-3:i] == ['some', 'of', 'the']:
                return True
    
    return False

def collect():
    count = 0
    w = open('collection.txt', 'w')
    for root, dirs, files in os.walk('../wikidata/'):
        for filename in files:
            if filename[-2:] != '-2':
                continue
            f = open(os.path.join(root, filename), 'r')
            for line in f:
                if line[0] != '\t':
                    # count += 1
                    w.write(line)
            f.close()
    w.close()
    # print count

def test():
    linker = EL()
    sentence = "what character does ellen play in finding nemo"
    result = linker.link(sentence)
    for e in result:
        print e.name, e.entity.id


if __name__ == '__main__':
    # run_wiiki_filter(1)
    collect()