from sparql_backend import backend
import threading
import time
import sys
from string import Template
import logging

logging.basicConfig(filename='query_log_2', level=logging.INFO)


class WorkerThread(threading.Thread):
    def __init__(self, id, host, port, queryList, outList=None):
        self.id = id
        threading.Thread.__init__(self)
        self.queryList = queryList
        self.outList = outList
        self.sparql = backend.SPARQLHTTPBackend(host, port, '/sparql')
        print 'thread %d created.' % (self.id) 

    def run(self):
        try:
            for query in self.queryList:
                result = '\t'.join(sum(self.sparql.query(query), []))
                logging.info('th-%d: %s', self.id, result)
                self.outList.append(result)
        except:
            print("Error in thread %d:" % self.id, sys.exc_info()[0])
            return


class SparqlDriver(object):
    def __init__(self, host='202.120.38.146', port='8688', num_thread=8):
        self.host = host
        self.port = port
        self.num_thread = num_thread

    def bulk_query(self, queryList):
        total_query = len(queryList)
        query_per_thread = total_query / self.num_thread
        threadList = [None] * self.num_thread
        resultList = [[]] * self.num_thread 
        for i in range(self.num_thread):
            begin = i * query_per_thread
            if i == self.num_thread - 1:
                end = len(queryList)
            else:
                end = begin + query_per_thread
            threadList[i] = WorkerThread(i, self.host, self.port, queryList[begin:end], resultList[i])

        start_time = time.time()
        # print threadList
        for i in range(self.num_thread):
            threadList[i].start()

        for i in range(self.num_thread):
            threadList[i].join()
        print time.time() - start_time

        return sum(resultList, [])


def test():
    queryList = []
    with open('basic.graph.train.tsv', 'r') as f:
        # for i in range(1000):
        #     line = f.readline()
        for line in f:
            id, score, focus, sklt = line[:-1].split('\t')
            queryList.append(gen_query(focus, sklt))

    driver = SparqlDriver(num_thread=8)
    result = driver.bulk_query(queryList)
    with open('sklt.txt', 'w') as f:
        for ans in result:
            f.write(ans)
            f.write('\n')

temp1 = Template('''
PREFIX fb: <http://rdf.freebase.com/ns/>
SELECT DISTINCT ?ans
WHERE {
    fb:${e} fb:${s} ?x .
    ?x fb:type.object.name ?ans .
}
LIMIT 100
''')
temp2 = Template('''
PREFIX fb: <http://rdf.freebase.com/ns/>
SELECT DISTINCT ?ans
WHERE {
    fb:${e} fb:${s1} ?x .
    ?x fb:${s2} ?y .
    ?y fb:type.object.name ?ans .
}
LIMIT 100
''')
def gen_query(focus, sklt):
    sklt = sklt.split('--')
    if len(sklt) == 1:
        return temp1.substitute(e=focus, s=sklt[0])
    if len(sklt) == 2:
        return temp2.substitute(e=focus, s1=sklt[0], s2=sklt[1])
    raise


if __name__ == '__main__':
    test()