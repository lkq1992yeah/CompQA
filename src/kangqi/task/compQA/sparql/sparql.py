"""
Author: Kangqi Luo
Goal: Executes SPARQL query, and maintain a cache of query results.
"""


from ..xy_bao.sparql_backend.backend import SPARQLHTTPBackend
# from .sparql_backend import SPARQLHTTPBackend          # I don't know why my backend doesn't work.

from kangqi.util.cache import DictCache
from kangqi.util.LogUtil import LogInfo


class SparqlDriver(object):

    def __init__(self, sparql_ip='202.120.38.146', sparql_port=8999,
                 cache_dir=None, use_cache=True, verbose=0):
        LogInfo.logs('Use sparql cache: %s, cache_dir: %s', use_cache, cache_dir)
        self.verbose = verbose
        self.use_cache = use_cache
        self.sparql = SPARQLHTTPBackend(sparql_ip, str(sparql_port), '/sparql')
        if self.use_cache:
            LogInfo.begin_track('Loading sparql caches ... ')
            self.cache = DictCache('%s/sparql_cache' % cache_dir)
            LogInfo.end_track()
        else:
            LogInfo.logs('Currently do not use sparql cache.')
        self.query_prefix = 'PREFIX fb: <http://rdf.freebase.com/ns/>'

    # Executes the sparql intention (condition: not found in the cache)
    def execute_sparql(self, query):
        # if self.verbose:
        #     LogInfo.logs('%s', query)
        try:
            ret = self.sparql.query(query, vb=self.verbose)
        except BaseException:
            ret = []
        if ret is None:
            ret = []
        return ret

    def perform_query(self, query):
        if self.use_cache:
            ret = self.cache.get(query)
            if ret is not None:
                return ret

        ret = self.execute_sparql(query)

        if self.use_cache:
            if ret is not None:
                self.cache.put(query, ret)
        return ret

    # only perform decomposition when there is only one column in the intention result
    @staticmethod
    def decomposition(query_ret):
        return [row[0] for row in query_ret]

    # given an entity, return all its types
    # will perform decomposition
    def query_type_given_entity(self, mid):
        query = (
            '%s SELECT DISTINCT ?o WHERE { '
            'fb:%s fb:type.object.type ?o . '
            '}' % (self.query_prefix, mid)
        )
        query_ret = self.perform_query(query)
        type_list = self.decomposition(query_ret)
        return type_list

    # used in one-hop expansion
    def query_pred_obj_given_subj(self, mid):
        query = (
            '%s SELECT DISTINCT ?p ?o WHERE { '
            'fb:%s ?p ?o . '
            '}' % (self.query_prefix, mid)
        )   # direction subj --> obj
        return self.perform_query(query)

    # used in entity constraint extraction: find all possible predicates pointing to the entity
    def query_pred_given_object(self, obj):
        query = (
            '%s SELECT DISTINCT ?p WHERE { '
            '?s ?p fb:%s . '
            '}' % (self.query_prefix, obj)
        )
        query_ret = self.perform_query(query)
        return self.decomposition(query_ret)

    # used to find touch entities for an entity constraint
    def query_subject_given_pred_obj(self, pred, obj):
        query = (
            '%s SELECT DISTINCT ?s WHERE { '
            '?s fb:%s fb:%s . '
            '}' % (self.query_prefix, pred, obj)
        )
        query_ret = self.perform_query(query)
        return self.decomposition(query_ret)


if __name__ == '__main__':
    LogInfo.begin_track('[sparql.py] starts ... ')
    driver = SparqlDriver(sparql_ip='202.120.38.146', sparql_port=8699,
                          use_cache=False)

    q1 = (
        'PREFIX fb: <http://rdf.freebase.com/ns/> '
        'SELECT DISTINCT ?name WHERE { '
        'fb:m.0d05w3 fb:type.object.name ?name . '
        '}'
    )
    q2 = '''PREFIX fb: <http://rdf.freebase.com/ns/>
    SELECT DISTINCT ?x0 ?n0 ?x1
    WHERE {
      ?x0 fb:geography.river.length ?x1 .
      OPTIONAL { ?x0 fb:type.object.name ?n0 }
    } ORDER BY DESC(?x1)
    LIMIT 100
    '''
    q3 = '''PREFIX fb: <http://rdf.freebase.com/ns/>
    SELECT DISTINCT ?x1 ?x2 ?n2 ?x3
    WHERE {
        fb:m.0dr_4 fb:film.film.starring ?x1 .
        ?x1 fb:film.performance.actor ?x2 .
        OPTIONAL { ?x2 fb:type.object.name ?n2 } .
        ?x2 fb:people.person.gender fb:m.05zppz .
        ?x2 fb:people.person.date_of_birth ?x3
    }
    ORDER BY DESC(?x3)
    LIMIT 100
    '''
    q4 = '''PREFIX fb: <http://rdf.freebase.com/ns/>
    SELECT DISTINCT ?x1 ?x2 ?ord WHERE {
        ?x1 fb:book.book.editions ?x2 .
        ?x2 fb:book.book_edition.publication_date ?ord .
    } ORDER BY DESC(?ord) LIMIT 100
    '''

    for q in [q1, q2, q3, q4]:
        ret_list = driver.perform_query(q)
        LogInfo.logs('return: %s', ret_list)
    LogInfo.end_track()
