"""
BasicQueryGraphGen: class for generating skeletons from a focus entity
"""
from sparql_backend import backend
from string import Template
from query_graph import *

class BasicQueryGraphGen:
    def __init__(self):
        self.sparql = backend.SPARQLHTTPBackend('202.120.38.146', '8688', '/sparql')
        self.templ1 = Template('''
        PREFIX fb: <http://rdf.freebase.com/ns/>
        SELECT DISTINCT ?pred
        WHERE {
            fb:${e} ?pred ?x .
            ?x fb:type.object.name ?name .
        } LIMIT 500
        ''')
        self.templ2 = Template('''
        PREFIX fb: <http://rdf.freebase.com/ns/>
        SELECT DISTINCT ?p1 ?p2
        WHERE {
            fb:${e} ?p1 ?y .
            ?y ?p2 ?x .
            ?x fb:type.object.name ?name .
            FILTER ( fb:${e} != ?x )
        } LIMIT 1500
        ''')

    def gen_basic_query_graph(self, entity):
        """
        entity: mid of focus entity
        """
        one_hop = self.sparql.query(self.templ1.substitute(e=entity))
        two_hop = self.sparql.query(self.templ2.substitute(e=entity))
        basic_graphs = []
        for g in one_hop:
            graph = QueryGraph()
            v1 = Vertex(is_constant=True, is_answer=False, ground=entity, id=None)
            v2 = Vertex(is_constant=False, is_answer=True, ground=None, id='x')
            edge = Edge(is_functional=False, ground=g[0], left=v1, right=v2)
            graph.add_edge(v1, v2, edge)
            graph.set_focus(entity)
            basic_graphs.append(graph)
        for g in two_hop:
            graph = QueryGraph()
            v1 = Vertex(is_constant=True, is_answer=False, ground=entity, id=None)
            v2 = Vertex(is_constant=False, is_answer=False, ground=None, id=graph.get_id())
            v3 = Vertex(is_constant=False, is_answer=True, ground=None, id='x')
            e1 = Edge(is_functional=False, ground=g[0], left=v1, right=v2)
            e2 = Edge(is_functional=False, ground=g[1], left=v2, right=v3)
            graph.add_edge(v1, v2, e1)
            graph.add_edge(v2, v3, e2)
            graph.set_focus(entity)
            basic_graphs.append(graph)
        return basic_graphs

def test():
    worker = BasicQueryGraphGen()
    bgraphs = worker.gen_basic_query_graph('m.0k6zx9y')
    for g in bgraphs:
        print '--'.join([e.ground for e in g.edges])


if __name__ == '__main__':
    test()