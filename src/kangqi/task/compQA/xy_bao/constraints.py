"""
This file defines the constraints that can be added to a schema
Each constraint class has two methods: 
 1) static method *detect(basic_graph, entities, question)* to detect constraints in a question and return the constraint instances
 2) *bind(self, graph)* apply the constraint to a schema and return a set of new schemas
 note: the *basic_graph* parameter for *detect* is actually useless, but I just leave it.
"""
from query_graph import *
from sparql_backend import backend
from string import Template
from fb_util import *
from nltk import word_tokenize, pos_tag

sparql = backend.SPARQLHTTPBackend('202.120.38.146', '8688', '/sparql')

class Constraint(object):
    @staticmethod
    def detect(basic_graph, entities, question):
        pass

    def bind(self, graph):
        pass

class EntityConstraint(Constraint):
    @staticmethod
    def detect(basic_graph, entities, question):
        return [EntityConstraint(e) for e in entities]

    def __init__(self, entity):
        """
        More than just storing the entity, we retrieve the triplets that contain this entity for later computation
        """
        self.entity = entity
        query1 = '''
        PREFIX fb: <http://rdf.freebase.com/ns/>
        SELECT DISTINCT ?rel ?y
        WHERE {
            fb:%s ?rel ?y .
        }
        ''' % (entity)
        query_result1 = sparql.query(query1)
        self.kb_context1 = filter(lambda x: len(x) == 2 and is_mid(x[1]), query_result1)
        query2 = '''
        PREFIX fb: <http://rdf.freebase.com/ns/>
        SELECT DISTINCT ?rel ?x
        WHERE {
            ?x ?rel fb:%s .
        }
        ''' % (entity)
        query_result2 = sparql.query(query2)
        self.kb_context2 = filter(lambda x: len(x) == 2 and is_mid(x[1]), query_result2)

    def bind(self, graph):
        ret = []
        if self.entity == graph.focus:
            return ret
        v_list, candidates = graph.get_candidate_entity_full(sparql)

        for v in graph.vertices:
            if v.ground:
                continue

            rels1 = []
            if v.ground:
                for ro in self.kb_context1:
                    if ro[1] == v.ground:
                        rels1.append(ro[0])
            else:
                v_index = v_list.index(v.id)
                v_cands = set([x[v_index] for x in candidates])
                for ro in self.kb_context1:
                    if ro[1] in v_cands:
                        rels1.append(ro[0])
            for rel in set(rels1):
                ngraph = QueryGraph()
                ngraph.deepcopy_from(graph)
                nv = Vertex(is_constant=True, is_answer=False, ground=self.entity, id=None)
                ngraph.add_edge(nv, v, Edge(False, rel, nv, v))
                ret.append(ngraph)

            rels2 = []
            if v.ground:
                for ro in self.kb_context2:
                    if ro[1] == v.ground:
                        rels2.append(ro[0])
            else:
                v_index = v_list.index(v.id)
                v_cands = set([x[v_index] for x in candidates])
                for ro in self.kb_context2:
                    if ro[1] in v_cands:
                        rels2.append(ro[0])
            for rel in set(rels2):
                ngraph = QueryGraph()
                ngraph.deepcopy_from(graph)
                nv = Vertex(is_constant=True, is_answer=False, ground=self.entity, id=None)
                ngraph.add_edge(v, nv, Edge(False, rel, v, nv))
                ret.append(ngraph)

        return ret
    
    def __repr__(self):
        return 'EntityConstraint(%s)' % (self.entity)

class ExplicitTimeConstraint(Constraint):
    @staticmethod
    def detect(basic_graph, entities, question):
        time_mention = detect_time(question)
        if time_mention:
            if ' after ' in question or 'later than' in question:
                return [ExplicitTimeConstraint(time_mention, '>')]
            elif ' before ' in question or 'earlier than' in question:
                return [ExplicitTimeConstraint(time_mention, '<')]
            else:
                return [ExplicitTimeConstraint(time_mention, '=')]
        else:
            return []
    
    def __init__(self, time, op):
        self.time = time
        self.op = op

    def __repr(self):
        return 'TimeConstraint(%s, %s)' % (self.op, self.time)

    def bind(self, graph):
        ret = []
        v_list, candidates = graph.get_candidate_entity_full(sparql)
        for v in graph.vertices:
            if v.ground:
                continue

            rels = []
            if v.ground:
                cands = [v.ground]
            else:
                v_index = v_list.index(v.id)
                cands = set([x[v_index] for x in candidates])
            for cand in cands:
                query = '''
                PREFIX fb: <http://rdf.freebase.com/ns/>
                SELECT DISTINCT ?p ?type
                WHERE {
                    fb:%s ?p ?obj .
                    ?p fb:type.property.expected_type ?type .
                }
                ''' % (cand)
                query_result = sparql.query(query)
                for pt in query_result:
                    if pt[1] == 'type.datetime':
                        rels.append(pt[0])
            for rel in set(rels):
                ngraph = QueryGraph()
                ngraph.deepcopy_from(graph)
                nv1 = Vertex(is_constant=False, is_answer=False, ground=None, id=ngraph.get_id())
                ngraph.add_edge(v, nv1, Edge(is_functional=False, ground=rel, left=v, right=nv1))
                nv2 = Vertex(is_constant=True, is_answer=False, ground=self.time, id=None)
                ngraph.add_edge(nv1, nv2, Edge(is_functional=True, ground=self.op, left=nv1, right=nv2))
                ret.append(ngraph)
        return ret

class OrdinalConstraint(Constraint):
    @staticmethod
    def detect(basic_graph, entities, question):
        tokens = word_tokenize(question)
        tags = pos_tag(tokens)
        for i in range(len(tags)):
            token, tag = tags[i]
            if tag == 'JJS' or tag == 'RBS' or token == 'first' or token == 'last':
                if tag == 'JJS':
                    return [OrdinalConstraint('1', None, token)]
                if tag == 'RBS' or token == 'first' or token == 'last':
                    nextToken = tags[i+1][0]
                    return [OrdinalConstraint('1', None, nextToken)]
        return []

    def __init__(self, n, op, word):
        self.n = n
        self.op = op
        self.word = word
        # print n, op, word

    def __repr__(self):
        return 'OrdinalConstraint(%s, %s, %s)' % (self.op, self.n, self.word)

    def bind(self, graph):
        ret = []
        v_list, candidates = graph.get_candidate_entity_full(sparql)
        for v in graph.vertices:
            if v.ground:
                continue

            rels = []
            if v.ground:
                cands = [v.ground]
            else:
                v_index = v_list.index(v.id)
                cands = set([x[v_index] for x in candidates])
            for cand in cands:
                query = '''
                PREFIX fb: <http://rdf.freebase.com/ns/>
                SELECT DISTINCT ?p ?type
                WHERE {
                    fb:%s ?p ?obj .
                    ?p fb:type.property.expected_type ?type .
                }
                ''' % (cand)
                query_result = sparql.query(query)
                for pt in query_result:
                    if pt[1] == 'type.datetime' or pt[1] == 'type.int' or pt[1] == 'type.float':
                        rels.append(pt[0])

            for rel in set(rels):
                if self.op:
                    ngraph = QueryGraph()
                    ngraph.deepcopy_from(graph)
                    nv1 = Vertex(is_constant=False, is_answer=False, ground=None, id=ngraph.get_id())
                    ngraph.add_edge(v, nv1, Edge(False, rel, v, nv1))
                    nv2 = Vertex(is_constant=True, is_answer=False, ground=self.n, id=None)
                    ngraph.add_edge(nv1, nv2, Edge(is_functional=True, ground=self.op, left=nv1, right=nv2))
                    ret.append(ngraph)
                else:
                    ngraph1 = QueryGraph()
                    ngraph1.deepcopy_from(graph)
                    nv11 = Vertex(is_constant=False, is_answer=False, ground=None, id=ngraph1.get_id())
                    ngraph1.add_edge(v, nv11, Edge(False, rel, v, nv11))
                    nv21 = Vertex(is_constant=True, is_answer=False, ground=self.n, id=None)
                    ngraph1.add_edge(nv11, nv21, Edge(is_functional=True, ground='MaxAtN', left=nv11, right=nv21))
                    ret.append(ngraph1)
                    ngraph2 = QueryGraph()
                    ngraph2.deepcopy_from(graph)
                    nv12 = Vertex(is_constant=False, is_answer=False, ground=None, id=ngraph2.get_id())
                    ngraph2.add_edge(v, nv12, Edge(False, rel, v, nv12))
                    nv22 = Vertex(is_constant=True, is_answer=False, ground=self.n, id=None)
                    ngraph2.add_edge(nv12, nv22, Edge(is_functional=True, ground='MinAtN', left=nv12, right=nv22))
                    ret.append(ngraph2)
                
        return ret
        

class TypeConstraint(Constraint):
    @staticmethod
    def detect(basic_graph, entities, question):
        constraints = []
        types = extract_types(question)
        for t in types:
            constraints.append(TypeConstraint(t))
        return constraints

    def __init__(self, type):
        self.type = type

    def bind(self, graph):
        for v in graph.vetices:
            if v.id == 'x':
                target = v
                break
        nv = Vertex(is_constant=True, is_answer=False, ground=self.type, id=None)
        ngraph = QueryGraph()
        ngraph.deepcopy_from(graph)
        ngraph.add_edge(target, nv, Edge(is_functional=False, ground='type.object.type', left=target, right=nv))
        return [ngraph]

class AggregationConstraint(Constraint):
    @staticmethod
    def detect(basic_graph, entities, question):
        if question.startswith('how many') or question.__contains__('number of') or question.__contains__('count of'):
            return [AggregationConstraint()]
        else:
            return []

    def bind(self, graph):
        ngraph = QueryGraph()
        ngraph.deepcopy_from(graph)
        ngraph.aggregation = True
        return [ngraph]
        

