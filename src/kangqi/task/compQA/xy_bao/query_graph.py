"""
data structures for query graph
"""
from copy import copy, deepcopy


class Vertex:
    def __init__(self, is_constant, is_answer, ground, id):
        self.is_constant = is_constant
        self.is_answer = is_answer
        self.ground = ground
        self.id = id

    def __repr__(self):
        # if self.ground:
        #     return self.ground
        # return self.id
        return self.ground or self.id

class Edge:
    def __init__(self, is_functional, ground, left, right):
        self.is_functional = is_functional
        self.ground = ground
        self.left = left
        self.right = right

    def __repr__(self):
        return self.left + '--' + self.ground + '--' + self.right

class QueryGraph:
    def __init__(self):
        self.vertices = []
        self.edges = []
        self.id_count = 0
        self.aggregation = False
        self.focus = None

    @staticmethod
    def init_from_text(st):
        """
        reconstruct query graph from text 
        """
        graph = QueryGraph()
        for line in st:
            if not (line.startswith('<<<') or line.startswith('>>>')):
                subj, rel, obj = line.split('--')
                if subj == 'x' or subj[0] == 'y' and subj[1] in '0123456789':
                    nv1 = Vertex(is_constant=False, is_answer=(subj == 'x'), ground=None, id=subj)
                else:
                    nv1 = Vertex(is_constant=True, is_answer=False, ground=subj, id=None)
                if obj == 'x' or obj[0] == 'y' and obj[1] in '0123456789':
                    nv2 = Vertex(is_constant=False, is_answer=(obj == 'x'), ground=None, id=obj)
                else:
                    nv2 = Vertex(is_constant=True, is_answer=False, ground=obj, id=None)
                if rel in ['<', '>', '=', 'MaxAtN', 'MinAtN']:
                    edge = Edge(is_functional=True, ground=rel, left=nv1, right=nv2)
                else:
                    edge = Edge(is_functional=False, ground=rel, left=nv1, right=nv2)
                graph.add_edge(nv1, nv2, edge)
        return graph
                    

    def deepcopy_from(self, g):
        self.vertices = deepcopy(g.vertices)
        self.edges = deepcopy(g.edges)
        self.id_count = g.id_count
        self.aggregation = g.aggregation
        self.focus = g.focus

    def get_id(self):
        """
        return the current available variable node id
        """
        id = "y%d" % self.id_count
        self.id_count += 1
        return id

    def add_vertex(self, nv):
        for v in self.vertices:
            if (v.ground or v.id) == (nv.ground or nv.id):
                return v
        self.vertices.append(nv)
        return nv

    def add_edge(self, v1, v2, e):
        # if not v1 in self.vertices:
        #     self.vertices.append(v1)
        # if not v2 in self.vertices:
        #     self.vertices.append(v2)
        self.add_vertex(v1)
        self.add_vertex(v2)
        self.edges.append(e)

    def set_focus(self, f):
        self.focus = f

    def __repr__(self):
        repr = '<<<<<<\n'
        for edge in self.edges:
            s = str(edge.left) + '--' + edge.ground + '--' + str(edge.right) + '\n'
            repr += s
        return repr + '>>>>>>\n'

    def to_sparql(self):
        """
        return sparql query of the query graph
        TODO: functional relations
        """
        sql = '''
        PREFIX fb: <http://rdf.freebase.com/ns/>
        SELECT DISTINCT ?x
        WHERE {\n
        '''
        for edge in self.edges:
            if edge.is_functional:
                continue
            if edge.left.ground:
                left = 'fb:' + edge.left.ground
            else:
                left = '?' + edge.left.id
            if edge.right.ground:
                right = 'fb:' + edge.right.ground
            else:
                right = '?' + edge.right.id
            rel = 'fb:' + edge.ground

            sql += '\t%s %s %s .\n' % (left, rel, right)

        sql += '} LIMIT 100'
        # print sql
        return sql

    def to_sparql_name(self):
        """
        return sparql query that retrieve name rather than mid
        """
        sql = '''
        PREFIX fb: <http://rdf.freebase.com/ns/>
        SELECT DISTINCT ?name
        WHERE {\n
        '''
        for edge in self.edges:
            if edge.is_functional:
                continue
            if edge.left.ground:
                left = 'fb:' + edge.left.ground
            else:
                left = '?' + edge.left.id
            if edge.right.ground:
                right = 'fb:' + edge.right.ground
            else:
                right = '?' + edge.right.id
            rel = 'fb:' + edge.ground

            sql += '\t%s %s %s .\n' % (left, rel, right)
        sql += '\t?x fb:type.object.name ?name .\n'
        sql += '} LIMIT 100'
        # print sql
        return sql

    def get_candidate_entity(self, v, sparql):
        """
        get the candidates of vertex v
        """ 
        if v.ground:
            return [v.ground]
        id = v.id
        sql = self.to_sparql()
        sql = sql.replace('SELECT DISTINCT ?x', 'SELECT DISTINCT ?' + id)
        # print sql
        cand = sparql.query(sql)
        # print 'cand', cand
        return sum(cand, [])

    def get_candidate_entity_full(self, sparql):
        """
        get the candidates of all variable vertices
        """
        v_list = []
        for v in self.vertices:
            if v.id:
                v_list.append(v.id)
        sql = self.to_sparql()
        sql = sql.replace('SELECT DISTINCT ?x', 'SELECT DISTINCT ' + ' '.join(['?' + v for v in v_list]))
        cand = sparql.query(sql)
        # print v_list, 'SELECT DISTINCT ' + ' '.join(['?' + v for v in v_list]), sql
        # print sql
        return v_list, cand            
       
