"""
By Xusheng, 6/30/2017
Schema definition for tensorflow use.
Pls contact me if you have any questions or suggestions
"""

class Node(object):

    def __init__(self, id, name, embedding):
        self.id = id
        self.name = name
        self.embedding = embedding
        # skeleton_node/edge means next skeleton node/edge
        # towards object
        self.skeleton_node = None
        self.skeleton_edge = None
        # constraint_node/edge means linked constraint node/edge
        self.constraint_node = None
        self.constraint_edge = None

    def set_skeleton_node(self, node):
        self.skeleton_node = node

    def set_skeleton_edge(self, edge):
        self.skeleton_edge = edge

    def set_constraint_node(self, node):
        self.constraint_node = node

    def set_constraint_edge(self, edge):
        self.constraint_edge = edge


class Edge(object):

    def __init__(self, id, name, embedding):
        self.id = id
        self.name = name
        self.embedding = embedding


class Schema(object):

    def __init__(self, node):
        self.subj_node = node
        self.constraints = list()

    # tell me which nodes are constraints
    # for xusheng's version of model
    def add_constraint_node(self, node):
        self.constraints.append(node)

    def encode_form_text(self, text):
        # todo
        return




