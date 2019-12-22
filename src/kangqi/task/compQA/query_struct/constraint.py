# -*- coding: utf-8 -*-


class Constraint(object):

    # comp: will be used in time constraint (before 2012, in 2012, after 2012)
    # and will also be used in ordinal constraint (ASC, DESC)
    def __init__(self, x, p, comp, o, constr_type, linking_item):
        self.x, self.p, self.comp, self.o, self.constr_type, self.linking_item = \
            x, p, comp, o, constr_type, linking_item

    def to_string(self):
        p_str = self.p
        if isinstance(p_str, unicode): p_str = p_str.encode('utf-8')
        o_str = self.o
        if isinstance(o_str, unicode): o_str = o_str.encode('utf-8')
        if self.constr_type == 'Entity':
            o_str += ' (%s)' %self.linking_item.name.encode('utf-8')

        return '%s: x%d, %s, %s %s' %(self.constr_type, self.x, p_str, self.comp, o_str)
