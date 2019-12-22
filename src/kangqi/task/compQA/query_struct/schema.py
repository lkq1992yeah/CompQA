# -*- coding: utf-8 -*-

'''
Author: Kangqi Luo
Goal: define a intention structure, containing the following information:
    1. focus entity (interval, mid, prob)
    2. path (sequence of predicates)
    3. constraints (<x, p, comp, o>)
'''

from kangqi.util.LogUtil import LogInfo

class Schema(object):

    def __init__(self):
        self.focus_item = None
        self.constraints = []
        self.path = []      # sequence of predicate IDs.

    def get_display_str(self):
        str_list = []
        focus_str = 'Focus: %s (%s)' %(
                self.focus_item.name.encode('utf-8'),
                self.focus_item.entity.id.encode('utf-8')
                ) if self.focus_item is not None else 'Focus: None'
        path_str = 'Path: ' + ', '.join(self.path)
        str_list.append(focus_str)
        str_list.append(path_str)
        for constraint in self.constraints:
            str_list.append(constraint.to_string())
        return str_list

    def display(self):
        LogInfo.begin_track('Showing schema detail: ')
        for disp in self.get_display_str():
            LogInfo.logs(disp)
        LogInfo.end_track()

    def to_string(self):
        return '\t'.join(self.get_display_str())

    # convert this kind of schema into xy's line format
    # focus --> y0 --> y1 --> ... --> x
    def to_xy_schema_line(self):
        elem_list = []
        for pred_pos, pred in enumerate(self.path):     # split path
            subj = self.focus_item.entity.id.encode('utf-8') \
                if pred_pos == 0 else 'y%d' % (pred_pos - 1)            # y0, y1, y2 ...
            obj = 'x' if pred_pos == len(self.path) - 1 else 'y%d' % pred_pos
            elem_list.append('%s--%s--%s' % (subj, pred, obj))

        y_idx = len(self.path) - 1  # count the index of auxiliary node
        for constr in self.constraints:                 # split each constraint
            subj = 'x' if constr.x == len(self.path) \
                else 'y%d' % (constr.x - 1)
            obj = constr.o
            if constr.constr_type in ('Entity', 'Type'):
                elem_list.append('%s--%s--%s' % (subj, constr.p, obj))
            else:
                elem_list.append('%s--%s--y%d' % (subj, constr.p, y_idx))
                elem_list.append('y%d--%s--%s' % (y_idx, constr.comp, obj))     # max/min/gt/lt
                y_idx += 1
        return '\t'.join(elem_list)

    # convert the schema into sparql query.
    # returns:
    # 1. sparql intention stored in a list, each item is a line of intention code.
    # 2. the (ordered) list of variables (headers) of this sparql query.
    # TODO: Add reverse predicate control.
    def get_sparql_str(self):
        var_set = set([])
        ordinal_constr = None
        constr_list = []
        for idx in range(len(self.path)):
            var_set.add('?x%d' %(idx + 1))
        # first deal with traditional constraints
        for constr in self.constraints:
            var_pos = constr.x
            var_symbol = '?x%d' %var_pos
            var_set.add(var_symbol)
            if constr.constr_type == 'Ordinal':
                ordinal_constr = constr
                continue        # we later deal with ordinal constraint
            elif constr.constr_type in ['Entity', 'Type']:
                constr_list.append('%s fb:%s fb:%s .' %(var_symbol, constr.p, constr.o))
            elif constr.constr_type == 'Time':
                # currently we just focus on year information
                tm_var_symbol = '?y%d' %var_pos
                var_set.add(tm_var_symbol)
                constr_list.append('%s fb:%s %s .' %(var_symbol, constr.p, tm_var_symbol))
                year = constr.o
                if year is not None:
                    first_day = '"%s-01-01"^^xsd:dateTime' %year
                    last_day = '"%s-12-31"^^xsd:dateTime' %year
                    comp = constr.comp
                    if comp == '==':
                        constr_list.append('FILTER (%s >= %s) .' %(tm_var_symbol, first_day))
                        constr_list.append('FILTER (%s <= %s) .' %(tm_var_symbol, last_day))
                    elif comp == '>=':
                        constr_list.append('FILTER (%s >= %s) .' %(tm_var_symbol, first_day))
                    elif comp == '>':
                        constr_list.append('FILTER (%s > %s) .' %(tm_var_symbol, last_day))
                    elif comp == '<':
                        constr_list.append('FILTER (%s < %s) .' %(tm_var_symbol, first_day))
                    else:
                        LogInfo.logs('Found unknown comp in time constraint: %s', comp)
                else:
                    pass    # We encountered a time constraint for ground query test. Thus, it's OK to just add a solid edge.
            else:
                LogInfo.logs('Found unknown constraint type: %s', constr.constr_type)
        # Second: show target name
        target_var_symbol = '?x%d' %len(self.path)
        target_var_name_symbol = '?n%d' %len(self.path)
        var_set.add(target_var_symbol); var_set.add(target_var_name_symbol)
        constr_list.append(
            'OPTIONAL { %s fb:type.object.name %s } .' %(target_var_symbol, target_var_name_symbol))
        # Third: deal with ordinal constraints
        if ordinal_constr is not None:
            ordinal_var_symbol = '?x%d' %ordinal_constr.x
            ordinal_target_symbol = '?ord'
            var_set.add(ordinal_target_symbol)
            constr_list.append(
                '%s fb:%s %s .' %(ordinal_var_symbol, ordinal_constr.p, ordinal_target_symbol))

        var_list = sorted(var_set) # first n..., then x..., and finally y....
        var_str = ' '.join(var_list)
        sparql_str_list = []
        sparql_str_list.append('PREFIX fb: <http://rdf.freebase.com/ns/>')
        sparql_str_list.append('SELECT DISTINCT %s WHERE {' %var_str)
        for path_idx in range(len(self.path)):  # add path information
            pred = self.path[path_idx]
            if path_idx == 0:
                sparql_str_list.append(
                    'fb:%s fb:%s ?x%d .' %(self.focus_item.entity.id, pred, path_idx + 1))
            else:
                sparql_str_list.append('?x%d fb:%s ?x%d .' %(path_idx, pred, path_idx + 1))
        sparql_str_list += constr_list
        if ordinal_constr is None or ordinal_constr.o == None:
            # Either we don't have ordinal constraints, or we just use it for ground query test
            sparql_str_list.append('}')
        else:
            lmt = 100
            sparql_str_list.append(
                '} ORDER BY %s(?ord) LIMIT %d OFFSET %d' %(
                ordinal_constr.comp, lmt, ordinal_constr.o - 1))
        return sparql_str_list, var_list

    def display_sparql(self):
        LogInfo.begin_track('Showing SPARQL: ')
        for disp in self.get_sparql_str()[0]:
            LogInfo.logs(disp)
        LogInfo.end_track()

    def to_sparql(self):
        return '\t'.join(self.get_sparql_str()[0])