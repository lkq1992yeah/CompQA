# -*- coding: utf-8 -*-

#==============================================================================
# Author: Kangqi Luo
# Goal: Given a schema and the gold answer (set), return the loss score (measuring the similarity)
# Ordinary questions: just use F1
# If used ordinal constraint: allow a decay based on reciprocal rank diff.
#==============================================================================

import re

from kangqi.util.LogUtil import LogInfo


class LossCalculator(object):

    # send a sparql query driver as input
    def __init__(self, driver):
        self.driver = driver
        self.year_re = re.compile(r'[0-9]{4}')
        pass

    def f1(self, predict_set, answer_set):
        p = r = f1 = 0.0
        if len(predict_set) != 0:
            joint = predict_set & answer_set
            p = 1.0 * len(joint) / len(predict_set)
            r = 1.0 * len(joint) / len(answer_set)
            f1 = 2.0 * p * r / (p + r) if p > 0.0 else 0.0
        # LogInfo.logs('p = %.6f, r = %.6f, f1 = %.6f', p, r, f1)
        return {'p': p, 'r': r, 'f1': f1}


    # we are calculating the score based on F1.
    def calculate(self, schema, answer_set, vb=0):
        sparql_str_list, var_list = schema.get_sparql_str()
        sparql_str = ' '.join(sparql_str_list)
        query_result = self.driver.perform_query(sparql_str)
        if vb >= 1:
            LogInfo.logs('%d ground result extracted, showing examples:', len(query_result))
            LogInfo.logs('%s', query_result[:10])

        # when output the answer, we omit all the focus / constraint entities in the SPARQL
        forbidden_mid_set = set([])
        if schema.focus_item is not None:
            focus_mid = schema.focus_item.entity.id
            forbidden_mid_set.add(schema.focus_item.entity.id)
        for constr in schema.constraints:
            if constr.constr_type == 'Entity':
                forbidden_mid_set.add(constr.o)     # constraint object mid
        forbidden_name_set = set([])        # the names of forbidden entities, provided through querying process

        predict_set = set([])
        var_sz = len(var_list)
        target_var = ''
        for var in var_list:
            if var.startswith('?n'):
                target_var = var.replace('n', 'x')
                break
        if target_var == '':
            LogInfo.logs('Error: cannot get the target variable!')
        else:
            concern_col = -1
            for col_idx in range(var_sz):
                if var_list[col_idx] == target_var:
                    concern_col = col_idx
                    break
            if concern_col == -1:
                LogInfo.logs('Error: concern_col = -1')
            else:
                if vb >= 1:
                    LogInfo.logs('Concern column = %d (%s)', concern_col, var_list[concern_col])

                for row in query_result:
                    try:    # some query result has error (caused by \n in a string)
                        target_mid = row[concern_col]
                        if target_mid.startswith('m.'):
                            target_name = row[0]        # get its name in FB
                            if target_mid in forbidden_mid_set:
                                forbidden_name_set.add(target_name)
                            if target_name != '':
                                predict_set.add(target_name)
                        else:       # the answer may be a number, string or datetime
                            ans_name = target_mid
                            if re.match(self.year_re, ans_name[0: 4]):
                                ans_name = ans_name[0 : 4]
                                # if we found a DT, then we just keep its year info.
                            predict_set.add(ans_name)
                    except IndexError:
                        pass

        if '' in predict_set:
            predict_set.remove('')      # won't keep blank name
        if len(predict_set - forbidden_name_set) != 0:
            predict_set -= forbidden_name_set
            # we remove the forbidden_mid_set from answers,
            # only if there have remaining entities as the output

        if vb >= 1:
            sample_predict_set = set(list(predict_set)[:50])
            LogInfo.logs('Predict size = %d: %s%s',
                         len(predict_set),
                         sample_predict_set,
                         ' ...' if len(sample_predict_set) < len(predict_set) else '')
            LogInfo.logs('Answers size = %d: %s', len(answer_set), answer_set)
        score_dict = self.f1(predict_set, answer_set)
        return predict_set, score_dict


