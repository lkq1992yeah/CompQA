import copy
import time
import numpy as np
from datetime import datetime

from ..dataset.kq_schema import CompqSchema
from ..eff_candgen.combinator import predicate_filtering
from ..util.fb_helper import get_pred_name, inverse_predicate, get_begin_dt_pred, \
    is_type_contained_by, is_type_contains, is_type_ignored

from kangqi.util.LogUtil import LogInfo
from kangqi.util.time_track import TimeTracker as Tt


class CandidateSearcher:

    """ Consider both from Lukov Lexicon and S-MART"""

    """
    Main Framework:
    Given the question and gather_linkings (from either Lexicon or S-MART),
    we first try to apply fuzzy matching technique and retrieve instantiations of different templates.
    For each coarse schema (with fixed focus, constraint entity and predicates),
    we calculate its segmentation score and apply filtering (if necessary, but not performed in this class).
    Afterwards, instead of fuzzy matching, we enumerate legal type/time/ordinal constraints
    over each coarse schema, which would make the generation step quicker.
    """

    def __init__(self, query_service, wd_emb_util, vb):
        self.query_service = query_service
        self.wd_emb_util = wd_emb_util
        self.top_similar_ords = 4
        self.pred_emb_dict = {}         # <predicate, embedding>, act as a cache
        self.vb = vb
        # vb = 0: total slience
        # vb = 1: show basic flow of the process
        # vb = 4: show data received from the query_service

    def find_coarse_schemas(self, entity_linkings, conflict_matrix, aggregate=False):
        """
        Just consider focus and entity constraint, this function generates a list of coarse schemas.
        Regard it as semi-product of the final one.
        :param entity_linkings:     LinkData for entities only.
        :param conflict_matrix:     indicating conflicts between linkings.
        :param aggregate:           whether to force apply aggregation to all candidate schemas
        :return:    coarse schemas, containing focus, constraints and paths.
        """
        el_size = len(entity_linkings)
        gl_size = len(conflict_matrix)
        coarse_state_list = []          # [(compq_schema, visit_arr)]
        for path_len in (1, 2):
            for mf_idx, main_focus in enumerate(entity_linkings):
                state_marker = ['Len-%d||F%d/%d' % (path_len, mf_idx + 1, el_size)]
                gl_pos = main_focus.gl_pos
                visit_arr = [0] * gl_size   # indicating how many conflicts at the particular searching state
                cur_comb = [(0, mf_idx)]    # collects all focus / e-constraint used in the current state
                for conf_idx in conflict_matrix[gl_pos]:
                    visit_arr[conf_idx] += 1
                self.coarse_search(path_len=path_len,
                                   entity_linkings=entity_linkings,
                                   conflict_matrix=conflict_matrix,
                                   cur_el_idx=-1, cur_comb=cur_comb,
                                   visit_arr=visit_arr,
                                   coarse_state_list=coarse_state_list,
                                   state_marker=state_marker,
                                   aggregate=aggregate)
        return coarse_state_list

    def coarse_search(self, path_len, entity_linkings, conflict_matrix,
                      cur_el_idx, cur_comb, visit_arr, coarse_state_list, state_marker, aggregate):
        if self.vb >= 1:
            LogInfo.begin_track('[%s] (%s)', '||'.join(state_marker),
                                datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        fuzzy_query = self.build_fuzzy_query(path_len, entity_linkings, cur_comb)
        if self.vb >= 1:
            LogInfo.logs('Fuzzy query: %s', fuzzy_query)

        st = time.time()
        Tt.start('query_sparql')
        query_ret = self.query_service.query_sparql(fuzzy_query)
        Tt.record('query_sparql')
        if self.vb >= 4:
            LogInfo.logs('Recv >>>: %s', query_ret)
        filt_query_ret = predicate_filtering(query_ret=query_ret, path_len=path_len)
        if self.vb >= 1:
            LogInfo.logs('Filt_Query_Ret = %d / %d (%.3fs)', len(filt_query_ret), len(query_ret), time.time() - st)
        if len(filt_query_ret) == 0:
            if self.vb >= 1:
                LogInfo.end_track()
            return  # no need to search deeper

        coarse_schemas = self.build_coarse_schemas(path_len=path_len, el_linkings=entity_linkings,
                                                   cur_comb=cur_comb, filt_query_ret=filt_query_ret,
                                                   aggregate=aggregate)
        for sc in coarse_schemas:
            coarse_state_list.append((sc, list(visit_arr)))     # save coarse schemas into the output state

        # Ready to search deeper
        el_size = len(entity_linkings)
        for nxt_el_idx in range(cur_el_idx + 1, el_size):
            gl_pos = entity_linkings[nxt_el_idx].gl_pos
            if visit_arr[gl_pos] != 0:                  # cannot be visited due to conflict
                continue
            for conf_idx in conflict_matrix[gl_pos]:    # ready to enter the next state
                visit_arr[conf_idx] += 1
            for attach_idx in range(1, path_len+1):     # enumerate each possible attach position
                nxt_comb = list(cur_comb)
                nxt_comb.append((attach_idx, nxt_el_idx))
                state_marker.append('%d/%d-%d' % (nxt_el_idx + 1, el_size, attach_idx))
                self.coarse_search(path_len=path_len,
                                   entity_linkings=entity_linkings,
                                   conflict_matrix=conflict_matrix,
                                   cur_el_idx=nxt_el_idx,
                                   cur_comb=nxt_comb,
                                   visit_arr=visit_arr,
                                   coarse_state_list=coarse_state_list,
                                   state_marker=state_marker,
                                   aggregate=aggregate)
                del state_marker[-1]
            for conf_idx in conflict_matrix[gl_pos]:  # return back
                visit_arr[conf_idx] -= 1
        # Ends of DFS
        if self.vb >= 1:
            LogInfo.end_track()

    def find_typed_schemas(self, type_linkings, conflict_matrix, start_schema, visit_arr, simple_type_match):
        """
        Given the current schema, try adding only one type to the answer node.
        Adding on mediator node is not allowed.
        simple_type_match == True: Following Bao, no type filtering.
        """
        local_states = []
        start_schema.infer_var_types()  # build the explicit types for o1 and o2
        if not simple_type_match:           # perform type filtering
            use_type_linkings = self.type_filtering(start_schema, type_linkings)
        else:                               # no filtering at all
            use_type_linkings = type_linkings
        for tl_data in use_type_linkings:
            gl_pos = tl_data.gl_pos
            if visit_arr[gl_pos] != 0:  # cannot be visited due to conflict
                continue
            typed_visit_arr = list(visit_arr)  # new state after applying types
            for conf_idx in conflict_matrix[gl_pos]:
                typed_visit_arr[conf_idx] += 1
            typed_sc = copy.deepcopy(start_schema)
            typed_sc.raw_paths.append(('Type', tl_data, ['type.object.type']))
            typed_sc.var_types[-1].add(tl_data.value)       # the current type becomes explicit
            local_states.append((typed_sc, typed_visit_arr))
        return local_states

    @staticmethod
    def find_timed_schemas(time_linkings, conflict_matrix, start_schema, visit_arr, simple_time_match):
        """
        Given the current schema, try adding time constraint into previous schemas.
        Based on our observation, we don't need to consider adding more than one time constraints,
        which makes the code simpler and avoids the recursive function.
        simple_time_match == True: Following Bao, won't use time interval information.
        """
        local_states = []
        start_schema.infer_time_preds()     # got all time-related predicates
        for tml_data in time_linkings:
            gl_pos = tml_data.gl_pos
            if visit_arr[gl_pos] != 0:          # cannot be visited due to conflict
                continue
            timed_visit_arr = list(visit_arr)   # new state after selecting this time span
            for conf_idx in conflict_matrix[gl_pos]:
                timed_visit_arr[conf_idx] += 1
            for attach_idx in range(1, start_schema.hops+1):        # enumerate ?o1 and ?o2
                for time_pred in start_schema.var_time_preds[attach_idx-1]:     # enumerate each time predicate
                    """
                    Kangqi on 180308:
                    we consider time intervals.
                    Suppose we take "gov.gov_position.held.from" as the time predicate,
                    we are actually using both "from" and "to" to locate the time interval,
                    instead of using them individually.
                    Check the detail in sc.build_sparql()
                    """
                    if not simple_time_match:   # let's consider time interval
                        if get_begin_dt_pred(time_pred) != '':
                            continue    # won't pick the ending time predicate, should pair with beginning time pred.
                    timed_sc = copy.deepcopy(start_schema)
                    pred_seq = [time_pred]
                    if timed_sc.hops == 2 and attach_idx == 1:  # constraint at mediator
                        pred_seq = [timed_sc.inv_main_pred_seq[0]] + pred_seq
                    timed_sc.raw_paths.append(('Time', tml_data, pred_seq))
                    local_states.append((timed_sc, timed_visit_arr))
        return local_states

    def find_ordinal_schemas(self, ordinal_linkings, start_schema, visit_arr):
        """
        Given the current schema, try adding only one ordinal constraint into previous schemas.
        Since this is the final stage, the output visit_arr is no longer required.
        """
        local_states = []
        start_schema.infer_ordinal_preds()      # get all ordinal-related predicates

        # detail checking code
        # if 'geography.river' in start_schema.var_types[-1]:
        #     LogInfo.logs('Hit-Inside: %s', start_schema.disp_raw_path())
        #     LogInfo.logs('var_types: %s', start_schema.var_types)
        #     LogInfo.logs('var_ord_preds: %s', start_schema.var_ord_preds)

        """ First collect Top-K most similar ordinal predicates from all ordinal mentions """
        # We worry about generating too many schemas here, therefore we won't make combination any more
        ordinal_sim_tups = []       # (ord_idx, attach_idx, ordinal_pred, sim)
        for ord_idx, ord_data in enumerate(ordinal_linkings):
            gl_pos = ord_data.gl_pos
            if visit_arr[gl_pos] != 0:  # cannot be visited due to conflict
                continue
            target_word = ord_data.mention.split(' ')[-1]  # last word of the ordinal mention
            target_emb = self.wd_emb_util.get_phrase_emb(target_word)
            if target_emb is None:      # unknown word as the ordinal mention (shouldn't be)
                continue
            for attach_idx in range(1, start_schema.hops + 1):      # enumerate each attach_idx and ordinal_predicate
                for ordinal_pred in start_schema.var_ord_preds[attach_idx - 1]:
                    if ordinal_pred not in self.pred_emb_dict:
                        pred_name = get_pred_name(ordinal_pred)
                        pred_emb = self.wd_emb_util.get_phrase_emb(pred_name)
                        self.pred_emb_dict[ordinal_pred] = pred_emb
                    else:
                        pred_emb = self.pred_emb_dict[ordinal_pred]
                    if pred_emb is None:
                        continue
                    sim = np.sum(target_emb * pred_emb).astype(float)
                    ordinal_sim_tups.append((ord_idx, attach_idx, ordinal_pred, sim))

        # Current: pick top-4 ordinal constraints (ideally speaking, 2 max- and 2 min-)
        ordinal_sim_tups.sort(key=lambda _tup: _tup[-1], reverse=True)
        pick_tups = ordinal_sim_tups[:self.top_similar_ords]

        # if len(ordinal_sim_tups) > 0:
        #     LogInfo.begin_track('Check ordinal similarity of %d tuples:', len(ordinal_sim_tups))
        #     for ord_idx, attach_idx, ordinal_pred, sim in ordinal_sim_tups:
        #         ord_data = ordinal_linkings[ord_idx]
        #         LogInfo.logs('%s --> %s %.6f', ord_data.mention.encode('utf-8'), ordinal_pred, sim)
        #     LogInfo.end_track()

        for ord_idx, attach_idx, ordinal_pred, sim in pick_tups:
            ord_data = ordinal_linkings[ord_idx]
            final_sc = copy.deepcopy(start_schema)
            pred_seq = [ordinal_pred]
            if final_sc.hops == 2 and attach_idx == 1:  # constraint at mediator
                pred_seq = [final_sc.inv_main_pred_seq[0]] + pred_seq
            final_sc.raw_paths.append(('Ordinal', ord_data, pred_seq))
            local_states.append((final_sc, None))
        return local_states

    # =================== Utility functions ================== #

    @staticmethod
    def build_fuzzy_query(path_len, entity_linkings, cur_comb):
        assert cur_comb[0][0] == 0      # the first attach information must be the focus
        where_main_line = ''
        where_constr_lines = []

        add_p_idx = path_len
        for attach_idx, el_idx in cur_comb:
            mid = entity_linkings[el_idx].value
            if attach_idx == 0:     # main line
                where_main_line = 'fb:%s ?p1 ?o1 .' % mid
                if path_len == 2:
                    where_main_line += ' ?o1 ?p2 ?o2 .'
            else:
                add_p_idx += 1
                where_constr_lines.append('?o%d ?p%d fb:%s .' % (attach_idx, add_p_idx, mid))
        where_constr_lines.sort()

        select_line = 'SELECT DISTINCT'
        for p_idx in range(add_p_idx):
            select_line += ' ?p%d' % (p_idx+1)

        sparql_lines = [select_line, 'WHERE {', where_main_line] + where_constr_lines + ['}']
        sparql_str = ' '.join(sparql_lines)
        return sparql_str

    @staticmethod
    def build_coarse_schemas(path_len, el_linkings, cur_comb, filt_query_ret, aggregate):
        # just build the schema, but not querying F1 now.
        coarse_sc_list = []
        for row in filt_query_ret:
            sc = CompqSchema()
            sc.hops = path_len
            sc.aggregate = aggregate
            sc.main_pred_seq = row[:path_len]       # [p1, p2]
            sc.inv_main_pred_seq = [inverse_predicate(pred) for pred in sc.main_pred_seq]    # [!p1, !p2]
            sc.inv_main_pred_seq.reverse()          # [!p2, !p1]
            sc.raw_paths = []
            col_idx = path_len
            for attach_idx, el_idx in cur_comb:
                el_data = el_linkings[el_idx]
                if attach_idx == 0:
                    sc.raw_paths.append(('Main', el_data, sc.main_pred_seq))
                else:
                    pred = row[col_idx]
                    pred_seq = [pred]
                    if path_len == 2 and attach_idx == 1:   # constraint at mediator
                        pred_seq = [sc.inv_main_pred_seq[0]] + pred_seq
                    sc.raw_paths.append(('Entity', el_data, pred_seq))
                    col_idx += 1        # scan another constraint
            coarse_sc_list.append(sc)
        return coarse_sc_list

    @staticmethod
    def type_filtering(sc, type_linkings):
        filt_type_linkings = []
        ans_type_set = sc.var_types[-1]
        for tl_data in type_linkings:
            tp = tl_data.value
            if is_type_ignored(tp):
                continue
            flag = False
            for ans_tp in ans_type_set:
                if tp == ans_tp:
                    flag = True
                elif is_type_contained_by(tp, ans_tp):
                    flag = True
                elif is_type_contains(tp, ans_tp):
                    flag = True
                if flag:
                    filt_type_linkings.append(tl_data)
                    break
        # all valid types are kept.
        return filt_type_linkings
