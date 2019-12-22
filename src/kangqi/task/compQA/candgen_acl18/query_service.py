"""
Author: Kangqi Luo
Date: 180208
Goal: The service for handling all SPARQL queries & maintain all queried schemas.
"""

import re
import os
import sys
import json
import codecs
from datetime import datetime

from SimpleXMLRPCServer import SimpleXMLRPCServer
from SimpleXMLRPCServer import SimpleXMLRPCRequestHandler

from ..dataset.u import load_compq, load_webq
from ..eval.official_eval import compute_f1

from ..xy_bao.sparql_backend.backend import SPARQLHTTPBackend

from kangqi.util.LogUtil import LogInfo


def show_time():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


class QueryService:

    # cache_dir = 'runnings/acl18_cache', lock=None
    def __init__(self, sparql_cache_fp, qsc_cache_fp, vb=0):
        LogInfo.begin_track('QueryService initialize ... ')
        self.year_re = re.compile(r'^[1-2][0-9][0-9][0-9]$')
        self.query_prefix = 'PREFIX fb: <http://rdf.freebase.com/ns/> '
        self.pref_len = len(self.query_prefix)

        self.sparql_dict = {}       # key: sparql, value: query_ret
        self.q_sc_dict = {}         # key: q_idx + sparql + count_or_not, value: p, r, f1, ans_size
        self.q_gold_dict = {}       # key: q_id (WebQ-xxx, CompQ-xxx), value: gold answers (with possible preprocess)

        self.sparql_buffer = []
        self.q_sc_buffer = []

        # self.lock = lock        # control multiprocess
        self.backend = SPARQLHTTPBackend('202.120.38.146', '8999', '/sparql', cache_enabled=False)
        self.vb = vb
        # vb = 0: silence
        # vb = 1: show request information or cache hit information
        # vb = 2: show the full request information, no matter hit or not
        # vb = 3: show detail return value

        compq_list = load_compq()
        webq_list = load_webq()
        for mark, qa_list in [('CompQ', compq_list), ('WebQ', webq_list)]:
            for idx, qa in enumerate(qa_list):
                gold_list = qa['targetValue']
                q_id = '%s_%d' % (mark, idx)
                self.q_gold_dict[q_id] = gold_list
        LogInfo.logs('%d QA loaded from WebQ & CompQ.', len(self.q_gold_dict))

        self.sparql_cache_fp = sparql_cache_fp
        self.qsc_cache_fp = qsc_cache_fp
        # if not os.path.isfile(self.sparql_cache_fp):
        #     os.mknod(self.sparql_cache_fp)
        # if not os.path.isfile(self.qsc_cache_fp):
        #     os.mknod(self.qsc_cache_fp)

        LogInfo.begin_track('Loading SPARQL cache ...')
        if os.path.isfile(self.sparql_cache_fp):
            with codecs.open(self.sparql_cache_fp, 'r', 'utf-8') as br:
                while True:
                    line = br.readline()
                    if line is None or line == '':
                        break
                    key, query_ret = json.loads(line)
                    self.sparql_dict[key] = query_ret
        LogInfo.logs('%d SPARQL cache loaded.', len(self.sparql_dict))
        LogInfo.end_track()

        LogInfo.begin_track('Loading <q_sc, stat> cache ...')
        if os.path.isfile(self.qsc_cache_fp):
            with codecs.open(self.qsc_cache_fp, 'r', 'utf-8') as br:
                while True:
                    line = br.readline()
                    if line is None or line == '':
                        break
                    key, stat = json.loads(line)
                    self.q_sc_dict[key] = stat
        LogInfo.logs('%d <q_sc, stat> cache loaded.', len(self.q_sc_dict))
        LogInfo.end_track()

        LogInfo.end_track('Initialize complete.')

    """ ====================== SPARQL related ====================== """

    def shrink_query(self, sparql_query):
        key = sparql_query
        if key.startswith(self.query_prefix):
            key = key[self.pref_len:]
        return key

    def kernel_query(self, key, repeat_count=10):
        sparql_str = self.query_prefix + key
        try_times = 0
        query_ret = None
        """ 180323: Try several times, if encountered exception """
        while try_times < repeat_count:
            try_times += 1
            query_ret = self.backend.query(sparql_str)
            if self.vb >= 1:
                if query_ret is not None:
                    LogInfo.logs('Query return lines = %d', len(query_ret))
                else:
                    LogInfo.logs('Query return None (exception encountered)'
                                 '[try_times=%d/%d]', try_times, repeat_count)
            if query_ret is not None:
                break
        return query_ret

    @staticmethod
    def extract_forbidden_mid(sparql_query):
        # extract all entities in the SPARQL query.
        forbidden_mid_set = set([])
        cur_st = 0
        while True:
            st = sparql_query.find('fb:m.', cur_st)
            if st == -1:
                break
            ed = sparql_query.find(' ', st + 5)
            mid = sparql_query[st+3: ed]
            forbidden_mid_set.add(mid)
            cur_st = ed
        return forbidden_mid_set

    def answer_post_process_new(self, query_ret, forbidden_mid_set, ret_symbol_list,
                                tm_comp, tm_value, allow_forever, ord_comp, ord_rank):
        """
        Deal with time filtering and ranking in this post-process.
        We can handle complex queries, such as time constraints on an interval, or ranking with ties.
        1. tm_comp: < / == / > / None
        2. tm_value: xxxx_yyyy / (blank)
        3. allow_forever: Fyes / Fdyn / Fhalf / Fno / (blank)
        4. ord_comp: min / max / None
        5. ord_rank: xxx / (blank)
        """
        """ Step 1: make sure what does each column represent """
        query_ret = filter(lambda _tup: len(_tup) == len(ret_symbol_list), query_ret)       # remove error lines
        ans_mid_pos = ans_name_pos = tm_begin_pos = tm_end_pos = ord_pos = -1
        for idx, symbol in enumerate(ret_symbol_list):
            if symbol in ('?o1', '?o2'):
                ans_mid_pos = idx
            elif symbol in ('?n1', '?n2'):
                ans_name_pos = idx
            elif symbol == '?tm1':
                tm_begin_pos = idx
            elif symbol == '?tm2':
                tm_end_pos = idx
            elif symbol == '?ord':
                ord_pos = idx
        assert ans_mid_pos != -1        # there must have ?o1 or ?o2

        """ Step 2: Filter by time """
        if tm_begin_pos == -1:      # no time constraint available
            tm_filter_ret = query_ret
        else:       # compare interval [a, b] and [c, d]
            close_open_interval = False
            if allow_forever.endswith('-co'):       # represent time interval in [begin, end) style
                close_open_interval = True
                allow_forever = allow_forever[:-3]
            assert allow_forever in ('Fyes', 'Fdyn', 'Fhalf', 'Fno')
            assert tm_comp in ('<', '==', '>')
            strict_ret = []     # both begin and end are given
            half_inf_ret = []   # one side is not given, treat as inf / -inf
            all_inf_ret = []    # both sides are not given, treat as unchangable fact
            target_begin_year, target_end_year = [int(x) for x in tm_value.split('_')]
            if tm_end_pos == -1:
                tm_end_pos = tm_begin_pos       # the schema returns a single time, not an interval
            for row in query_ret:
                begin_tm_str = row[tm_begin_pos]
                end_tm_str = row[tm_end_pos]
                begin_year = int(begin_tm_str[:4]) if re.match(self.year_re, begin_tm_str[:4]) else -12345
                end_year = int(end_tm_str[:4]) if re.match(self.year_re, end_tm_str[:4]) else 12345
                info_cate = 'strict'
                if begin_year == -12345 and end_year == 12345:
                    info_cate = 'all_inf'
                elif begin_year == -12345 or end_year == 12345:
                    info_cate = 'half_inf'

                # Taken "forever" into consideration, calculate the real state between target and output times
                if not close_open_interval:         # [begin, end]
                    if end_year < target_begin_year:
                        state = '<'
                    elif begin_year > target_end_year:
                        state = '>'
                    else:
                        state = '=='        # in, or say, during / overlap.
                else:                               # [begin, end) for both target and query result times
                    """ adjust the interval for the [begin, end) format """
                    if end_year <= begin_year:
                        end_year = begin_year + 1       # at least make sense for the first year
                    if target_end_year <= target_begin_year:
                        target_end_year = target_begin_year + 1
                    """ then compare [begin, end) with [target_begin, target_end) """
                    if end_year <= target_begin_year:
                        state = '<'
                    elif begin_year >= target_end_year:
                        state = '>'
                    else:
                        state = '=='
                if state == tm_comp:
                    if info_cate == 'strict':
                        strict_ret.append(row)
                    elif info_cate == 'half_inf':
                        half_inf_ret.append(row)
                    else:
                        all_inf_ret.append(row)
            """ Merge strict / half_inf / all_inf rows based on different strategies """
            if allow_forever == 'Fno':      # ignore half_inf or all_inf
                tm_filter_ret = strict_ret
            elif allow_forever == 'Fhalf':
                tm_filter_ret = strict_ret + half_inf_ret
            elif allow_forever == 'Fyes':   # consider all cases
                tm_filter_ret = strict_ret + half_inf_ret + all_inf_ret
            else:       # Fdyn, consider all cases, but with priority
                if len(strict_ret) > 0:
                    tm_filter_ret = strict_ret
                elif len(half_inf_ret) > 0:
                    tm_filter_ret = half_inf_ret
                else:
                    tm_filter_ret = all_inf_ret
        if self.vb >= 3:
            LogInfo.logs('After time filter = %d', len(tm_filter_ret))
            LogInfo.logs(tm_filter_ret)

        """ Step 3: Ranking """
        if ord_pos == -1:
            ordinal_ret = tm_filter_ret             # no ordinal filtering available
        else:
            ordinal_ret = []
            ord_rank = int(ord_rank)
            assert ord_comp in ('min', 'max')
            if ord_rank <= len(tm_filter_ret):
                for row in tm_filter_ret:
                    ord_str = row[ord_pos]
                    """ Try convert string value into float, could be int/float/DT value """
                    val = None
                    try:                    # int/float/DT-year-only
                        val = float(ord_str)
                    except ValueError:      # datetime represented in YYYY-MM-DD style
                        hyphen_pos = ord_str.find('-', 1)   # ignore the beginning "-", which represents "negative".
                        if hyphen_pos != -1:        # remove MM-DD information
                            val = float(ord_str[:hyphen_pos])      # only picking year from datetime
                        else:       # still something wrong
                            LogInfo.logs('Warning: unexpected ordinal value "%s"', ord_str)
                    if val is not None:
                        row[ord_pos] = val
                        ordinal_ret.append(row)
                reverse = ord_comp == 'max'
                ordinal_ret.sort(key=lambda _tup: _tup[ord_pos], reverse=reverse)
                if self.vb >= 3:
                    LogInfo.logs('Sort by ordinal constraint ...')
                    LogInfo.logs(ordinal_ret)
                target_ord_value = ordinal_ret[ord_rank-1][ord_pos]
                LogInfo.logs('target_ord_value = %s', target_ord_value)
                ordinal_ret = filter(lambda _tup: _tup[ord_pos] == target_ord_value, ordinal_ret)
        if self.vb >= 3:
            LogInfo.logs('After ordinal filter = %d', len(ordinal_ret))
            LogInfo.logs(ordinal_ret)

        """ Step 4: Final answer collection """
        forbidden_ans_set = set([])     # all the names occurred in the query result whose mid is forbidden
        normal_ans_set = set([])        # all the remaining normal names
        for row in ordinal_ret:
            ans_mid = row[ans_mid_pos]  # could be mid/int/float/datetime
            if ans_name_pos == -1:          # the answer is int/float/datetime
                ans_name = ans_mid
                if re.match(self.year_re, ans_name[:4]):
                    ans_name = ans_name[:4]
                    # if we found a DT, then we just keep its year info.
                normal_ans_set.add(ans_name)
            else:
                ans_name = row[ans_name_pos]
                if ans_mid in forbidden_mid_set:
                    forbidden_ans_set.add(ans_name)
                else:
                    normal_ans_set.add(ans_name)

        if len(normal_ans_set) > 0:  # the normal answers have a strict higher priority.
            final_ans_set = normal_ans_set
        else:  # we take the forbidden answer as output, only if we have no other choice.
            final_ans_set = forbidden_ans_set
        LogInfo.logs('Final Answer: %s', final_ans_set)
        return final_ans_set

    # def answer_post_process(self, forbidden_mid_set, query_ret):
    #     """
    #     Copied from eff_candgen/answer_query.py
    #     After retrieving the query result, we filter some answers through heuristics,
    #     and construct the final answer list.
    #     :param forbidden_mid_set:       all m.xx in sparql_query
    #     :param query_ret:               raw SPARQL result.
    #     :return: the final answer list
    #     """
    #     forbidden_name_set = set([])    # all the names occurred in the query result whose mid is forbidden
    #     normal_name_set = set([])       # all the remaining normal names
    #
    #     for row in query_ret:  # copied from loss_calc.py
    #         try:  # some query result has error (caused by \n in a string)
    #             target_mid = row[0]
    #             if target_mid.startswith('m.'):
    #                 target_name = row[1]  # get its name in FB
    #                 if target_name == '':  # ignore entities without a proper name
    #                     continue
    #                 if target_mid in forbidden_mid_set:
    #                     forbidden_name_set.add(target_name)
    #                 else:
    #                     normal_name_set.add(target_name)
    #             else:  # the answer may be a number, string or datetime
    #                 ans_name = target_mid
    #                 if re.match(self.year_re, ans_name[0: 4]):
    #                     ans_name = ans_name[0: 4]
    #                     # if we found a DT, then we just keep its year info.
    #                 normal_name_set.add(ans_name)
    #         except IndexError:  # some query result has an IndexError (caused by \n in a string)
    #             pass
    #
    #     if len(normal_name_set) > 0:  # the normal answers have a strict higher priority.
    #         final_ans_set = normal_name_set
    #     else:  # we take the forbidden answer as output, only if we have no other choice.
    #         final_ans_set = forbidden_name_set
    #     return final_ans_set

    @staticmethod
    def compq_answer_normalize(ans):
        """ Change hyphen and en-dash into '-', and then lower case. """
        return re.sub(u'[\u2013\u2212]', '-', ans).lower()

    """ ==================== Registered Functions ==================== """

    def query_sparql(self, sparql_query):
        key = self.shrink_query(sparql_query)
        hit = key in self.sparql_dict
        show_request = (not hit or self.vb >= 2)    # going to perform a real query, or just want to show more
        if show_request:
            LogInfo.begin_track('[%s] SPARQL Request:', show_time())
            LogInfo.logs(key)
        if hit and self.vb >= 1:
            if not show_request:
                LogInfo.logs('[%s] SPARQL hit!', show_time())
            else:
                LogInfo.logs('SPARQL hit!')
        if hit:
            query_ret = self.sparql_dict[key]
        else:
            query_ret = self.kernel_query(key=key)

            # if self.lock is not None:
            #     self.lock.acquire()
            if query_ret is not None:       # ignore schemas returning None
                self.sparql_dict[key] = query_ret
                self.sparql_buffer.append((key, query_ret))
            # if self.lock is not None:
            #     self.lock.release()

        if show_request and self.vb >= 3:
            LogInfo.logs(query_ret)
        if show_request:
            LogInfo.end_track()

        final_query_ret = query_ret or []
        return final_query_ret          # won't send None back, but use empty list as instead

    # def query_q_sc_stat(self, q_id, sparql_query, aggregate=False):
    #     """
    #     :param q_id:            CompQ-xxxx, or WebQ-xxx
    #     :param sparql_query:    ordinary SPARQL query
    #     :param aggregate:       whether manually perform COUNT(*) or not
    #     :return: [ans_size, p, r, f1]
    #     """
    #     key = self.shrink_query(sparql_query)
    #     if not aggregate:
    #         q_sc_key = '%s|%s' % (q_id, key)
    #     else:
    #         q_sc_key = '%s|Agg|%s' % (q_id, key)

    def query_q_sc_stat(self, q_sc_key):
        """
        q_sc_key: q_id | SPARQL | tm_comp | tm_value | allow_forever | ord_comp | ord_rank | aggregate
        1. tm_comp: < / == / > / None
        2. tm_value: xxxx_yyyy / (blank)
        3. allow_forever: Fyes / Fdyn / Fhalf / Fno / (blank)
        4. ord_comp: min / max / None
        5. ord_rank: xxx / (blank)
        6. aggregate: Agg / None
        """
        hit = q_sc_key in self.q_sc_dict
        show_request = (not hit or self.vb >= 2)  # going to perform a real query, or just want to show more
        if show_request:
            LogInfo.begin_track('[%s] Q_Schema Request:', show_time())
            LogInfo.logs(q_sc_key)
        if hit and self.vb >= 1:
            if not show_request:
                LogInfo.logs('[%s] Q_Schema hit!', show_time())
            else:
                LogInfo.logs('Q_Schema hit!')
        if hit:
            stat = self.q_sc_dict[q_sc_key]
            spt = stat.split('_')
            ans_size = int(spt[0])
            p, r, f1 = [float(x) for x in spt[1:]]
        else:
            """
            We don't save the detail result of a non-fuzzy SPARQL query in the F1-query scenario,
            that is because we don't expect that a detail SPARQL query occurs many times in different questions.
            If would cost too much memory if we force to save all non-fuzzy query results,
            the main motivation of using cache is to support the client program quickly running multiple times,
            not for saving time ACROSS DIFFERENT QUESTIONS.
            """
            (q_id, sparql_query,
             tm_comp, tm_value, allow_forever,
             ord_comp, ord_rank, agg) = q_sc_key.split('|')
            assert sparql_query.startswith('SELECT DISTINCT ')
            assert ' WHERE ' in sparql_query
            symbol_str = sparql_query[16: sparql_query.find(' WHERE ')]
            ret_symbol_list = symbol_str.split(' ')
            gold_list = self.q_gold_dict[q_id]
            forbidden_mid_set = self.extract_forbidden_mid(sparql_query=sparql_query)
            if self.vb >= 1:
                LogInfo.logs('Forbidden mid: %s', forbidden_mid_set)
            key = self.shrink_query(sparql_query=sparql_query)
            query_ret = self.kernel_query(key)
            if query_ret is None:   # encountered error, and won't save q_sc_stat.
                ans_size = 0
                p = r = f1 = 0.
            else:
                # predict_value = self.answer_post_process(
                #     forbidden_mid_set=forbidden_mid_set, query_ret=query_ret)
                predict_value = self.answer_post_process_new(
                    query_ret=query_ret,
                    forbidden_mid_set=forbidden_mid_set,
                    ret_symbol_list=ret_symbol_list,
                    tm_comp=tm_comp, tm_value=tm_value,
                    allow_forever=allow_forever,
                    ord_comp=ord_comp, ord_rank=ord_rank
                )
                predict_list = list(predict_value)      # change from set to list
                if agg == 'Agg':                        # manually perform COUNT(*)
                    distint_answers = len(predict_value)
                    predict_list = [str(distint_answers)]   # only one answer: the number of distinct targets.

                ans_size = len(predict_list)
                if q_id.startswith('Webq'):
                    r, p, f1 = compute_f1(gold_list, predict_list)
                else:  # CompQ
                    """
                    1. force lowercase, both gold and predict
                    2. hyphen normalize: -, \u2013, \u2212 
                    Won't the affect the values to be stored in the file.
                    """
                    eval_gold_list = [self.compq_answer_normalize(x) for x in gold_list]
                    eval_predict_list = [self.compq_answer_normalize(x) for x in predict_list]
                    r, p, f1 = compute_f1(eval_gold_list, eval_predict_list)
                stat = '%d_%.6f_%.6f_%.6f' % (ans_size, p, r, f1)

                # if self.lock is not None:
                #     self.lock.acquire()
                self.q_sc_dict[q_sc_key] = stat
                self.q_sc_buffer.append((q_sc_key, stat))
                # if self.lock is not None:
                #     self.lock.release()

        ret_info = [ans_size, p, r, f1]
        if show_request:
            LogInfo.logs('Answers = %d, P = %.6f, R = %.6f, F1 = %.6f', ans_size, p, r, f1)
            LogInfo.end_track()
        return ret_info

    def save_buffer(self):
        # if self.lock is not None:
        #     self.lock.acquire()
        with codecs.open(self.sparql_cache_fp, 'a', 'utf-8') as bw:
            for tup in self.sparql_buffer:
                bw.write(json.dumps(tup))
                bw.write('\n')
        with codecs.open(self.qsc_cache_fp, 'a', 'utf-8') as bw:
            for tup in self.q_sc_buffer:
                bw.write(json.dumps(tup))
                bw.write('\n')
        self.sparql_buffer = []
        self.q_sc_buffer = []
        LogInfo.logs('[%s] buffer saved.', show_time())
        # if self.lock is not None:
        #     self.lock.release()
        return 0


# Restrict to a particular path.
class RequestHandler(SimpleXMLRPCRequestHandler):
    rpc_paths = ('/RPC2',)


def main():
    # Create server
    srv_port = int(sys.argv[1])
    cache_dir = sys.argv[2]
    vb = int(sys.argv[3])
    LogInfo.logs('srv_port = %d, vb = %d', srv_port, vb)
    server = SimpleXMLRPCServer(("0.0.0.0", srv_port), requestHandler=RequestHandler)
    server.register_introspection_functions()

    service_inst = QueryService(sparql_cache_fp=cache_dir+'/sparql.cache',
                                qsc_cache_fp=cache_dir+'/q_sc_stat.cache',
                                vb=vb)
    server.register_function(service_inst.query_sparql)
    server.register_function(service_inst.query_q_sc_stat)
    server.register_function(service_inst.save_buffer)
    LogInfo.logs('Functions registered: %s', server.system_listMethods())

    # Run the server's main loop
    LogInfo.logs('Begin serving ... ')
    server.serve_forever()


def test():
    service_inst = QueryService(
        sparql_cache_fp='runnings/acl18_cache/tmp/sparql.cache',
        qsc_cache_fp='runnings/acl18_cache/tmp/q_sc_stat.cache',
        vb=3
    )

    # q_id = 'CompQ_1352'
    # sparql = 'SELECT DISTINCT ?o2 ?n2 ?tm1 ?tm2 ?ord WHERE { ' \
    #          'fb:m.07sz1 fb:law.court.judges ?o1 . ' \
    #          '?o1 fb:law.judicial_tenure.judge ?o2 . ' \
    #          '?o2 fb:type.object.name ?n2 . ' \
    #          '?o1 fb:law.judicial_tenure.from_date ?ord . ' \
    #          'OPTIONAL { ?o1 fb:law.judicial_tenure.from_date ?tm1 . } . ' \
    #          'OPTIONAL { ?o1 fb:law.judicial_tenure.to_date ?tm2 . } . ' \
    #          '}'
    # tm_comp = '=='
    # tm_value = '2009_2009'
    # allow_forever = 'Fno'
    # ord_comp = 'max'
    # ord_rank = '1'
    # agg = 'None'

    # q_id = 'CompQ_1783'
    # sparql = 'SELECT DISTINCT ?o2 ?n2 ?ord WHERE { ' \
    #          'fb:m.01d5z fb:baseball.baseball_team.team_stats ?o1 . ' \
    #          '?o1 fb:baseball.baseball_team_stats.season ?o2 . ' \
    #          '?o1 fb:baseball.baseball_team_stats.wins ?ord . ' \
    #          '?o2 fb:type.object.name ?n2 . } '
    # tm_comp = 'None'
    # tm_value = ''
    # allow_forever = ''
    # ord_comp = 'max'
    # ord_rank = '1'
    # agg = 'None'

    q_id = 'CompQ_1705'
    sparql = 'SELECT DISTINCT ?o1 ?n1 ?ord WHERE { ' \
             'fb:m.06x5s fb:time.recurring_event.instances ?o1 . ' \
             '?o1 fb:sports.sports_championship_event.champion fb:m.05tfm . ' \
             '?o1 fb:type.object.name ?n1 . ' \
             '?o1 fb:time.event.end_date ?ord . ' \
             '}'
    tm_comp = 'None'
    tm_value = ''
    allow_forever = ''
    ord_comp = 'max'
    ord_rank = '1'
    agg = 'None'

    q_sc_key = '|'.join([q_id, sparql,
                         tm_comp, tm_value, allow_forever,
                         ord_comp, ord_rank, agg])
    service_inst.query_q_sc_stat(q_sc_key=q_sc_key)


if __name__ == '__main__':
    # main()
    test()
