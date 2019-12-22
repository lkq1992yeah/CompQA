import xmlrpclib

from kangqi.util.LogUtil import LogInfo


def main():
    srv_proxy = xmlrpclib.ServerProxy('http://202.120.38.146:9610')
    LogInfo.logs('Query Service online.')

    fuzzy_sparql = 'PREFIX fb: <http://rdf.freebase.com/ns/> ' \
                   'SELECT DISTINCT ?p1 ?p2 WHERE { ' \
                   'fb:m.06w2sn5 ?p1 ?o1 . ?o1 ?p2 ?o2 . ' \
                   '}'
    LogInfo.begin_track('Querying [%s]:', fuzzy_sparql)
    query_ret = srv_proxy.query_sparql(fuzzy_sparql)
    LogInfo.logs('query_ret: %d lines.', query_ret)
    # LogInfo.logs('query_ret: %d lines.', len(query_ret))
    # for row in query_ret:
    #     LogInfo.logs(row)
    LogInfo.end_track()
    srv_proxy.save_buffer()


if __name__ == '__main__':
    main()
