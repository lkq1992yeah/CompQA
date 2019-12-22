from ..sparql.sparql import SparqlDriver

from kangqi.util.LogUtil import LogInfo


def main():
    sparql_driver = SparqlDriver(sparql_ip='202.120.38.146',
                                 sparql_port=8999,
                                 use_cache=False,
                                 verbose=0)
    # test_list = ['m.02lt8']
    # # test_list = ['m.03647x', 'm.09l3p']
    # for e in test_list:
    #     LogInfo.begin_track('Trying %s:', e)
    #     for select_str in ('DISTINCT ?p1 ?p2', '?p1 ?p2', 'DISTINCT ?p1 ?p2 ?o2'):
    #         LogInfo.begin_track('[%s]:', select_str)
    #         sparql_lines = [sparql_driver.query_prefix]
    #         sparql_lines.append('SELECT %s WHERE {' % select_str)
    #         sparql_lines.append('fb:%s ?p1 ?o1 .' % e)
    #         sparql_lines.append('?o1 ?p2 ?o2 .')
    #         sparql_lines.append('}')
    #         sparql_str = ' '.join(sparql_lines)
    #         LogInfo.logs(sparql_str)
    #         query_ret = sparql_driver.perform_query(sparql_str)
    #         LogInfo.end_track('Lines = %d', len(query_ret))
    #     LogInfo.end_track()

    query = """
    PREFIX fb: <http://rdf.freebase.com/ns/> 
    SELECT DISTINCT ?var1 ?var2 ?v2_name WHERE {
      fb:m.0824r fb:location.location.contains ?var1 . 
      ?var1 fb:location.location.contains ?var2 . 
      ?var2 fb:type.object.type fb:location.place_with_neighborhoods . 
      ?var1 fb:location.location.area ?area .
      OPTIONAL {?var2 fb:type.object.name ?v2_name} .
    } ORDER BY DESC(?area) LIMIT 1
    """
    LogInfo.logs(query)
    query_ret = sparql_driver.perform_query(query)
    LogInfo.end_track('Lines = %d', len(query_ret))
    LogInfo.logs(query_ret)

if __name__ == '__main__':
    main()
