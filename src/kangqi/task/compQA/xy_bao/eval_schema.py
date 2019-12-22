import os
from sparql_backend import backend
from query_graph import *
from eval import *
import json


def parse_golden(s):
    result = []
    start = s.find('(description ')
    while start > 0:
        end = s.find(')', start + 1)
        dscpt = s[start + 13 : end]
        if dscpt[0] == '\"':
            dscpt = dscpt[1:-1]
        result.append(dscpt)
        start = s.find('(description ', end + 1)
    return result

def eval_schema_block(block):
    # load gold answer
    golden = []
    with open('/home/xianyang/Webquestions/Json/webquestions.examples.train.json', 'r') as f:
        webq = json.load(f)
        for q in webq:
            golden.append(parse_golden(q['targetValue']))

    dir = 'schema/train/%s/' % (block)
    files = os.listdir(dir)
    for f in files:
        if f.endswith('_2') or f + '_2' in files:
            continue
        eval_schema_file(dir + f, dir + f + '_2', golden[int(f)])

def eval_schema_file(fin, fout, gold):
    sparql = backend.SPARQLHTTPBackend('202.120.38.146', '8688', '/sparql')
    # print(gold)
    with open(fin, 'r') as f:
        lines = map(lambda x: x.strip(), f.readlines())
        sklts = []
        i = 0
        while i < len(lines):
            if lines[i] == 'sklt:':
                sklt_start = i + 1
                j = i + 1
                while lines[j] != '>>>>>>':
                    j += 1
                sklt_end = j + 1
                schemas = []
                j = sklt_end
                schema_start = 0
                while j < len(lines) and lines[j] != 'sklt:':
                    if lines[j] == '<<<<<<':
                        schema_start = j
                    elif lines[j] == '>>>>>>':
                        schemas.append((schema_start, j+1))
                    j += 1
                sklts.append(((sklt_start, sklt_end), schemas))
                i = j - 1
            i += 1
                
    all_sklts = []

    for sklt in sklts:
        sklt_text = lines[sklt[0][0]+1 : sklt[0][1]-1]   
        sklt_graph = QueryGraph.init_from_text(sklt_text)
        sklt_sparql = sklt_graph.to_sparql_name()
        # print sklt_sparql
        sklt_query_result = sum(sparql.query(sklt_sparql), [])
        # print(sklt_query_result)
        (P, R, F1) = computeF1(gold, sklt_query_result)
        schemas_list = []
        for schema in sklt[1]:
            schema_text = lines[schema[0]+1 : schema[1]-1]
            repeat = False
            for fs in schemas_list:
                if set(fs['schema'].split('\t')) == set(schema_text):
                    repeat = True
                    break
            if repeat:
                continue
            if F1 > 0:
                schema_graph = QueryGraph.init_from_text(schema_text)
                schema_sparql = schema_graph.to_sparql_name()
                schema_query_result = sum(sparql.query(schema_sparql), [])
                # print(schema_query_result)
                (sP, sR, sF1) = computeF1(gold, schema_query_result)
                schemas_list.append({'schema': '\t'.join(schema_text), 'P': sP, 'R': sR, 'F1': sF1})
            else:
                schemas_list.append({'schema': '\t'.join(schema_text)})
        sklt_dict = {'basicgraph': '\t'.join(sklt_text), 'P': P, 'R': R, 'F1': F1, 'schemas': schemas_list}
        all_sklts.append(sklt_dict)

    with open(fout, 'w') as f:
        json.dump(all_sklts, f, sort_keys=True, indent=4, separators=(',', ': '))


if __name__ == '__main__':
    # eval_schema_block('0-99')
    blocks = os.listdir('schema/train/')
    from multiprocessing.dummy import Pool

    pool = Pool(8)
    pool.map(eval_schema_block, blocks)
