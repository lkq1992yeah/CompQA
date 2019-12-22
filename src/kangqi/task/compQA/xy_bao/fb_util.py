
def normalize_entity(e):
    if e.startswith('/'):
        e = e[1:].replace('/', '.')
    if e[0] == 'm':
        return 'fb:' + e
    else:
        return '\"%s\"' % e

def is_mid(s):
    return s.startswith('m.')

def is_numerical(s):
    pass

def is_time(s):
    pass

def link_to_num_or_time(p, backend):
    query = '''
    PREFIX fb: <http://rdf.freebase.com/ns/>
    SELECT DISTINCT ?type
    WHERE {
        fb:%s fb:type.property.expected_type ?type .
    }
    ''' % (p)
    query_result = backend.query(query)
    if 'type.datetime' in sum(query_result, []):
        return True
    return Falses

def get_relation_of_entity_pair(e1, e2, backend):
    if not is_mid(e1) or not is_mid(e2):
        return []
    query = Template('''
    PREFIX fb: <http://rdf.freebase.com/ns/>
    SELECT DISTINCT ?rel 
    WHERE {
        ${e1} ?rel ${e2} .
    } LIMIT 100
    ''').substitute(e1=normalize_entity(e1), e2=normalize_entity(e2))
    # print query
    result = backend.query(query)
    return sum(result, [])

def get_relation_of_entity(e, backend):
    if not is_mid(e):
        return []
    query = Template('''
    PREFIX fb: <http://rdf.freebase.com/ns/>
    SELECT DISTINCT ?rel 
    WHERE {
        ${e} ?rel ?obj .
    } LIMIT 100
    ''').substitute(e=normalize_entity(e))
    # print query
    result = backend.query(query)
    return sum(result, [])

def detect_time(question):
    def is_num(s):
        return reduce(lambda x, y: x and y, [c in '0123456789' for c in s]) 
    # only year right now
    tokens = question[:-1].split()
    for t in tokens:
        if len(t) == 4 and is_num(t):
            return t
        if len(t) == 3 and t[2] == 's' and is_num(t[:-1]):
            return '19' + t[:-1]
        if len(t) == 4 and t[2:] == '\'s' and is_num(t[:-2]):
            return '19' + t[:-2]
    return None

def extract_types(question):
    return ['UNKNOWN_TYPE']

ordinal_list = ['first', 'second', 'third']
def is_ordinal_number(s):
    if s in ordinal_list:
        return True, ordinal_list.index(s), '+'
    if s == 'last':
        return True, 1, '-'
