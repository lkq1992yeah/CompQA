import codecs
from kangqi.util.LogUtil import LogInfo


# each SMART instance maps to a EL entry
class SMART(object):

    def __init__(self, line_u, offset):
        spt = line_u.strip().split('\t')
        q_id = spt[0]
        self.q_idx = int(q_id[q_id.find('-')+1:]) + offset
        self.surface_form = spt[1]
        self.st_pos = int(spt[2])
        self.length = int(spt[3])
        self.mid = mid_transform(spt[4])
        self.e_name = spt[5]
        self.score = float(spt[6])


q_links_dict = {}


def mid_transform(mid):
    # /people/person/place_of_birth --> people.person.place_of_birth
    if not mid.startswith('/'):
        return mid
    return mid[1:].replace('/', '.')


# Load S-MART entity linking result for WebQuestions
def load_webq_linking_data(el_dir='/home/data/Webquestions/S-MART'):
    if len(q_links_dict) != 0:
        return q_links_dict
    for Tvt, offset in (('train', 0), ('test', 3778)):
        fp = '%s/webquestions.examples.%s.e2e.top10.filter.tsv' % (el_dir, Tvt)
        with codecs.open(fp, 'r', 'utf-8') as br:
            for line_u in br.readlines():
                smart_item = SMART(line_u=line_u, offset=offset)
                q_idx = smart_item.q_idx
                if q_idx not in q_links_dict:
                    q_links_dict[q_idx] = []
                q_links_dict[q_idx].append(smart_item)
    LogInfo.logs('%d S-MART results loaded for %d questions.',
                 sum([len(x) for x in q_links_dict.values()]), len(q_links_dict))
    return q_links_dict


def load_compq_linking_data(el_dir='/home/data/ComplexQuestions/S-MART'):
    if len(q_links_dict) != 0:
        return q_links_dict
    for Tvt, offset in (('train', 0), ('test', 1300)):
        fp = '%s/compQ.%s.s-mart.tsv' % (el_dir, Tvt)
        with codecs.open(fp, 'r', 'utf-8') as br:
            for line_u in br.readlines():
                smart_item = SMART(line_u=line_u, offset=offset)
                q_idx = smart_item.q_idx
                if q_idx not in q_links_dict:
                    q_links_dict[q_idx] = []
                q_links_dict[q_idx].append(smart_item)
    LogInfo.logs('%d S-MART results loaded for %d questions.',
                 sum([len(x) for x in q_links_dict.values()]), len(q_links_dict))
    return q_links_dict


if __name__ == '__main__':
    load_webq_linking_data()
