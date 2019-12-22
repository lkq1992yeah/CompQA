# -*- coding:utf-8 -*-

import re
import codecs

from kangqi.util.LogUtil import LogInfo


data_dir = 'data/fb_metadata'

type_pred_dict = {}     # <type, set([pred])>
type_name_dict = {}
sup_type_dict = {}
sub_type_dict = {}
med_type_set = set([])
ordinal_type_set = {'type.int', 'type.float', 'type.datetime'}
ignore_type_domain_set = {'base', 'common', 'freebase', 'm', 'type', 'user'}

pred_domain_dict = {}
pred_range_dict = {}
pred_name_dict = {}
pred_inverse_dict = {}
time_pred_dict = {}     # <pred, (target_direction, target_pred)>
ignore_pred_domain_set = {'common', 'freebase', 'm', 'type'}

entity_name_dict = {}


def remove_parenthesis(name):
    while True:
        lf_pos = name.find('(')
        if lf_pos == -1:
            break
        rt_pos = name.find(')', lf_pos+1)
        if rt_pos == -1:
            rt_pos = len(name) - 1
        name = name[:lf_pos] + name[rt_pos+1:]
    return name


def adjust_name(name):
    name = name.lower()
    # name = name.replace('(s)', '')  # remove such thing in type or predicate names
    name = remove_parenthesis(name)
    name = re.sub(r'[/|\\,.?!@#$%^&*():;"]', '', name)          # remove puncs
    name = re.sub(' +', ' ', name).strip()                      # remove extra blanks
    return name


""" ============ Type related function ============== """


def load_mediator():
    if len(med_type_set) == 0:
        med_fp = data_dir + '/mediator/mediators.tsv'
        with codecs.open(med_fp, 'r', 'utf-8') as br:
            for line in br.readlines():
                med_type_set.add(line.strip())
        LogInfo.logs('FBHelper: %d mediator types loaded.', len(med_type_set))
    return med_type_set


def is_mediator_type(t):
    load_mediator()
    return t in med_type_set


def load_sup_sub_types():
    if len(sup_type_dict) == 0 or len(sub_type_dict) == 0:
        type_uhash = {}
        with codecs.open(data_dir + '/superType/type_dict.tsv', 'r', 'utf-8') as br:
            for line in br.readlines():
                spt = line.strip().split('\t')
                idx = int(spt[0])
                mid = spt[1]
                type_uhash[idx] = mid
        with codecs.open(data_dir + '/superType/version_0.9.txt', 'r', 'utf-8') as br:
            pairs = 0
            for line in br.readlines():
                idx_list = map(lambda x: int(x), line.strip().split('\t'))
                ch_idx = idx_list[0]
                ch_type = type_uhash[ch_idx]
                for fa_idx in idx_list[1:]:         # ignore child type itself
                    pairs += 1
                    fa_type = type_uhash[fa_idx]
                    sup_type_dict.setdefault(ch_type, set([])).add(fa_type)
                    sub_type_dict.setdefault(fa_type, set([])).add(ch_type)
        LogInfo.logs('FBHelper: %d sub/super type pairs loaded.', pairs)
    return sup_type_dict, sub_type_dict


def is_type_contained_by(ta, tb):       # whether tb is a super type of ta
    load_sup_sub_types()
    return tb in sup_type_dict.get(ta, set([]))


def is_type_contains(ta, tb):           # whether tb is a sub type of tb
    load_sup_sub_types()
    return tb in sub_type_dict.get(ta, set([]))


def load_type_name():
    if len(type_name_dict) == 0:
        with codecs.open(data_dir + '/TS-name.txt', 'r', 'utf-8') as br:
            for line in br.readlines():
                spt = line.strip().split('\t')
                if len(spt) < 2:
                    continue
                tp, raw_name = spt
                tp_name = adjust_name(raw_name)     # lowercased, remove punc, remove (s)
                type_name_dict[tp] = tp_name
        LogInfo.logs('FBHelper: %d type names loaded.', len(type_name_dict))
    return type_name_dict


def get_type_name(tp):
    load_type_name()
    return type_name_dict.get(tp, '')


def is_type_ignored(tp):
    pref = tp[:tp.find('.')]
    return pref in ignore_type_domain_set


""" ================== Predicate related functions =================== """


def load_domain_range():
    if len(pred_range_dict) == 0 or len(pred_domain_dict) == 0:
        meta_fp = data_dir + '/PS-TP-triple.txt'
        with codecs.open(meta_fp, 'r', 'utf-8') as br:
            for line in br.readlines():
                s, p, o = line.strip().split('\t')
                if p == 'type.property.schema':
                    pred_domain_dict[s] = o
                else:
                    pred_range_dict[s] = o
        LogInfo.logs('FBHelper: %d domain + %d range info loaded.', len(pred_domain_dict), len(pred_range_dict))
    return pred_domain_dict, pred_range_dict


def get_domain(pred):
    load_domain_range()
    if pred[0] == '!':      # inverse predicate
        return pred_range_dict.get(pred[1:], '')
    else:
        return pred_domain_dict.get(pred, '')


def get_range(pred):
    load_domain_range()
    if pred[0] == '!':      # inverse predicate
        return pred_domain_dict.get(pred[1:], '')
    else:
        return pred_range_dict.get(pred, '')


def load_inverse():
    if len(pred_inverse_dict) == 0:
        with codecs.open(data_dir + '/inverses.tsv', 'r', 'utf-8') as br:
            for line in br.readlines():
                spt = line.strip().split('\t')
                pred_inverse_dict[spt[0]] = spt[1]
                pred_inverse_dict[spt[1]] = spt[0]
        LogInfo.logs('FBHelper: %d inverse predicate info loaded.', len(pred_inverse_dict))
    return pred_inverse_dict


def inverse_predicate(pred):
    load_inverse()
    if pred in pred_inverse_dict:
        return pred_inverse_dict[pred]
    elif pred.startswith('!'):
        return pred[1:]
    return '!' + pred


def load_pred_name():
    if len(pred_name_dict) == 0 or len(type_pred_dict) == 0:
        with codecs.open(data_dir + '/PS-name.txt', 'r', 'utf-8') as br:
            for line in br.readlines():
                spt = line.strip().split('\t')
                if len(spt) < 2:
                    continue
                pred, raw_name = spt
                pred_name = adjust_name(raw_name)
                tp = pred[:pred.rfind('.')]
                type_pred_dict.setdefault(tp, set([])).add(pred)
                pred_name_dict[pred] = pred_name
        LogInfo.logs('FBHelper: %d predicate names loaded.', len(pred_name_dict))
    return pred_name_dict


def get_pred_name(pred):
    load_pred_name()
    use_pred = pred[1:] if pred[0] == '!' else pred     # consider inverse predicates
    if use_pred.startswith('m.__') and use_pred.endswith('__'):     # virtual predicate
        return use_pred[4: -2]
    else:       # normal predicates
        return pred_name_dict.get(use_pred, '')


def is_pred_ignored(pred):
    use_pred = pred[1:] if pred[0] == '!' else pred
    pref = use_pred[:use_pred.find('.')]
    return pref in ignore_pred_domain_set


def is_mediator_as_expect(pred):
    t = get_range(pred)
    return is_mediator_type(t)


def get_preds_given_type(tp):     # return all predicates under this type
    load_pred_name()
    return type_pred_dict.get(tp, set([]))


def get_dt_preds_given_type(tp):
    load_domain_range()
    load_pred_name()
    ret_set = set([])
    for pred in get_preds_given_type(tp):
        if pred_range_dict.get(pred, '') == 'type.datetime':
            ret_set.add(pred)
    return ret_set


def get_ord_preds_given_type(tp):
    load_domain_range()
    load_pred_name()
    ret_set = set([])
    for pred in get_preds_given_type(tp):
        if pred_range_dict.get(pred, '') in ordinal_type_set:
            ret_set.add(pred)
    return ret_set


def load_time_pair_preds():
    if len(time_pred_dict) == 0:
        with codecs.open(data_dir + '/time_pairs.txt', 'r', 'utf-8') as br:
            for line in br.readlines():
                spt = line.strip().split('\t')
                if len(spt) < 2:
                    continue
                tm_begin_pred, tm_end_pred = spt
                time_pred_dict[tm_begin_pred] = ('end', tm_end_pred)
                time_pred_dict[tm_end_pred] = ('begin', tm_begin_pred)
        LogInfo.logs('FBHelper: %d time pair predicates loaded.', len(time_pred_dict))
    return time_pred_dict


def get_end_dt_pred(pred):
    # Example: sports.pro_athlete.career_start --> career_end
    # always from start side to end side
    load_time_pair_preds()
    target_tup = time_pred_dict.get(pred, ('', ''))
    if target_tup[0] == 'end':
        return target_tup[1]
    return ''


def get_begin_dt_pred(pred):
    # always from end side to begin side
    load_time_pair_preds()
    target_tup = time_pred_dict.get(pred, ('', ''))
    if target_tup[0] == 'begin':
        return target_tup[1]
    return ''


""" ================ Entity related functions =================== """


def load_entity_name(concern_e_set):
    if len(entity_name_dict) == 0:
        with codecs.open(data_dir + '/S-NAP-ENO-triple.txt', 'r', 'utf-8') as br:
            for line in br.readlines():
                spt = line.strip().split('\t')
                if len(spt) < 3:
                    continue
                if not spt[0].startswith('m.'):
                    continue
                if spt[1] != 'type.object.name':
                    continue
                if concern_e_set is not None and spt[0] not in concern_e_set:
                    continue
                entity_name_dict[spt[0]] = adjust_name(spt[2])
    LogInfo.logs('FBHelper: %d entity names collected from %d concern mids.',
                 len(entity_name_dict), len(concern_e_set))


def get_entity_name(mid):
    # load_entity_name()      # we don't know which entities are concerned
    return entity_name_dict.get(mid, '')


def get_item_name(mid):     # used when it's not clear what kind of object the mid is
    p_name = get_pred_name(mid)
    if p_name != '':
        return p_name
    t_name = get_type_name(mid)
    if t_name != '':
        return t_name
    if mid.startswith('m.') and not mid.startswith('m.0'):      # it's a type/predicate, but not an entity
        last_dot = mid.rfind('.')
        last_part = mid[last_dot+1:]
        return last_part.split('_')     # simply pick the name from id
    return get_entity_name(mid)


#
# class FreebaseHelper(object):
#
#     def __init__(self, data_dir):
#         self.data_dir = data_dir
#         self.domain_dict = {}
#         self.range_dict = {}
#         self.inverse_dict = {}
#         self.sup_type_dict = {}         # type --> super types
#         self.sub_type_dict = {}         # type --> sub types
#
#         self.name_dict = None
#
#
#     def load_names(self, e_set, t_set, p_set):
#         LogInfo.begin_track('Loading names ... ')
#         self.name_dict = {}
#
#         # deal with virtual mids (only available in predicates)
#         # m.__before__, m.__after__, m.__max__, m.__min__, m.__1__, ......
#         for pred in p_set:
#             if pred.startswith('m.__') and pred.endswith('__'):
#                 virtual_pred_name = pred[4: -2]
#                 self.name_dict[pred] = virtual_pred_name
#
#         with codecs.open(self.data_dir + '/PS-name.txt', 'r', 'utf-8') as br:
#             for line in br.readlines():
#                 spt = line.strip().split('\t')
#                 if len(spt) < 2:
#                     continue
#                 if spt[0] not in p_set and ('!'+spt[0]) not in p_set:
#                     # considering both forward and backward predicates
#                     continue
#                 self.name_dict[spt[0]] = adjust_name(spt[1])
#         LogInfo.logs('Predicate scanned, collected %d names.', len(self.name_dict))
#
#
#
#         LogInfo.end_track()
#
#     def get_item_name(self, mid):
#         if mid.startswith('!'):
#             mid = mid[1:]
#         return self.name_dict.get(mid, '')
