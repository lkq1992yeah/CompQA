"""
evaluation utils
"""

import numpy as np
from sklearn.metrics import f1_score
from xusheng.util.log_util import LogInfo


def eval_acc_pn(score_list, PN):
    case_num = int(len(score_list) / PN)
    # LogInfo.begin_track("Case num: %d", case_num)
    if case_num == 0:
        # LogInfo.end_track("[Error!]")
        return 0.0, 0, 0
    correct = 0
    for i in range(case_num):
        failed = False
        pos_score = score_list[i*PN]
        scores = str(pos_score)
        for j in range(i*PN+1, (i+1)*PN):
            neg_score = score_list[j]
            scores += " " + str(neg_score)
            if pos_score <= neg_score:
                failed = True
                break
        # LogInfo.logs("Case %d: %s", i+1, scores)
        if not failed:
            correct += 1
    accuracy = float(correct / case_num)
    # LogInfo.end_track("Accuracy: %.4f(%d/%d)", accuracy, correct, case_num)
    return accuracy, correct, case_num


def eval_classify_f1(raw_score, y_true, average='macro'):
    y_pred = np.argmax(raw_score, axis=1)
    LogInfo.logs(y_pred)
    f1 = f1_score(y_true=y_true, y_pred=y_pred, average=average)
    return f1


def eval_seq_crf(y_pred_, y_true_, method='precision'):
    """
    Evaluation for sequence labeling, without "Outside"
    under specific conditions (3-class)
    :param y_pred_: [B, T, ]
    :param y_true_: [B, T, ]
    :param method: precision/ recall
    :return: f1 score
    """
    # LogInfo.logs("y_pred: %s", '\n'.join([str(x) for x in y_pred_]))
    # LogInfo.logs("y_true: %s", '\n'.join([str(x) for x in y_true_]))

    tag_dict = ['O', 'PL_B', 'PL_I', 'PK_B', 'PK_I', 'PV_B', 'PV_I']
    LogInfo.begin_track("Eval seq %s...", method)
    if method == 'precision':
        y_pred = np.array(y_pred_)
        y_true = np.array(y_true_)
    elif method == 'recall':
        y_pred = np.array(y_true_)
        y_true = np.array(y_pred_)

    correct = {'PL': 0, 'PK': 0, 'PV': 0}
    act_cnt = {'PL': 0, 'PK': 0, 'PV': 0}
    for line_pred, line_true in zip(y_pred, y_true):
        i = 0
        cnt = len(line_pred)
        while i < cnt:
            tag_num = line_pred[i]
            tag = tag_dict[tag_num]
            if tag == 'O':
                i += 1
                continue
            else:
                kind = tag[:2]
                sign = tag[3]
            if sign == 'B':
                j = i + 1
                while j < cnt:
                    next_tag = tag_dict[line_pred[j]]
                    if next_tag[:2] == kind and next_tag[3] == 'I':
                        j += 1
                    else:
                        break
            else:
                i += 1
                continue

            act_cnt[kind] += 1

            act_label = ' '.join([str(x) for x in line_true[i:j]])
            proposed_label = ' '.join([str(x) for x in line_pred[i:j]])
            if act_label == proposed_label and (j == cnt or line_true[j] != line_true[i]+1):
                correct[kind] += 1
            i = j

    ret = dict()
    keys = act_cnt.keys()
    correct_total = 0
    cnt_total = 0
    for key in keys:
        if act_cnt[key] == 0:
            ret[key] = 0.0
        else:
            ret[key] = correct[key] * 1.0 / act_cnt[key]
        LogInfo.logs("%s : %.4f(%d/%d)", key, ret[key], correct[key], act_cnt[key])
        correct_total += correct[key]
        cnt_total += act_cnt[key]
        if cnt_total == 0:
            overall = 0.0
        else:
            overall = correct_total * 1.0 / cnt_total
    LogInfo.logs("Over-all %s: %.4f(%d/%d)", method, overall,
                 correct_total, cnt_total)
    LogInfo.end_track()
    return overall

def eval_seq_crf_with_o(y_pred_, y_true_, method='precision'):
    """
    Evaluation for sequence labeling, including "Outside"
    under specific conditions (3-class)
    :param y_pred_: [B, T, ]
    :param y_true_: [B, T, ]
    :param method: precision/ recall
    :return: f1 score
    """
    # LogInfo.logs("y_pred: %s", '\n'.join([str(x) for x in y_pred_]))
    # LogInfo.logs("y_true: %s", '\n'.join([str(x) for x in y_true_]))

    tag_dict = ['O', 'PL_B', 'PL_I', 'PK_B', 'PK_I', 'PV_B', 'PV_I']
    LogInfo.begin_track("Eval seq %s...", method)
    if method == 'precision':
        y_pred = np.array(y_pred_)
        y_true = np.array(y_true_)
    elif method == 'recall':
        y_pred = np.array(y_true_)
        y_true = np.array(y_pred_)

    correct = {'O': 0, 'PL': 0, 'PK': 0, 'PV': 0}
    act_cnt = {'O': 0, 'PL': 0, 'PK': 0, 'PV': 0}
    for line_pred, line_true in zip(y_pred, y_true):
        i = 0
        cnt = len(line_pred)
        while i < cnt:
            tag_num = line_pred[i]
            tag = tag_dict[tag_num]
            if tag == 'O':
                # regard "Outside" as fourth tag
                act_cnt['O'] += 1
                if line_true[i] == line_pred[i]:
                    correct['O'] += 1
                i += 1
                continue
            else:
                kind = tag[:2]
                sign = tag[3]
            if sign == 'B':
                j = i + 1
                while j < cnt:
                    next_tag = tag_dict[line_pred[j]]
                    if next_tag[:2] == kind and next_tag[3] == 'I':
                        j += 1
                    else:
                        break
            else:
                i += 1
                continue

            act_cnt[kind] += 1

            act_label = ' '.join([str(x) for x in line_true[i:j]])
            proposed_label = ' '.join([str(x) for x in line_pred[i:j]])
            if act_label == proposed_label and (j == cnt or line_true[j] != line_true[i]+1):
                correct[kind] += 1
            i = j

    ret = dict()
    keys = act_cnt.keys()
    correct_total = 0
    cnt_total = 0
    for key in keys:
        if act_cnt[key] == 0:
            ret[key] = 0.0
        else:
            ret[key] = correct[key] * 1.0 / act_cnt[key]
        LogInfo.logs("%s : %.4f(%d/%d)", key, ret[key], correct[key], act_cnt[key])
        correct_total += correct[key]
        cnt_total += act_cnt[key]
        if cnt_total == 0:
            overall = 0.0
        else:
            overall = correct_total * 1.0 / cnt_total
    LogInfo.logs("Over-all %s: %.4f(%d/%d)", method, overall,
                 correct_total, cnt_total)
    LogInfo.end_track()
    return overall

def eval_seq_crf_with_o_atis(y_pred_, y_true_, method='precision'):
    """
    Evaluation for ATIS dataset, including "Outside"
    under specific conditions (3-class)
    :param y_pred_: [B, T, ]
    :param y_true_: [B, T, ]
    :param method: precision/ recall
    :return: f1 score
    """
    # LogInfo.logs("y_pred: %s", '\n'.join([str(x) for x in y_pred_]))
    # LogInfo.logs("y_true: %s", '\n'.join([str(x) for x in y_true_]))

    tag_dict = ['O', 'B-day_number', 'B-stoploc.state_code', 'B-toloc.state_code', 'B-time_relative', 'B-fromloc.state_code', 'B-stoploc.airport_code', 'B-airline_code', 'B-connect', 'B-depart_time.period_mod', 'B-flight', 'B-arrive_time.period_mod', 'B-booking_class', 'B-month_name', 'B-return_date.day_name', 'B-depart_date.month_name', 'B-arrive_date.today_relative', 'B-return_time.period_of_day', 'B-aircraft_code', 'B-arrive_date.date_relative', 'B-state_code', 'B-days_code', 'B-airport_code', 'B-period_of_day', 'B-arrive_date.day_name', 'B-flight_days', 'B-return_time.period_mod', 'B-fromloc.airport_code', 'B-arrive_date.month_name', 'B-mod', 'B-stoploc.airport_name', 'B-compartment', 'B-toloc.airport_code', 'B-depart_date.date_relative', 'B-day_name', 'B-or', 'B-depart_date.year', 'B-depart_date.day_name', 'B-toloc.country_name', 'B-return_date.month_name', 'B-meal',
                'B-stoploc.city_name', 'I-stoploc.city_name', 'B-round_trip', 'I-round_trip', 'B-state_name', 'I-state_name', 'B-fromloc.city_name', 'I-fromloc.city_name', 'B-airline_name', 'I-airline_name', 'B-flight_stop', 'I-flight_stop', 'B-fromloc.airport_name', 'I-fromloc.airport_name', 'B-arrive_time.start_time', 'I-arrive_time.start_time', 'B-cost_relative', 'I-cost_relative', 'B-city_name', 'I-city_name', 'B-arrive_time.end_time', 'I-arrive_time.end_time', 'B-meal_code', 'I-meal_code', 'B-depart_date.day_number', 'I-depart_date.day_number', 'B-meal_description', 'I-meal_description', 'B-arrive_time.time', 'I-arrive_time.time', 'B-depart_date.today_relative', 'I-depart_date.today_relative', 'B-fare_amount', 'I-fare_amount', 'B-airport_name', 'I-airport_name', 'B-flight_time', 'I-flight_time', 'B-flight_number', 'I-flight_number', 'B-toloc.airport_name', 'I-toloc.airport_name', 'B-flight_mod', 'I-flight_mod', 'B-depart_time.time_relative', 'I-depart_time.time_relative', 'B-return_date.date_relative', 'I-return_date.date_relative', 'B-economy', 'I-economy', 'B-class_type', 'I-class_type', 'B-toloc.state_name', 'I-toloc.state_name', 'B-arrive_time.period_of_day', 'I-arrive_time.period_of_day', 'B-toloc.city_name', 'I-toloc.city_name', 'B-depart_time.start_time', 'I-depart_time.start_time', 'B-return_date.day_number', 'I-return_date.day_number', 'B-today_relative', 'I-today_relative', 'B-depart_time.end_time', 'I-depart_time.end_time', 'B-fromloc.state_name', 'I-fromloc.state_name', 'B-depart_time.time', 'I-depart_time.time', 'B-return_date.today_relative', 'I-return_date.today_relative', 'B-fare_basis_code', 'I-fare_basis_code', 'B-arrive_date.day_number', 'I-arrive_date.day_number', 'B-restriction_code', 'I-restriction_code', 'B-transport_type', 'I-transport_type', 'B-time', 'I-time', 'B-arrive_time.time_relative', 'I-arrive_time.time_relative', 'B-depart_time.period_of_day', 'I-depart_time.period_of_day']

    LogInfo.begin_track("Eval seq %s on %d tags...", method, len(tag_dict))
    if method == 'precision':
        y_pred = np.array(y_pred_)
        y_true = np.array(y_true_)
    elif method == 'recall':
        y_pred = np.array(y_true_)
        y_true = np.array(y_pred_)

    names = set()
    for tag in tag_dict:
        if tag == 'O':
            names.add('O')
        else:
            names.add(tag[2:])
    LogInfo.logs("%d different terms", len(names))
    correct = dict()
    act_cnt = dict()
    for name in names:
        correct[name] = 0
        act_cnt[name] = 0

    for line_pred, line_true in zip(y_pred, y_true):
        i = 0
        cnt = len(line_pred)
        while i < cnt:
            tag_num = line_pred[i]
            tag = tag_dict[tag_num]
            if tag_num <= 40:
                # tags with "B" without "I", including "O"
                if tag_num == 0:
                    kind = 'O'
                else:
                    kind = tag[2:]
                act_cnt[kind] += 1
                if line_true[i] == line_pred[i]:
                    correct[kind] += 1
                i += 1
                continue
            else:
                kind = tag[2:]
                sign = tag[0]
            if sign == 'B':
                j = i + 1
                while j < cnt:
                    next_tag = tag_dict[line_pred[j]]
                    if next_tag[2:] == kind and next_tag[0] == 'I':
                        j += 1
                    else:
                        break
            else:
                i += 1
                continue

            act_cnt[kind] += 1

            act_label = ' '.join([str(x) for x in line_true[i:j]])
            proposed_label = ' '.join([str(x) for x in line_pred[i:j]])
            if act_label == proposed_label and (j == cnt or line_true[j] != line_true[i]+1):
                correct[kind] += 1
            i = j

    ret = dict()
    keys = act_cnt.keys()
    correct_total = 0
    cnt_total = 0
    for key in keys:
        if act_cnt[key] == 0:
            ret[key] = 0.0
        else:
            ret[key] = correct[key] * 1.0 / act_cnt[key]
            LogInfo.logs("%s : %.4f(%d/%d)", key, ret[key], correct[key], act_cnt[key])
        correct_total += correct[key]
        cnt_total += act_cnt[key]
        if cnt_total == 0:
            overall = 0.0
        else:
            overall = correct_total * 1.0 / cnt_total
    LogInfo.logs("Over-all %s: %.4f(%d/%d)", method, overall,
                 correct_total, cnt_total)
    LogInfo.end_track()
    return overall


def eval_seq_softmax(raw_score, y_true, method='precision'):
    """
    Evaluation for sequence labeling
    under specific conditions (3-class)
    :param raw_score: [B, T, class_dim]
    :param y_true: [B, T, ]
    :param method: precision/ recall
    :return: f1 score
    """
    tag_dict = ['O', 'PL_B', 'PL_I', 'PK_B', 'PK_I', 'PV_B', 'PV_I']
    LogInfo.begin_track("Eval seq %s...", method)
    if method == 'precision':
        y_pred = np.argmax(raw_score, axis=1).reshape((-1))
        y_true = np.array(y_true).reshape((-1))
    elif method == 'recall':
        y_pred = np.array(y_true).reshape((-1))
        y_true = np.argmax(raw_score, axis=1).reshape((-1))

    # LogInfo.logs("y_pred: [%s]", ' '.join([str(x) for x in y_pred]))
    # LogInfo.logs("y_true: [%s]", ' '.join([str(x) for x in y_true]))
    # LogInfo.logs("y_pred: [%s]", y_pred)
    # LogInfo.logs("y_true: [%s]", y_true)

    cnt = len(y_pred)
    i = 0
    correct = {'PL': 0, 'PK': 0, 'PV': 0}
    act_cnt = {'PL': 0, 'PK': 0, 'PV': 0}
    while i < cnt:
        tag_num = y_pred[i]
        tag = tag_dict[tag_num]
        if tag == 'O':
            i += 1
            continue
        else:
            kind = tag[:2]
            sign = tag[3]
        if sign == 'B':
            j = i + 1
            while j < cnt:
                next_tag = tag_dict[y_pred[j]]
                if next_tag[:2] == kind and next_tag[3] == 'I':
                    j += 1
                else:
                    break
        else:
            i += 1
            continue

        act_cnt[kind] += 1

        act_label = ' '.join([str(x) for x in y_true[i:j]])
        proposed_label = ' '.join([str(x) for x in y_pred[i:j]])
        if act_label == proposed_label and (j == cnt or y_true[j] != y_true[i]+1):
            correct[kind] += 1
        i = j

    ret = dict()
    keys = act_cnt.keys()
    correct_total = 0
    cnt_total = 0
    for key in keys:
        if act_cnt[key] == 0:
            ret[key] = 0.0
        else:
            ret[key] = correct[key] * 1.0 / act_cnt[key]
        LogInfo.logs("%s : %.4f(%d/%d)", key, ret[key], correct[key], act_cnt[key])
        correct_total += correct[key]
        cnt_total += act_cnt[key]
        if cnt_total == 0:
            overall = 0.0
        else:
            overall = correct_total * 1.0 / cnt_total
    LogInfo.logs("Over-all %s: %.4f(%d/%d)", method, overall,
                 correct_total, cnt_total)
    LogInfo.end_track()
    return overall


def eval_link_f1(raw_score, label, PN):

    return 0

if __name__ == '__main__':
    y_ = [[1, 0, 0, 0, 0, 0, 0],
          [0, 1, 0, 0, 0, 0, 0],
          [0, 0, 1, 0, 0, 0, 0],
          [0, 0, 1, 3, 0, 0, 0],
          [1, 0, 0, 0, 0, 0, 0],
          [1, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 1, 0, 0, 0],
          [1, 0, 0, 0, 0, 2, 0],
          [1, 0, 0, 0, 0, 0, 2],
          [1, 0, 0, 0, 0, 0, 0],
          [1, 0, 0, 0, 0, 0, 2],
          [1, 0, 0, 2, 0, 0, 0],
          [1, 0, 0, 2, 4, 0, 0],
          [1, 2, 0, 0, 0, 0, 0],
          [1, 0, 2, 0, 0, 0, 0],
          ]
    y_ = [[0, 1, 2, 0, 0, 0, 41, 42, 42, 5, 6, 3, 4, 1, 0]]
    y  = [[0, 1, 2, 2, 0, 0, 41, 42, 42, 0, 6, 3, 3, 1, 1]]
    LogInfo.logs(eval_seq_crf_with_o_atis(y_, y, method='precision'))
    LogInfo.logs(eval_seq_crf_with_o_atis(y_, y, method='recall'))

    with open("./tags", 'r') as fin:
        tags = dict()
        for line in fin:
            line = line.strip()
            if line == 'O':
                continue
            name = line[2:]
            if name in tags:
                tags[name] += 1
            else:
                tags[name] = 1
    LogInfo.logs("%d tags", len(tags))
    BI = list()
    B = list()
    for key, val in tags.items():
        if val == 2:
            BI.append(key)
        else:
            B.append(key)
    LogInfo.logs("%d BI tags", len(BI))
    LogInfo.logs("%d B tags", len(B))
    fout = open("label_idx.txt", 'w')
    ret = "\'O\', "
    for name in B:
        ret += "\'B-" + name + "\', "
        fout.write("B-" + name + "\n")
    LogInfo.logs(ret)
    ret = ""
    for name in BI:
        ret += "\'B-" + name + "\', "
        fout.write("B-" + name + "\n")
        ret += "\'I-" + name + "\', "
        fout.write("I-" + name + "\n")
    LogInfo.logs(ret)
    fout.close()

