import copy
import random

import numpy as np

from data.test_data import get_test_data
from kangqi.util.LogUtil import LogInfo
from xusheng.util.tf_util import get_variance  # , print_detail_2d
from xusheng.util.tf_util import normalize


# Iterative prediction
# every time  choose one cell to find best candidate bring up the score
# repeat until score converges

# -------------------------------- Testing ------------------------------------- #

def test(model, session, split, verbose, test_data=None,
         norm=0, candidate_num=50):
    rows = 16
    cols = 6
    if verbose:
        LogInfo.begin_track("Loading %s data...", split)
    if test_data is None:
        test_data = get_test_data(split, verbose)
    if verbose:
        LogInfo.logs("Loaded.")
        LogInfo.logs("Cell shape: %s", np.array(test_data['cell']).shape)
        LogInfo.logs("Entity shape: %s", np.array(test_data['entity']).shape)
        LogInfo.logs("Coherence shape: %s", np.array(test_data['coherence']).shape)
        LogInfo.logs("Context shape: %s", np.array(test_data['context']).shape)
        LogInfo.logs("Ground truth shape: %s", np.array(test_data['truth']).shape)
        LogInfo.end_track()

    LogInfo.begin_track("[%s] start evaluating...", split)
    test_size = len(test_data['cell'])
    pre_list = list()
    init_correct_sum = 0
    upper_correct_sum = 0
    best_correct_sum = 0
    cell_num_sum = 0
    initial_pre_list = list()
    upper_pre_list = list()
    better = 0
    equal = 0

    for i in range(test_size):
        if verbose:
            LogInfo.begin_track("Testing #%d/%d table...", i+1, test_size)
        single_table = dict()  # every time input one single table for testing

        single_table['cell'] = np.array(test_data['cell'][i:i+1]).reshape((-1, 100))
        single_table['entity'] = np.array(test_data['entity'][i:i+1]).reshape((-1, 100))
        single_table['coherence'] = np.array(test_data['coherence'][i:i+1]).reshape((-1, 100))
        # coherence normalization
        if norm == 1:
            single_table['coherence'] = normalize(single_table['coherence'])
        elif norm == 2:
            single_table['coherence'] = np.sum(single_table['coherence'], axis=1).reshape((-1, 1))
        single_table['context'] = np.array(test_data['context'][i:i+1]).reshape((-1, 100))
        # shell table, replace entity with truth & upper bound to get a score
        shell_table = dict()
        shell_table['cell'] = single_table['cell']
        shell_table['context'] = single_table['context']
        shell_table['coherence'] = single_table['coherence']

        candi_data = test_data['candidate'][i]
        truth_data = np.array(test_data['truth'][i]).reshape((-1, 100))
        cell_num = len(candi_data)

        # initial precision
        correct = 0
        for idx in range(rows * cols):
            if truth_data[idx][0] != 0 and truth_data[idx][1] != 0 and \
                            np.sum(np.abs(truth_data[idx] - single_table['entity'][idx])) < 1e-6:
                # np.all(truth_data[xi][xj] == single_table['entity'][xi * cols + xj]):
                correct += 1
        init_correct = correct
        init_precision = float(correct) / cell_num
        if verbose:
            LogInfo.logs("Precision for initial table #%d/%d: %.4f(%d/%d)",
                         i + 1, test_size, init_precision, init_correct, cell_num)
        initial_pre_list.append(init_precision)

        # upper bound precision
        upper_bound_table = copy.deepcopy(single_table['entity'])
        correct = 0
        for idx in range(cell_num):
            candidates = candi_data[idx]
            row_pos = candidates['row']
            col_pos = candidates['col']
            vecs = np.array(candidates['vec'])[:candidate_num]
            x = row_pos*cols+col_pos
            if np.sum(np.abs(upper_bound_table[x]-truth_data[x])) < 1e-6:
                correct += 1
            else:
                for vec in vecs:
                    if np.sum(np.abs(vec-truth_data[x])) < 1e-6:
                        correct += 1
                        upper_bound_table[x] = vec
                        break
        upper_correct = correct
        upper_precision = float(correct) / cell_num
        if verbose:
            LogInfo.logs("Precision for upper-bound table #%d/%d: %.4f(%d/%d)",
                         i + 1, test_size, upper_precision, upper_correct, cell_num)
        upper_pre_list.append(upper_precision)

        # initial score
        running_ret = model.eval(session=session, input_data=single_table)
        initial_score = running_ret['eval_score']
        if verbose:
            LogInfo.logs("Eval score for initial table: %.4f.", initial_score)

        # truth score
        shell_table['entity'] = truth_data
        running_ret = model.eval(session=session, input_data=shell_table)
        truth_score = running_ret['eval_score']
        if verbose:
            LogInfo.logs("Eval score for truth table: %.4f.", truth_score)

        # upper bound score
        shell_table['entity'] = upper_bound_table
        running_ret = model.eval(session=session, input_data=shell_table)
        upper_score = running_ret['eval_score']
        if verbose:
            LogInfo.logs("Eval score for upper bound table: %.4f.", upper_score)

        best_correct = 0
        best_precision = 0
        x = range(cell_num)

        # ************************* ITERATIVE PREDICTION START *************************
        rounds = 0
        silence = 0
        round_table = copy.deepcopy(single_table)
        base_score = initial_score
        while silence < 3:
            rounds += 1
            silence += 1
            if verbose:
                LogInfo.begin_track("Round #%d for table #%d/%d...",
                                    rounds, i+1, test_size)

            random.shuffle(x)  # random choose replacement order among cells
            for j in x:
                cell_candidates =candi_data[j]
                row_pos = cell_candidates['row']
                col_pos = cell_candidates['col']
                vecs = cell_candidates['vec'][:candidate_num]
                tmp_table = copy.deepcopy(round_table)  # deep copy

                # ********* NEW: in-time update ********
                tmp_table['cell'] = np.stack([tmp_table['cell']] * candidate_num, axis=0)
                tmp_table['entity'] = np.stack([tmp_table['entity']] * candidate_num, axis=0)
                tmp_table['coherence'] = np.stack([tmp_table['coherence']] * candidate_num, axis=0)
                tmp_table['context'] = np.stack([tmp_table['context']] * candidate_num, axis=0)
                for idx, vec in enumerate(vecs):
                    tmp_table['entity'][idx][row_pos*cols+col_pos] = vec
                    tmp = tmp_table['entity'][idx].reshape((1, rows, cols, 100))
                    if norm == 2:
                        tmp_table['coherence'][idx] = np.sum(get_variance(tmp), axis=1).reshape((-1, 1))
                    else:
                        tmp_table['coherence'][idx] = get_variance(tmp)

                tmp_table['cell'] = np.reshape(tmp_table['cell'], (-1, 100))
                tmp_table['entity'] = np.reshape(tmp_table['entity'], (-1, 100))
                if norm == 2:
                    tmp_table['coherence'] = np.reshape(tmp_table['coherence'], (-1, 1))
                else:
                    tmp_table['coherence'] = np.reshape(tmp_table['coherence'], (-1, 100))
                if norm == 1:
                    tmp_table['coherence'] = normalize(tmp_table['coherence'])
                tmp_table['context'] = np.reshape(tmp_table['context'], (-1, 100))
                running_ret = model.eval(session=session, input_data=tmp_table)
                scores = np.array(running_ret['eval_score'])
                score = scores.max()
                index = scores.argmax()
                if score > base_score:
                    silence = 0
                    if verbose:
                        LogInfo.logs("====> Got higher score %.4f, update cell #%d/%d.",
                                     score, j+1, cell_num)
                    base_score = score
                    round_table['entity'][row_pos*cols+col_pos] = vecs[index]
                    # LogInfo.logs("%d_%d_%d: --> %d.", i+92, row_pos, col_pos, index)
                # ********* NEW: in-time update ********

                # ********* OLD: in-time update ********
                # for vec in vecs:
                #     # step 1. change entity matrix
                #     tmp_table['entity'][row_pos*cols+col_pos] = vec
                #     # step 2. change coherence matrix
                #     tmp = tmp_table['entity'].reshape((1, rows, cols, 100))
                #     tmp_table['coherence'] = get_variance(tmp)
                #     # step 3. re-run model, get a new score
                #     running_ret = model.eval(tmp_table)
                #     score = running_ret['eval_score']
                #     if score-base_score > 0:
                #         silence = 0
                #         if verbose:
                #             LogInfo.logs("====> Got higher score %.4f, update cell #%d/%d.",
                #                          score, j+1, cell_num)
                #         base_score = score
                #         round_table['entity'][row_pos*cols+col_pos] = vec
                # ********* OLD: in-time update ********

            # compare round table to ground truth
            correct = 0
            for idx in range(rows * cols):
                if truth_data[idx][0] != 0 and truth_data[idx][1] != 0 and \
                                np.sum(np.abs(truth_data[idx] - round_table['entity'][idx])) < 1e-6:
                    # np.all(truth_data[xi][xj] == single_table['entity'][xi * cols + xj]):
                    correct += 1
            precision = float(correct) / cell_num
            if verbose:
                LogInfo.logs("Precision for round %d @ table #%d/%d: %.4f(%d/%d)",
                             rounds, i+1, test_size, precision, correct, cell_num)
            best_correct = max(best_correct, correct)
            best_precision = max(best_precision, precision)
            if verbose:
                LogInfo.end_track()

        # summary for one testing table
        if verbose:
            LogInfo.logs("Precision for initial table #%d/%d: %.4f(%d/%d).",
                         i + 1, test_size, init_precision, init_correct, cell_num)
            LogInfo.logs("Precision for upper-bound table #%d/%d: %.4f(%d/%d)",
                         i + 1, test_size, upper_precision, upper_correct, cell_num)
            LogInfo.logs("Precision for table #%d/%d: %.4f(%d/%d) after %d rounds.",
                         i+1, test_size, best_precision, best_correct, cell_num, rounds)
        if best_correct > init_correct:
            better += 1
        elif best_correct == init_correct:
            equal += 1
        pre_list.append(best_precision)
        init_correct_sum += init_correct
        upper_correct_sum += upper_correct
        best_correct_sum += best_correct
        cell_num_sum += cell_num
        if verbose:
            LogInfo.end_track()
        if verbose:
            LogInfo.logs("[s-#%d/%d] i:%.2f, u:%.2f, t:%.2f, m:%.2f", i+1, test_size,
                         initial_score, upper_score, truth_score, base_score)
            LogInfo.logs("[p-#%d/%d] i:%.2f(%d/%d), u:%.2f(%d/%d), m:%.2f(%d/%d)", i+1, test_size,
                         init_precision, init_correct, cell_num,
                         upper_precision, upper_correct, cell_num,
                         best_precision, best_correct, cell_num)

        # ************************* ITERATIVE PREDICTION END *************************

    # summary for all testing tables
    LogInfo.begin_track("[%s] Macro-precision of %d tables", split, test_size)
    LogInfo.logs("Initial precision: %.4f.", sum(initial_pre_list)/test_size)
    LogInfo.logs("Upper-bound precision: %.4f.", sum(upper_pre_list)/test_size)
    LogInfo.logs("Model precision: >> %.4f <<", sum(pre_list) / test_size)

    LogInfo.end_track()

    LogInfo.begin_track("[%s] Micro-precision of %d tables", split, test_size)
    LogInfo.logs("Initial precision: %.4f (%d/%d).",
                 float(init_correct_sum) / cell_num_sum, init_correct_sum, cell_num_sum)
    LogInfo.logs("Upper-bound precision: %.4f (%d/%d).",
                 float(upper_correct_sum) / cell_num_sum, upper_correct_sum, cell_num_sum)
    if split == "valid":
        LogInfo.logs("Model precision: >>> %.4f (%d/%d) <<<",
                     float(best_correct_sum) / cell_num_sum, best_correct_sum, cell_num_sum)
    else:
        LogInfo.logs("Model precision: >> %.4f (%d/%d) <<",
                     float(best_correct_sum) / cell_num_sum, best_correct_sum, cell_num_sum)
    LogInfo.end_track()

    LogInfo.logs("[%s] Better %d/%d, Equal: %d/%d, Worse: %d/%d.", split,
                 better, test_size, equal, test_size, test_size-better-equal, test_size)
    LogInfo.end_track()
    return float(best_correct_sum) / cell_num_sum


if __name__=="__main__":

    LogInfo.logs("Pls copy from ModelTrainer...")
    # model.load("%s/model/ep%d" % (root_path, int(sys.argv[3])))

    # ==== KQ: Checking weights ==== #
    # LogInfo.begin_track('Showing weight information: ')
    # final_weight_tf = myModel.weights['wo']
    # final_weight_opt = myModel.sess.run(final_weight_tf).reshape([-1])
    #    LogInfo.logs('Weight_Output %s: %s', final_weight_opt.shape, final_weight_opt)

    # groups = len(final_weight_opt) / myModel.d_hidden
    # for group_idx in range(groups):
    #     g_vec = final_weight_opt[group_idx * myModel.d_hidden : (group_idx + 1) * myModel.d_hidden]
    #     LogInfo.logs('Group = %d: shape = %s, mean = %.6f, std = %.6f, min = %.6f, max = %.6f',
    #                  group_idx, g_vec.shape, g_vec.mean(), g_vec.std(), g_vec.min(), g_vec.max())
    #     LogInfo.logs('%s', g_vec)
    #
    # LogInfo.end_track()
    # exit()
    # train_data, valid_data, test_data, train_eval_data = \
    #     load_joint_from_file(joint_data_fp, protocol)
    # test_enhanced(model=model, split="test", verbose=False, test_data=test_data)
