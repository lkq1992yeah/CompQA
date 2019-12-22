# -*- coding: utf-8 -*-

#==============================================================================
# Author: Kangqi Luo
# Goal: Convert <qa, [(schema, score)]> into arrays sent to Tensorflow part.
#==============================================================================

import numpy as np
import cPickle

from .w2v import Word2Vec

from kangqi.util.LogUtil import LogInfo


class DataSaver(object):

    def __init__(self, sent_max_len, n_word_emb, path_max_len, n_path_emb, w2v_fp):
        self.el_score_disc = Discretizer([0.1 * x for x in range(1, 10)]) # [0.1, ..., 0.9]
        self.pop_disc = Discretizer([1] + [pow(10, x) for x in range(2, 7)]) # [1, 100, 1K, 10K, 100K, 1M]
        self.n_el = self.el_score_disc.len + self.pop_disc.len + 2  # EL score, entity pop, perfect match

        self.sent_max_len = sent_max_len
        self.path_max_len = path_max_len
        self.n_word_emb = n_word_emb
        self.n_path_emb = n_path_emb

        assert n_path_emb == 3 * n_word_emb
        # currently, the predicate is represented by concatenating domain/type/property w2v results.

        self.w2v = Word2Vec(w2v_fp, self.n_word_emb)


    # PN: the total number of positive + negative cases.
    # Prepare necessary numpy arrays, and save into pydump file.
    # qa_schema_dict: <qa, (schema, (lower_predict_set, score))>
    def save_qa_data(self, qa_schema_dict, PN, save_fp, Tvt, sc_mode='Skeleton'):
        sz = len(qa_schema_dict)

        goal_list = []      # The list storing the final numpy tensors.
        goal_name_list = [] # storing the name of each tensor.

        q_tensor3 = np.zeros((sz, self.sent_max_len, self.n_word_emb), dtype='float32')   # (cases, sent_max_len, n_word_emb)
        el_tensor3 = np.zeros((sz, PN, self.n_el), dtype='float32')   # (cases, PN, n_el)
        path_tensor4 = np.zeros((sz, PN, self.path_max_len, self.n_path_emb), dtype='float32') # (cases, PN, path_max_len, n_path_emb)
        score_tensor3 = np.zeros((sz, PN, 3), dtype='float32')    # (cases, PN, 3) (P, R, F1)
        mask_matrix = np.zeros((sz, PN), dtype='float32')   # mask array
        goal_list += [q_tensor3, el_tensor3, path_tensor4, score_tensor3, mask_matrix]
        goal_name_list += ['q_tensor3', 'el_tensor3', 'path_tensor4', 'score_tensor3', 'mask_matrix']

        if sc_mode == 'Sk+Ordinal':
            # check .ordinal_ours for detail information of each tensor.
            ord_x_matrix = np.zeros((sz, PN), dtype='int32')
            ord_pred_tensor3 = np.zeros((sz, PN, self.n_path_emb), dtype='float32')
            ord_op_tensor3 = np.zeros((sz, PN, 2), dtype='float32')
            ord_obj_tensor3 = np.zeros((sz, PN, self.n_word_emb), dtype='float32')
            ord_mask_matrix = np.zeros((sz, PN), dtype='float32')
            goal_list += [ord_x_matrix, ord_pred_tensor3,
                          ord_op_tensor3, ord_obj_tensor3, ord_mask_matrix]
            goal_name_list += ['ord_x_matrix', 'ord_pred_tensor3',
                               'ord_op_tensor3', 'ord_obj_tensor3', 'ord_mask_matrix']

        LogInfo.begin_track('Saving %d <qa, [schema, score]> data: ', sz)
        srt_qa_schema_list = sorted(qa_schema_dict.items(), lambda x, y: cmp(x[0].q_idx, y[0].q_idx))

        for case_idx in range(sz):
            local_list = [] # storing local numpy tensors for one QA case.

            qa, schema_score_list = srt_qa_schema_list[case_idx]
            LogInfo.begin_track('Entering QA %d / %d: ', case_idx + 1, sz)
            q_matrix = self.q_encoding(qa.tokens)
            self.gather_tensors(local_list, [q_matrix], ['q_matrix'])

            if Tvt == 'T':
                # schema filtering, since we will use at most "PN" schemas.
                # Perform ranking first, then perform truncating
                schema_score_list.sort(lambda x, y: -cmp(x[1][1]['f1'], y[1][1]['f1']))
                filt_schema_score_list = schema_score_list[0 : PN]
            else:
                # First perform truncating (since we don't know the gold answer in advance)
                # Then perform ranking, because we need to incorporate with HingeLoss model
                filt_schema_score_list = schema_score_list[0 : PN]
                filt_schema_score_list.sort(lambda x, y: -cmp(x[1][1]['f1'], y[1][1]['f1']))
            LogInfo.logs('%d candidate schema will be used (PN=%d)', len(filt_schema_score_list), PN)

            basic_list = (
                el_matrix, path_tensor3,
                score_matrix, mask_vec
            ) = self.schema_encoding(filt_schema_score_list, PN)
            basic_name_list = ['el_matrix', 'path_tensor3', 'score_matrix', 'mask_vec']
            self.gather_tensors(local_list, basic_list, basic_name_list)

            if sc_mode == 'Sk+Ordinal':
                ord_list = (
                    ord_x_vec, ord_pred_matrix,
                    ord_op_matrix, ord_obj_matrix, ord_mask_vec
                ) = self.ordinal_encoding(filt_schema_score_list, PN)
                ord_name_list = ['ord_x_vec', 'ord_pred_matrix',
                                 'ord_op_matrix', 'ord_obj_matrix', 'ord_mask_vec']
                self.gather_tensors(local_list, ord_list, ord_name_list)

            for goal_item, local_item in zip(goal_list, local_list):
                goal_item[case_idx] = local_item    # store to the global numpy array
            LogInfo.end_track()                     # end of each QA
        LogInfo.end_track()     # end of all QAs

        LogInfo.begin_track('Saving to %s ... ', save_fp)
        with open(save_fp, 'wb') as bw:
            for item, name in zip(goal_list, goal_name_list):
                cPickle.dump(name, bw)
                np.save(bw, item)
                LogInfo.logs('%s: %s saved.', name, item.shape)
        LogInfo.end_track()

    # put tensors together into local_list
    def gather_tensors(self, local_list, new_tensor_list, new_tensor_name_list):
        for tensor, name in zip(new_tensor_list, new_tensor_name_list):
            LogInfo.logs('%s %s generated.', name, tensor.shape)
        local_list += new_tensor_list

    # input: tokens of one sentence; output: its embedding array.
    def q_encoding(self, tokens):
        ret_matrix = np.zeros((self.sent_max_len, self.n_word_emb), dtype='float32')
        tok_sz = len(tokens)
        used_len = 0
        for idx in range(tok_sz):
            if used_len >= self.sent_max_len: break   # truncate the sentence.
            tok = tokens[idx]
            wd = tok.token
            if isinstance(wd, unicode):
                wd = wd.encode('utf-8')
            wd_vec = self.w2v.get(wd)
            if wd_vec is None or len(wd_vec) == 0: continue   # this word is missing
            assert len(wd_vec) == self.n_word_emb
            ret_matrix[used_len] = wd_vec
            used_len += 1
        return ret_matrix

    # input: a list of (schema, score) information
    # output: different kinds of numpy arrays as the input of Tensorflow part.
    # ** We ensure that len(schema_list) <= PN, since we've filtered some of them.
    # schema_score_list: [(schema, (lower_predict_set, score_dict))]
    def schema_encoding(self, schema_score_list, PN):
        schema_list = [kv[0] for kv in schema_score_list]
        score_list = [kv[1] for kv in schema_score_list]       # [(lower_predict_set, score_dict)]

        el_matrix = np.zeros((PN, self.n_el), dtype='float32')   # (PN, n_el)
        path_tensor3 = np.zeros((PN, self.path_max_len, self.n_path_emb), dtype='float32') # (PN, path_max_len, n_path_emb)
        score_matrix = np.zeros((PN, 3), dtype='float32')    # (PN, 3)  (P, R, F1)
        mask_vec = np.zeros((PN, ), dtype='float32')

        for sc_idx in range(len(schema_list)):
            lower_predict_set, score_dict = score_list[sc_idx]
            p_score = score_dict['p']
            r_score = score_dict['r']
            f1_score = score_dict['f1']
            score_matrix[sc_idx][0] = p_score
            score_matrix[sc_idx][1] = r_score
            score_matrix[sc_idx][2] = f1_score
            mask_vec[sc_idx] = 1
            schema = schema_list[sc_idx]
            el_vec = self.el_encoding(schema)
            el_matrix[sc_idx] = el_vec
            path_matrix = self.path_encoding(schema)
            path_tensor3[sc_idx] = path_matrix

        return el_matrix, path_tensor3, score_matrix, mask_vec


    # EL contains: entity name, entity types, surface_score, popularity, and perfect_match
    # Let's first consider surface_score, popularity and perfect_match.
    def el_encoding(self, schema):
        el_item = schema.focus_item
        surface_score = el_item.surface_score
        popularity = el_item.score
        perfect_match = el_item.perfect_match
        # mid = el_item.entity.id

        score_vec = self.el_score_disc.convert(surface_score)
        pop_vec = self.pop_disc.convert(popularity)
        perfect_match_vec = np.zeros((2, ), dtype='float32')
        if perfect_match:
            perfect_match_vec[1] = 1
        else:
            perfect_match_vec[0] = 1

        el_vec = np.concatenate([score_vec, pop_vec, perfect_match_vec])    # (n_el, )
        assert el_vec.shape == (self.n_el, )
        return el_vec

    # Path contains: a sequence of predicates
    def path_encoding(self, schema):
        ret_matrix = np.zeros((self.path_max_len, 3 * self.n_word_emb), dtype='float32')
        path = schema.path
        LogInfo.begin_track('path: %s: ', path)
        for pred_idx in range(len(path)):
            pred = path[pred_idx]
            pred_vec = self.predicate_encoding(pred)
            assert len(pred_vec) == self.n_path_emb
            ret_matrix[pred_idx] = pred_vec
        LogInfo.end_track()
        return ret_matrix

    # predicate --> hidden vector
    # Currently, we concatenate domain / type / property's w2v
    def predicate_encoding(self, pred):
        surface_vec_list = []
        for surface in pred.split('.'): # iterative domain / type / property
            wd_list = surface.split('_')
            avg_wd_vec = np.zeros((self.n_word_emb, ), dtype='float32')
            keep_cnt = 0
            for wd in wd_list:
                wd_vec = self.w2v.get(wd)
                if wd_vec is None or len(wd_vec) == 0: continue
                assert wd_vec.shape == avg_wd_vec.shape
                avg_wd_vec += wd_vec
                keep_cnt += 1
            LogInfo.logs('pred = %s, surface = %s, keep_cnt = %d.', pred, surface, keep_cnt)
            if keep_cnt > 0:
                avg_wd_vec /= keep_cnt     # get average w2v for the item (domain / type / property)
            surface_vec_list.append(avg_wd_vec)
        while len(surface_vec_list) < 3:
            # We found some predicates don't obey domain.type.property format "topic_server.population_number"
            # So we just pad the predicate surface vector
            surface_vec_list.append(np.zeros((self.n_word_emb, ), dtype='float32'))
        pred_vec = np.concatenate(surface_vec_list)    # (3 * n_word_emb, )
        return pred_vec

    # input: a list of (schema, score) information
    # output: All tensors related to ordinal constraints.
    # We also ensure that len(schema_score_list) <= PN
    def ordinal_encoding(self, schema_score_list, PN):
        schema_list = [kv[0] for kv in schema_score_list]

        ord_x_vec = np.zeros((PN,), dtype='int32')
        ord_pred_matrix = np.zeros((PN, self.n_path_emb), dtype='float32')
        ord_op_matrix = np.zeros((PN, 2), dtype='float32')
        ord_obj_matrix = np.zeros((PN, self.n_word_emb), dtype='float32')
        ord_mask_vec = np.zeros((PN,), dtype='float32')

        for sc_idx in range(len(schema_list)):
            schema = schema_list[sc_idx]
            ordinal_constr = None
            for constr in schema.constraints:
                if constr.constr_type == 'Ordinal':
                    ordinal_constr = constr
                    break
            # Now we've get the ordinal constraint

            if ordinal_constr is not None:
                var_pos = ordinal_constr.x
                pred = ordinal_constr.p
                op = ordinal_constr.comp
                obj_wd = ordinal_constr.linking_item.tokens[0].token # rank word
                assert op in ('ASC', 'DESC')

                ord_mask_vec[sc_idx] = 1.0
                ord_x_vec[sc_idx] = var_pos - 1 # locate the index of RNN state
                pred_vec = self.predicate_encoding(pred)
                assert len(pred_vec) == self.n_path_emb
                ord_pred_matrix[sc_idx] = pred_vec
                if op == 'ASC':
                    ord_op_matrix[sc_idx, 0] = 1.0
                else:
                    ord_op_matrix[sc_idx, 1] = 1.0
                obj_wd_vec = self.w2v.get(obj_wd)
                if obj_wd_vec is not None:
                    ord_obj_matrix[sc_idx] = obj_wd_vec

        return ord_x_vec, ord_pred_matrix, ord_op_matrix, ord_obj_matrix, ord_mask_vec


#==============================================================================
# Below: Utility codes
#==============================================================================

def load_numpy_input(pydump_fp):
    LogInfo.begin_track('Loading input from %s: ', pydump_fp)
    np_list = []
    with open(pydump_fp, 'rb') as br:
        while True:
            try:
                np_tensor = np.load(br)
                LogInfo.logs('Tensor: %s loaded.', np_tensor.shape)
                np_list.append(np_tensor)
            except (IOError, EOFError) as ex:
                LogInfo.logs('%s Reach EOF.', type(ex))
                break
    LogInfo.end_track()
    return np_list

# Each numpy array is associated with the name
def load_numpy_input_with_names(pydump_fp):
    LogInfo.begin_track('Loading input from %s: ', pydump_fp)
    np_list = []
    with open(pydump_fp, 'rb') as br:
        while True:
            try:
                tensor_name = cPickle.load(br)
                np_tensor = np.load(br)
                LogInfo.logs('%s: %s loaded.', tensor_name, np_tensor.shape)
                np_list.append(np_tensor)
            except (IOError, EOFError) as ex:
                LogInfo.logs('%s Reach EOF.', type(ex))
                break
    LogInfo.end_track()
    return np_list

# Goal: convert a float number into 1-hot vector
class Discretizer(object):

    def __init__(self, split_list):
        split_list.sort()
        self.split_list = split_list
        self.len = len(split_list) + 1

    def convert(self, score):
        ret_vec = np.zeros((self.len, ), dtype='float32')
        if score < self.split_list[0]:
            ret_vec[0] = 1
        elif score >= self.split_list[-1]:
            ret_vec[-1] = 1
        else:
            for i in range(len(self.split_list) - 1):
                if score >= self.split_list[i] and score < self.split_list[i + 1]:
                    ret_vec[i + 1] = 1
                    break
        return ret_vec

    # distribution_vec: sum of several discretized vectors
    def show_distribution(self, score_list):
        sz = len(score_list)
        LogInfo.begin_track('Showing distribution over %d data: ', sz)
        distribution_vec = np.zeros((self.len, ), dtype='float32')
        for score in score_list:
            distribution_vec += self.convert(score)
        # LogInfo.logs('%s', distribution_vec)
        for idx in range(self.len):
            val = distribution_vec[idx]
            LogInfo.logs('[%s, %s): %d / %d (%.3f%%)',
                         '-inf' if idx == 0 else str(self.split_list[idx - 1]),
                         str(self.split_list[idx]) if idx < self.len - 1 else 'inf',
                         int(val), sz, 100.0 * val / sz)
        LogInfo.end_track()



if __name__ == '__main__':
    # split_list = [0.1 * x for x in range(1, 10)]
    # val_list = [0.09999, 0.10, 0.51, 1.00]

    split_list = [1] + [pow(10, x) for x in range(2, 7)]
    val_list = [0, 58, 100, 127, 9999.999, 1000, 5899, 10234, 1048576, 10485760]

    disc = Discretizer(split_list)
    for val in val_list:
        LogInfo.logs('%g: %s, %s', val, disc.convert(val), disc.convert(val).shape)


