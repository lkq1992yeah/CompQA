"""
Author: Kangqi Luo
Date: 180420
Goal: A simpler Base class for data building, fight for EMNLP18!!!
"""

import numpy as np
from .base_schema_eval_dataloader import BaseSchemaEvalDataLoader
from .base_schema_optm_dataloader import BaseSchemaOptmDataLoader

from kangqi.util.LogUtil import LogInfo


class BaseSchemaDLBuilder:

    def __init__(self, schema_dataset, compq_mt_model,
                 neg_pick_config, shuffle=False,
                 optm_dataloader=BaseSchemaOptmDataLoader,
                 eval_dataloader=BaseSchemaEvalDataLoader):
        self.schema_dataset = schema_dataset
        self.compq_mt_model = compq_mt_model
        self.neg_pick_config = neg_pick_config
        self.shuffle = shuffle
        """
        The builder is based on the actual data (schema_dataset) and input specification (compq_mt_model).
        In terms of optm/eval data pick for different tasks, we follow the same strategy.
        """
        self.active_optm_dataloader = optm_dataloader
        self.active_eval_dataloader = eval_dataloader

    def build_task_dataloaders(self, task_name, optm_batch_size, eval_batch_size):
        LogInfo.begin_track('Building dataloader for task [%3s]:', task_name)

        input_tensor_names = getattr(self.compq_mt_model, '%s_eval_input_names' % task_name)
        active_input_tensor_dict = {k: self.compq_mt_model.input_tensor_dict[k]
                                    for k in input_tensor_names}
        LogInfo.logs('Active input tensors: %s', input_tensor_names)

        data_name = self.schema_dataset.data_name
        train_pos = valid_pos = 0
        if data_name == 'WebQ':
            train_pos = 3023
            valid_pos = 3778
        elif data_name == 'CompQ':
            train_pos = 1000
            valid_pos = 1300
        elif data_name == 'SimpQ':
            train_pos = 75910
            valid_pos = 86755
        q_optm_pairs_dict, q_evals_dict = self.prepare_optm_eval_data(task_name=task_name, train_pos=train_pos)

        full_q_idx_list = sorted(q_optm_pairs_dict.keys())
        train_q_idx_list = filter(lambda x: x < train_pos, full_q_idx_list)
        valid_q_idx_list = filter(lambda x: train_pos <= x < valid_pos, full_q_idx_list)
        test_q_idx_list = filter(lambda x: x >= valid_pos, full_q_idx_list)

        modes = ['train', 'valid', 'test']
        q_idx_lists = [train_q_idx_list, valid_q_idx_list, test_q_idx_list]

        optm_dl = self.active_optm_dataloader(
            q_optm_pairs_dict={q_idx: q_optm_pairs_dict[q_idx] for q_idx in train_q_idx_list},
            input_tensor_dict=active_input_tensor_dict,
            schema_dataset=self.schema_dataset,
            task_name=task_name, mode='optm',
            shuffle=self.shuffle,
            batch_size=optm_batch_size*2
        )
        eval_dl_list = []
        for mode, cur_q_idx_list in zip(modes, q_idx_lists):
            cur_q_evals_dict = {q_idx: q_evals_dict[q_idx] for q_idx in cur_q_idx_list}
            eval_dl = self.active_eval_dataloader(
                q_evals_dict=cur_q_evals_dict,
                input_tensor_dict=active_input_tensor_dict,
                schema_dataset=self.schema_dataset,
                task_name=task_name, mode=mode,
                shuffle=self.shuffle, batch_size=eval_batch_size
            )
            eval_dl_list.append(eval_dl)

        LogInfo.end_track('[%3s]: Optm/T/v/t dataloader returned.', task_name)
        # optm_dl, eval_train_dl, eval_valid_dl, eval_test_dl
        final_dl_list = [optm_dl] + eval_dl_list
        return final_dl_list

    def prepare_optm_eval_data(self, task_name, train_pos):
        """
        Given the fixed S-MART based schemas and dynamic generated schemas,
        build the Optm/T/v/t data loader for the specific epoch.
        That's to say, we control the negative sampling strategy here.
        """
        """ Negative strategy: 
        1. fix: (threshold + random sample) 
        2. dynamic: (theshold + weighed sample based on delta)
        """
        """
        Be careful of deuplications.   
        (dedup at schema / entity / relation level)
        (duplicate data must have the identical numpy rep. at the specific level)
        """

        assert task_name in ('el', 'rm', 'full')
        runtime_score_key = task_name + '_score'

        schema_dataset = self.schema_dataset
        neg_f1_ths = self.neg_pick_config['neg_f1_ths']
        neg_max_sample = self.neg_pick_config['neg_max_sample']
        strategy = self.neg_pick_config.get('strategy', 'fix')
        cool_down = self.neg_pick_config.get('cool_down', 1.0)
        LogInfo.logs('neg_pick_config: %s', self.neg_pick_config)
        LogInfo.logs('runtime_score_key = [%s]', runtime_score_key)
        assert strategy in ('Dyn', 'Fix')

        q_optm_pairs_dict = {}  # < q_idx, [(pos_sc, neg_sc)] >
        q_evals_dict = {}       # < q_idx, [sc] >
        total_optm_pair_size = 0
        total_eval_sc_size = 0

        for scan_idx, q_idx in enumerate(schema_dataset.q_idx_list):
            is_in_train = q_idx < train_pos
            # LogInfo.logs('current q_idx: %d', q_idx)
            if scan_idx % 1000 == 0:
                LogInfo.logs('[%3s] scanned %d / %d questions. optm_pairs = %d, eval_sc = %d.',
                             task_name, scan_idx, len(schema_dataset.q_idx_list),
                             total_optm_pair_size, total_eval_sc_size)

            cand_list = schema_dataset.smart_q_cand_dict[q_idx]
            dedup_sc_tups = self.schema_dedup_for_specific_task(
                cand_list=cand_list, task_name=task_name
            )   # [(sc, sc.f1)]
            np.random.shuffle(dedup_sc_tups)    # shuffle, avoid data leaking.
            pos_tups = filter(lambda _tup: _tup[-1] >= neg_f1_ths, dedup_sc_tups)
            neg_tups = filter(lambda _tup: _tup[-1] < neg_f1_ths, dedup_sc_tups)
            # pos_size = len(pos_tups)
            # neg_size = len(neg_tups)
            # LogInfo.logs('q_idx = %d, pos_size = %d, neg_size = %d', q_idx, pos_size, neg_size)
            for sc, _ in dedup_sc_tups:
                self.input_feat_gen(sc, is_in_train=is_in_train)     # Generate input features here

            eval_sc_list = filter(lambda _sc: _sc.ans_size > 0, [tup[0] for tup in dedup_sc_tups])
            """ 180314: Remove schema from evaluation set, if ans_size = 0 """
            optm_sc_pair_list = []
            for sc1, f1_1 in pos_tups:
                if strategy == 'Fix':
                    for sc2, f1_2 in pos_tups:       # both sc+ and sc- come from positive list
                        if f1_1 > f1_2:
                            optm_sc_pair_list.append((sc1, sc2))
                    """ 180420: shuffle negative list for each positive schema """
                    np.random.shuffle(neg_tups)
                    for sc2, _ in neg_tups[:neg_max_sample]:    # sc- comes from negative list
                        optm_sc_pair_list.append((sc1, sc2))
                else:
                    sample_tups = []
                    runtime_score_1 = (0. if sc1.run_info is None else
                                       sc1.run_info.get(runtime_score_key, 0.))
                    for sc2, f1_2 in pos_tups + neg_tups:
                        if f1_1 > f1_2:
                            runtime_score_2 = (0. if sc2.run_info is None else
                                               sc2.run_info.get(runtime_score_key, 0.))
                            delta = runtime_score_2 - runtime_score_1   # the larger, the more critical
                            sample_tups.append((sc1, sc2, delta))
                    local_picked_tups = self.weighted_sampling(sample_tups=sample_tups,
                                                               neg_max_sample=neg_max_sample,
                                                               cool_down=cool_down)
                    optm_sc_pair_list += local_picked_tups
            q_optm_pairs_dict[q_idx] = optm_sc_pair_list
            q_evals_dict[q_idx] = eval_sc_list
            total_optm_pair_size += len(optm_sc_pair_list)
            total_eval_sc_size += len(eval_sc_list)

        LogInfo.logs('[%3s] In total: optm_pairs = %d, eval_sc = %d.',
                     task_name, total_optm_pair_size, total_eval_sc_size)
        return q_optm_pairs_dict, q_evals_dict

    def schema_dedup_for_specific_task(self, cand_list, task_name):
        """
        Given all candidates of a question, remove duplicate schemas w.r.t specific task.
        :return: [(sc, specific_f1)]
        """
        assert task_name in ('el', 'rm', 'full')
        key_cands_dict = {}
        ret_tup_list = []
        for sc in cand_list:
            key = ''
            if task_name == 'rm':
                key = self.get_rm_key(sc)
            elif task_name == 'el':
                key = self.get_el_key(sc)
            elif task_name == 'full':
                key = self.get_full_key(sc)
                # technically speaking, there's no duplicate schemas
                # when using dealing with full task, but we just write like this.
            key_cands_dict.setdefault(key, []).append(sc)     # separate schemas by task-specific key

        for key, cand_list in key_cands_dict.items():
            """ All schemas under the same key could be shrink into one candidate """
            max_f1 = max([sc.f1 for sc in cand_list])
            for sc in cand_list:
                if task_name == 'rm':
                    sc.rm_f1 = max_f1
                elif task_name == 'el':
                    sc.el_f1 = max_f1
            first_sc = cand_list[0]     # just pick the first schema as representative one
            ret_tup_list.append((first_sc, max_f1))
        return ret_tup_list

    def get_rm_key(self, sc):
        raise NotImplementedError

    def get_el_key(self, sc):
        raise NotImplementedError

    def get_full_key(self, sc):
        raise NotImplementedError

    def input_feat_gen(self, sc, is_in_train, **kwargs):
        raise NotImplementedError

    @staticmethod
    def weighted_sampling(sample_tups, neg_max_sample, cool_down):
        """ weighted sampling without replacements """
        if len(sample_tups) == 0:
            return []
        delta = np.array([tup[-1] for tup in sample_tups])
        cd_delta = delta * cool_down
        raw_prob = np.exp(cd_delta)
        prob = raw_prob / np.sum(raw_prob)
        # LogInfo.logs('delta: %s', delta)
        # LogInfo.logs('prob: %s', prob)
        # sample_size = min(neg_max_sample, len(sample_tups))
        sample_size = neg_max_sample
        pick_idx_list = np.random.choice(a=len(sample_tups),
                                         size=sample_size,
                                         replace=True,
                                         p=prob)
        pick_tups = []
        for idx in pick_idx_list:
            tup = sample_tups[idx]
            pick_tups.append((tup[0], tup[1]))
        return pick_tups
