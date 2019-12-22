import numpy as np

from schema_optm_dataloader import SchemaOptmDataLoader
from schema_eval_dataloader import SchemaEvalDataLoader

from kangqi.util.LogUtil import LogInfo


def prepare_optm_eval_data(schema_dataset, task_name,
                           neg_f1_ths, neg_max_sample):
    """
    Given the fixed S-MART based schemas and dynamic generated schemas,
    build the Optm/T/v/t data loader for the specific epoch.
    That's to say, we control the negative sampling strategy here.
    """
    """
    Negative strategy: 
        Sample at most "neg_max_samples" data from smart & dynamic dataset.
        Schemas in dynamic set always receive a higher priority.
        Following Angel's idea, all negative samples make the total contribution as the number of positive samples.
        That is, neg_weight = min(1., len(pos_samples) / len(neg_samples))
    """
    """
    Build the big np_data_list, merge from the two datasets.
    What we need in each dataset:
        q_cand_dict: don't forget to mark the use_idx of each schema
        np_data_list: numpy arrays of all inputs (v, v_len, mask, sc_len, xxxx)
    S-MART: fixed, so please pre-calculate all features and just use it.
    Dyn: calculate on-the-fly, and remember all history schemas.
         then how to manage the dynamic np_data_list ?? (maybe np_data_dict, or just save into each schema)
         be careful with different level of features (q level, non-q level)
         (dedup at schema / entity / relation level)
         (duplicate data must have the identical numpy rep. at the specific level)
    """
    assert task_name in ('seg', 'el', 'rm', 'full')

    q_optm_pairs_dict = {}      # < q_idx, [(pos_sc, neg_sc, weight)] >
    q_evals_dict = {}           # < q_idx, [sc] >
    total_optm_pair_size = 0
    total_eval_sc_size = 0
    for scan_idx, q_idx in enumerate(schema_dataset.q_idx_list):
        if scan_idx % 200 == 0:
            LogInfo.logs('[%3s] scanned %d / %d questions. optm_pairs = %d, eval_sc = %d.',
                         task_name, scan_idx, len(schema_dataset.q_idx_list),
                         total_optm_pair_size, total_eval_sc_size)

        smart_cand_list = schema_dataset.smart_q_cand_dict[q_idx]
        dedup_sc_tups = schema_dedup_for_specific_task(
            smart_cand_list=smart_cand_list,
            dynamic_cand_list=[],
            task_name=task_name,
            el_use_type=False
        )   # [(mark, sc, sc.f1)]
        np.random.shuffle(dedup_sc_tups)
        # shuffle the order of schema, avoid data leaking.

        pos_tups = filter(lambda _tup: _tup[2] >= neg_f1_ths, dedup_sc_tups)
        neg_tups = filter(lambda _tup: _tup[2] < neg_f1_ths, dedup_sc_tups)
        pos_size = len(pos_tups)
        # neg_size = len(neg_tups)
        # LogInfo.logs('q_idx = %d, pos_size = %d, neg_size = %d', q_idx, pos_size, neg_size)
        np.random.shuffle(neg_tups)
        dyn_neg_tups = filter(lambda _tup: _tup[0] == 'dynamic', neg_tups)
        fix_neg_tups = filter(lambda _tup: _tup[0] == 'smart', neg_tups)

        if neg_max_sample == -1:
            """ by default: always keeping pos_size = neg_size """
            neg_max_sample = pos_size
        if len(dyn_neg_tups) > neg_max_sample:      # pick all dyn_schemas before picking fix_schemas
            pick_neg_tups = dyn_neg_tups[:neg_max_sample]
        else:
            rem_size = neg_max_sample - len(dyn_neg_tups)
            pick_neg_tups = dyn_neg_tups + fix_neg_tups[:rem_size]

        """ 180313: Let's make things easier: just set equally weighted. """
        neg_weight = 1.

        final_pick_tups = pos_tups + pick_neg_tups
        optm_sc_pair_list = []
        for mark1, sc1, score_1 in final_pick_tups:
            for mark2, sc2, score_2 in final_pick_tups:
                if score_1 < neg_f1_ths:
                    continue
                if score_1 > score_2:
                    weight = 1. if score_2 >= neg_f1_ths else neg_weight
                    # tune down the contribution when a poor schema acts as a negative schema.
                    optm_sc_pair_list.append((sc1, sc2, weight))

        eval_tups = dedup_sc_tups
        eval_sc_list = [tup[1] for tup in eval_tups]
        """ 180314: Remove schema from evaluation set, if ans_size = 0 """
        eval_sc_list = filter(lambda _sc: _sc.ans_size > 0, eval_sc_list)

        q_optm_pairs_dict[q_idx] = optm_sc_pair_list
        q_evals_dict[q_idx] = eval_sc_list
        total_optm_pair_size += len(optm_sc_pair_list)
        total_eval_sc_size += len(eval_sc_list)

    LogInfo.logs('[%3s] In total: optm_pairs = %d, eval_sc = %d.',
                 task_name, total_optm_pair_size, total_eval_sc_size)
    return q_optm_pairs_dict, q_evals_dict


def schema_dedup_for_specific_task(smart_cand_list, dynamic_cand_list, task_name, el_use_type):
    """
    Given the candidates from both side, remove duplicate schemas w.r.t specific task.
    For each duplicate group, always keep dynamic schemas first.
    :return: [(mark, sc, specific_f1)]
    """
    assert task_name in ('el', 'rm', 'full')
    key_tup_dict = {}
    ret_tup_list = []
    for mark, cand_list in [('dynamic', dynamic_cand_list), ('smart', smart_cand_list)]:
        for sc in cand_list:
            key = ''
            if task_name == 'rm':
                key = sc.get_rm_key()
            elif task_name == 'el':
                key = sc.get_el_key(el_use_type)
            elif task_name == 'full':
                key = sc.get_full_key(el_use_type)
                # technically speaking, there's no duplicate schemas
                # when using dealing with full task, but we just write like this.
            key_tup_dict.setdefault(key, []).append((mark, sc))     # separate schemas by task-specific key

    for key, tups in key_tup_dict.items():
        """ All schemas under the same key could be shrinked into one candidate """
        max_f1 = max([sc.f1 for mark, sc in tups])
        first_dynamic_sc = first_smart_sc = None
        for mark, sc in tups:
            if task_name == 'rm':
                sc.rm_f1 = max_f1
            elif task_name == 'el':
                sc.el_f1 = max_f1
            if mark == 'dynamic' and first_dynamic_sc is None:
                first_dynamic_sc = sc
            if mark == 'smart' and first_smart_sc is None:
                first_smart_sc = sc
        if first_dynamic_sc is not None:        # dynamic schemas first
            ret_tup_list.append(('dynamic', first_dynamic_sc, max_f1))
        elif first_smart_sc is not None:
            ret_tup_list.append(('smart', first_smart_sc, max_f1))
    return ret_tup_list


def build_task_dataloaders(feat_gen, task_name,
                           schema_dataset, compq_mt_model,
                           optm_batch_size, eval_batch_size,
                           neg_f1_ths=0.1, neg_max_sample=20):
    LogInfo.begin_track('Building dataloader for task [%3s]:', task_name)
    LogInfo.logs('neg_f1_ths = %.6f', neg_f1_ths)
    LogInfo.logs('neg_max_sample = %.6f', neg_max_sample)
    data_name = schema_dataset.data_name
    q_optm_pairs_dict, q_evals_dict = prepare_optm_eval_data(
        schema_dataset=schema_dataset, task_name=task_name,
        neg_f1_ths=neg_f1_ths, neg_max_sample=neg_max_sample
    )
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

    full_q_idx_list = sorted(q_optm_pairs_dict.keys())
    train_q_idx_list = filter(lambda x: x < train_pos, full_q_idx_list)
    valid_q_idx_list = filter(lambda x: train_pos <= x < valid_pos, full_q_idx_list)
    test_q_idx_list = filter(lambda x: x >= valid_pos, full_q_idx_list)

    optm_dl = SchemaOptmDataLoader(
        schema_dataset=schema_dataset, compq_mt_model=compq_mt_model,
        q_optm_pairs_dict={q_idx: q_optm_pairs_dict[q_idx] for q_idx in train_q_idx_list},
        task_name=task_name, mode='optm',
        shuffle=False, batch_size=optm_batch_size*2, feat_gen=feat_gen
        # Couldn't shuffle! Because the positive and negative schemas are stored in different lines.
        # If we shuffle, then the model fails to bind the correct positive and negative cases together!!
    )
    eval_train_dl = SchemaEvalDataLoader(
        schema_dataset=schema_dataset, compq_mt_model=compq_mt_model,
        q_evals_dict={q_idx: q_evals_dict[q_idx] for q_idx in train_q_idx_list},
        task_name=task_name, mode='train',
        shuffle=False, batch_size=eval_batch_size, feat_gen=feat_gen
    )
    eval_valid_dl = SchemaEvalDataLoader(
        schema_dataset=schema_dataset, compq_mt_model=compq_mt_model,
        q_evals_dict={q_idx: q_evals_dict[q_idx] for q_idx in valid_q_idx_list},
        task_name=task_name, mode='valid',
        shuffle=False, batch_size=eval_batch_size, feat_gen=feat_gen
    )
    eval_test_dl = SchemaEvalDataLoader(
        schema_dataset=schema_dataset, compq_mt_model=compq_mt_model,
        q_evals_dict={q_idx: q_evals_dict[q_idx] for q_idx in test_q_idx_list},
        task_name=task_name, mode='test',
        shuffle=False, batch_size=eval_batch_size, feat_gen=feat_gen
    )
    LogInfo.end_track('[%3s]: Optm/T/v/t dataloader returned.', task_name)
    return optm_dl, eval_train_dl, eval_valid_dl, eval_test_dl
