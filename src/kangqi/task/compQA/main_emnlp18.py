import os
import codecs
import shutil
import argparse
import numpy as np
import tensorflow as tf
from ast import literal_eval
from datetime import datetime

from dataset.dataset_emnlp18 import SchemaDatasetEMNLP18
from dataset.dl_compq_emnlp18.schema_dl_builder import build_task_dataloaders
from dataset.dl_compq_emnlp18.input_feat_generator import InputFeatureGenerator

from model.compq_emnlp18.compq_mt_model import CompqMultiTaskModel

from learner.pyltr.pyltr_worker import ltr_whole_process
from learner.compq_e2e_old import \
    EntityLinkingOptimizer, EntityLinkingEvaluator, \
    RelationMatchingOptimizer, RelationMatchingEvaluator, \
    FullTaskOptimizer, FullTaskEvaluator

from .u import delete_dir, save_model, load_model

from util.word_emb import WordEmbeddingUtil

from kangqi.util.LogUtil import LogInfo


working_ip = '202.120.38.146'
parser_port_dict = {'Blackhole': 9601, 'Darkstar': 8601}
sparql_port_dict = {'Blackhole': 8999, 'Darkstar': 8699}

parser = argparse.ArgumentParser(description='QA Model Training')

parser.add_argument('--machine', default='Blackhole', choices=['Blackhole', 'Darkstar'])
parser.add_argument('--word_emb', default='glove')
parser.add_argument('--dim_emb', type=int, default=300, help='word/predicate embedding dimension')
parser.add_argument('--data_config', help='dataset config')
parser.add_argument('--neg_f1_ths', default=0.1, type=float)
parser.add_argument('--neg_max_sample', default=20, type=int)

parser.add_argument('--seg_config', help='segment config')
parser.add_argument('--rm_config', help='relation matching config')
parser.add_argument('--model_config', help='general model config')
parser.add_argument('--resume_model_name', default='model_best',
                    help='the directory name of the model which you wan to resume learning')

parser.add_argument('--optm_batch_size', type=int, default=128, help='optm_batch size')
parser.add_argument('--eval_batch_size', type=int, default=32, help='eval_batch size')
parser.add_argument('--max_epoch', type=int, default=30, help='max epochs')
parser.add_argument('--max_patience', type=int, default=10000, help='max patience')

parser.add_argument('--full_optm_method', choices=['ltr', 'full', 'rm', 'el'])
parser.add_argument('--pre_train_steps', type=int, default=10)

parser.add_argument('--output_dir', help='output dir, including results, models and others')
parser.add_argument('--save_best', help='save best model only', action='store_true')
parser.add_argument('--test_only', action='store_true')

parser.add_argument('--gpu_fraction', type=float, default=0.25, help='GPU fraction limit')
parser.add_argument('--verbose', type=int, default=0, help='verbose level')


def main(args):
    # ==== Optm & Eval register ==== #
    # ltr: learning-to-rank; full: fully-connected layer as the last layer
    full_optm_method = args.full_optm_method
    if full_optm_method in ('el', 'rm'):        # sub-task only mode
        optm_tasks = eval_tasks = [full_optm_method]
    else:                                       # ltr or full
        # optm_tasks = ['el', 'rm', 'full']
        # eval_tasks = ['el', 'rm', 'full']
        full_optm_method = 'rm'         # TODO: full task is a bit tricky here
        optm_tasks = ['el', 'rm']
        eval_tasks = ['el', 'rm']       # TODO: full task is a bit tricky here
        # all sub-tasks needed, including full
        # we need direct comparison between full and ltr
        if full_optm_method == 'ltr':
            eval_tasks.append('ltr')
    LogInfo.logs('full_optm_method: %s', full_optm_method)
    LogInfo.logs('optimize tasks: %s', optm_tasks)
    LogInfo.logs('evaluate tasks: %s', eval_tasks)

    # ==== Loading Necessary Util ==== #
    LogInfo.begin_track('Loading Utils ... ')
    wd_emb_util = WordEmbeddingUtil(wd_emb=args.word_emb, dim_emb=args.dim_emb)
    wd_emb_util.load_word_indices()
    wd_emb_util.load_mid_indices()
    LogInfo.end_track()

    # ==== Loading Dataset ==== #
    LogInfo.begin_track('Creating Dataset ... ')
    data_config = literal_eval(args.data_config)
    data_config['wd_emb_util'] = wd_emb_util
    data_config['verbose'] = args.verbose
    schema_dataset = SchemaDatasetEMNLP18(**data_config)
    schema_dataset.load_smart_cands()
    """ load data before constructing model, as we generate lookup dict in the loading phase """
    LogInfo.end_track()

    # ==== Building Model ==== #
    LogInfo.begin_track('Building Model and Session ... ')
    gpu_options = tf.GPUOptions(allow_growth=True,
                                per_process_gpu_memory_fraction=args.gpu_fraction)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options,
                                            intra_op_parallelism_threads=8))
    model_config = literal_eval(args.model_config)
    for key in ('qw_max_len', 'pw_max_len', 'path_max_size',
                'pw_voc_inputs', 'pw_voc_length',
                'pw_voc_domain', 'entity_type_matrix'):
        model_config[key] = getattr(schema_dataset, key)
    for key in ('n_words', 'dim_emb'):
        model_config[key] = getattr(wd_emb_util, key)
    model_config['el_feat_size'] = 3        # TODO: manually assigned
    model_config['extra_feat_size'] = 16
    if full_optm_method != 'full':
        model_config['full_back_prop'] = False      # make sure rm/el optimizes during all epochs
    full_back_prop = model_config['full_back_prop']
    compq_mt_model = CompqMultiTaskModel(**model_config)

    LogInfo.begin_track('Showing final parameters: ')
    for var in tf.global_variables():
        LogInfo.logs('%s: %s', var.name, var.get_shape().as_list())
    LogInfo.end_track()

    focus_param_name_list = ['el_kernel/out_fc/weights', 'el_kernel/out_fc/biases',
                             'full_task/final_fc/weights', 'full_task/final_fc/biases',
                             'abcnn1_rm_kernel/sim_ths', 'abcnn2_rm_kernel/sim_ths']
    focus_param_list = []
    with tf.variable_scope('', reuse=tf.AUTO_REUSE):
        for param_name in focus_param_name_list:
            try:
                var = tf.get_variable(name=param_name)
                focus_param_list.append(var)
            except ValueError:
                pass
    LogInfo.begin_track('Showing %d concern parameters: ', len(focus_param_list))
    for name, tensor in zip(focus_param_name_list, focus_param_list):
        LogInfo.logs('%s --> %s', name, tensor.get_shape().as_list())
    LogInfo.end_track()

    saver = tf.train.Saver()
    LogInfo.begin_track('Running global_variables_initializer ...')
    start_epoch = 0
    best_valid_f1 = 0.
    resume_flag = False
    model_dir = None
    if args.resume_model_name not in ('', 'None'):
        model_dir = '%s/%s' % (args.output_dir, args.resume_model_name)
        if os.path.exists(model_dir):
            resume_flag = True
    if resume_flag:
        start_epoch, best_valid_f1 = load_model(saver=saver, sess=sess, model_dir=model_dir)
    else:
        wd_emb_util.load_word_embeddings()
        wd_emb_util.load_mid_embeddings()
        sess.run(tf.global_variables_initializer(),
                 feed_dict={compq_mt_model.w_embedding_init: wd_emb_util.word_emb_matrix})
        # compq_mt_model.m_embedding_init: wd_emb_util.mid_emb_matrix})
    LogInfo.end_track('Start Epoch = %d', start_epoch)
    LogInfo.end_track('Model build complete.')

    # ==== Register optm / eval ==== #
    feat_gen = InputFeatureGenerator(schema_dataset=schema_dataset)
    el_optimizer = EntityLinkingOptimizer(compq_mt_model=compq_mt_model, sess=sess, ob_batch_num=100)
    el_evaluator = EntityLinkingEvaluator(compq_mt_model=compq_mt_model, sess=sess, ob_batch_num=100)
    rm_optimizer = RelationMatchingOptimizer(compq_mt_model=compq_mt_model, sess=sess, ob_batch_num=100)
    rm_evaluator = RelationMatchingEvaluator(compq_mt_model=compq_mt_model, sess=sess, ob_batch_num=100)
    # full_optimizer = FullTaskOptimizer(compq_mt_model=compq_mt_model, sess=sess, ob_batch_num=100)
    # full_evaluator = FullTaskEvaluator(compq_mt_model=compq_mt_model, sess=sess, ob_batch_num=100)
    LogInfo.logs('Optimizer & Evaluator defined for RM, EL and FULL.')

    # ==== Iteration begins ==== #
    output_dir = args.output_dir
    if not os.path.exists(output_dir + '/detail'):
        os.makedirs(output_dir + '/detail')
    if not os.path.exists(output_dir + '/result'):
        os.makedirs(output_dir + '/result')

    LogInfo.begin_track('Learning start ...')
    patience = args.max_patience

    status_fp = output_dir + '/status.csv'
    raw_header_list = ['Epoch']
    for task_name in ('ltr', 'full', 'el', 'rm'):   # all possibilities of Optm/T/v/t
        local_header_list = []
        if task_name in optm_tasks:
            local_header_list.append('%s_loss' % task_name)
        if task_name in eval_tasks:
            for mark in 'Tvt':
                local_header_list.append('%s_%s_F1' % (task_name, mark))
        if len(local_header_list) > 0:
            raw_header_list.append(' |  ')
        raw_header_list += local_header_list
    raw_header_list += [' |  ', 'Status', 'Time']
    disp_header_list = []
    no_tab = True
    for idx, header in enumerate(raw_header_list):      # dynamic add \t into raw headers
        if not (no_tab or header.endswith(' ')):
            disp_header_list.append('\t')
        disp_header_list.append(header)
        no_tab = header.endswith(' ')
    with open(status_fp, 'a') as bw:
        write_str = ''.join(disp_header_list)
        bw.write(write_str + '\n')

    if full_back_prop:
        LogInfo.logs('full_back_prop = %s, pre_train_steps = %d.', full_back_prop, args.pre_train_steps)
    else:
        LogInfo.logs('no pre-train available.')
    for epoch in range(start_epoch+1, args.max_epoch+1):
        if patience == 0:
            LogInfo.logs('Early stopping at epoch = %d.', epoch)
            break
        update_flag = False
        disp_item_dict = {'Epoch': epoch}

        LogInfo.begin_track('Epoch %d / %d', epoch, args.max_epoch)

        LogInfo.begin_track('Generating dynamic schemas ...')
        task_dls_dict = {}
        for task_name in eval_tasks:
            task_dls_dict[task_name] = build_task_dataloaders(
                feat_gen=feat_gen, task_name=task_name,
                schema_dataset=schema_dataset, compq_mt_model=compq_mt_model,
                optm_batch_size=args.optm_batch_size, eval_batch_size=args.eval_batch_size,
                neg_f1_ths=args.neg_f1_ths, neg_max_sample=args.neg_max_sample
            )       # [task_optm_dl, task_eval_train_dl, ...]
        el_dl_list = task_dls_dict.get('el')
        rm_dl_list = task_dls_dict.get('rm')
        # full_dl_list = task_dls_dict.get('full')        # these variables could be None
        LogInfo.end_track()

        if not args.test_only:      # won't perform training when just testing
            """ ==== Sub-task optimizing ==== """
            if epoch <= args.pre_train_steps or not full_back_prop:
                # pre-train stage, or always need train & update
                LogInfo.begin_track('Multi-task optimizing ... ')
                optm_schedule_list = []
                if 'el' in optm_tasks:
                    el_optimizer.reset_optm_info()
                    optm_schedule_list += [('el', x) for x in range(el_dl_list[0].n_batch)]
                    LogInfo.logs('[ el]: n_rows = %d, n_batch = %d.', len(el_dl_list[0]), el_dl_list[0].n_batch)
                if 'rm' in optm_tasks:
                    rm_optimizer.reset_optm_info()
                    optm_schedule_list += [('rm', x) for x in range(rm_dl_list[0].n_batch)]
                    LogInfo.logs('[ rm]: n_rows = %d, n_batch = %d.', len(rm_dl_list[0]), rm_dl_list[0].n_batch)
                np.random.shuffle(optm_schedule_list)
                LogInfo.logs('EL & RM task shuffled.')

                for task_name, batch_idx in optm_schedule_list:
                    if task_name == 'el':
                        el_optimizer.optimize(optm_dl=el_dl_list[0], batch_idx=batch_idx)
                    if task_name == 'rm':
                        rm_optimizer.optimize(optm_dl=rm_dl_list[0], batch_idx=batch_idx)

                if 'el' in optm_tasks:
                    LogInfo.logs('[ el] loss = %.6f', el_optimizer.ret_loss)
                    disp_item_dict['el_loss'] = el_optimizer.ret_loss
                if 'rm' in optm_tasks:
                    LogInfo.logs('[ rm] loss = %.6f', rm_optimizer.ret_loss)
                    disp_item_dict['rm_loss'] = rm_optimizer.ret_loss
                LogInfo.end_track()     # End of optm.

        """ ==== Sub-task evluation, if possible ==== """
        if epoch <= args.pre_train_steps or not full_back_prop:
            for task, task_dl_list, evaluator in [
                ('el', el_dl_list, el_evaluator),
                ('rm', rm_dl_list, rm_evaluator)
            ]:
                if task not in eval_tasks:
                    continue
                LogInfo.begin_track('Evaluation for [%s]:', task)
                for mark, eval_dl in zip('Tvt', task_dl_list[1:]):
                    LogInfo.begin_track('Eval-%s ...', mark)
                    disp_key = '%s_%s_F1' % (task, mark)
                    detail_fp = '%s/detail/%s.%s.tmp' % (output_dir, task, mark)    # detail/rm.T.tmp
                    disp_item_dict[disp_key] = evaluator.evaluate_all(eval_dl=eval_dl, detail_fp=detail_fp)
                    LogInfo.end_track()
                LogInfo.end_track()

        # """ ==== full optimization & evaluation, also prepare data for ltr ==== """
        # if epoch > args.pre_train_steps or not full_back_prop:
        #     pyltr_data_list = []  # save T/v/t <q, [cand]> formation for the use of pyltr
        #     if 'full' in eval_tasks:
        #         LogInfo.begin_track('Full-task Optm & Eval:')
        #         if 'full' in optm_tasks:
        #             LogInfo.begin_track('Optimizing ...')
        #             LogInfo.logs('[full]: n_rows = %d, n_batch = %d.', len(full_dl_list[0]), full_dl_list[0].n_batch)
        #             full_optimizer.optimize_all(optm_dl=full_dl_list[0])  # quickly optimize the full model
        #             LogInfo.logs('[full] loss = %.6f', full_optimizer.ret_loss)
        #             disp_item_dict['full_loss'] = full_optimizer.ret_loss
        #             LogInfo.end_track()
        #         for mark, eval_dl in zip('Tvt', full_dl_list[1:]):
        #             LogInfo.begin_track('Eval-%s ...', mark)
        #             disp_key = 'full_%s_F1' % mark
        #             detail_fp = '%s/detail/full.%s.tmp' % (output_dir, mark)
        #             disp_item_dict[disp_key] = full_evaluator.evaluate_all(eval_dl=eval_dl, detail_fp=detail_fp)
        #             pyltr_data_list.append(full_evaluator.ret_q_score_dict)
        #             LogInfo.end_track()
        #         LogInfo.end_track()

        """ ==== LTR optimization & evaluation (non-TF code) ==== """
        # if 'ltr' in eval_tasks:
        #     LogInfo.begin_track('LTR Optm & Eval:')
        #     assert len(pyltr_data_list) == 3
        #     LogInfo.logs('rich_feats_concat collected for all T/v/t schemas.')
        #     LogInfo.begin_track('Ready for ltr running ... ')
        #     ltr_metric_list = ltr_whole_process(pyltr_data_list=pyltr_data_list,
        #                                         eval_dl_list=full_dl_list[1:],
        #                                         output_dir=output_dir)
        #     LogInfo.end_track()
        #     for mark_idx, mark in enumerate(['T', 'v', 't']):
        #         key = 'ltr_%s_F1' % mark
        #         disp_item_dict[key] = ltr_metric_list[mark_idx]
        #     LogInfo.end_track()

        if not args.test_only:
            """ Display & save states (results, details, params) """
            validate_focus = '%s_v_F1' % full_optm_method
            if validate_focus in disp_item_dict:
                cur_valid_f1 = disp_item_dict[validate_focus]
                if cur_valid_f1 > best_valid_f1:
                    best_valid_f1 = cur_valid_f1
                    update_flag = True
                    patience = args.max_patience
                else:
                    patience -= 1
                LogInfo.logs('Model %s, best %s = %.6f [patience = %d]',
                             'updated' if update_flag else 'stayed',
                             validate_focus, cur_valid_f1, patience)
                disp_item_dict['Status'] = 'UPDATE' if update_flag else str(patience)
            else:
                disp_item_dict['Status'] = '------'

            disp_item_dict['Time'] = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
            with open(status_fp, 'a') as bw:
                write_str = ''
                for item_idx, header in enumerate(disp_header_list):
                    if header.endswith(' ') or header == '\t':        # just a split
                        write_str += header
                    else:
                        val = disp_item_dict.get(header, '--------')
                        if isinstance(val, float):
                            write_str += '%8.6f' % val
                        else:
                            write_str += str(val)
                bw.write(write_str + '\n')

            LogInfo.logs('Output concern parameters ... ')
            param_result_list = sess.run(focus_param_list)  # don't need any feeds, since we focus on parameters
            with open(output_dir + '/detail/param.%03d' % epoch, 'w') as bw:
                LogInfo.redirect(bw)
                np.set_printoptions(threshold=np.nan)
                for param_name, param_result in zip(focus_param_name_list, param_result_list):
                    LogInfo.logs('%s: shape = %s ', param_name, param_result.shape)
                    LogInfo.logs(param_result)
                    LogInfo.logs('============================\n')
                np.set_printoptions()
                LogInfo.stop_redirect()

            if update_flag:     # save the latest details
                for mode in 'Tvt':
                    for task in ('rm', 'el', 'full', 'ltr'):
                        src = '%s/detail/%s.%s.tmp' % (output_dir, task, mode)
                        dest = '%s/detail/%s.%s.best' % (output_dir, task, mode)
                        if os.path.isfile(src):
                            shutil.move(src, dest)
                if args.save_best:
                    save_best_dir = '%s/model_best' % output_dir
                    delete_dir(save_best_dir)
                    save_model(saver=saver, sess=sess, model_dir=save_best_dir,
                               epoch=epoch, valid_metric=best_valid_f1)
        else:
            """ Output rank information for all schemas """
            LogInfo.begin_track('Saving overall testing results between Q-words and main paths: ')
            mp_qw_dict = {}
            qw_mp_dict = {}
            qa_list = schema_dataset.qa_list
            count = 0
            for q_idx, sc_list in schema_dataset.smart_q_cand_dict.items():
                qa = qa_list[q_idx]
                raw_tok_list = [tok.token.lower() for tok in qa['tokens']]
                for sc in sc_list:
                    if sc.run_info is None or 'rm_score' not in sc.run_info:
                        continue
                    count += 1
                    score = sc.run_info['rm_score']
                    main_path = sc.main_pred_seq
                    main_path_str = '-->'.join(main_path)
                    main_path_words = sc.path_words_list[0]
                    main_path_words_str = ' | '.join(main_path_words)
                    mp = '[%s] [%s]' % (main_path_str, main_path_words_str)
                    rm_tok_list = RelationMatchingEvaluator.prepare_rm_tok_list(sc=sc, raw_tok_list=raw_tok_list)
                    rm_tok_str = ' '.join(rm_tok_list).replace('<PAD>', '').strip()
                    qw = 'Q-%04d [%s]' % (q_idx, rm_tok_str)
                    mp_qw_dict.setdefault(mp, []).append((qw, score))
                    qw_mp_dict.setdefault(qw, []).append((mp, score))
            LogInfo.logs('%d qw + %d path --> %d pairs ready to save.', len(qw_mp_dict), len(mp_qw_dict), count)

            with codecs.open(output_dir+'/detail/mp_qw_results.txt', 'w', 'utf-8') as bw:
                mp_list = sorted(mp_qw_dict.keys())
                for mp in mp_list:
                    qw_score_tups = mp_qw_dict[mp]
                    bw.write(mp + '\n')
                    qw_score_tups.sort(key=lambda _tup: _tup[-1], reverse=True)
                    for rank_idx, (qw, score) in enumerate(qw_score_tups[:100]):
                        bw.write('  Rank=%04d    score=%9.6f    %s\n' % (rank_idx+1, score, qw))
                    bw.write('\n===================================\n\n')
            LogInfo.logs('<path, qwords, score> saved.')

            with codecs.open(output_dir+'/detail/qw_mp_results.txt', 'w', 'utf-8') as bw:
                qw_list = sorted(qw_mp_dict.keys())
                for qw in qw_list:
                    mp_score_tups = qw_mp_dict[qw]
                    bw.write(qw + '\n')
                    mp_score_tups.sort(key=lambda _tup: _tup[-1], reverse=True)
                    for rank_idx, (mp, score) in enumerate(mp_score_tups[:100]):
                        bw.write('  Rank=%04d    score=%9.6f    %s\n' % (rank_idx+1, score, mp))
                    bw.write('\n===================================\n\n')
            LogInfo.logs('<qwords, path, score> saved.')

            LogInfo.end_track()

            LogInfo.end_track()         # jump out of the epoch iteration
            break       # test-only mode, no need to perform more things.

        LogInfo.end_track()     # end of epoch
    LogInfo.end_track()         # end of learning


if __name__ == '__main__':
    LogInfo.begin_track('[kangqi.task.compQA.main_acl18] running ...')
    _args = parser.parse_args()
    main(_args)
    LogInfo.end_track('All Done.')
