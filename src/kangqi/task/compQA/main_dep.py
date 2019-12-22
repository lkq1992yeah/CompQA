import os
import shutil
import argparse
import numpy as np
import tensorflow as tf
from ast import literal_eval
from datetime import datetime

from dataset.dataset_dep import SchemaDatasetDep
from dataset.dl_compq_e2e.dep_schema_dl_builder import DepSchemaDLBuilder
from model.compq_dep.compq_mt_model import CompqMultiTaskModel

from learner.compq_e2e.e2e_evaluator import BaseEvaluator
from learner.compq_e2e.e2e_optimizer import BaseOptimizer
from learner.compq_e2e.el_disp_helper import show_el_detail_without_type
from learner.compq_e2e.rm_disp_helper import show_basic_rm_info
from learner.compq_e2e.full_disp_helper import show_basic_full_info

from util.word_emb import WordEmbeddingUtil
from .u import delete_dir, save_model, load_model, construct_display_header

from kangqi.util.LogUtil import LogInfo


working_ip = '202.120.38.146'
parser_port_dict = {'Blackhole': 9601, 'Darkstar': 8601}
sparql_port_dict = {'Blackhole': 8999, 'Darkstar': 8699}

parser = argparse.ArgumentParser(description='QA Model Training')

parser.add_argument('--machine', default='Blackhole', choices=['Blackhole', 'Darkstar'])
parser.add_argument('--word_emb', default='glove')
parser.add_argument('--dim_emb', type=int, default=300, help='word/predicate embedding dimension')
parser.add_argument('--data_config', help='dataset config')

parser.add_argument('--dep_simulate', default='False')

parser.add_argument('--neg_pick_config', help='negative sampling config')
# parser.add_argument('--neg_f1_ths', default=0.1, type=float)
# parser.add_argument('--neg_max_sample', default=20, type=int)

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
        optm_tasks = ['el', 'rm', 'full']
        eval_tasks = ['el', 'rm', 'full']
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
    LogInfo.end_track()

    # ==== Loading Dataset ==== #
    LogInfo.begin_track('Creating Dataset ... ')
    data_config = literal_eval(args.data_config)
    data_config['el_feat_size'] = 3
    data_config['extra_feat_size'] = 16
    data_config['wd_emb_util'] = wd_emb_util
    data_config['verbose'] = args.verbose
    schema_dataset = SchemaDatasetDep(**data_config)
    schema_dataset.load_all_data()
    """ load data before constructing model, as we generate lookup dict in the loading phase """
    LogInfo.end_track()

    # ==== Building Model ==== #
    LogInfo.begin_track('Building Model and Session ... ')
    gpu_options = tf.GPUOptions(allow_growth=True,
                                per_process_gpu_memory_fraction=args.gpu_fraction)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options,
                                            intra_op_parallelism_threads=8))
    model_config = literal_eval(args.model_config)
    for key in ('qw_max_len', 'pw_max_len', 'path_max_size', 'pseq_max_len',
                'el_feat_size', 'extra_feat_size'):
        model_config[key] = getattr(schema_dataset, key)
    model_config['n_words'] = len(schema_dataset.active_dicts['word'])
    model_config['n_paths'] = len(schema_dataset.active_dicts['path'])
    model_config['n_mids'] = len(schema_dataset.active_dicts['mid'])
    model_config['dim_emb'] = wd_emb_util.dim_emb
    full_back_prop = model_config['full_back_prop']
    compq_mt_model = CompqMultiTaskModel(**model_config)

    LogInfo.begin_track('Showing final parameters: ')
    for var in tf.global_variables():
        LogInfo.logs('%s: %s', var.name, var.get_shape().as_list())
    LogInfo.end_track()

    attempt_param_list = ['rm_task/rm_final_merge/sent_repr/out_fc/weights',
                          'rm_task/rm_final_merge/sent_repr/out_fc/biases',
                          'rm_task/rm_final_merge/alpha',
                          'el_task/out_fc/weights', 'el_task/out_fc/biases',
                          'full_task/out_fc/weights', 'full_task/out_fc/biases']
    focus_param_list = []
    focus_param_name_list = []
    with tf.variable_scope('', reuse=tf.AUTO_REUSE):
        for param_name in attempt_param_list:
            try:
                var = tf.get_variable(name=param_name)
                focus_param_list.append(var)
                focus_param_name_list.append(param_name)
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
        dep_simulate = True if args.dep_simulate == 'True' else False
        wd_emb_mat = wd_emb_util.produce_active_word_embedding(
            active_word_dict=schema_dataset.active_dicts['word'],
            dep_simulate=dep_simulate
        )
        pa_emb_mat = np.random.uniform(low=-0.1, high=0.1,
                                       size=(model_config['n_paths'], model_config['dim_emb'])).astype('float32')
        mid_emb_mat = np.random.uniform(low=-0.1, high=0.1,
                                        size=(model_config['n_mids'], model_config['dim_emb'])).astype('float32')
        LogInfo.logs('%s random path embedding created.', pa_emb_mat.shape)
        LogInfo.logs('%s random mid embedding created.', mid_emb_mat.shape)
        sess.run(tf.global_variables_initializer(),
                 feed_dict={compq_mt_model.w_embedding_init: wd_emb_mat,
                            compq_mt_model.p_embedding_init: pa_emb_mat,
                            compq_mt_model.m_embedding_init: mid_emb_mat})
    LogInfo.end_track('Start Epoch = %d', start_epoch)
    LogInfo.end_track('Model build complete.')

    # ==== Register optm / eval ==== #
    rm_optimizer = BaseOptimizer(task_name='rm', compq_mt_model=compq_mt_model, sess=sess)
    el_optimizer = BaseOptimizer(task_name='el', compq_mt_model=compq_mt_model, sess=sess)
    full_optimizer = BaseOptimizer(task_name='full', compq_mt_model=compq_mt_model, sess=sess)
    rm_evaluator = BaseEvaluator(task_name='rm', compq_mt_model=compq_mt_model,
                                 sess=sess, detail_disp_func=show_basic_rm_info)
    el_evaluator = BaseEvaluator(task_name='el', compq_mt_model=compq_mt_model,
                                 sess=sess, detail_disp_func=show_el_detail_without_type)
    full_evaluator = BaseEvaluator(task_name='full', compq_mt_model=compq_mt_model,
                                   sess=sess, detail_disp_func=show_basic_full_info)
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
    disp_header_list = construct_display_header(optm_tasks=optm_tasks, eval_tasks=eval_tasks)
    with open(status_fp, 'a') as bw:
        write_str = ''.join(disp_header_list)
        bw.write(write_str + '\n')

    if full_back_prop:
        LogInfo.logs('full_back_prop = %s, pre_train_steps = %d.', full_back_prop, args.pre_train_steps)
    else:
        LogInfo.logs('no pre-train available.')

    sent_usage = model_config['sent_usage']
    dep_or_cp = 'dep'
    if sent_usage.startswith('cp'):
        dep_or_cp = 'cp'
    dl_builder = DepSchemaDLBuilder(schema_dataset=schema_dataset, compq_mt_model=compq_mt_model,
                                    neg_pick_config=literal_eval(args.neg_pick_config),
                                    parser_port=parser_port_dict[args.machine],
                                    dep_or_cp=dep_or_cp)
    for epoch in range(start_epoch+1, args.max_epoch+1):
        if patience == 0:
            LogInfo.logs('Early stopping at epoch = %d.', epoch)
            break
        update_flag = False
        disp_item_dict = {'Epoch': epoch}

        LogInfo.begin_track('Epoch %d / %d', epoch, args.max_epoch)

        LogInfo.begin_track('Generating schemas ...')
        task_dls_dict = {}
        for task_name in eval_tasks:
            task_dls_dict[task_name] = dl_builder.build_task_dataloaders(
                task_name=task_name,
                optm_batch_size=args.optm_batch_size,
                eval_batch_size=args.eval_batch_size
            )
            # [task_optm_dl, task_eval_train_dl, ...]
        el_dl_list = task_dls_dict.get('el')
        rm_dl_list = task_dls_dict.get('rm')
        full_dl_list = task_dls_dict.get('full')        # these variables could be None
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
                    result_fp = '%s/result/%s.%s.%03d' % (output_dir, task, mark, epoch)    # result/rm.T.001
                    disp_item_dict[disp_key] = evaluator.evaluate_all(
                        eval_dl=eval_dl,
                        detail_fp=detail_fp,
                        result_fp=result_fp
                    )
                    LogInfo.end_track()
                LogInfo.end_track()

        """ ==== full optimization & evaluation, also prepare data for ltr ==== """
        if epoch > args.pre_train_steps or not full_back_prop:
            # pyltr_data_list = []  # save T/v/t <q, [cand]> formation for the use of pyltr
            if 'full' in eval_tasks:
                LogInfo.begin_track('Full-task Optm & Eval:')
                if 'full' in optm_tasks and not args.test_only:
                    LogInfo.begin_track('Optimizing ...')
                    LogInfo.logs('[full]: n_rows = %d, n_batch = %d.', len(full_dl_list[0]), full_dl_list[0].n_batch)
                    full_optimizer.optimize_all(optm_dl=full_dl_list[0])  # quickly optimize the full model
                    LogInfo.logs('[full] loss = %.6f', full_optimizer.ret_loss)
                    disp_item_dict['full_loss'] = full_optimizer.ret_loss
                    LogInfo.end_track()
                for mark, eval_dl in zip('Tvt', full_dl_list[1:]):
                    LogInfo.begin_track('Eval-%s ...', mark)
                    disp_key = 'full_%s_F1' % mark
                    detail_fp = '%s/detail/full.%s.tmp' % (output_dir, mark)
                    result_fp = '%s/result/full.%s.%03d' % (output_dir, mark, epoch)    # result/full.T.001
                    disp_item_dict[disp_key] = full_evaluator.evaluate_all(
                        eval_dl=eval_dl,
                        detail_fp=detail_fp,
                        result_fp=result_fp
                    )
                    # pyltr_data_list.append(full_evaluator.ret_q_score_dict)
                    LogInfo.end_track()
                LogInfo.end_track()

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

        LogInfo.end_track()     # end of epoch
    LogInfo.end_track()         # end of learning


if __name__ == '__main__':
    LogInfo.begin_track('[kangqi.task.compQA.main_dep] running ...')
    _args = parser.parse_args()
    main(_args)
    LogInfo.end_track('All Done.')
