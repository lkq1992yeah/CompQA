"""
Author: Kangqi Luo
Goal: Main entrance of the complex QA experiments.
Put into use in 17/12/20, we leverage the new schema representation, and change models
"""

import os
import time
import shutil
import argparse
import tensorflow as tf
from ast import literal_eval
from datetime import datetime

from dataset.dataset import QScDataset
from dataset.kq_schema_helper import add_relation_only_metric
from dataset.dl_compq import CompqPairDataLoader, CompqSingleDataLoader

from model.compq_overall import EntityLinkingKernel, CompqModel
from model.compq_overall import SeparatedRelationMatchingKernel, CompactRelationMatchingKernel

from util.word_emb import WordEmbeddingUtil

from kangqi.util.LogUtil import LogInfo
from kangqi.util.time_track import TimeTracker as Tt


working_ip = '202.120.38.146'
parser_port_dict = {'Blackhole': 9601, 'Darkstar': 8601}
sparql_port_dict = {'Blackhole': 8999, 'Darkstar': 8699}

parser = argparse.ArgumentParser(description='QA Model Training')

parser.add_argument('--machine', default='Blackhole', choices=['Blackhole', 'Darkstar'])
parser.add_argument('--eval_mode', default='normal', choices=['normal', 'relation_only'])
parser.add_argument('--word_emb', default='glove')
parser.add_argument('--fb_meta_dir', default='data/fb_metadata', help='directory of Freebase metadata')
parser.add_argument('--dim_emb', type=int, default=300, help='word/predicate embedding dimension')
parser.add_argument('--data_config', help='dataset config')
parser.add_argument('--dl_neg_mode', help='negative sampling mode of DataLoader')

parser.add_argument('--rm_config', help='relation matching config')
parser.add_argument('--model_config', help='general model config')
parser.add_argument('--resume_model_name', default='None',
                    help='the directory name of the model which you wan to resume learning')

parser.add_argument('--optm_batch_size', type=int, default=128, help='optm_batch size')
parser.add_argument('--eval_batch_size', type=int, default=32, help='eval_batch size')
parser.add_argument('--max_epoch', type=int, default=30, help='max epochs')
parser.add_argument('--max_patience', type=int, default=10000, help='max patience')

parser.add_argument('--output_dir', help='output dir, including results, models and others')
parser.add_argument('--save_epoch', help='save model after each epoch', action='store_true')
parser.add_argument('--save_best', help='save best model only', action='store_true')

parser.add_argument('--gpu_fraction', type=float, default=0.25, help='GPU fraction limit')
parser.add_argument('--verbose', type=int, default=0, help='verbose level')


def main(args):
    LogInfo.begin_track('Learning starts ... ')

    # ==== Loading Necessary Util ==== #
    LogInfo.begin_track('Loading Utils ... ')
    wd_emb_util = WordEmbeddingUtil(wd_emb=args.word_emb, dim_emb=args.dim_emb)
    LogInfo.end_track()

    # ==== Loading Dataset ==== #
    data_config = literal_eval(args.data_config)    # including data_name, dir, max_length and others
    data_config['wd_emb_util'] = wd_emb_util
    # data_config['kb_emb_util'] = kb_emb_util
    data_config['verbose'] = args.verbose
    dataset = QScDataset(**data_config)
    dataset.load_size()  # load size info

    # ==== Build Model First ==== #
    LogInfo.begin_track('Building Model and Session ... ')
    gpu_options = tf.GPUOptions(allow_growth=True,
                                per_process_gpu_memory_fraction=args.gpu_fraction)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options,
                                            intra_op_parallelism_threads=8))
    rm_config = literal_eval(args.rm_config)    # Relation Matching
    rm_name = rm_config['name']
    del rm_config['name']
    assert rm_name in ('Compact', 'Separated')

    rm_config['n_words'] = dataset.word_size
    rm_config['n_mids'] = dataset.mid_size
    rm_config['dim_emb'] = args.dim_emb
    rm_config['q_max_len'] = dataset.q_max_len
    rm_config['sc_max_len'] = dataset.sc_max_len
    rm_config['path_max_len'] = dataset.path_max_len
    rm_config['pword_max_len'] = dataset.path_max_len * dataset.item_max_len
    rm_config['verbose'] = args.verbose
    if rm_name == 'Compact':
        LogInfo.logs('RelationMatchingKernel: Compact')
        rm_kernel = CompactRelationMatchingKernel(**rm_config)
    else:
        LogInfo.logs('RelationMatchingKernel: Separated')
        rm_kernel = SeparatedRelationMatchingKernel(**rm_config)
    el_kernel = EntityLinkingKernel(
        e_max_size=dataset.e_max_size, e_feat_len=dataset.e_feat_len, verbose=args.verbose)

    model_config = literal_eval(args.model_config)
    model_config['sess'] = sess
    model_config['objective'] = args.eval_mode      # relation_only / normal
    model_config['relation_kernel'] = rm_kernel
    model_config['entity_kernel'] = el_kernel
    model_config['extra_len'] = dataset.extra_len
    model_config['verbose'] = args.verbose
    compq_model = CompqModel(**model_config)

    LogInfo.begin_track('Showing final parameters: ')
    for var in tf.global_variables():
        LogInfo.logs('%s: %s', var.name, var.get_shape().as_list())
    LogInfo.end_track()
    saver = tf.train.Saver()

    LogInfo.begin_track('Parameter initializing ... ')
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
        dataset.load_init_emb()  # loading parameters for embedding initialize
        LogInfo.logs('Running global_variables_initializer ...')
        sess.run(tf.global_variables_initializer(),
                 feed_dict={rm_kernel.w_embedding_init: dataset.word_init_emb,
                            rm_kernel.m_embedding_init: dataset.mid_init_emb})
    LogInfo.end_track('Start Epoch = %d', start_epoch)
    LogInfo.end_track('Model Built.')
    tf.get_default_graph().finalize()

    # ==== Constructing Data_Loader ==== #
    LogInfo.begin_track('Creating DataLoader ... ')
    dataset.load_cands()  # first loading all the candidates
    if args.eval_mode == 'relation_only':
        ro_change = 0
        for cand_list in dataset.q_cand_dict.values():
            ro_change += add_relation_only_metric(cand_list)    # for "RelationOnly" evaluation
        LogInfo.logs('RelationOnly F1 change: %d schemas affected.', ro_change)

    optm_dl_config = {'dataset': dataset, 'mode': 'train',
                      'batch_size': args.optm_batch_size, 'proc_ob_num': 5000, 'verbose': args.verbose}
    eval_dl_config = dict(optm_dl_config)
    spt = args.dl_neg_mode.split('-')       # Neg-${POOR_CONTRIB}-${POOR_MAX_SAMPLE}
    optm_dl_config['poor_contribution'] = int(spt[1])
    optm_dl_config['poor_max_sample'] = int(spt[2])
    optm_dl_config['shuffle'] = False
    optm_train_data = CompqPairDataLoader(**optm_dl_config)

    eval_dl_config['batch_size'] = args.eval_batch_size
    eval_data_group = []
    for mode in ('train', 'valid', 'test'):
        eval_dl_config['mode'] = mode
        eval_data = CompqSingleDataLoader(**eval_dl_config)
        eval_data.renew_data_list()
        eval_data_group.append(eval_data)
    (eval_train_data, eval_valid_data, eval_test_data) = eval_data_group
    LogInfo.end_track()  # End of loading data & dataset

    # ==== Free memories ==== #
    for item in (wd_emb_util, dataset.wd_emb_util, data_config):
        del item

    # ==== Ready for learning ==== #
    LogInfo.begin_track('Learning start ... ')
    output_dir = args.output_dir
    if not os.path.exists(output_dir + '/detail'):
        os.makedirs(output_dir + '/detail')
    if not os.path.exists(output_dir + '/result'):
        os.makedirs(output_dir + '/result')
    if os.path.isdir(output_dir + '/TB'):
        shutil.rmtree(output_dir + '/TB')
    tf.summary.FileWriter(output_dir + '/TB/optm', sess.graph)      # saving model graph information
    # optm_summary_writer = tf.summary.FileWriter(output_dir + '/TB/optm', sess.graph)
    # eval_train_summary_writer = tf.summary.FileWriter(output_dir + '/TB/eval_train', sess.graph)
    # eval_valid_summary_writer = tf.summary.FileWriter(output_dir + '/TB/eval_valid', sess.graph)
    # eval_test_summary_writer = tf.summary.FileWriter(output_dir + '/TB/eval_test', sess.graph)
    # LogInfo.logs('TensorBoard writer defined.')
    # TensorBoard imformation

    status_fp = output_dir + '/status.csv'
    with open(status_fp, 'a') as bw:
        bw.write('%s\t%s\t%s\t%s\t%s\t%s\t%s\n' % (
            'Epoch', 'T_loss', 'T_F1', 'v_F1', 'Status', 't_F1',
            datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
        ))
    patience = args.max_patience

    for epoch in range(start_epoch + 1, args.max_epoch + 1):
        if patience == 0:
            LogInfo.logs('Early stopping at epoch = %d.', epoch)
            break

        LogInfo.begin_track('Epoch %d / %d', epoch, args.max_epoch)
        update_flag = False

        LogInfo.begin_track('Optimizing ...')
        train_loss = compq_model.optimize(optm_train_data, epoch, ob_batch_num=1)
        LogInfo.end_track('T_loss = %.6f', train_loss)

        LogInfo.begin_track('Eval-Training ...')
        train_f1 = compq_model.evaluate(eval_train_data, epoch, ob_batch_num=50,
                                        detail_fp=output_dir + '/detail/train_%03d.txt' % epoch)
        LogInfo.end_track('T_F1 = %.6f', train_f1)

        LogInfo.begin_track('Eval-Validating ...')
        valid_f1 = compq_model.evaluate(eval_valid_data, epoch, ob_batch_num=50,
                                        detail_fp=output_dir + '/detail/valid_%03d.txt' % epoch)
        LogInfo.logs('v_F1 = %.6f', valid_f1)
        if valid_f1 > best_valid_f1:
            best_valid_f1 = valid_f1
            update_flag = True
            patience = args.max_patience
        else:
            patience -= 1
        LogInfo.logs('Model %s, best v_F1 = %.6f [patience = %d]',
                     'updated' if update_flag else 'stayed',
                     valid_f1,
                     patience)
        LogInfo.end_track()

        LogInfo.begin_track('Eval-Testing ... ')
        test_f1 = compq_model.evaluate(eval_test_data, epoch, ob_batch_num=20,
                                       detail_fp=output_dir + '/detail/test_%03d.txt' % epoch,
                                       result_fp=output_dir + '/result/test_schema_%03d.txt' % epoch)
        LogInfo.end_track('t_F1 = %.6f', test_f1)

        with open(status_fp, 'a') as bw:
            bw.write('%d\t%8.6f\t%8.6f\t%8.6f\t%s\t%8.6f\t%s\n' % (
                epoch, train_loss, train_f1, valid_f1,
                'UPDATE' if update_flag else str(patience), test_f1,
                datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
            ))
        save_epoch_dir = '%s/model_epoch_%d' % (output_dir, epoch)
        save_best_dir = '%s/model_best' % output_dir
        if args.save_epoch:
            delete_dir(save_epoch_dir)
            save_model(saver=saver, sess=sess, model_dir=save_epoch_dir, epoch=epoch, valid_metric=valid_f1)
            if update_flag and args.save_best:  # just create a symbolic link
                delete_dir(save_best_dir)
                os.symlink(save_epoch_dir, save_best_dir)  # symlink at directory level
        elif update_flag and args.save_best:
            delete_dir(save_best_dir)
            save_model(saver=saver, sess=sess, model_dir=save_best_dir, epoch=epoch, valid_metric=valid_f1)

        LogInfo.end_track()  # End of epoch
    LogInfo.end_track()  # End of learning

    Tt.display()
    LogInfo.end_track('All Done.')


def delete_dir(target_dir):
    if os.path.islink(target_dir):
        os.remove(target_dir)
    elif os.path.isdir(target_dir):
        shutil.rmtree(target_dir)


def save_model(saver, sess, model_dir, epoch, valid_metric):
    t0 = time.time()
    LogInfo.logs('Saving model into [%s] ...', model_dir)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_fp = model_dir + '/model'
    saver.save(sess, model_fp)
    LogInfo.logs('Saved [%.3fs]', time.time() - t0)

    with open(model_dir + '/epoch', 'w') as bw:
        bw.write('%d\n' % epoch)
        bw.write('%.6f\n' % valid_metric)


def load_model(saver, sess, model_dir):
    t0 = time.time()
    LogInfo.logs('Loading model from [%s] ...', model_dir)
    model_fp = model_dir + '/model'
    saver.restore(sess, model_fp)
    LogInfo.logs('Loaded [%.3fs]', time.time() - t0)

    start_epoch = 0
    valid_metric = 0.
    epoch_nb_fp = model_dir + '/epoch'
    if os.path.isfile(epoch_nb_fp):
        with open(epoch_nb_fp, 'r') as br:
            start_epoch = int(br.readline().strip())
            valid_metric = float(br.readline().strip())
    return start_epoch, valid_metric


if __name__ == '__main__':
    _args = parser.parse_args()
    main(_args)
