# -*- coding:utf-8 -*-

"""
Author: Kangqi Luo
Goal: Main program of complex QA
** DEPRECATED ** (Used before 17/12/01)
"""

import os
import time
import shutil
import argparse
import tensorflow as tf
from ast import literal_eval
from datetime import datetime

import dataset as ds
from dataset.xy_dataset import QScDataset

import model as md

from util.word_emb import WordEmbeddingUtil
from util.kb_emb import KBEmbeddingUtil
from util.fb_helper import FreebaseHelper

from kangqi.util.LogUtil import LogInfo


parser = argparse.ArgumentParser(description='QA Model Training')
parser.add_argument('--webq_fp',
                    help='webquestions file',
                    default='/home/kangqi/Webquestions/Json/webquestions.examples.json')

parser.add_argument('--machine', default='Blackhole', choices=['Blackhole', 'Darkstar'])
parser.add_argument('--word_emb', default='glove')
parser.add_argument('--kb_emb_dir', default='data/kb_emb/FB2M', help='directory of knowledge base embedding')
parser.add_argument('--fb_meta_dir', default='data/fb_metadata', help='directory of Freebase metadata')
parser.add_argument('--data_config', help='dataset config')

parser.add_argument('--eval_dataloader', default='QScEvalDataLoader', choices=ds.__eval_all__,
                    help='DataLoader architecture: ' + ' | '.join(ds.__eval_all__) + ' (default: QScEvalDataLoader)')
parser.add_argument('--optm_dataloader', default='QScPairDataLoader', choices=ds.__optm_all__,
                    help='DataLoader architecture: ' + ' | '.join(ds.__optm_all__) + ' (default: QScPairDataLoader)')
parser.add_argument('--sampling_config', help='configuration of positive / negative data sampling')

parser.add_argument('--optm_model', default='QScHingeModel', choices=md.__optm_all__,
                    help='Model architecture: ' + ' | '.join(md.__optm_all__) + ' (default: QScHingeModel)')
parser.add_argument('--eval_model', default='QScDynamicEvalModel', choices=md.__eval_all__,
                    help='Model architecture: ' + ' | '.join(md.__eval_all__) + ' (default: QScDynamicEvalModel)')
parser.add_argument('--model_config', help='model config')
parser.add_argument('--optm_config', help='optimization config')
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
    eval_model_config = literal_eval(args.model_config)

    # ==== Loading Necessary Util ==== #
    LogInfo.begin_track('Loading Utils ... ')
    # with open(args.webq_fp, 'r') as br:
    #     webq_data = json.load(br)
    #     webq_list = [webq['utterance'] for webq in webq_data]
    # LogInfo.logs('%d questions loaded from [%s].', len(webq_list), args.webq_fp)
    parser_ip = '202.120.38.146'
    parser_port = 9601 if args.machine == 'Blackhole' else 8601
    wd_emb_util = WordEmbeddingUtil(wd_emb=args.word_emb, dim_wd_emb=eval_model_config['dim_wd_emb'],
                                    parser_ip=parser_ip, parser_port=parser_port)
    kb_emb_util = KBEmbeddingUtil(kb_emb_dir=args.kb_emb_dir, dim_kb_emb=eval_model_config['dim_kb_emb'])
    fb_helper = FreebaseHelper(args.fb_meta_dir)
    LogInfo.end_track()

    # ==== Loading Dataset ==== #
    data_config = literal_eval(args.data_config)
    data_config['wd_emb_util'] = wd_emb_util
    data_config['kb_emb_util'] = kb_emb_util
    data_config['fb_helper'] = fb_helper
    data_config['verbose'] = args.verbose
    data_config.setdefault('protocol_name', 'XY')
    for spec in ('q_max_len', 'sc_max_len', 'path_max_len', 'item_max_len'):
        data_config[spec] = eval_model_config[spec]
    dataset = QScDataset(**data_config)
    dataset.load_size()     # load size info

    # ==== Build Model First ==== #
    LogInfo.begin_track('Building Model and Session ... ')
    gpu_options = tf.GPUOptions(allow_growth=True,
                                per_process_gpu_memory_fraction=args.gpu_fraction)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options,
                                            intra_op_parallelism_threads=8))
    eval_model_config['sess'] = sess
    eval_model_config['n_words'] = dataset.w_size
    eval_model_config['n_entities'] = dataset.e_size
    eval_model_config['n_preds'] = dataset.p_size
    eval_model_config['verbose'] = args.verbose
    eval_model = getattr(md, args.eval_model)(**eval_model_config)

    optm_model_config = literal_eval(args.optm_config)
    for k, v in eval_model_config.items():
        optm_model_config[k] = v
    optm_model = getattr(md, args.optm_model)(**optm_model_config)

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
        dataset.load_init_emb()     # loading parameters for embedding initialize
        # wd_emb_util.load_embeddings()
        # kb_emb_util.load_entity_embeddings()
        # kb_emb_util.load_predicate_embeddings()
        # w_emb = np.zeros((dataset.w_size, wd_emb_util.dim_wd_emb), dtype='float32')
        # e_emb = np.zeros((dataset.e_size, kb_emb_util.dim_kb_emb), dtype='float32')
        # p_emb = np.zeros((dataset.p_size, kb_emb_util.dim_kb_emb), dtype='float32')
        # w_emb[:wd_emb_util.n_words] = wd_emb_util.emb_matrix
        # e_emb[:kb_emb_util.n_entities] = kb_emb_util.e_emb_matrix
        # p_emb[:kb_emb_util.n_preds] = kb_emb_util.p_emb_matrix
        # LogInfo.logs('w_emb: %d --> %d initialized.', wd_emb_util.n_words, dataset.w_size)
        # LogInfo.logs('e_emb: %d --> %d initialized.', kb_emb_util.n_entities, dataset.e_size)
        # LogInfo.logs('p_emb: %d --> %d initialized.', kb_emb_util.n_preds, dataset.p_size)
        LogInfo.logs('Running global_variables_initializer ...')
        sess.run(tf.global_variables_initializer(),
                 feed_dict={eval_model.w_embedding_init: dataset.w_init_emb,
                            eval_model.e_embedding_init: dataset.e_init_emb,
                            eval_model.p_embedding_init: dataset.p_init_emb})

    LogInfo.end_track('Start Epoch = %d', start_epoch)
    LogInfo.end_track()

    # ==== Constructing Data_Loader ==== #
    LogInfo.begin_track('Creating DataLoader ... ')

    dataset.load_cands()    # first loading all the candidates
    loader_config = {k: eval_model_config[k] for k in ('q_max_len', 'sc_max_len', 'path_max_len', 'item_max_len')}
    loader_config['dataset'] = dataset
    loader_config['verbose'] = args.verbose

    optm_train_loader_config = dict(loader_config)
    optm_train_loader_config['mode'] = 'train'
    optm_train_loader_config['sampling_config'] = literal_eval(args.sampling_config)
    optm_train_loader_config['batch_size'] = args.optm_batch_size
    optm_train_data = getattr(ds, args.optm_dataloader)(**optm_train_loader_config)

    eval_loader_config = dict(loader_config)
    eval_loader_config['batch_size'] = args.eval_batch_size
    eval_data_group = []
    for mark in ('train', 'valid', 'test'):
        eval_loader_config['mode'] = mark
        eval_data = getattr(ds, args.eval_dataloader)(**eval_loader_config)
        eval_data.renew_data_list()
        eval_data_group.append(eval_data)
    (eval_train_data, eval_valid_data, eval_test_data) = eval_data_group

    LogInfo.end_track()     # End of loading data & dataset

    # ==== Free memories ==== #
    del wd_emb_util
    del kb_emb_util
    del fb_helper
    del data_config
    del dataset.wd_emb_util
    del dataset.kb_emb_util

    # ==== Ready for learning ==== #
    LogInfo.begin_track('Learning start ... ')
    output_dir = args.output_dir
    if not os.path.exists(output_dir + '/detail'):
        os.makedirs(output_dir + '/detail')
    if os.path.isdir(output_dir + '/TB'):
        shutil.rmtree(output_dir + '/TB')
    optm_summary_writer = tf.summary.FileWriter(output_dir + '/TB/optm', sess.graph)
    eval_train_summary_writer = tf.summary.FileWriter(output_dir + '/TB/eval_train', sess.graph)
    eval_valid_summary_writer = tf.summary.FileWriter(output_dir + '/TB/eval_valid', sess.graph)
    eval_test_summary_writer = tf.summary.FileWriter(output_dir + '/TB/eval_test', sess.graph)
    LogInfo.logs('TensorBoard writer defined.')
    # TensorBoard imformation

    status_fp = output_dir + '/status.csv'
    with open(status_fp, 'a') as bw:
        bw.write('%s\t%s\t%s\t%s\t%s\t%s\t%s\n' % (
            'Epoch', 'T_loss', 'T_F1', 'v_F1', 'Status', 't_F1',
            datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
        ))
    patience = args.max_patience

    for epoch in range(start_epoch+1, args.max_epoch+1):
        if patience == 0:
            LogInfo.logs('Early stopping at epoch = %d.', epoch)
            break

        LogInfo.begin_track('Epoch %d / %d', epoch, args.max_epoch)
        update_flag = False

        LogInfo.begin_track('Optimizing ...')
        train_loss = optm_model.optimize(optm_train_data, epoch, optm_summary_writer)
        LogInfo.end_track('T_loss = %.6f', train_loss)

        LogInfo.begin_track('Eval-Training ...')
        train_f1 = eval_model.evaluate(eval_train_data, epoch,
                                       detail_fp=output_dir+'/detail/train_%03d.txt' % epoch,
                                       summary_writer=eval_train_summary_writer)
        LogInfo.end_track('T_F1 = %.6f', train_f1)

        LogInfo.begin_track('Eval-Validating ...')
        valid_f1 = eval_model.evaluate(eval_valid_data, epoch,
                                       detail_fp=output_dir+'/detail/valid_%03d.txt' % epoch,
                                       summary_writer=eval_valid_summary_writer)
        LogInfo.logs('v_F1 = %.6f', valid_f1)
        if valid_f1 > best_valid_f1:
            best_valid_f1 = valid_f1
            update_flag = True
            patience = args.max_patience
        else:
            patience -= 1
        LogInfo.logs('Model %s, best v_F1 = %.6f [patience = %d]',
                     'updated' if update_flag else 'stayed', valid_f1, patience)
        LogInfo.end_track()

        LogInfo.begin_track('Eval-Testing ... ')
        test_f1 = eval_model.evaluate(eval_test_data, epoch,
                                      detail_fp=output_dir + '/detail/test_%03d.txt' % epoch,
                                      summary_writer=eval_test_summary_writer)
        LogInfo.end_track('t_F1 = %.6f', test_f1)

        with open(status_fp, 'a') as bw:
            bw.write('%d\t%.6f\t%.6f\t%.6f\t%s\t%.6f\t%s\n' % (
                epoch, train_loss, train_f1, valid_f1,
                'UPDATE' if update_flag else str(patience), test_f1,
                datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
            ))
        save_epoch_dir = '%s/model_epoch_%d' % (output_dir, epoch)
        save_best_dir = '%s/model_best' % output_dir
        if args.save_epoch:
            delete_dir(save_epoch_dir)
            save_model(saver=saver, sess=sess, model_dir=save_epoch_dir, epoch=epoch, valid_metric=valid_f1)
            if update_flag and args.save_best:                  # just create a symbolic link
                delete_dir(save_best_dir)
                os.symlink(save_epoch_dir, save_best_dir)       # symlink at directory level
        elif update_flag and args.save_best:
            delete_dir(save_best_dir)
            save_model(saver=saver, sess=sess, model_dir=save_best_dir, epoch=epoch, valid_metric=valid_f1)

        LogInfo.end_track()         # End of epoch
    LogInfo.end_track()             # End of learning

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
