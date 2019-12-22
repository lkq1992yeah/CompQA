"""
Author: Kangqi Luo
Goal: Main entrance of the yu2017improved experiments.
Relation Detection Only
"""

import os
import shutil
import argparse
import tensorflow as tf
from ast import literal_eval
from datetime import datetime

# import dataset as ds
# import model as md
from dataset.xy_dataset import QScDataset
from dataset import SimpqSingleDataLoader, SimpqPairDataLoader
from model.simpq_rel_detect_yu2017 import SimpqEvalModel, SimpqOptmModel

from main_old import load_model, save_model, delete_dir

from util.word_emb import WordEmbeddingUtil
from util.kb_emb import KBEmbeddingUtil
from util.fb_helper import FreebaseHelper

from kangqi.util.LogUtil import LogInfo

parser = argparse.ArgumentParser(description='QA Model Training')

# parser.add_argument('--machine', default='Blackhole', choices=['Blackhole', 'Darkstar'])
parser.add_argument('--word_emb', default='glove')
parser.add_argument('--kb_emb_dir', default='data/kb_emb/FB2M', help='directory of knowledge base embedding')
parser.add_argument('--fb_meta_dir', default='data/fb_metadata', help='directory of Freebase metadata')
parser.add_argument('--dim_emb', type=int, default=300, help='word/predicate embedding dimension')
parser.add_argument('--data_config', help='dataset config')

# parser.add_argument('--eval_dataloader', default='QScEvalDataLoader', choices=ds.__eval_all__,
#                     help='DataLoader architecture: ' + ' | '.join(ds.__eval_all__) + ' (default: QScEvalDataLoader)')
# parser.add_argument('--optm_dataloader', default='QScPairDataLoader', choices=ds.__optm_all__,
#                     help='DataLoader architecture: ' + ' | '.join(ds.__optm_all__) + ' (default: QScPairDataLoader)')
# parser.add_argument('--sampling_config', help='configuration of positive / negative data sampling')
#
# parser.add_argument('--optm_model', default='QScHingeModel', choices=md.__optm_all__,
#                     help='Model architecture: ' + ' | '.join(md.__optm_all__) + ' (default: QScHingeModel)')
# parser.add_argument('--eval_model', default='QScDynamicEvalModel', choices=md.__eval_all__,
#                     help='Model architecture: ' + ' | '.join(md.__eval_all__) + ' (default: QScDynamicEvalModel)')

parser.add_argument('--model_config', help='model config')
# parser.add_argument('--optm_config', help='optimization config')
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
    wd_emb_util = WordEmbeddingUtil(wd_emb=args.word_emb, dim_wd_emb=args.dim_emb)
    kb_emb_util = KBEmbeddingUtil(kb_emb_dir=args.kb_emb_dir, dim_kb_emb=args.dim_emb)
    fb_helper = FreebaseHelper(args.fb_meta_dir)
    LogInfo.end_track()

    # ==== Loading Dataset ==== #
    data_config = literal_eval(args.data_config)    # including data_name, dir, max_length and others
    data_config['wd_emb_util'] = wd_emb_util
    data_config['kb_emb_util'] = kb_emb_util
    data_config['fb_helper'] = fb_helper
    data_config['verbose'] = args.verbose
    data_config.setdefault('protocol_name', 'XY')
    dataset = QScDataset(**data_config)
    dataset.load_size()  # load size info

    # ==== Build Model First ==== #
    LogInfo.begin_track('Building Model and Session ... ')
    gpu_options = tf.GPUOptions(allow_growth=True,
                                per_process_gpu_memory_fraction=args.gpu_fraction)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options,
                                            intra_op_parallelism_threads=8))
    optm_md_config = literal_eval(args.model_config)
    optm_md_config['sess'] = sess
    optm_md_config['n_words'] = dataset.w_size
    optm_md_config['n_preds'] = dataset.p_size
    optm_md_config['dim_emb'] = args.dim_emb
    optm_md_config['q_max_len'] = dataset.q_max_len
    optm_md_config['path_max_len'] = dataset.path_max_len
    optm_md_config['pword_max_len'] = dataset.path_max_len * dataset.item_max_len
    optm_md_config['verbose'] = args.verbose
    optm_model = SimpqOptmModel(**optm_md_config)

    eval_md_config = dict(optm_md_config)
    for k in ('margin', 'learning_rate', 'optm_name'):
        del eval_md_config[k]
    eval_model = SimpqEvalModel(**eval_md_config)

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
                 feed_dict={optm_model.w_embedding_init: dataset.w_init_emb,
                            optm_model.p_embedding_init: dataset.p_init_emb})
        # Be cautious: we must feed the tensor in optm_model, not in eval_model.
    LogInfo.end_track('Start Epoch = %d', start_epoch)
    LogInfo.end_track('Model Built.')

    # ==== Constructing Data_Loader ==== #
    LogInfo.begin_track('Creating DataLoader ... ')
    dataset.load_cands()  # first loading all the candidates
    q_max_len = dataset.q_max_len
    dl_config = {'dataset': dataset, 'mode': 'train', 'q_max_len': q_max_len,
                 'batch_size': args.optm_batch_size, 'proc_ob_num': 5000, 'verbose': args.verbose}
    optm_train_data = SimpqPairDataLoader(**dl_config)
    optm_train_data.renew_data_list()

    dl_config['batch_size'] = args.eval_batch_size
    eval_data_group = []
    for mode in ('train', 'valid', 'test'):
        dl_config['mode'] = mode
        eval_data = SimpqSingleDataLoader(**dl_config)
        eval_data.renew_data_list()
        eval_data_group.append(eval_data)
    (eval_train_data, eval_valid_data, eval_test_data) = eval_data_group
    LogInfo.end_track()  # End of loading data & dataset

    # ==== Free memories ==== #
    for item in (wd_emb_util, dataset.wd_emb_util,
                 kb_emb_util, dataset.kb_emb_util, fb_helper, data_config):
        del item

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

    for epoch in range(start_epoch + 1, args.max_epoch + 1):
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
                                       detail_fp=output_dir + '/detail/train_%03d.txt' % epoch,
                                       summary_writer=eval_train_summary_writer)
        LogInfo.end_track('T_F1 = %.6f', train_f1)

        LogInfo.begin_track('Eval-Validating ...')
        valid_f1 = eval_model.evaluate(eval_valid_data, epoch,
                                       detail_fp=output_dir + '/detail/valid_%03d.txt' % epoch,
                                       summary_writer=eval_valid_summary_writer)
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
            if update_flag and args.save_best:  # just create a symbolic link
                delete_dir(save_best_dir)
                os.symlink(save_epoch_dir, save_best_dir)  # symlink at directory level
        elif update_flag and args.save_best:
            delete_dir(save_best_dir)
            save_model(saver=saver, sess=sess, model_dir=save_best_dir, epoch=epoch, valid_metric=valid_f1)

        LogInfo.end_track()  # End of epoch
    LogInfo.end_track()  # End of learning

    LogInfo.end_track('All Done.')


if __name__ == '__main__':
    _args = parser.parse_args()
    main(_args)
