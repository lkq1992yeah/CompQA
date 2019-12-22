import codecs
import argparse
import tensorflow as tf
from ast import literal_eval

from ..dataset.dataset_dep import SchemaDatasetDep
from ..model.compq_dep.compq_mt_model import CompqMultiTaskModel

from ..util.word_emb import WordEmbeddingUtil
from ..u import load_model

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
    compq_mt_model = CompqMultiTaskModel(**model_config)

    LogInfo.begin_track('Showing final parameters: ')
    for var in tf.global_variables():
        LogInfo.logs('%s: %s', var.name, var.get_shape().as_list())
    LogInfo.end_track()

    saver = tf.train.Saver()
    LogInfo.begin_track('Outputting vectors ...')
    model_dir = '%s/%s' % (args.output_dir, args.resume_model_name)
    start_epoch, best_valid_f1 = load_model(saver=saver, sess=sess, model_dir=model_dir)
    LogInfo.end_track('Start Epoch = %d', start_epoch)

    active_word_dict = schema_dataset.active_dicts['word']
    dep_simulate = True if args.dep_simulate == 'True' else False
    init_w_emb_mat = wd_emb_util.produce_active_word_embedding(
        active_word_dict=schema_dataset.active_dicts['word'],
        dep_simulate=dep_simulate
    )
    upd_w_emb_mat = sess.run(compq_mt_model.w_embedding)
    LogInfo.logs('Updated word embedding fetched.')

    emb_value_fp = '%s/w_emb_%03d.value' % (args.output_dir, start_epoch)
    emb_label_fp = '%s/w_emb_%03d.label' % (args.output_dir, start_epoch)
    with codecs.open(emb_value_fp, 'w', 'utf-8') as bw_value, codecs.open(emb_label_fp, 'w', 'utf-8') as bw_label:
        for word, idx in active_word_dict.items():
            upd_vec = upd_w_emb_mat[idx]
            init_vec = init_w_emb_mat[idx]
            upd_vec_str = '\t'.join(['%.4f' % x for x in upd_vec])
            init_vec_str = '\t'.join(['%.4f' % x for x in init_vec])
            bw_value.write(init_vec_str + '\n')
            bw_value.write(upd_vec_str + '\n')
            bw_label.write(word + '\n')
            bw_label.write('%s__%03d\n' % (word, start_epoch))
    LogInfo.logs('Embedding info saved.')
    LogInfo.end_track('Model build complete.')


if __name__ == '__main__':
    LogInfo.begin_track('[kangqi.task.compQA.analyze.embedding_output] running ...')
    _args = parser.parse_args()
    main(_args)
    LogInfo.end_track('All Done.')
