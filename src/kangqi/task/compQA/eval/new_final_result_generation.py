# import codecs
# import argparse
# import tensorflow as tf
# from ast import literal_eval
#
# from ..dataset.dataset_dep import SchemaDatasetDep
# from ..model.compq_dep.compq_mt_model import CompqMultiTaskModel
#
# from ..util.word_emb import WordEmbeddingUtil
# from ..u import load_model
#
# from kangqi.util.LogUtil import LogInfo
#
#
# parser = argparse.ArgumentParser(description='QA Model Training')
#
# parser.add_argument
# parser.add_argument('--machine', default='Blackhole', choices=['Blackhole', 'Darkstar'])
# parser.add_argument('--word_emb', default='glove')
# parser.add_argument('--dim_emb', type=int, default=300, help='word/predicate embedding dimension')
# parser.add_argument('--data_config', help='dataset config')
#
# parser.add_argument('--dep_simulate', default='False')
#
# parser.add_argument('--neg_pick_config', help='negative sampling config')
#
# parser.add_argument('--seg_config', help='segment config')
# parser.add_argument('--rm_config', help='relation matching config')
# parser.add_argument('--model_config', help='general model config')
# parser.add_argument('--resume_model_name', default='model_best',
#                     help='the directory name of the model which you wan to resume learning')
#
# parser.add_argument('--optm_batch_size', type=int, default=128, help='optm_batch size')
# parser.add_argument('--eval_batch_size', type=int, default=32, help='eval_batch size')
# parser.add_argument('--max_epoch', type=int, default=30, help='max epochs')
# parser.add_argument('--max_patience', type=int, default=10000, help='max patience')
#
# parser.add_argument('--full_optm_method', choices=['ltr', 'full', 'rm', 'el'])
# parser.add_argument('--pre_train_steps', type=int, default=10)
#
# parser.add_argument('--output_dir', help='output dir, including results, models and others')
# parser.add_argument('--save_best', help='save best model only', action='store_true')
# parser.add_argument('--test_only', action='store_true')
#
# parser.add_argument('--gpu_fraction', type=float, default=0.25, help='GPU fraction limit')
# parser.add_argument('--verbose', type=int, default=0, help='verbose level')
#
#
# def main():
#
