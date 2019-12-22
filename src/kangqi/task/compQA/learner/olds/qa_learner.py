# Author: Kangqi Luo
# Goal: The learning framework of QA

import tensorflow as tf
import numpy as np
import cPickle
import sys, os


from kangqi.util.LogUtil import LogInfo
from kangqi.util.config import load_configs

from xusheng.task.qa.schema import Schema

# The code is not for pre-train!!


# Note: given embedding: w2v is fixed
#       given word index: embedding can be adjusted

# def load_q_skeleton_data(data_size, max_q_wds, max_sk_hops, max_sk_wds):
#     q_tensor3 = np.zeros((data_size, max_len, n_emb), dtype='float32')
#     sk_tensor3 = np.zeros((data_size, PN, max_hop))

# Load questions and candidate schemas
# Question: np tensor of (data_size, max_len, embedding)
def load_q_schema_data(data_size, max_len, n_emb, max_sc_hops, PN):
    LogInfo.begin_track('Data preparation: ')

    # [ [schema candidate] ], a matrix of Schema objects
    # each questions has a limited number of candidates
    cand_schemas_list = [ [] ]
    # TODO: Load schemas, and convert schemas into tensor (ds, PN, 3) 3=pred, cons_p, cons_e

    q_tensor3 = np.zeros((data_size, max_len, n_emb), dtype='float32')
    cand_tensor4 = np.zeros((data_size, PN, max_sc_hops, 3), dtype='int')
    # candidate schema tensor: need look up
    # we can write a pseudo-RNN (since the length is so short)
    gold_tensor3 = np.zeros((data_size, PN, 3), dtype='float32')    # P/R/F1
    mask_matrix = np.zeros((data_size, PN), dtype='float32')
    np_list = (q_tensor3, gold_tensor3, mask_matrix)
    # TODO: Load q_tensor3, gold_tensor3, mask_tensor3 from files (Kangqi do this)
    #       Need w2v results.

    assert np.shape(q_tensor3) == (data_size, max_len, n_emb)
    assert np.shape(gold_tensor3) == (data_size, PN, 3)
    assert np.shape(mask_matrix) == (data_size, PN)

    LogInfo.end_track()
    return np_list, cand_schemas_list
    

def train(info_tup_list, input_tf_list):
    learner = DefaultLearner(
        n_epoch,
        max_patience,
        batch_size,
        info_tup_list,
        input_tf_list,
        log_fp,
        sess,
        train_step,
        metric_tf,
        final_loss_tf)
    learner.learn(log_header, loss_as_everage=False)


def Main():
    eval_dir = ''   # TODO
    config_dict = load_configs(eval_dir + '/param_config')
    # TODO: Define parameters

    np_list = load_q_schema_data(data_size, max_len, n_emb, PN) # TODO: parameters
    Tvt_split_list = [0, 3022, 3778, 5810]  # TODO: Webq
    for idx, Tvt in enumerate(['T', 'v', 't']):
        indices = range(Tvt_split_list[idx], Tvt_split_list[idx+1])





if __name__ == '__main__':
    LogInfo.begin_track('src.kangqi.task.compQA.qa_learner starts ... ')
    Main()
    LogInfo.end_track()

