import random
import sys

import numpy as np
import tensorflow as tf

from test import test
from data.train_data import get_train_data
from kangqi.task.tabel.data.prepare_joint import load_joint_from_file
from kangqi.util.LogUtil import LogInfo
from kangqi.util.config import load_configs
from model.exp_model import Model1, Model2, Model3, Model4, Model5, \
    Model1_rank, Model2_rank, Model3_rank, Model4_rank, Model5_rank, \
    Model3_rank_avg, Model6_rank, Model7_rank, Model8_rank
from xusheng.util.tf_util import normalize

# Training different model for CIKM submission
# 1. Read config file
# 2. Load data
# 3. Load corresponding model
# 4. Train the model
# 5. Validate & Test every some epochs

def get_batch(data, idx):
    batch_data = dict()
    batch_data['cell'] = \
        data['cell'][idx * PN * rows * cols: (idx + real_batch_size) * PN * rows * cols]
    batch_data['entity'] = \
        data['entity'][idx * PN * rows * cols: (idx + real_batch_size) * PN * rows * cols]
    batch_data['coherence'] = \
        data['coherence'][idx * PN: (idx + real_batch_size) * PN]
    batch_data['context'] = \
        data['context'][idx * PN * rows * cols: (idx + real_batch_size) * PN * rows * cols]
    batch_data['corrupt'] = \
        data['corrupt'][idx: idx + real_batch_size]
    return batch_data

if __name__=='__main__':
    setting_dir = sys.argv[1]
    try_dir = sys.argv[2]
    root_path = 'runnings/tabel/e2e/%s/%s' % (setting_dir, try_dir)
    config_fp = '%s/param_config' % root_path
    config_dict = load_configs(config_fp)

    # data path
    joint_data_fp = config_dict['joint_data_fp']
    # model selection
    model_num = int(config_dict['model'])
    trans = config_dict['translation_setting']
    nonj = config_dict['non_joint_setting']
    coherence_norm = 0
    if 'coherence_norm' in config_dict:
        coherence_norm = int(config_dict['coherence_norm'])
    # table settings
    candidate_num = 50
    if 'candidate_num' in config_dict:
        candidate_num = int(config_dict['candidate_num'])
    epochs = int(config_dict['epochs'])
    display_step = int(config_dict['display_step'])
    valid_step = int(config_dict['valid_step'])
    test_step = int(config_dict['test_step'])
    PN = int(config_dict['PN'])  # table_PN
    real_batch_size = int(config_dict['batch_size'])
    batch_size = real_batch_size * PN  # real batch size * PN
    rows = int(config_dict['rows'])
    cols = int(config_dict['cols'])
    dim = rows + cols - 2  # length of context
    # model params
    d_w2v = int(config_dict['w2v_dim'])
    d_hidden = int(config_dict['hidden_dim'])
    d_final = int(config_dict['final_dim'])
    margin = float(config_dict['margin'])
    learning_rate = float(config_dict['learning_rate'])
    keep_prob = float(config_dict['keep_prob'])
    # evaluation strategy
    train_acc = int(config_dict['train_acc'])
    eval_threshold = float(config_dict['eval_threshold'])

    # Model Selection

    graph = tf.Graph()
    with graph.as_default():
        if model_num == 1:
            LogInfo.logs("Model 1, use surface feature only.")
            model = Model1(trans_pre_train=trans, nonj_pre_train=nonj,
                           batch_size=batch_size, margin=margin,
                           PN = PN, rows=rows, columns=cols,
                           d_w2v=d_w2v, d_hidden=d_hidden, d_final=d_final,
                           lr=learning_rate)

        elif model_num == 10:
            LogInfo.logs("Model 1_rank, use surface & RankNet loss")
            model = Model1_rank(trans_pre_train=trans, nonj_pre_train=nonj,
                                batch_size=batch_size, margin=margin,
                                PN = PN, rows=rows, columns=cols,
                                d_w2v=d_w2v, d_hidden=d_hidden, d_final=d_final,
                                lr=learning_rate)

        elif model_num == 2:
            LogInfo.logs("Model 2, use surface feature & context feature.")
            model = Model2(trans_pre_train=trans, nonj_pre_train=nonj,
                           batch_size=batch_size, margin=margin,
                           PN = PN, rows=rows, columns=cols,
                           d_w2v=d_w2v, d_hidden=d_hidden, d_final=d_final,
                           lr=learning_rate)

        elif model_num == 20:
            LogInfo.logs("Model 2_rank, use surface & context feature & RankNet loss")
            model = Model2_rank(trans_pre_train=trans, nonj_pre_train=nonj,
                                batch_size=batch_size, margin=margin,
                                PN = PN, rows=rows, columns=cols,
                                d_w2v=d_w2v, d_hidden=d_hidden, d_final=d_final,
                                lr=learning_rate)

        elif model_num == 3:
            LogInfo.logs("Model 3, use surface feature & context feature & Kenny's coherence feature.")
            model = Model3(trans_pre_train=trans, nonj_pre_train=nonj,
                           batch_size=batch_size, margin=margin,
                           PN = PN, rows=rows, columns=cols,
                           d_w2v=d_w2v, d_hidden=d_hidden, d_final=d_final,
                           lr=learning_rate)

        elif model_num == 30:
            LogInfo.logs("Model 3_rank, use surface & context & coherence feature & RankNet loss")
            model = Model3_rank(trans_pre_train=trans, nonj_pre_train=nonj,
                                batch_size=batch_size, margin=margin,
                                PN = PN, rows=rows, columns=cols,
                                d_w2v=d_w2v, d_hidden=d_hidden, d_final=d_final,
                                lr=learning_rate, istrain=True, keep_prob=keep_prob)
            LogInfo.logs("Train model generated.")

            tf.get_variable_scope().reuse_variables()

            model_test = Model3_rank(trans_pre_train=trans, nonj_pre_train=nonj,
                                     batch_size=batch_size, margin=margin,
                                     PN = PN, rows=rows, columns=cols,
                                     d_w2v=d_w2v, d_hidden=d_hidden, d_final=d_final,
                                     lr=learning_rate, istrain=False, keep_prob=keep_prob)
            LogInfo.logs("Eval model generated.")

        elif model_num == 80:
            LogInfo.logs("Model 8_rank, Monolingual, use surface & context & coherence feature & RankNet loss")
            model = Model8_rank(trans_pre_train=trans, nonj_pre_train=nonj,
                                batch_size=batch_size, margin=margin,
                                PN = PN, rows=rows, columns=cols,
                                d_w2v=d_w2v, d_hidden=d_hidden, d_final=d_final,
                                lr=learning_rate, istrain=True, keep_prob=keep_prob)
            LogInfo.logs("Train model generated.")

            tf.get_variable_scope().reuse_variables()

            model_test = Model8_rank(trans_pre_train=trans, nonj_pre_train=nonj,
                                     batch_size=batch_size, margin=margin,
                                     PN=PN, rows=rows, columns=cols,
                                     d_w2v=d_w2v, d_hidden=d_hidden, d_final=d_final,
                                     lr=learning_rate, istrain=False, keep_prob=keep_prob)
            LogInfo.logs("Eval model generated.")


        elif model_num == 300:
            LogInfo.logs("Model 3_rank_avg, "
                         "use surface & context & avg coherence feature & RankNet loss")
            model = Model3_rank_avg(trans_pre_train=trans, nonj_pre_train=nonj,
                                    batch_size=batch_size, margin=margin,
                                    PN = PN, rows=rows, columns=cols,
                                    d_w2v=d_w2v, d_hidden=d_hidden, d_final=d_final,
                                    lr=learning_rate)

        elif model_num == 4:
            LogInfo.logs("Model 4, use context feature only.")
            model = Model4(trans_pre_train=trans, nonj_pre_train=nonj,
                           batch_size=batch_size, margin=margin,
                           PN = PN, rows=rows, columns=cols,
                           d_w2v=d_w2v, d_hidden=d_hidden, d_final=d_final,
                           lr=learning_rate)

        elif model_num == 40:
            LogInfo.logs("Model 4_rank, use context & RankNet loss")
            model = Model4_rank(trans_pre_train=trans, nonj_pre_train=nonj,
                                batch_size=batch_size, margin=margin,
                                PN = PN, rows=rows, columns=cols,
                                d_w2v=d_w2v, d_hidden=d_hidden, d_final=d_final,
                                lr=learning_rate)

        elif model_num == 5:
            LogInfo.logs("Model 5, use Kenny's coherence feature only.")
            model = Model5(trans_pre_train=trans, nonj_pre_train=nonj,
                           batch_size=batch_size, margin=margin,
                           PN = PN, rows=rows, columns=cols,
                           d_w2v=d_w2v, d_hidden=d_hidden, d_final=d_final,
                           lr=learning_rate)

        elif model_num == 50:
            LogInfo.logs("Model 5_rank, use coherence feature & RankNet loss")
            model = Model5_rank(trans_pre_train=trans, nonj_pre_train=nonj,
                                batch_size=batch_size, margin=margin,
                                PN = PN, rows=rows, columns=cols,
                                d_w2v=d_w2v, d_hidden=d_hidden, d_final=d_final,
                                lr=learning_rate)

        elif model_num == 60:
            LogInfo.logs("Model 6_rank, use surface & coherence feature & RankNet loss")
            model = Model6_rank(trans_pre_train=trans, nonj_pre_train=nonj,
                                batch_size=batch_size, margin=margin,
                                PN = PN, rows=rows, columns=cols,
                                d_w2v=d_w2v, d_hidden=d_hidden, d_final=d_final,
                                lr=learning_rate)

        elif model_num == 70:
            LogInfo.logs("Model 7_rank, use context & coherence feature & RankNet loss")
            model = Model7_rank(trans_pre_train=trans, nonj_pre_train=nonj,
                                batch_size=batch_size, margin=margin,
                                PN = PN, rows=rows, columns=cols,
                                d_w2v=d_w2v, d_hidden=d_hidden, d_final=d_final,
                                lr=learning_rate)

    # -------------------------------- Training ------------------------------------- #

    LogInfo.begin_track("Loading training data...")
    if joint_data_fp == 'None':
        train_data = get_train_data()
        valid_data = None
        test_data = None
    else:
        protocol = int(config_dict['protocol'])
        train_data, valid_data, test_data, train_eval_data = \
            load_joint_from_file(joint_data_fp, protocol)

    table_num = len(train_data['cell']) / PN
    LogInfo.logs("Loaded.")
    train_data['cell'] = np.array(train_data['cell']).reshape((-1, 100))
    train_data['entity'] = np.array(train_data['entity']).reshape((-1, 100))
    train_data['coherence'] = np.array(train_data['coherence']).reshape((-1, 100))
    if coherence_norm == 1:
        train_data['coherence'] = normalize(train_data['coherence'])
    elif coherence_norm == 2:
        train_data['coherence'] = np.sum(train_data['coherence'], axis=1).reshape((-1, 1))

    train_data['context'] = np.array(train_data['context']).reshape((-1, 100))
    # corrupt data remains the same shape (table_num, PN)
    train_data['corrupt'] = np.array(train_data['corrupt'])

    LogInfo.logs("After reshaping...")
    LogInfo.logs("Cell shape: %s", train_data['cell'].shape)
    LogInfo.logs("Entity shape: %s", train_data['entity'].shape)
    LogInfo.logs("Coherence shape: %s", train_data['coherence'].shape)
    LogInfo.logs("Context shape: %s", train_data['context'].shape)
    LogInfo.logs("Corrupt shape: %s", train_data['corrupt'].shape)
    LogInfo.end_track()

    # TF config
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True

    LogInfo.begin_track("Start training...")
    total_batch = table_num - real_batch_size + 1
    best_ret = 0.0
    waiting = 0
    with tf.Session(graph=graph, config=tf_config) as session:
        session.run(tf.global_variables_initializer())

        # pre-train weights assign START
        if trans != "random":
            LogInfo.logs("Loading pre-trained translation weights...")
            model.get_translation_weights(session=session, setting=trans)
        else:
            LogInfo.logs("W/o pre-trained translation weights.")

        if nonj != "random":
            LogInfo.logs("Loading pre-trained non-joint weights...")
            model.get_nonjoint_weights(session=session, setting=nonj)
        else:
            LogInfo.logs("W/o pre-trained non-joint weights.")
        # pre-train weights assign END

        for epoch in range(epochs):
            LogInfo.begin_track("Epoch %d/%d", epoch+1, epochs)
            # Loop over all batches, random order
            x = range(total_batch)
            random.shuffle(x)
            for idx, batch_id in enumerate(x):
                batch_train_data = get_batch(train_data, batch_id)
                # batch_train_data = get_sample_batch(train_data, batch_id)
                if idx == 0:
                    LogInfo.logs("Batch train data shape: %s, %s, %s, %s, %s",
                                 np.array(batch_train_data['cell']).shape,
                                 np.array(batch_train_data['entity']).shape,
                                 np.array(batch_train_data['coherence']).shape,
                                 np.array(batch_train_data['context']).shape,
                                 np.array(batch_train_data['corrupt']).shape)
                running_ret = model.train(session=session, input_data=batch_train_data)
                if idx % display_step == 0:
                    LogInfo.logs("Iter %d/%d, loss: %s", idx+1, total_batch, running_ret['loss'])
                    # print_detail_2d(running_ret['w_trans'])
                    # print_detail_2d(running_ret['vec_two'])
                    # LogInfo.logs("loss_matrix: %s", running_ret['loss_matrix'])
                    # LogInfo.logs("scores: %s", running_ret['scores'])
            LogInfo.end_track()

            waiting += 1
            if waiting > 20:
                LogInfo.end_track("Validation no longer improves, aborting training...")
                break
            # Validation Testing
            if (epoch+1) % valid_step == 0:
                if train_acc == 1:
                    _ = test(model=model_test,
                             session=session,
                             split="train",
                             verbose=False,
                             test_data=train_eval_data,
                             norm=coherence_norm,
                             candidate_num=candidate_num)
                ret = test(model=model_test,
                           session=session,
                           split="valid",
                           verbose=False,
                           test_data=valid_data,
                           norm=coherence_norm,
                           candidate_num=candidate_num)
                if ret > best_ret:
                    waiting = 0
                    best_ret = ret
                    model.save(session=session, directory="%s/model" % root_path)
                    # Testing
                    _ = test(model=model_test,
                             session=session,
                             split="test",
                             verbose=False,
                             test_data=test_data,
                             norm=coherence_norm,
                             candidate_num=candidate_num)
        LogInfo.end_track()

