"""
Training the model IntentionIdentifier
"""

import sys
import numpy as np
import tensorflow as tf

from xusheng.task.intention.data.data import DataLoader, BatchGenerator
from xusheng.task.intention.model.identifier import IntentionIdentifier
from xusheng.util.config import ConfigDict
from xusheng.util.data_util import VocabularyLoader
from xusheng.util.eval_util import eval_acc_pn
from xusheng.util.log_util import LogInfo

if __name__ == '__main__':
    setting_dir = sys.argv[1]
    try_dir = sys.argv[2]
    root_path = 'runnings/%s/%s' % (setting_dir, try_dir)
    config_path = '%s/param_config' % root_path
    config = ConfigDict(config_path)

    vocab_loader = VocabularyLoader()
    vocab_loader.load_vocab(config.get("vocab_fp"), config.get("embedding_dim"), 'utf-8')
    config.add("vocab_size", vocab_loader.vocab_size)
    LogInfo.logs("Embedding shape: %s.", vocab_loader.vocab_embedding.shape)

    data_loader = DataLoader(config.get("max_seq_len"), vocab_loader.vocab_index_dict)
    data_loader.load(config.get("data_fp"), 'utf-8')

    LogInfo.logs("Create train, valid, test split...")
    train_size = int(0.997 * data_loader.data_size)
    valid_size = int(0.001 * data_loader.data_size)
    test_size = data_loader.data_size - train_size - valid_size

    train_data = data_loader.data[:train_size]

    valid_data = data_loader.data[train_size:train_size+valid_size]
    context_idx_valid, context_seq_valid, pinlei_idx_valid = zip(*valid_data)
    context_idx_valid = np.array(context_idx_valid)
    context_seq_valid = np.array(context_seq_valid)
    pinlei_idx_valid = np.array(pinlei_idx_valid)
    valid_data = [context_idx_valid, context_seq_valid, pinlei_idx_valid]

    test_data = data_loader.data[train_size+valid_size:]
    context_idx_test, context_seq_test, pinlei_idx_test = zip(*test_data)
    context_idx_test = np.array(context_idx_test)
    context_seq_test = np.array(context_seq_test)
    pinlei_idx_test = np.array(pinlei_idx_test)
    test_data = [context_idx_test, context_seq_test, pinlei_idx_test]

    data_loader.data.clear()
    LogInfo.logs("train: valid: test = %d: %d: %d.", train_size, valid_size, test_size)
    # LogInfo.logs("train data: %s", train_data)
    # LogInfo.logs("valid data: %s", valid_data)
    # LogInfo.logs("test data: %s", test_data)

    batch_generator = BatchGenerator(train_data, config.get("batch_size"))

    LogInfo.begin_track("Create models...")
    graph = tf.Graph()
    with graph.as_default():
        train_model = IntentionIdentifier(config=config,
                                          mode=tf.contrib.learn.ModeKeys.TRAIN,
                                          embedding_vocab=vocab_loader.vocab_embedding)
        LogInfo.logs("Train model created.")

        # all get_variable parameters will be reused in eval
        tf.get_variable_scope().reuse_variables()

        eval_model = IntentionIdentifier(config=config,
                                         mode=tf.contrib.learn.ModeKeys.EVAL,
                                         embedding_vocab=vocab_loader.vocab_embedding)
        LogInfo.logs("Eval model created.")
    LogInfo.end_track()

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True

    best_valid_acc = 0.0
    waiting = 0

    LogInfo.begin_track("Start training...")
    with tf.Session(graph=graph, config=tf_config) as session:
        session.run(tf.global_variables_initializer())
        for epoch in range(config.get("epoch")):
            LogInfo.begin_track("Epoch %d/%d...", epoch, config.get("epoch"))
            batch_generator.reset_batch_pointer()
            for batch in range(batch_generator.num_batches):
                batch_data = batch_generator.next_batch()
                # LogInfo.logs("Context_idx_shape: %s.", batch_data[0].shape)
                # LogInfo.logs("Context_seq_shape: %s.", batch_data[1].shape)
                # LogInfo.logs("Pinlei_idx_shape: %s.", batch_data[2].shape)
                # LogInfo.logs("batch data: %s", batch_data)
                return_list = train_model.train(session, batch_data)
                loss = return_list[1]
                score_list = return_list[2][:20]
                context_embedding = return_list[3]
                pinlei_embedding = return_list[4]
                context_slice = return_list[5]
                encoder_output = return_list[6]
                query_hidden = return_list[7]
                pinlei_hidden = return_list[8]
                LogInfo.logs("Batch %d/%d ==> loss: %.4f",
                             batch, batch_generator.num_batches, loss)
                # LogInfo.logs("score: %s.", score_list)
                # LogInfo.logs("context_embedding: %s.", context_embedding)
                # LogInfo.logs("pinlei_embedding: %s.", pinlei_embedding)
                # LogInfo.logs("context_slice: %s.", context_slice)
                # LogInfo.logs("encoder_output: %s.", encoder_output)
                # LogInfo.logs("query_hidden: %s.", query_hidden)
                # LogInfo.logs("pinlei_hidden: %s.", pinlei_hidden)

            LogInfo.end_track()

            if epoch % config.get("eval_step") == 0:
                waiting += 1
                LogInfo.begin_track("Eval on validation set...")
                # LogInfo.logs("Context_idx_shape: %s.", valid_data[0].shape)
                # LogInfo.logs("Context_seq_shape: %s.", valid_data[1].shape)
                # LogInfo.logs("Pinlei_idx_shape: %s.", valid_data[2].shape)
                score_list = eval_model.eval(session, valid_data)
                valid_acc, valid_corr, valid_case = \
                    eval_acc_pn(score_list, config.get("PN"))
                LogInfo.logs("Valid accuracy: %.4f(%d/%d)",
                             valid_acc, valid_corr, valid_case)
                LogInfo.end_track()
                if valid_acc > best_valid_acc:
                    best_valid_acc = valid_acc
                    LogInfo.begin_track("Eval on testing set...")
                    # LogInfo.logs("Context_idx_shape: %s.", test_data[0].shape)
                    # LogInfo.logs("Context_seq_shape: %s.", test_data[1].shape)
                    # LogInfo.logs("Pinlei_idx_shape: %s.", test_data[2].shape)
                    score_list = eval_model.eval(session, test_data)
                    test_acc, test_corr, test_case = \
                        eval_acc_pn(score_list, config.get("PN"))
                    LogInfo.logs("Test accuracy: %.4f(%d/%d)",
                                 test_acc, test_corr, test_case)
                    LogInfo.end_track()
                    waiting = 0
                    eval_model.save(session, root_path + "/model")

            if waiting == 20:
                LogInfo.logs("Valid result no longer improves, abort training...")
                break
    LogInfo.end_track()













