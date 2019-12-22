"""
Training the multi-task model
"""

import sys

import numpy as np
import tensorflow as tf
from xusheng.task.nlu.model.multi_task import MultiTaskModel

from xusheng.task.nlu.abort.data.data import DataLoader, BatchGenerator, candidate_generate
from xusheng.util.config import ConfigDict
from xusheng.util.data_util import VocabularyLoader
from xusheng.util.eval_util import eval_classify_f1, \
    eval_seq, eval_link_f1
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
    train_size = int(config.get("train_split") * data_loader.data_size)
    valid_size = int(config.get("valid_split") * data_loader.data_size)
    test_size = data_loader.data_size - train_size - valid_size

    train_data = data_loader.data[:train_size]

    query_idx_v, query_len_v, label_v, intent_v, link_mask_v, entity_idx_v = \
        zip(*data_loader.data[train_size:train_size+valid_size])

    query_idx_t, query_len_t, label_t, intent_t, link_mask_t, entity_idx_t = \
        zip(*data_loader.data[train_size+valid_size:])

    data_loader.data.clear()
    LogInfo.logs("train: valid: test = %d: %d: %d.", train_size, valid_size, test_size)
    # LogInfo.logs("train data: %s", train_data)
    # LogInfo.logs("valid data: %s", valid_data)
    # LogInfo.logs("test data: %s", test_data)

    batch_generator = BatchGenerator(train_data, config.get("batch_size"))

    LogInfo.begin_track("Create models...")
    graph = tf.Graph()
    with graph.as_default():
        train_model = MultiTaskModel(config=config,
                                     mode=tf.contrib.learn.ModeKeys.TRAIN,
                                     embedding_vocab=vocab_loader.vocab_embedding)
        LogInfo.logs("Train model created.")

        # all get_variable parameters will be reused in eval
        tf.get_variable_scope().reuse_variables()

        eval_model = MultiTaskModel(config=config,
                                    mode=tf.contrib.learn.ModeKeys.EVAL,
                                    embedding_vocab=vocab_loader.vocab_embedding)
        LogInfo.logs("Eval model created.")
    LogInfo.end_track()

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True

    best_valid_intent_f1 = 0.0
    best_valid_label_f1 = 0.0
    best_valid_link_f1 = 0.0
    waiting = 0

    LogInfo.begin_track("Start training...")
    with tf.Session(graph=graph, config=tf_config) as session:
        session.run(tf.global_variables_initializer())
        for epoch in range(config.get("epoch")):
            LogInfo.begin_track("Epoch %d/%d...", epoch, config.get("epoch"))
            batch_generator.reset_batch_pointer()
            for batch in range(batch_generator.num_batches):
                batch_data = batch_generator.next_batch()
                return_list = train_model.train(session, batch_data)
                intent_loss = return_list[3]
                label_loss = return_list[4]
                link_loss = return_list[5]
                LogInfo.logs("Batch %d/%d ==> loss: %.4f, %.4f, %.4f",
                             batch, batch_generator.num_batches,
                             intent_loss, label_loss, link_loss)
            LogInfo.end_track()

            if epoch % config.get("eval_step") == 0:
                waiting += 1
                LogInfo.begin_track("Eval on validation set...")
                valid_data_intent_label = [
                    np.array(query_idx_v),
                    np.array(query_len_v)
                ]
                intent_list, label_list = \
                    eval_model.eval_intent_and_label(session, valid_data_intent_label)

                # get link mask based on nlu result
                # and do candidate generation accordingly
                new_query_idx_v, new_query_len_v, new_link_mask_v, new_entity_idx_v = \
                    candidate_generate(label_list, query_idx_v, query_len_v,
                                       vocab_loader, config.get("PN"))
                valid_data_link = [
                    np.array(new_query_idx_v),
                    np.array(new_query_len_v),
                    np.array(new_link_mask_v),
                    np.array(new_entity_idx_v)
                ]
                link_score_list = eval_model.eval_link(session, valid_data_link)

                valid_intent_f1 = eval_classify_f1(intent_list, intent_v, 'macro')
                valid_label_f1 = eval_seq(label_list, label_v)
                valid_link_f1 = eval_link_f1(link_score_list, entity_idx_v, config.get("PN"))
                LogInfo.logs("macro F1-intent/F1-label/F1-link: %.4f/%.4f/%.4f",
                             valid_intent_f1, valid_label_f1, valid_link_f1)
                LogInfo.end_track()
                # valid result improved, testing on test set
                if valid_intent_f1 > best_valid_intent_f1 \
                        or valid_label_f1 > best_valid_label_f1 \
                        or valid_link_f1 > best_valid_link_f1:
                    best_valid_intent_f1 = valid_intent_f1
                    best_valid_label_f1 = best_valid_label_f1
                    best_valid_link_f1 = best_valid_link_f1
                    LogInfo.begin_track("Eval on testing set...")
                    test_data_intent_label = [
                        np.array(query_idx_t),
                        np.array(query_len_t)
                    ]
                    intent_list, label_list = \
                        eval_model.eval_intent_and_label(session, test_data_intent_label)

                    # get link mask based on nlu result
                    # and do candidate generation accordingly
                    new_query_idx_t, new_query_len_t, new_link_mask_t, new_entity_idx_t = \
                        candidate_generate(label_list, query_idx_t, query_len_t,
                                           vocab_loader, config.get("PN"))
                    test_data_link = [
                        np.array(new_query_idx_t),
                        np.array(new_query_len_t),
                        np.array(new_link_mask_t),
                        np.array(new_entity_idx_t)
                    ]
                    link_score_list = eval_model.eval_link(session, test_data_link)

                    test_intent_f1 = eval_classify_f1(intent_list, intent_t, 'macro')
                    test_label_f1 = eval_seq(label_list, label_t)
                    test_link_f1 = eval_link_f1(link_score_list, entity_idx_t, config.get("PN"))
                    LogInfo.logs("macro F1-intent/F1-label/F1-link: %.4f/%.4f/%.4f",
                                 test_intent_f1, test_label_f1, test_link_f1)
                    LogInfo.end_track()
                    waiting = 0
                    eval_model.save(session, root_path + "/model")

            if waiting == 20:
                LogInfo.logs("Valid result no longer improves, abort training...")
                break
    LogInfo.end_track()

