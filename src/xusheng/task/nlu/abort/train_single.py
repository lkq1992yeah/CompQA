"""
Training the single-task model
currently support NER (sequence labeling task)
"""

import sys

import numpy as np
import tensorflow as tf
from xusheng.task.nlu.data.data import DataLoader, BatchGenerator

from xusheng.task.nlu.abort.model.single_task import NER
from xusheng.util.config import ConfigDict
from xusheng.util.data_util import VocabularyLoader
from xusheng.util.eval_util import eval_seq_crf
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

    query_idx_v, query_len_v, label_v, _, _, _ = \
        zip(*data_loader.data[train_size:train_size+valid_size])

    query_idx_t, query_len_t, label_t, _, _, _ = \
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
        train_model = NER(config=config,
                          mode=tf.contrib.learn.ModeKeys.TRAIN,
                          embedding_vocab=vocab_loader.vocab_embedding)
        LogInfo.logs("Train model created.")

        # all get_variable parameters will be reused in eval
        tf.get_variable_scope().reuse_variables()

        eval_model = NER(config=config,
                         mode=tf.contrib.learn.ModeKeys.EVAL,
                         embedding_vocab=vocab_loader.vocab_embedding)
        LogInfo.logs("Eval model created.")
    LogInfo.end_track()

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True

    best_valid_f1 = 0.0
    waiting = 0

    LogInfo.begin_track("Start training...")
    with tf.Session(graph=graph, config=tf_config) as session:
        session.run(tf.global_variables_initializer())
        for epoch in range(1, config.get("epoch")):
            LogInfo.begin_track("Epoch %d/%d...", epoch, config.get("epoch"))
            batch_generator.reset_batch_pointer()
            for batch in range(batch_generator.num_batches):
                batch_data = batch_generator.next_batch()
                _, loss, logits, _ = train_model.train(session, batch_data)

                LogInfo.logs("Batch %d/%d ==> loss: %.4f",
                             batch+1, batch_generator.num_batches, loss)
            LogInfo.end_track()

            if epoch % config.get("eval_step") == 0:
                waiting += 1
                LogInfo.begin_track("Eval on validation set...")
                valid_data = [
                    np.array(query_idx_v),
                    np.array(query_len_v)
                ]
                logits, = eval_model.eval(session, valid_data)
                tag_list = eval_model.decode(logits, query_len_v)
                precision = eval_seq_crf(tag_list,
                                         [x[:y] for x, y in zip(label_v, query_len_v)],
                                         method='precision')
                recall = eval_seq_crf(tag_list,
                                      [x[:y] for x, y in zip(label_v, query_len_v)],
                                      method='recall')
                if precision == 0 and recall == 0:
                    valid_f1 = 0.0
                else:
                    valid_f1 = 2 * precision * recall / (precision + recall)
                LogInfo.logs("F1-valid: %.4f", valid_f1)
                LogInfo.end_track()
                # valid result improved, testing on test set
                if valid_f1 > best_valid_f1:
                    best_valid_f1 = valid_f1
                    LogInfo.begin_track("Eval on testing set...")
                    test_data = [
                        np.array(query_idx_t),
                        np.array(query_len_t)
                    ]
                    logits, = eval_model.eval(session, test_data)
                    tag_list = eval_model.decode(logits, query_len_t)
                    precision = eval_seq_crf(tag_list,
                                             [x[:y] for x, y in zip(label_t, query_len_t)],
                                             method='precision')
                    recall = eval_seq_crf(tag_list,
                                          [x[:y] for x, y in zip(label_t, query_len_t)],
                                          method='recall')
                    if precision == 0 and recall == 0:
                        test_f1 = 0.0
                    else:
                        test_f1 = 2 * precision * recall / (precision + recall)
                    LogInfo.logs("F1-test: %.4f", test_f1)
                    LogInfo.end_track()
                    waiting = 0
                    eval_model.save(session, root_path + "/model")

            if waiting == config.get("waiting"):
                LogInfo.logs("Valid result no longer improves, abort training...")
                break
    LogInfo.end_track()


