"""
Testing the trained best model to identify multi-pinlei queries
"""

import sys
import numpy as np
import tensorflow as tf
import codecs

from xusheng.task.intention.data.data import DataLoader
from xusheng.task.intention.model.identifier import IntentionIdentifier
from xusheng.util.config import ConfigDict
from xusheng.util.data_util import VocabularyLoader
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
    data_loader.load(config.get("test_data_fp"), 'utf-8')

    LogInfo.begin_track("Create models...")
    graph = tf.Graph()
    with graph.as_default():
        test_model = IntentionIdentifier(config=config,
                                         mode=tf.contrib.learn.ModeKeys.EVAL,
                                         embedding_vocab=vocab_loader.vocab_embedding)
        LogInfo.logs("Test model created.")
    LogInfo.end_track()

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True

    LogInfo.begin_track("Start testing...")
    with tf.Session(graph=graph, config=tf_config) as session:
        test_model.load(session, root_path + "/model/best_model")
        batch_size = config.get("test_batch_size")
        batch_num = int(data_loader.data_size / batch_size)
        score_list = list()
        for i in range(batch_num+1):
            LogInfo.logs("Testing batch %d...", i+1)
            if (i+1)*batch_size > data_loader.data_size:
                test_data_batch = data_loader.data[i*batch_size: data_loader.data_size]
            else:
                test_data_batch = data_loader.data[i*batch_size: (i+1)*batch_size]
            context_idx, context_seq, pinlei_idx = zip(*test_data_batch)
            context_idx = np.array(context_idx)
            context_seq = np.array(context_seq)
            pinlei_idx = np.array(pinlei_idx)
            feed_data_batch = [context_idx, context_seq, pinlei_idx]
            score_batch = test_model.eval(session, feed_data_batch)
            score_list.extend(score_batch)
    LogInfo.end_track("%d score got.", len(score_list))

    LogInfo.begin_track("Showing testing result...")
    fin1 = codecs.open(config.get("test_data_fp") + ".check", 'r', encoding='utf-8')
    fin2 = codecs.open(config.get("test_data_fp") + ".name", 'r', encoding='utf-8')
    lines1 = fin1.readlines()
    lines2 = fin2.readlines()
    cnt = 0
    for line1, line2 in zip(lines1, lines2):
        query = line1.strip()
        pinlei = line2.strip().split("\t")[1]
        LogInfo.logs("%s ===> %s ===> %.4f", query, pinlei, score_list[cnt])
        cnt += 1
    LogInfo.end_track()















