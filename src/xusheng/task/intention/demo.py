"""
Testing the trained best model to identify multi-pinlei queries
Test data is from demo keyboard input
"""

import sys
import numpy as np
import tensorflow as tf
import codecs

from xusheng.task.intention.data.data import DataLoader
from xusheng.task.intention.model.identifier import IntentionIdentifier
from xusheng.task.intention.data.misc import MultiPinleiEvalDataAdapter
from xusheng.util.config import ConfigDict
from xusheng.util.data_util import VocabularyLoader
from xusheng.util.log_util import LogInfo

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

if __name__ == '__main__':
    setting_dir = sys.argv[1]
    try_dir = sys.argv[2]

    # load configuration
    root_path = 'runnings/%s/%s' % (setting_dir, try_dir)
    config_path = '%s/param_config' % root_path
    config = ConfigDict(config_path)

    # load vocabulary
    vocab_loader = VocabularyLoader()
    vocab_loader.load_vocab(config.get("vocab_fp"), config.get("embedding_dim"), 'utf-8')
    config.add("vocab_size", vocab_loader.vocab_size)
    LogInfo.logs("Embedding shape: %s.", vocab_loader.vocab_embedding.shape)

    # load pinlei
    data_adapter = MultiPinleiEvalDataAdapter()
    data_adapter.load_pinlei()

    # load auxiliary words
    auxiliary_words = set()
    with codecs.open(config.get("auxiliary_words_fp"), 'r', encoding='utf-8') as fin:
        for line in fin:
            auxiliary_words.add(line.strip())
    LogInfo.logs("%d auxiliary words loaded.", len(auxiliary_words))

    # load location words
    location_words = set()
    with codecs.open(config.get("location_words_fp"), 'r', encoding='utf-8') as fin:
        for line in fin:
            location_words.add(line.strip())
    LogInfo.logs("%d location words loaded.", len(location_words))

    # load pre fix words
    pre_words = set()
    with codecs.open(config.get("pre_words_fp"), 'r', encoding='utf-8') as fin:
        for line in fin:
            pre_words.add(line.strip())
    LogInfo.logs("%d pre words loaded.", len(pre_words))

    # load multi-pinlei co-occur relationship from jiahe
    pinlei_pairs = dict()
    with codecs.open(config.get("pair_data_fp"), 'r', encoding='utf-8') as fin:
        for line in fin:
            spt = line.strip().split()
            if len(spt) < 3:
                LogInfo.logs("[error] bad line: %s", line)
            # store the sub-pinlei to delete in following algorithm
            pinlei_pairs['[['+spt[0]+']] [['+spt[1]+']]'] = '[['+spt[1]+']]'
            pinlei_pairs['[['+spt[1]+']] [['+spt[0]+']]'] = '[['+spt[1]+']]'
    LogInfo.logs("%d pinlei pairs loaded.", len(pinlei_pairs))

    # data transformer
    data_feeder = DataLoader(config.get("max_seq_len"), vocab_loader.vocab_index_dict)

    # create model
    LogInfo.begin_track("Create models...")
    graph = tf.Graph()
    with graph.as_default():
        test_model = IntentionIdentifier(config=config,
                                         mode=tf.contrib.learn.ModeKeys.TRAIN,
                                         embedding_vocab=vocab_loader.vocab_embedding)
        LogInfo.logs("Test model created.")
    LogInfo.end_track()

    # tensorflow configuration
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True

    # testing started
    LogInfo.begin_track("Start testing...")
    with tf.Session(graph=graph, config=tf_config) as session:
        test_model.load(session, root_path + "/model/best_model")
        cnt = 0
        while True:
            LogInfo.logs("Please input your segmented (split by \' \') query:")
            query = input()
            LogInfo.begin_track("\nQuery: %s", query)
            if query == "exit" or query == "q":
                LogInfo.logs("Abort testing...")
                break
            cnt += 1
            raw, context, label = data_adapter.tag_pinlei(query)
            LogInfo.logs("Context: %s", context)
            LogInfo.logs("Pinlei: %s", " ".join(label))
            if len(label) == 0:
                LogInfo.logs("[error] Pinlei not found.")
                LogInfo.end_track()
                continue
            elif len(label) == 1:
                LogInfo.logs("Query Intent is %s.", label[0])
                LogInfo.end_track()
                continue
            else:
                pinlei_words = dict()
                query_list = raw.split(" ")
                for pinlei in label:
                    pinlei_words[pinlei] = query_list.index(pinlei)
                to_determine = list()
                solved = False
                # 1. rule, check auxiliary/location/pre-fix words
                for pinlei, idx in pinlei_words.items():
                    if idx+1 < len(query_list) and query_list[idx+1] in auxiliary_words \
                        or idx+2 < len(query_list) \
                            and query_list[idx+1] in location_words \
                            and query_list[idx+2] in auxiliary_words \
                            or idx > 0 and query_list[idx-1] in pre_words:
                        continue
                    else:
                        to_determine.append(pinlei)
                if len(to_determine) == 0:
                    to_determine = list(label)
                if len(to_determine) == 1:
                    LogInfo.logs("Query Intent is %s.", to_determine[0])
                    LogInfo.end_track()
                    continue
                # 2. use pinlei pair table
                while len(to_determine) > 1:
                    key = to_determine[0]+' '+to_determine[1]
                    if key in pinlei_pairs:
                        to_determine.remove(pinlei_pairs[key])
                    else:
                        break
                if len(to_determine) == 1:
                    LogInfo.logs("Query Intent is %s.", to_determine[0])
                    LogInfo.end_track()
                    continue

                if len(context) == 0:
                    last = 0
                    for pinlei in to_determine:
                        if pinlei_words[pinlei] > last:
                            last = pinlei_words[pinlei]
                    LogInfo.logs("Query Intent is %s.", query_list[last])
                    LogInfo.end_track()
                    continue

                context_idxs = list()
                context_seqs = list()
                pinlei_idxs = list()
                # attention: to_determine or label ?
                for pinlei in to_determine:
                    new_line = context + "\t" + pinlei
                    LogInfo.logs("Model Input: %s", new_line)
                    context_idx, context_seq, pinlei_idx = data_feeder.decode_line(new_line)
                    context_idxs.append(context_idx)
                    context_seqs.append(context_seq)
                    pinlei_idxs.append(pinlei_idx)
                LogInfo.begin_track("Feed dict = ")
                for context_idx, context_seq, pinlei_idx in zip(context_idxs, context_seqs, pinlei_idxs):
                    LogInfo.logs("%s\t%s\t%s", context_idx, context_seq, pinlei_idx)
                LogInfo.end_track()
                feed_data = [context_idxs, context_seqs, pinlei_idxs]
                score_list = test_model.eval(session, feed_data)
                LogInfo.logs("Result:")
                max_score = - 999999.0
                ret_pinlei = ""
                for pinlei, score in zip(to_determine, score_list):
                    LogInfo.logs("%s ===> score: %.4f", pinlei, score)
                    if score > max_score:
                        max_score = score
                        ret_pinlei = pinlei
                LogInfo.logs("Query Intent is %s.", ret_pinlei)
                LogInfo.end_track()
    LogInfo.end_track("%d query tested.", cnt)




