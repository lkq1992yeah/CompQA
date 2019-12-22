"""
Author: Kangqi Luo
Goal: Implment yu2017improved
Relation Detection Only
Each data: <q, r> (pn_size is not used in the model)
"""

import tensorflow as tf

from .u import seq_encoding, schema_encoding, get_merge_function
from ..u import show_tensor

from ..base_model import BaseModel
from xusheng.model.rnn_encoder import BidirectionalRNNEncoder

from ...dataset import SimpqSingleDataLoader

from kangqi.util.LogUtil import LogInfo


class SimpqEvalModel(BaseModel):

    def __init__(self, sess, n_words, n_preds, dim_emb,
                 q_max_len, path_max_len, pword_max_len,
                 dim_hidden, rnn_cell, merge_config,
                 reuse=tf.AUTO_REUSE, verbose=0):
        LogInfo.begin_track('SimpqEvalModel Building ...')
        super(SimpqEvalModel, self).__init__(sess=sess, verbose=verbose)

        # ======== declare sub-modules (the same as optm part) ======== #

        num_units = dim_hidden / 2  # bidirectional
        rnn_config = {'num_units': num_units, 'cell_class': rnn_cell}
        encoder_args = {'config': rnn_config, 'mode': tf.contrib.learn.ModeKeys.TRAIN}

        q_encoder = BidirectionalRNNEncoder(**encoder_args)
        pred_encoder = BidirectionalRNNEncoder(**encoder_args)
        pword_encoder = BidirectionalRNNEncoder(**encoder_args)
        merge_func = get_merge_function(merge_config=merge_config, dim_hidden=dim_hidden, reuse=reuse)
        LogInfo.logs('Sub-modules declared.')

        # ======== define tensors ======== #

        q_words_input = tf.placeholder(dtype=tf.int32,
                                       shape=[None, q_max_len],
                                       name='q_words_input')  # (data_size, q_max_len)
        q_words_len_input = tf.placeholder(dtype=tf.int32,
                                           shape=[None],
                                           name='q_words_len_input')  # (data_size, )
        preds_input = tf.placeholder(dtype=tf.int32,
                                     shape=[None, path_max_len],
                                     name='preds_input')            # (data_size, path_max_len)
        preds_len_input = tf.placeholder(dtype=tf.int32,
                                         shape=[None],
                                         name='preds_len_input')    # (data_size, )
        pwords_input = tf.placeholder(dtype=tf.int32,
                                      shape=[None, pword_max_len],
                                      name='pwords_input')          # (data_size, path_max_len)
        pwords_len_input = tf.placeholder(dtype=tf.int32,
                                          shape=[None],
                                          name='pwords_len_input')  # (data_size, )
        self.eval_input_tf_list = [
            q_words_input, q_words_len_input,
            preds_input, preds_len_input,
            pwords_input, pwords_len_input
        ]
        LogInfo.begin_track('Showing %d input tensors:', len(self.eval_input_tf_list))
        for tensor in self.eval_input_tf_list:
            show_tensor(tensor)
        LogInfo.end_track()

        # ======== start building model ======== #

        with tf.variable_scope('Embedding_Lookup', reuse=reuse):
            with tf.device("/cpu:0"):
                self.w_embedding_init = tf.placeholder(dtype=tf.float32,
                                                       shape=(n_words, dim_emb),
                                                       name='w_embedding_init')
                self.p_embedding_init = tf.placeholder(dtype=tf.float32,
                                                       shape=(n_preds, dim_emb),
                                                       name='p_embedding_init')
                w_embedding = tf.get_variable(name='w_embedding',
                                              initializer=self.w_embedding_init)
                p_embedding = tf.get_variable(name='p_embedding',
                                              initializer=self.p_embedding_init)

                q_words_embedding = tf.nn.embedding_lookup(
                    params=w_embedding, ids=q_words_input, name='q_embedding'
                )  # (batch, q_max_len, dim_emb)
                preds_embedding = tf.nn.embedding_lookup(
                    params=p_embedding, ids=preds_input, name='preds_embedding'
                )  # (batch, path_max_len, dim_emb)
                pwords_embedding = tf.nn.embedding_lookup(
                    params=w_embedding, ids=pwords_input, name='pwords_embedding'
                )  # (batch, pword_max_len, dim_emb)

        with tf.variable_scope('Question', reuse=reuse):
            q_words_hidden = seq_encoding(
                emb_input=q_words_embedding,
                len_input=q_words_len_input,
                encoder=q_encoder, reuse=reuse)         # (data_size, q_max_len, dim_emb)
            q_hidden = tf.reduce_max(
                q_words_hidden, axis=1, name='q_hidden'
            )    # (data_size, dim_hidden)

        with tf.variable_scope('Schema', reuse=reuse):
            with tf.variable_scope('Path', reuse=reuse):
                preds_hidden = seq_encoding(
                    emb_input=preds_embedding,
                    len_input=preds_len_input,
                    encoder=pred_encoder, reuse=reuse)  # (data_size, path_max_len, dim_emb)
            with tf.variable_scope('Pword', reuse=reuse):
                pwords_hidden = seq_encoding(
                    emb_input=pwords_embedding,
                    len_input=pwords_len_input,
                    encoder=pword_encoder, reuse=reuse)  # (data_size, pword_max_len, dim_emb)
            schema_hidden = schema_encoding(
                preds_hidden=preds_hidden, preds_len=preds_len_input,
                pwords_hidden=pwords_hidden, pwords_len=pwords_len_input)

        with tf.variable_scope('Merge', reuse=reuse):
            # self.score = cosine_sim(lf_input=q_hidden, rt_input=schema_hidden)    # (data_size, )
            self.score = merge_func(q_hidden, schema_hidden)        # (data_size, )
            # Now final score defined.

        self.eval_summary = tf.summary.merge_all(key='eval')
        LogInfo.logs('* final score defined.')

        LogInfo.end_track()

    def evaluate(self, data_loader, epoch_idx, ob_batch_num=50, detail_fp=None, summary_writer=None):
        """
        Goal: output the score of each candidate.
        After scanning all the candidates, we check the score with the original dataset,
        and calculate the final accuracy.
        """
        if data_loader is None or len(data_loader) == 0:
            return 0.

        assert isinstance(data_loader, SimpqSingleDataLoader)
        self.prepare_data(data_loader=data_loader)

        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()

        score_list = []
        scan_size = 0
        for batch_idx in range(data_loader.n_batch):
            local_data_list, local_indices = data_loader.get_next_batch()
            local_size = len(local_data_list[0])    # the first dimension is always batch size
            fd = {input_tf: local_data for input_tf, local_data in zip(self.eval_input_tf_list, local_data_list)}

            local_score_list = self.sess.run(
                self.score, feed_dict=fd,
                options=run_options, run_metadata=run_metadata
            )
            score_list += local_score_list.tolist()
            scan_size += local_size
            if (batch_idx + 1) % ob_batch_num == 0:
                LogInfo.logs('[eval-%s-B%d/%d] scanned = %d/%d',
                             data_loader.mode,
                             batch_idx + 1,
                             data_loader.n_batch,
                             scan_size,
                             len(data_loader))

            if summary_writer is not None:
                # summary_writer.add_summary(summary, point)
                if batch_idx == 0:
                    summary_writer.add_run_metadata(run_metadata, 'epoch-%d' % epoch_idx)
                    # for running time info, just keep trace of the first batch of every epoch.

        # After scoring all candidates, now we count the final accuracy.
        assert len(score_list) == len(data_loader.cand_tup_list)
        q_predict_dict = {}     # <q, [(gold, predict)]>
        for (q_idx, cand), score in zip(data_loader.cand_tup_list, score_list):
            cand.run_info = {'score': score}
            q_predict_dict.setdefault(q_idx, []).append((cand.f1, score))

        correct = 0
        for q_idx, predict_tup_list in q_predict_dict.items():
            predict_tup_list.sort(key=lambda tup: tup[1], reverse=True)
            if predict_tup_list[0][0] == 1.:
                correct += 1
        LogInfo.logs('%d / %d questions correct.', correct, len(q_predict_dict))
        acc = 1. * correct / len(q_predict_dict)
        return acc
