"""
Author: Kangqi Luo
Date: 171221
Goal: For the debugging of tensorflow speedup issues.
Checkout the note on Evernote for detail.

The code itself copies from compq_optm_model.py
"""


import tensorflow as tf

from ..compq_overall.relation_matching_kernel import RelationMatchingKernel as CompqKernel

from ..u import show_tensor
from ..base_model import BaseModel
from ..simpq_rel_detect_yu2017.u import seq_encoding, seq_hidden_max_pooling, seq_hidden_averaging
from kangqi.util.tf.cosine_sim import cosine_sim

from kangqi.util.LogUtil import LogInfo
from kangqi.util.time_track import TimeTracker as Tt


class DebugOptmModel(BaseModel):

    def __init__(self, sess, compq_kernel,
                 margin, learning_rate, optm_name, debug_mode):
        LogInfo.begin_track('CompqOptmModel Building ...')
        assert isinstance(compq_kernel, CompqKernel)
        verbose = compq_kernel.verbose
        reuse = compq_kernel.reuse
        assert optm_name in ('Adam', 'Adadelta', 'Adagrad', 'GradientDescent')
        optm_name += 'Optimizer'

        assert debug_mode in ('Loss', 'Grad', 'Update', 'Raw')
        self.debug_mode = debug_mode
        # debug_mode: specify what does the model evaluates when optimizing
        # 1. Loss: just evaluate the avg_loss
        # 2. Grad: evaluate loss + gradient (computed by tf.gradient(), with gradient clipping)
        # 3. Update: evaluate loss + gradient + updating (apply_gradients )
        # 4. Raw: evaluate loss + updating (but use the original coding style, without calling tf.gradient())

        super(DebugOptmModel, self).__init__(sess=sess, verbose=verbose)        # TODO: ob_batch_num = 20

        # ======== define tensors ======== #

        q_max_len = compq_kernel.q_max_len
        path_max_len = compq_kernel.path_max_len
        pword_max_len = compq_kernel.pword_max_len

        self.optm_input_tf_list = []
        sc_tensor_groups = []  # [ pos_tensors, neg_tensors ]
        for cate in ('pos', 'neg'):
            qwords_input = tf.placeholder(dtype=tf.int32,
                                          shape=[None, q_max_len],
                                          name=cate + '_qwords_input')  # (data_size, q_max_len)
            qwords_len_input = tf.placeholder(dtype=tf.int32,
                                              shape=[None],
                                              name=cate + '_qwords_len_input')  # (data_size, )
            sc_len_input = tf.placeholder(dtype=tf.int32,
                                          shape=[None],
                                          name=cate + '_sc_len_input')  # (data_size, )
            preds_input = tf.placeholder(dtype=tf.int32,
                                         shape=[None, None, path_max_len],
                                         name=cate + '_preds_input')  # (data_size, sc_max_len, path_max_len)
            preds_len_input = tf.placeholder(dtype=tf.int32,
                                             shape=[None, None],
                                             name=cate + '_preds_len_input')  # (data_size, sc_max_len)
            pwords_input = tf.placeholder(dtype=tf.int32,
                                          shape=[None, None, pword_max_len],
                                          name=cate + '_pwords_input')  # (data_size, sc_max_len, pword_max_len)
            pwords_len_input = tf.placeholder(dtype=tf.int32,
                                              shape=[None, None],
                                              name=cate + '_pwords_len_input')  # (data_size, sc_max_len)
            tensor_group = [qwords_input, qwords_len_input, sc_len_input,
                            preds_input, preds_len_input,
                            pwords_input, pwords_len_input]
            sc_tensor_groups.append(tensor_group)
            self.optm_input_tf_list += tensor_group
        LogInfo.begin_track('Showing %d input tensors:', len(self.optm_input_tf_list))
        for tensor in self.optm_input_tf_list:
            show_tensor(tensor)
        LogInfo.end_track()

        # ======== start building model ======== #

        with tf.name_scope('Debug_Optm'):
            score_list = []     # store two tensors: positive and negative score
            for cate, sc_tensor_group in zip(('pos', 'neg'), sc_tensor_groups):
                LogInfo.begin_track('Calculate score at %s side ...', cate)
                [qwords_input, qwords_len_input, sc_len_input,
                 preds_input, preds_len_input,
                 pwords_input, pwords_len_input] = sc_tensor_group
                with tf.device("/cpu:0"):
                    qwords_embedding = tf.nn.embedding_lookup(
                        params=compq_kernel.w_embedding,
                        ids=qwords_input, name='q_embedding'
                    )  # (data_size, q_max_len, dim_emb)
                    preds_embedding = tf.nn.embedding_lookup(
                        params=compq_kernel.m_embedding,
                        ids=preds_input, name='preds_embedding'
                    )  # (data_size, sc_max_len, path_max_len, dim_emb)
                    pwords_embedding = tf.nn.embedding_lookup(
                        params=compq_kernel.w_embedding,
                        ids=pwords_input, name='pwords_embedding'
                    )  # (data_size, sc_max_len, pword_max_len, dim_emb)

                _, _, score = compq_kernel.get_score(
                    qwords_embedding=qwords_embedding, qwords_len=qwords_len_input, sc_len=sc_len_input,
                    preds_embedding=preds_embedding, preds_len=preds_len_input,
                    pwords_embedding=pwords_embedding, pwords_len=pwords_len_input
                )  # (data_size, ), currently we ignore the attention matrix information
                score_list.append(score)
                LogInfo.end_track()

            pos_score, neg_score = score_list
            margin_loss = tf.nn.relu(neg_score + margin - pos_score,
                                     name='margin_loss')
            self.avg_loss = tf.reduce_mean(margin_loss, name='avg_loss')

            optimizer = getattr(tf.train, optm_name)
            if self.debug_mode in ('Loss', 'Grad', 'Update'):
                params = tf.trainable_variables()  # code style from https://github.com/tensorflow/nmt
                gradients = tf.gradients(self.avg_loss, params)
                self.clipped_gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=1.0)
                self.optm_step = optimizer(learning_rate).apply_gradients(zip(self.clipped_gradients, params))
            else:       # Raw updating code style
                self.optm_step = optimizer(learning_rate).minimize(self.avg_loss)

        tf.summary.scalar('avg_loss', self.avg_loss, collections=['optm'])
        self.optm_summary = tf.summary.merge_all(key='optm')
        LogInfo.logs('* avg_loss and optm_step defined.')

        LogInfo.end_track()

    def optimize(self, data_loader, epoch_idx, ob_batch_num=10, summary_writer=None):
        if data_loader is None:
            return -1.

        LogInfo.logs('Debug mode: %s', self.debug_mode)
        Tt.start('optimize')
        Tt.start('prepare')
        self.prepare_data(data_loader=data_loader)
        # run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        # run_metadata = tf.RunMetadata()
        # tf.reset_default_graph()
        Tt.record('prepare')
        scan_size = 0
        ret_loss = 0.
        for batch_idx in range(data_loader.n_batch):
            # point = (epoch_idx - 1) * data_loader.n_batch + batch_idx
            Tt.start('allocate')
            local_data_list, _ = data_loader.get_next_batch()
            local_size = len(local_data_list[0])    # the first dimension is always batch size
            fd = {input_tf: local_data for input_tf, local_data in zip(self.optm_input_tf_list, local_data_list)}
            Tt.record('allocate')
            Tt.start('running')
            if self.debug_mode == 'Loss':
                local_loss, summary = self.sess.run(
                    [self.avg_loss, self.optm_summary],
                    feed_dict=fd,
                    # options=run_options,
                    # run_metadata=run_metadata
                )
            elif self.debug_mode == 'Grad':
                _, local_loss, summary = self.sess.run(
                    [self.clipped_gradients, self.avg_loss, self.optm_summary],
                    feed_dict=fd,
                    # options=run_options,
                    # run_metadata=run_metadata
                )
            else:       # Update or Raw
                _, local_loss, summary = self.sess.run(
                    [self.optm_step, self.avg_loss, self.optm_summary],
                    feed_dict=fd,
                    # options=run_options,
                    # run_metadata=run_metadata
                )
            Tt.record('running')

            Tt.start('display')
            ret_loss = (ret_loss * scan_size + local_loss * local_size) / (scan_size + local_size)
            scan_size += local_size
            if (batch_idx+1) % ob_batch_num == 0:
                LogInfo.logs('[optm-%s-B%d/%d] avg_loss = %.6f, scanned = %d/%d',
                             data_loader.mode,
                             batch_idx+1,
                             data_loader.n_batch,
                             ret_loss,
                             scan_size,
                             len(data_loader))
            # if summary_writer is not None:
            #     summary_writer.add_summary(summary, point)
            #     if batch_idx == 0:
            #         summary_writer.add_run_metadata(run_metadata, 'epoch-%d' % epoch_idx)
            Tt.record('display')

        Tt.record('optimize')
        return ret_loss
