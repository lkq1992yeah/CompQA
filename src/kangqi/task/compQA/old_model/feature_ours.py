# -*- coding: utf-8 -*-

#==============================================================================
# Author: Kangqi Luo
# Goal: Construct our model: stack different components together
# Return the whole features for a single <q, schema>
#==============================================================================

import tensorflow as tf

from . import QuestionCNN, SkeletonRNN, Ordinal_Ours

from kangqi.util.tf.tf_basics import weight_variable, bias_variable
from kangqi.util.LogUtil import LogInfo

class Hidden_Feature_Ours(object):

    def __init__(self, n_el,
                 sent_max_len, n_word_emb, n_window, n_cnn_hidden,
                 path_max_len, n_path_emb, n_rnn_hidden,
                 n_feat_sk, n_compat_hidden, activation, sc_mode):
        self.n_el = n_el
        self.n_feat_sk = n_feat_sk    # hidden feature capturing similarity bet. q and sk
        self.activation = activation

        # q_cnn part
        self.n_window = n_window
        self.n_maxlen = sent_max_len
        self.n_cnn_input = n_word_emb
        self.n_cnn_hidden = n_cnn_hidden
        self.q_cnn = QuestionCNN(
                n_maxlen=self.n_maxlen,
                n_input=self.n_cnn_input,
                n_hidden=self.n_cnn_hidden,
                n_window=self.n_window,
                activation=self.activation
        )

        # sk_rnn part
        self.n_steps = path_max_len
        self.n_rnn_input = self.n_path_emb = n_path_emb
        self.n_rnn_hidden = n_rnn_hidden
        self.sk_rnn = SkeletonRNN(
                n_steps=self.n_steps,
                n_input=self.n_rnn_input,
                n_hidden=self.n_rnn_hidden
        )

        self.n_compat_hidden = n_compat_hidden
        self.ordinal = Ordinal_Ours(
                n_word_emb=self.n_cnn_input,
                n_cnn_hidden=self.n_cnn_hidden,
                n_steps=self.n_steps,
                n_rnn_hidden=self.n_rnn_hidden,
                n_pred=self.n_path_emb,
                n_compat_hidden=self.n_compat_hidden,
                activation=self.activation
        )

        assert sc_mode in ('Skeleton', 'Sk+Ordinal')
        self.sc_mode = sc_mode


    # Note: data_size = ds = batch * PN
    # q_tf: (ds, n_maxlen, n_cnn_input)
    # sk_tf: (ds, n_steps, n_rnn_input)
    # el_tf: (ds, n_el)    # just an entity linking score (discrete version)
    # ordinal_tf_list consists of the following tensors:
    #     x_tf, pred_tf, op_tf, obj_tf, mask_tf. Check ordinal_ours.py for detail information.
    def build(self, q_tf, sk_tf, el_tf, ordinal_tf_list):
        part_tf_info_list = []  # collect feature tf and length for each part of the model

        part_tf_info_list.append((el_tf, self.n_el))
        # First part: entity linking feature

        q_cnn_tf = self.q_cnn.build(q_tf=q_tf) # (ds, n_cnn_hidden)
        sk_rnn_state_tf, sk_rnn_tf = \
            self.sk_rnn.build(sk_tf=sk_tf)
        # sk_rnn_state_tf: (ds, n_steps, 2 * n_rnn_hidden)
        # sk_rnn_tf: (ds, 2 * n_rnn_hidden)

        n_h_sk = self.n_cnn_hidden + 2 * self.n_rnn_hidden # RNN: FwBk output mode
        h_sk_tf = tf.concat(values=[q_cnn_tf, sk_rnn_tf], axis=1) # (ds, n_cnn_hid + n_rnn_hid)
        # h_sk_tf = tf.concat(1, [q_cnn_tf, sk_rnn_tf]) # (ds, n_cnn_hid + n_rnn_hid) # for 0.12.1
        W_feat_sk = weight_variable([n_h_sk, self.n_feat_sk])
        b_feat_sk = bias_variable([self.n_feat_sk])
        h_feat_sk_tf = self.activation(tf.matmul(h_sk_tf, W_feat_sk) + b_feat_sk)
        part_tf_info_list.append((h_feat_sk_tf, self.n_feat_sk))
        # h_feat_sk_tf (ds, n_feat_sk):
        # capturing similarity features between question and skeleton


        # ======== Now Define Constraint Hidden Tensors ======== #
        if self.sc_mode == 'Sk+Ordinal':
            W_compat = weight_variable([2 * self.n_rnn_hidden + self.n_path_emb, self.n_compat_hidden])
            b_compat = bias_variable([self.n_compat_hidden])
            W_pred = weight_variable([self.n_path_emb, self.n_cnn_hidden])
            b_pred = bias_variable([self.n_cnn_hidden])
            W_op = weight_variable([2, self.n_cnn_hidden])
            b_op = bias_variable([self.n_cnn_hidden])
            W_obj = weight_variable([self.n_cnn_input, self.n_cnn_hidden])
            b_obj = bias_variable([self.n_cnn_hidden])

            ord_x_tf, ord_pred_tf, ord_op_tf, ord_obj_tf, ord_mask_tf = ordinal_tf_list
            h_ordinal_tf = self.ordinal.build(
                    q_cnn_tf, sk_rnn_state_tf,
                    ord_x_tf, ord_pred_tf, ord_op_tf, ord_obj_tf, ord_mask_tf,
                    W_compat, b_compat, W_pred, b_pred,
                    W_op, b_op, W_obj, b_obj)
            part_tf_info_list.append((h_ordinal_tf, self.n_compat_hidden + 3))
            # h_ordinal_tf: (ds, n_compat_hidden + 3)
        # ====================================================== #

        # Now we build the overall features.
        n_h_whole = sum([tf_info[1] for tf_info in part_tf_info_list]) # sum length of each sub-feature
        part_tf_list = [tf_info[0] for tf_info in part_tf_info_list]
        h_whole_tf = tf.concat(values=part_tf_list, axis=1)
        # (ds, n_h_whole): the hidden feature of schemas

        LogInfo.logs('h_whole_tf compiled. %s', h_whole_tf.get_shape().as_list())

        W_whole = weight_variable([n_h_whole, 1])
        b_whole = bias_variable([1])
        score_tf_raw = tf.matmul(h_whole_tf, W_whole) + b_whole # (ds, 1) (linear output)
        score_tf = score_tf_raw
        # score_tf = tf.nn.relu(score_tf_raw)     # make sure that all score >= 0, which makes HingeLoss easier.

        LogInfo.logs('* Hidden_Feature_Ours built.')
        return score_tf
