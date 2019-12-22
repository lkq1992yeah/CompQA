# -*- coding: utf-8 -*-

#==============================================================================
# Author: Kangqi Luo
# Goal: Takes q and ordinal constraint <x, p, op, o> as input,
# returns the hidden feature capturing both lexical similarity and structural compatibility.
#==============================================================================


import tensorflow as tf

from kangqi.util.tf.cosine_sim import get_length_tf, get_cosine_tf
from kangqi.util.LogUtil import LogInfo

class Ordinal_Ours(object):

    def __init__(self, n_word_emb, n_cnn_hidden,
             n_steps, n_rnn_hidden,
             n_pred, n_compat_hidden, activation):
        # we don't need n_compat_hidden to be so big.
        self.n_word_emb = n_word_emb
        self.n_cnn_hidden = n_cnn_hidden
        self.n_steps = n_steps
        self.n_rnn_hidden = n_rnn_hidden
        self.n_pred = n_pred
        self.n_compat_hidden = n_compat_hidden
        self.activation = activation


    # Now let's build the function
    # Note: data_size = ds = batch * PN
    # q_cnn_tf: (ds, n_cnn_hidden): hidden feature vector of the question
    # sk_rnn_state_tf: (ds, n_steps, n_rnn_hidden): hidden state of each var in the path
    # x_tf: (ds,) as int32 (indicating position of variable on the skeleton)
    # pred_tf: (ds, n_pred): the ordinal predicate
    # op_tf: (ds, 2) indicating ASC or DESC
    # obj_tf: (ds, n_word_emb): the embedding of object
    # * Here the object is a rank, but we'd like to use the embedding of the corresponding word
    # * Therefore, we use n_word_emb as the dimension of object.
    # mask_tf: (ds,) indicating whether there's an ordinal constraint or not
    # (Option) If mask_tf = 0, then we could set the overall ordinal feature to be 0.

    # Some parameters are shared across different constraints.
    # W/b_compat: compatibility weight of <s, p> pairs (n_rnn_hidden + n_pred, n_compat_hidden)
    # W/b_pred: (n_pred, n_cnn_hidden)
    #           turning predicate into "n_cnn_hidden" vector,
    #           so we can calculate cosine similarity between pred and q.
    # W/b_op: (2, n_cnn_hidden) Similar as W/b_pred.
    # W/b_obj: (n_word_emb, n_cnn_hidden) Similar as params above.
    # (Option) We could add attention mechanism over pred / op matrices.

    # Output: (ds, n_compat_hidden + 3) tensor.
    # consists of compatibility feature, with 3 similarity feature.
    def build(self, q_cnn_tf, sk_rnn_state_tf,
              x_tf, pred_tf, op_tf, obj_tf, mask_tf,
              W_compat, b_compat, W_pred, b_pred,
              W_op, b_op, W_obj, b_obj):
        row_tf = tf.range(tf.shape(sk_rnn_state_tf)[0])
        coo_tf = tf.stack([row_tf, x_tf], axis=-1) # (ds, 2)
        # build the co-ordination tensor for retrieving the target hidden state from rnn

        subj_tf = tf.gather_nd(sk_rnn_state_tf, coo_tf) # (ds, n_rnn_hidden)
        compat_tf = tf.concat([subj_tf, pred_tf], axis=-1)
        # (s, n_rnn_hidden + n_pred), ready to generate hidden features for <s, p> compatibility

        h_compat_tf = self.activation(tf.matmul(compat_tf, W_compat) + b_compat)
        # (ds, n_compat_hidden) as the hidden feature of <s, p>

        h_pred_tf = self.activation(tf.matmul(pred_tf, W_pred) + b_pred)
        h_op_tf = self.activation(tf.matmul(op_tf, W_op) + b_op)
        h_obj_tf = self.activation(tf.matmul(obj_tf, W_obj) + b_obj)
        item_tf_list = [h_pred_tf, h_op_tf, h_obj_tf]
        # (ds, n_cnn_hidden) Projecting predicate / op / obj into the same vector space as q_cnn_tf

        # Now compute cosine similarity between q_cnn_tf and h_pred_tf/h_op_tf/h_obj_tf
        q_cnn_length_tf = get_length_tf(q_cnn_tf)
        item_len_tf_list = [get_length_tf(item_tf) for item_tf in item_tf_list]
        # both l2-norm tensor: (ds, 1)

        cos_tf_list = [
            get_cosine_tf(q_cnn_tf, item_tf, q_cnn_length_tf, item_len_tf)
            for item_tf, item_len_tf in zip(item_tf_list, item_len_tf_list)
        ]   # both cosine tf: (ds, 1)
        # Now we've got cosine similarity between q and predicate / op / obj
        # ready to merge into the final hidden vector

        combine_tf_list = [h_compat_tf] + cos_tf_list
        h_ordinal_tf = tf.concat(combine_tf_list, axis=-1)
        # combine_tf: (ds, n_compat_hidden + 3)

        LogInfo.logs('h_ordinal_tf compiled. %s', h_ordinal_tf.get_shape().as_list())
        LogInfo.logs('* Ordinal_Ours built.')
        return h_ordinal_tf


