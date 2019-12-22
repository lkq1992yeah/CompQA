"""
Author: Kangqi Luo
Goal: Define the common model-related procedures of relation matching, used in both train & test
We call it "compact", because we can represent the whole schemas as a unique vector representation.
The opposite side, "seperated", represent each individual path, and never combine into a overall repr.
"""

import tensorflow as tf

from xusheng.model.rnn_encoder import BidirectionalRNNEncoder
from xusheng.model.attention import CrossAttentionLayer
from ..module.seq_helper import seq_encoding, seq_hidden_averaging, seq_hidden_max_pooling

# from ..u import show_tensor

from kangqi.util.tf.cosine_sim import cosine_sim
# from kangqi.util.LogUtil import LogInfo


class CompactRelationMatchingKernel:
    """
    Note: Throughout all procedures, pn_size never appears, but combined in the first dimension of all tensors.
    """

    def __init__(self, n_words, n_mids, dim_emb,
                 q_max_len, sc_max_len, path_max_len, pword_max_len,
                 rnn_cell, num_units, input_keep_prob, output_keep_prob,
                 preds_agg_mode, pwords_agg_mode, path_merge_mode,
                 reuse=tf.AUTO_REUSE, verbose=0):
        self.reuse = reuse
        self.verbose = verbose

        self.q_max_len = q_max_len
        self.sc_max_len = sc_max_len
        self.path_max_len = path_max_len
        self.pword_max_len = pword_max_len
        self.dim_emb = dim_emb

        self.dim_att_hidden = 128       # force set now

        self.preds_agg_mode = preds_agg_mode
        self.pwords_agg_mode = pwords_agg_mode
        self.path_merge_mode = path_merge_mode
        assert not (preds_agg_mode == 'None' and pwords_agg_mode == 'None')
        # must use at least one information from predicates, either name sequence, or mid sequence

        self.dim_hidden = 2 * num_units     # bidirectional
        self.rnn_config = {
            'num_units': num_units,
            'cell_class': rnn_cell,
            'dropout_input_keep_prob': input_keep_prob,
            'dropout_output_keep_prob': output_keep_prob
        }

        with tf.variable_scope('Embedding_Lookup', reuse=reuse):
            with tf.device('/cpu:0'):
                self.w_embedding_init = tf.placeholder(dtype=tf.float32,
                                                       shape=(n_words, dim_emb),
                                                       name='w_embedding_init')
                self.m_embedding_init = tf.placeholder(dtype=tf.float32,
                                                       shape=(n_mids, dim_emb),
                                                       name='m_embedding_init')
                self.w_embedding = tf.get_variable(name='w_embedding',
                                                   initializer=self.w_embedding_init)
                self.m_embedding = tf.get_variable(name='m_embedding',
                                                   initializer=self.m_embedding_init)

    def get_score(self, mode, qwords_embedding, qwords_len, sc_len,
                  preds_embedding, preds_len, pwords_embedding, pwords_len):
        """
        Produce the final similarity score.
        This function is the most important part in the optm/eval model.
        Just use cosine similarity
        :param mode: tf.contrib.learn.ModeKeys.TRAIN/INFER, which affects the dropout setting
        :param qwords_embedding:    (ds, q_max_len, dim_emb)
        :param qwords_len:          (ds, )
        :param sc_len:              (ds, )
        :param preds_embedding:     (ds, sc_max_len, path_max_len, dim_emb)
        :param preds_len:           (ds, sc_max_len)
        :param pwords_embedding:    (ds, sc_max_len, pword_max_len, dim_emb)
        :param pwords_len:          (ds, sc_max_len)
        :return: (ds, ) as the final similarity score
        """
        assert mode in (tf.contrib.learn.ModeKeys.TRAIN, tf.contrib.learn.ModeKeys.INFER)

        if self.rnn_config['cell_class'] == 'None':
            # won't use any recurrent layer, but just using pure embedding as instead
            self.dim_hidden = self.dim_emb      # force set dim_hidden to be dim_emb
            q_encoder = pred_encoder = pword_encoder = None
        else:
            encoder_args = {'config': self.rnn_config, 'mode': mode}
            q_encoder = BidirectionalRNNEncoder(**encoder_args)
            pred_encoder = BidirectionalRNNEncoder(**encoder_args)
            pword_encoder = BidirectionalRNNEncoder(**encoder_args)
            """
            BidirectionalRNNEncoder will set the dropout according to the current mode (TRAIN/INFER)
            """

        with tf.name_scope('RelationMatchingKernel'):
            with tf.variable_scope('Question', reuse=self.reuse):
                if q_encoder is None:
                    qwords_hidden = qwords_embedding
                    # (ds, q_max_len, dim_hidden=dim_emb)
                else:
                    qwords_hidden = seq_encoding(
                        emb_input=qwords_embedding,
                        len_input=qwords_len,
                        encoder=q_encoder, reuse=self.reuse)   # (ds, q_max_len, dim_hidden)
                q_hidden = seq_hidden_max_pooling(seq_hidden_input=qwords_hidden,
                                                  len_input=qwords_len)
            # (ds, dim_hidden), will be used in the final cosine similarity calculation

            # Step 1:   split schemas into paths
            #           merge ds and sc_max_len into one dimension
            qwords_hidden = tf.reshape(
                tf.stack([qwords_hidden] * self.sc_max_len, axis=1),
                shape=(-1, self.q_max_len, self.dim_hidden),
                name='qwords_hidden'
            )       # (ds * sc_max_len, q_max_len, dim_hidden)
            qwords_len = tf.reshape(
                tf.stack([qwords_len] * self.sc_max_len, axis=1),
                shape=(-1,),
                name='qwords_len'
            )       # (ds * sc_max_len, )
            # Now combine ds and sc_max_len into one dimension

            comb_tensor_list = []
            for tensor_input in (preds_embedding, preds_len, pwords_embedding, pwords_len):
                ori_shape = tensor_input.get_shape().as_list()
                comb_shape = [-1] + ori_shape[2:]       # keep the dimensions after (ds, sc_max_len)
                # show_tensor(tensor_input)
                # LogInfo.logs('ori_shape: %s, comb_shape: %s', ori_shape, comb_shape)
                comb_tensor_list.append(tf.reshape(tensor_input, shape=comb_shape))
            [preds_embedding, preds_len, pwords_embedding, pwords_len] = comb_tensor_list
            # (ds * sc_max_len, xxxxxxx)
            # for tensor in comb_tensor_list:
            #     show_tensor(tensor)

            # Step 2: Compute basic hidden repr.
            # xxx_final_hidden: (ds * sc_max_len, dim_hidden)
            # (Optional) xxx_att_mat: (ds * sc_max_len, q_max_len, xxx_max_len)
            with tf.name_scope('Schema'):
                with tf.variable_scope('preds', reuse=self.reuse):
                    if pred_encoder is None:
                        preds_hidden = preds_embedding
                        # (ds * sc_max_len, path_max_len, dim_hidden=dim_emb)
                    else:
                        preds_hidden = seq_encoding(
                            emb_input=preds_embedding,
                            len_input=preds_len,
                            encoder=pred_encoder,
                            reuse=self.reuse
                        )  # (ds * sc_max_len, path_max_len, dim_hidden)
                    pred_final_hidden, pred_att_mat = self.aggregate_within_path(
                        qwords_hidden=qwords_hidden, qwords_len=qwords_len,
                        pitems_hidden=preds_hidden, pitems_len=preds_len,
                        item_max_len=self.path_max_len, item_agg_mode=self.preds_agg_mode
                    )
                with tf.variable_scope('pwords', reuse=self.reuse):
                    if pword_encoder is None:
                        pwords_hidden = pwords_embedding
                        # (ds * sc_max_len, pword_max_len, dim_hidden=dim_emb)
                    else:
                        pwords_hidden = seq_encoding(
                            emb_input=pwords_embedding,
                            len_input=pwords_len,
                            encoder=pword_encoder,
                            reuse=self.reuse
                        )  # (ds * sc_max_len, pword_max_len, dim_hidden)
                    pword_final_hidden, pword_att_mat = self.aggregate_within_path(
                        qwords_hidden=qwords_hidden, qwords_len=qwords_len,
                        pitems_hidden=pwords_hidden, pitems_len=pwords_len,
                        item_max_len=self.pword_max_len, item_agg_mode=self.pwords_agg_mode
                    )

                # Step 3:   1. merge preds and pwords
                #           2. combine paths into schemas
                #           3. produce the final score
                # path_merge_mode: Max: max pooling
                #                  Sum: simple summation
                with tf.name_scope('PathMerge'):
                    assert not (pword_final_hidden is None and pred_final_hidden is None)
                    if pword_final_hidden is None:      # information comes from pwords only
                        path_final_hidden = pred_final_hidden
                    elif pred_final_hidden is None:     # information comes from preds only
                        path_final_hidden = pword_final_hidden
                    else:               # combine the information from both pwords and preds
                        assert self.path_merge_mode in ('Sum', 'Max')
                        if self.path_merge_mode == 'Sum':
                            path_final_hidden = tf.add(pword_final_hidden, pred_final_hidden,
                                                       name='path_final_hidden')        # (ds * sc_max_len, dim_hidden)
                        else:
                            path_final_hidden = tf.reduce_max(
                                tf.stack([pword_final_hidden,
                                          pred_final_hidden], axis=0),      # (2, ds * sc_max_len, dim_hidden)
                                axis=0, name='path_final_hidden')           # (ds * sc_max_len, dim_hidden)
                    sc_path_hidden = tf.reshape(path_final_hidden,
                                                shape=[-1, self.sc_max_len, self.dim_hidden],
                                                name='sc_path_hidden')              # (ds, sc_max_len, dim_hidden)
                    # max pooling along all paths
                    sc_hidden = seq_hidden_max_pooling(seq_hidden_input=sc_path_hidden,
                                                       len_input=sc_len)        # (ds, dim_hidden)
            score = cosine_sim(lf_input=q_hidden, rt_input=sc_hidden)           # (ds, )

        if pred_att_mat is not None:
            pred_att_mat = tf.reshape(pred_att_mat, [-1, self.sc_max_len, self.q_max_len, self.path_max_len],
                                      name='pred_att_mat')          # (ds, sc_max_len, q_max_len, path_max_len)
        if pword_att_mat is not None:
            pword_att_mat = tf.reshape(pword_att_mat, [-1, self.sc_max_len, self.q_max_len, self.pword_max_len],
                                       name='pword_att_mat')        # (ds, sc_max_len, q_max_len, pword_max_len)
        return pred_att_mat, pword_att_mat, score

    def aggregate_within_path(self, qwords_hidden, qwords_len,
                              pitems_hidden, pitems_len, item_max_len, item_agg_mode):
        """
        Goal: Perform the aggregation procedure within in a path.
        Convert the item representation (-1, item_max_len, dim_hidden) to the final (-1, dim_hidden) output
        The "item" could be either preds or pwords
        :param qwords_hidden:   (-1, q_max_len, dim_hidden)
        :param qwords_len:      (-1, )
        :param pitems_hidden:   (-1, item_max_len, dim_hidden)
        :param pitems_len:      (-1, )
        :param item_max_len:    either path_max_len, or pword_max_len
        :param item_agg_mode:   "Max", "Avg", "CAtt"
                                Max: MaxPooling
                                Avg: Averaging along the sequence
                                CAtt: CrossAttention
        :return:    * the final representation (-1, dim_hidden) and,
                    * optional: the attention matrix (-1, q_max_len, item_max_len)
        """
        assert item_agg_mode in ('Max', 'Avg', 'CAtt', 'None')
        if item_agg_mode == 'CAtt':     # CrossAttention
            cross_att = CrossAttentionLayer(left_max_len=self.q_max_len,
                                            right_max_len=item_max_len,
                                            hidden_dim=self.dim_att_hidden)
            _, pitem_final_hidden, att_matrix, _ = cross_att.compute_attention(
                left_tensor=qwords_hidden, left_len=qwords_len,
                right_tensor=pitems_hidden, right_len=pitems_len
            )
            return pitem_final_hidden, att_matrix
        elif item_agg_mode == 'Max':
            pitem_final_hidden = seq_hidden_max_pooling(seq_hidden_input=pitems_hidden,
                                                        len_input=pitems_len)       # (-1, dim_hidden)

            return pitem_final_hidden, None
        elif item_agg_mode == 'Avg':
            pitem_final_hidden = seq_hidden_averaging(seq_hidden_input=pitems_hidden,
                                                      len_input=pitems_len)
            return pitem_final_hidden, None
        else:       # None: produces nothing
            return None, None
