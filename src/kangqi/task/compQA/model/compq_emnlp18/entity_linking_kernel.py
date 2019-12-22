import tensorflow as tf

from ..module.simple_attention import SimpleAttention
from ..module.seq_helper import seq_encoding, seq_encoding_with_aggregation

from xusheng.model.rnn_encoder import BidirectionalRNNEncoder
from kangqi.util.tf.cosine_sim import cosine_sim
from kangqi.util.LogUtil import LogInfo


class EntityLinkingKernel:

    def __init__(self, dim_emb, qw_max_len, pw_max_len,
                 seq_merge_mode, scoring_mode,
                 rnn_config=None, att_config=None):
        # qw_max_len, pw_max_len, dim_emb, el_max_len, dim_el_hidden, keep_prob, use_type):
        self.dim_emb = dim_emb
        self.qw_max_len = qw_max_len
        self.pw_max_len = pw_max_len
        self.rnn_config = rnn_config
        self.scoring_mode = scoring_mode
        self.seq_merge_mode = seq_merge_mode
        self.att_config = att_config

        if self.rnn_config is not None:
            self.rnn_config['reuse'] = tf.AUTO_REUSE
            self.dim_hidden = 2 * rnn_config['num_units']
        else:  # no RNN, just directly using embedding
            self.dim_hidden = dim_emb

    def forward(self, el_size, qw_emb, qw_len, pw_sup_emb, pw_sup_len, sup_size,
                type_trans, el_type_signa, el_indv_feats, el_comb_feats, mode):
        """
        Note: number of paths in a schema == number of entities in the schema
        :param el_size:         (ds, )
        :param qw_emb:          (ds, path_max_size, qw_max_len, dim_emb)
        :param qw_len:          (ds, path_max_size)
        :param pw_sup_emb:      (ds, path_max_size, sup_max_size, pw_max_len, dim_emb)
        :param pw_sup_len:      (ds, path_max_size, sup_max_size)
        :param sup_size:        (ds, path_max_size)
        :param type_trans:      (ds, path_max_size, sup_max_size, dim_type)
        :param el_type_signa:   (ds, el_max_size, dim_type)
        :param el_indv_feats:   (ds, el_max_size, el_feat_size)
        :param el_comb_feats:   (ds, 1)
        :param mode:    TRAIN / INFER
        """
        LogInfo.begin_track('Build kernel: [el_kernel]')
        assert mode in (tf.contrib.learn.ModeKeys.INFER, tf.contrib.learn.ModeKeys.TRAIN)

        rnn_encoder = None
        if self.rnn_config is not None:
            encoder_args = {'config': self.rnn_config, 'mode': mode}
            rnn_encoder = BidirectionalRNNEncoder(**encoder_args)

        raw_shape = tf.shape(pw_sup_len)
        dyn_el_max_size = raw_shape[1]
        dyn_sup_max_size = raw_shape[2]

        """ Possible reshapes """
        qw_emb = tf.reshape(qw_emb, [-1, self.qw_max_len, self.dim_emb])
        # (ds * el_max_size, qw_max_len, dim_emb)
        qw_len = tf.reshape(qw_len, [-1])       # (ds * el_max_size)

        pw_sup_emb = tf.reshape(pw_sup_emb, [-1, self.pw_max_len, self.dim_emb])
        # (ds * el_max_size * sup_max_size, pw_max_len, dim_emb)
        pw_sup_len = tf.reshape(pw_sup_len, [-1])

        """ Calculate attention / non-attention question representation """
        pw_sup_repr = seq_encoding_with_aggregation(emb_input=pw_sup_emb, len_input=pw_sup_len,
                                                    rnn_encoder=rnn_encoder,
                                                    seq_merge_mode=self.seq_merge_mode)
        # (ds*el_max_size*sup_max_size, dim_hidden)

        if self.att_config is not None:
            dim_att_len = self.att_config['dim_att_hidden']
            att_func = self.att_config['att_func']
            qw_hidden = seq_encoding(emb_input=qw_emb, len_input=qw_len, encoder=rnn_encoder)
            # (ds * el_max_size, qw_max_len, dim_hidden)
            qw_mask = tf.sequence_mask(lengths=qw_len,
                                       maxlen=self.qw_max_len,
                                       dtype=tf.float32,
                                       name='qw_mask')  # (DS, qw_max_len)
            tile_qw_hidden = tf.tile(
                tf.expand_dims(qw_hidden, axis=1),      # (ds*el_max_size, 1, qw_max_len, dim_hidden)
                multiples=[1, dyn_sup_max_size, 1, 1],
                name='tile_qw_hidden'
            )   # (ds*el_max_size, sup_max_size, qw_max_len, dim_hidden)
            tile_qw_mask = tf.tile(
                tf.expand_dims(qw_mask, axis=1),
                multiples=[1, dyn_sup_max_size, 1],
                name='tile_qw_mask'
            )   # (ds*el_max_size, sup_max_size, qw_max_len)

            expand_qw_mask = tf.reshape(tile_qw_mask, [-1, self.qw_max_len])
            expand_qw_hidden = tf.reshape(tile_qw_hidden, [-1, self.qw_max_len, self.dim_hidden])
            # (ds*el_max_size*sup_max_size, qw_max_len, dim_hidden)

            simple_att = SimpleAttention(lf_max_len=self.qw_max_len,
                                         dim_att_hidden=dim_att_len,
                                         att_func=att_func)
            qw_att_repr, _, _ = simple_att.forward(lf_input=expand_qw_hidden,
                                                   lf_mask=expand_qw_mask,
                                                   fix_rt_input=pw_sup_repr)
            # (ds*el_max_size*sup_max_size, dim_hidden)
            final_qw_repr = qw_att_repr
        else:
            qw_repr = seq_encoding_with_aggregation(emb_input=qw_emb, len_input=qw_len,
                                                    rnn_encoder=rnn_encoder,
                                                    seq_merge_mode=self.seq_merge_mode)
            # (ds*el_max_size, dim_hidden)
            tile_qw_repr = tf.tile(
                tf.expand_dims(qw_repr, axis=1),
                multiples=[1, dyn_sup_max_size, 1],
                name='tile_qw_repr'
            )   # (ds*el_max_size, sup_max_size, dim_hidden)
            expand_qw_repr = tf.reshape(tile_qw_repr, [-1, self.dim_hidden])
            final_qw_repr = expand_qw_repr

        with tf.variable_scope('el_kernel', reuse=tf.AUTO_REUSE):
            """ Calculate cosine similarity, and turning into type distribution """
            sim_score = cosine_sim(lf_input=final_qw_repr, rt_input=pw_sup_repr)    # (ds*el_max_size, sup_max_size)
            sim_score = tf.reshape(sim_score, shape=raw_shape, name='sim_score')    # (ds, el_max_size, sup_max_size)
            sup_mask = tf.sequence_mask(lengths=sup_size,
                                        maxlen=dyn_sup_max_size,
                                        dtype=tf.float32,
                                        name='sup_mask')        # (ds, el_max_size, sup_max_size)
            mask_score = sup_mask * sim_score + (1. - sup_mask) * tf.float32.min
            pred_prob = tf.nn.softmax(logits=mask_score, name='pred_prob')  # (ds, el_max_size, sup_max_size)
            type_prob = tf.matmul(
                a=tf.expand_dims(pred_prob, axis=2),  # (ds, el_max_size, 1, sup_max_size)
                b=type_trans                          # (ds, el_max_size, sup_max_size, dim_type)
            )       # (ds, el_max_size, 1, dim_type)
            type_prob = tf.squeeze(input=type_prob, axis=2, name='type_prob')   # (ds, el_max_size, dim_type)
            type_match_score = tf.reduce_sum(el_type_signa*type_prob,
                                             axis=-1, keep_dims=True,
                                             name='type_match_score')   # (ds, el_max_size, 1)

            """ Feature concat and produce scores """
            el_indv_concat = tf.concat([type_match_score, el_indv_feats],
                                       axis=-1, name='el_indv_concat')  # (ds, el_max_size, 1+el_feat_size)
            el_mask = tf.sequence_mask(lengths=el_size, maxlen=dyn_el_max_size,
                                       dtype=tf.float32, name='el_mask')    # (ds, el_max_size)
            sum_indv_feats = tf.reduce_sum(
                el_indv_concat * tf.expand_dims(el_mask, axis=-1),
                axis=1, name='sum_indv_feats'
            )   # (ds, 1+el_feat_size)
            final_feats = tf.concat([sum_indv_feats, el_comb_feats], axis=-1, name='final_feats')
            # (ds, 1+el_max_size+1) --> type_match + indv_feats + comb_feat
            el_score = tf.contrib.layers.fully_connected(
                inputs=final_feats,
                num_outputs=1,
                activation_fn=None,
                scope='out_fc',
                reuse=tf.AUTO_REUSE
            )  # (ds, 1), representing type matching score

        LogInfo.end_track()
        return el_score, final_feats
