import tensorflow as tf

# from ..module import att_layer
from ..module.seq_helper import seq_hidden_averaging

from kangqi.util.LogUtil import LogInfo


class EntityLinkingKernel:

    def __init__(self, qw_max_len, el_max_len, dim_el_hidden, keep_prob, use_type):
        self.qw_max_len = qw_max_len
        self.el_max_len = el_max_len
        self.dim_el_hidden = dim_el_hidden
        self.keep_prob = keep_prob
        self.use_type = use_type
        # self.att_func_call = getattr(att_layer, 'cross_att_' + att_func)
        # TODO: bring attention instead of avg

    def forward(self, qw_emb, qw_len, el_type_emb, el_feats, el_len, mode):
        """
        :param qw_emb:      (ds, qw_max_len, dim_emb)
        :param qw_len:      (ds, )
        :param el_type_emb: (ds, el_max_len, dim_emb)
        :param el_feats:    (ds, el_max_len, el_feat_size)
        :param el_len:      (ds, )
        :param mode:    TRAIN / INFER
        :return:
        """
        LogInfo.begin_track('Build kernel: [el_kernel]')
        LogInfo.logs('use_type = %s, dim_el_hidden = %d, keep_prob = %.1f',
                     self.use_type, self.dim_el_hidden, self.keep_prob)
        assert mode in (tf.contrib.learn.ModeKeys.INFER, tf.contrib.learn.ModeKeys.TRAIN)

        with tf.variable_scope('el_kernel', reuse=tf.AUTO_REUSE):
            """ Step 1: build the Q repr for each entity, and concat with type information """
            q_hidden = seq_hidden_averaging(seq_hidden_input=qw_emb, len_input=qw_len)
            # (ds, dim_emb)
            q_hidden = tf.stack([q_hidden] * self.el_max_len, axis=1, name='q_hidden')
            # (ds, el_max_len, dim_emb)
            # TODO: using attention instead of Avg+Stack
            el_type_concat = tf.concat([q_hidden, el_type_emb], axis=-1, name='el_type_concat')
            # (ds, el_max_len, 2*dim_emb)

            """ Step 2: get the hidden vector representing the type association """
            if self.use_type:
                # Dropout the input of the type_hidden FC layer
                if mode == tf.contrib.learn.ModeKeys.TRAIN:
                    el_type_concat = tf.nn.dropout(el_type_concat, keep_prob=self.keep_prob, name='el_type_concat')
                el_type_hidden = tf.contrib.layers.fully_connected(
                    inputs=el_type_concat,
                    num_outputs=self.dim_el_hidden,
                    activation_fn=tf.nn.relu,
                    scope='hidden_fc',
                    reuse=tf.AUTO_REUSE
                )       # (ds, el_max_len, dim_el_hidden), representing type match
                if mode == tf.contrib.learn.ModeKeys.TRAIN:
                    el_type_hidden = tf.nn.dropout(
                        el_type_hidden, keep_prob=self.keep_prob, name='el_type_hidden')
                el_type_score = tf.contrib.layers.fully_connected(
                    inputs=el_type_hidden,
                    num_outputs=1,
                    activation_fn=None,
                    scope='type_fc',
                    reuse=tf.AUTO_REUSE
                )       # (ds, el_max_len, 1), representing type matching score
                el_feats_concat = tf.concat([el_type_score, el_feats], axis=-1, name='el_feats_concat')
                # (ds, el_max_len, 1 + el_feat_size)
            else:
                el_feats_concat = el_feats

            """ Step 3: combine the sparse features and build the final score """
            el_raw_score = tf.contrib.layers.fully_connected(
                inputs=el_feats_concat,
                num_outputs=1,
                activation_fn=None,
                scope='out_fc',
                reuse=tf.AUTO_REUSE
            )       # (ds, el_max_len, 1)
            el_raw_score = tf.squeeze(el_raw_score, axis=-1, name='el_raw_score')   # (ds, el_max_len)
            el_mask = tf.sequence_mask(lengths=el_len,
                                       maxlen=self.el_max_len,
                                       dtype=tf.float32,
                                       name='el_mask')      # (ds, el_max_len)
            el_score = tf.reduce_sum(el_raw_score * el_mask, axis=-1, name='el_score')  # (ds, )

        LogInfo.end_track()
        return el_score, el_feats_concat, el_raw_score
