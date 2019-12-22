"""
Author: Kangqi Luo
Goal: Define the common model-related procedures of entity linking part
"""


import tensorflow as tf

from ..module.seq_helper import seq_hidden_averaging
# from ..u import show_tensor


class EntityLinkingKernel:

    def __init__(self, e_max_size, e_feat_len, reuse=tf.AUTO_REUSE, verbose=0):
        self.reuse = reuse
        self.verbose = verbose

        self.e_max_size = e_max_size
        self.e_feat_len = e_feat_len

    def get_score(self, mode, qwords_embedding, qwords_len,
                  efeats, etypes_embedding, e_mask):
        """
        Produce the final entity linking score
        :param mode: TRAIN/INFER
        :param qwords_embedding:    (ds, q_max_len, dim_emb)
        :param qwords_len:          (ds, )
        :param efeats:              (ds, e_max_size, e_feat_len)
        :param etypes_embedding:    (ds, e_max_size, dim_emb)
        :param e_mask:              (ds, e_max_size)
        :return: (ds, ) as the final linking score
        """
        assert mode in (tf.contrib.learn.ModeKeys.TRAIN, tf.contrib.learn.ModeKeys.INFER)

        with tf.name_scope('EntityLinkingKernel'):
            q_avg_embedding = seq_hidden_averaging(seq_hidden_input=qwords_embedding,
                                                   len_input=qwords_len)    # (data_size, dim_emb)
            q_avg_embedding = tf.stack([q_avg_embedding] * self.e_max_size,
                                       axis=1, name='q_avg_emgedding')  # (data_size, e_max_size, dim_emb)
            # show_tensor(q_avg_embedding)
            etypes_dot = tf.reduce_sum(q_avg_embedding * etypes_embedding,
                                       axis=-1, keep_dims=True,
                                       name='etypes_dot')  # (data_size, e_max_size, 1)
            use_efeats = tf.concat([efeats, etypes_dot],
                                   axis=-1, name='use_efeats')  # (data_size, e_max_size, e_feat_len+1)
            # show_tensor(use_efeats)
            flat_efeats = tf.reshape(use_efeats,
                                     shape=[-1, self.e_feat_len+1],
                                     name='flat_efeats')    # (data_size * e_max_size, e_feat_len+1)
            # show_tensor(flat_efeats)
            hidden_output = tf.contrib.layers.fully_connected(
                inputs=flat_efeats,
                num_outputs=1,
                activation_fn=None,
                scope='linking_fc',
                reuse=self.reuse
            )  # (data_size * e_max_size, 1)
            # show_tensor(hidden_output)
            hidden_output = tf.reshape(hidden_output,
                                       shape=[-1, self.e_max_size],
                                       name='hidden_output')    # (data_size, e_max_size)
            # show_tensor(hidden_output)
            linking_score = tf.reduce_sum(hidden_output * e_mask,
                                          axis=-1, name='linking_score')    # (data_size, )
            # show_tensor(linking_score)
            # Currently using sum, not averaging

        return linking_score
