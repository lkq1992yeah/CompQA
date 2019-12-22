import tensorflow as tf


from ..module.seq_helper import seq_hidden_max_pooling
from kangqi.util.tf.cosine_sim import cosine_sim


class BaseRelationMatchingKernel:

    def __init__(self, qw_max_len, sc_max_len, p_max_len, pw_max_len, dim_emb,
                 path_usage, dim_att_hidden, att_func, att_merge_mode, seq_merge_mode,
                 scoring_mode, rnn_config, cnn_config=None, residual=False):
        self.qw_max_len = qw_max_len
        self.sc_max_len = sc_max_len
        self.pw_max_len = pw_max_len
        self.p_max_len = p_max_len
        self.dim_emb = dim_emb

        self.path_usage = path_usage

        self.dim_att_hidden = dim_att_hidden
        self.att_func = att_func
        self.att_merge_mode = att_merge_mode
        self.seq_merge_mode = seq_merge_mode    # only used when not using CrossAttention
        self.residual = residual                # only used in AttRelationMatchingKernel

        self.scoring_mode = scoring_mode
        assert scoring_mode in ('compact', 'separated')

        self.rnn_config = rnn_config
        self.cnn_config = cnn_config

        self.repr_mode = 'raw'
        if self.rnn_config is not None:
            assert self.cnn_config is None
            self.repr_mode = 'rnn'
            self.rnn_config['reuse'] = tf.AUTO_REUSE
            self.dim_hidden = 2 * rnn_config['num_units']
        elif self.cnn_config is not None:
            assert self.rnn_config is None
            self.repr_mode = 'cnn'
            self.dim_hidden = cnn_config['filters']
        else:           # no RNN / CNN, just directly using embedding
            self.dim_hidden = dim_emb

    def forward(self, qw_emb, qw_len, sc_len, p_emb, pw_emb, p_len, pw_len, mode):
        raise NotImplementedError

    @staticmethod
    def final_merge(q_rep, path_rep, sc_len, sc_max_len, dim_hidden, scoring_mode):
        """
        :param q_rep:           (ds * sc_max_len, dim_hidden)
        :param path_rep:        (ds * sc_max_len, dim_hidden), pay attention to the first dimension!
        :param sc_len:          (ds, )
        :param sc_max_len:      sc_max_len
        :param dim_hidden:      dim_hidden
        :param scoring_mode:    compact / separated
        """
        if scoring_mode == 'compact':
            # aggregate by max-pooling, then overall cosine
            q_att_rep = tf.reshape(q_rep, shape=[-1, sc_max_len, dim_hidden],
                                   name='q_att_rep')  # (ds, sc_max_len, dim_hidden)
            path_att_rep = tf.reshape(path_rep, shape=[-1, sc_max_len, dim_hidden],
                                      name='path_att_rep')  # (ds, sc_max_len, dim_hidden)
            q_final_rep = seq_hidden_max_pooling(seq_hidden_input=q_att_rep,
                                                 len_input=sc_len)
            path_final_rep = seq_hidden_max_pooling(seq_hidden_input=path_att_rep,
                                                    len_input=sc_len)  # (ds, dim_hidden)
            score = cosine_sim(lf_input=q_final_rep, rt_input=path_final_rep)  # (ds, )
            return {'rm_score': score}
        else:
            # separately calculate cosine, then sum together (with threshold control)
            raw_score = cosine_sim(lf_input=q_rep, rt_input=path_rep)  # (ds * sc_max_len, )
            raw_score = tf.reshape(raw_score, shape=[-1, sc_max_len], name='raw_score')  # (ds, sc_max_len)
            sim_ths = tf.get_variable(name='sim_ths', dtype=tf.float32, shape=[])
            path_score = tf.subtract(raw_score, sim_ths, name='path_score')  # add penalty to each potential seq.
            sc_mask = tf.sequence_mask(lengths=sc_len,
                                       maxlen=sc_max_len,
                                       dtype=tf.float32,
                                       name='sc_mask')  # (ds, sc_max_len) as mask
            score = tf.reduce_sum(path_score * sc_mask, axis=-1, name='score')  # (ds, )
            return {'rm_score': score, 'rm_path_score': path_score}
