import tensorflow as tf
import numpy as np

from kangqi.task.compQA.model.module.seq_helper import seq_encoding
from xusheng.model.rnn_encoder import BidirectionalRNNEncoder
from kangqi.util.LogUtil import LogInfo


max_len = 10
dim_emb = 30
n_words = 500
dim_hidden = 16


v_input = tf.placeholder(tf.int32, shape=[None, max_len])
v_len = tf.placeholder(tf.int32, shape=[None])
rnn_config = {'cell_class': 'GRU',
              'num_units': dim_hidden,
              'reuse': tf.AUTO_REUSE}
encoder_args = {'config': rnn_config, 'mode': tf.contrib.learn.ModeKeys.INFER}
rnn_encoder = BidirectionalRNNEncoder(**encoder_args)

with tf.variable_scope('embedding_lookup', reuse=tf.AUTO_REUSE):
    with tf.device('/cpu:0'):
        w_embedding_init = tf.placeholder(dtype=tf.float32, shape=(n_words, dim_emb), name='w_embedding_init')
        w_embedding = tf.get_variable(name='w_embedding', initializer=w_embedding_init)
        v_emb = tf.nn.embedding_lookup(params=w_embedding, ids=v_input)
    v_hidden = seq_encoding(emb_input=v_emb, len_input=v_len, encoder=rnn_encoder, reuse=tf.AUTO_REUSE)
    LogInfo.logs('v_hidden: %s', v_hidden.get_shape().as_list())

sess = tf.Session()
w_embedding_init_np = np.random.randn(n_words, dim_emb)
sess.run(tf.global_variables_initializer(), feed_dict={w_embedding_init: w_embedding_init_np})

v_input_np = [
    [3, 4, 5, 0, 0, 0, 0, 0, 0, 0],
    [3, 4, 5, 1, 1, 1, 1, 1, 1, 1],
    [3, 4, 5, 4, 3, 0, 0, 0, 0, 0]
]
v_len_np = [3, 3, 5]

v_hidden_out = sess.run(v_hidden, {v_input: v_input_np, v_len: v_len_np})
np.set_printoptions(threshold=np.nan)
LogInfo.logs('Output: %s', v_hidden_out.shape)
LogInfo.logs(np.sum(v_hidden_out ** 2, axis=-1))
