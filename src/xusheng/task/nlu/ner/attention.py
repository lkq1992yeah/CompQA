import abc
import six
import tensorflow as tf


@six.add_metaclass(abc.ABCMeta)
class AttentionLayer(object):
    def __init__(self, keys_input_dim, query_input_dim, att_dim):
        self.keys_input_dim = keys_input_dim
        self.query_input_dim = query_input_dim
        self.att_dim = att_dim

        self.W_att_keys = tf.get_variable("att_keys",
                                          shape=[self.keys_input_dim, self.att_dim],
                                          initializer=tf.truncated_normal_initializer(stddev=0.01),
                                          dtype=tf.float32)
        self.W_att_query = tf.get_variable("att_query",
                                           shape=[self.query_input_dim, self.att_dim],
                                           initializer=tf.truncated_normal_initializer(stddev=0.01),
                                           dtype=tf.float32)
        self.v_att = tf.get_variable("v_att",
                                     shape=[self.att_dim],
                                     initializer=tf.truncated_normal_initializer(stddev=0.01),
                                     dtype=tf.float32)

    @abc.abstractmethod
    def score_fn(self, keys, query):
        """Computes the attention score"""
        raise NotImplementedError

    def _build(self, query, values, values_length):
        """Computes attention scores and outputs.
        Args:
      query: The query used to calculate attention scores.
        In abstractive_summay this is typically the current state of the decoder.
        A tensor of shape `[B, ...]`
      keys: The keys used to calculate attention scores. In abstractive_summay, these
        are typically the outputs of the encoder and equivalent to `values`.
        A tensor of shape `[B, T, ...]` where each element in the `T`
        dimension corresponds to the key for that value.
      values: The elements to compute attention over. In abstractive_summay, this is
        typically the sequence of encoder outputs.
        A tensor of shape `[B, T, input_dim]`.
      values_length: An int32 tensor of shape `[B]` defining the sequence
        length of the attention values.

        Returns:
      A tuple `(scores, context)`.
      `scores` is vector of length `T` where each element is the
      normalized "score" of the corresponding `inputs` element.
      `context` is the final attention layer output corresponding to
      the weighted inputs.
      A tensor fo shape `[B, input_dim]`.
        """
        values_depth = values.get_shape().as_list()[-1]
        values_batch_size = tf.shape(values)[0]

        # Fully connected layers to transform both keys and query
        # into a tensor with `num_units` units
        values_tmp = tf.reshape(values, [-1, values_depth])
        att_keys = tf.matmul(values_tmp, self.W_att_keys)
        att_keys = tf.reshape(att_keys, [values_batch_size, -1, values_depth])
        att_query = tf.matmul(query, self.W_att_query)

        scores = self.score_fn(att_keys, att_query)

        # Replace all scores for padded inputs with tf.float32.min
        num_scores = tf.shape(scores)[1]
        scores_mask = tf.sequence_mask(
            lengths=tf.to_int32(values_length),
            maxlen=tf.to_int32(num_scores),
            dtype=tf.float32)
        scores = scores * scores_mask + ((1.0 - scores_mask) * tf.float32.min)

        # Normalize the scores
        scores_normalized = tf.nn.softmax(scores, name="scores_normalized")

        # Calculate the weighted average of the attention inputs
        # according to the scores
        context = tf.expand_dims(scores_normalized, 2) * values
        context = tf.reduce_sum(context, 1, name="context")
        context.set_shape([None, values_depth])

        return context


class AttentionLayerDot(AttentionLayer):
    """An attention layer that calculates attention scores using
    a dot product.
    returns:
        shape is [B, T]
    """
    def score_fn(self, keys, query):
        return tf.reduce_sum(keys * tf.expand_dims(query, 1), [2])


class AttentionLayerBahdanau(AttentionLayer):
    """An attention layer that calculates attention scores using
    a parameterized multiplication.
    returns:
        shape is [B, T]
    """
    def score_fn(self, keys, query):
        return tf.reduce_sum(self.v_att * tf.tanh(keys + tf.expand_dims(query, 1)), [2])


class AttentionLayerAvg(object):
    def __init__(self):
        pass

    def _build(self, values, values_length):
        '''
        values: The elements to compute attention over. In abstractive_summay, this is
        typically the sequence of encoder outputs.
        A tensor of shape `[B, T, input_dim]`.
        '''
        values_depth = values.get_shape().as_list()[-1]
        unrolling_num = values.get_shape().as_list()[1]

        att_mask = tf.sequence_mask(
            lengths=tf.to_int32(values_length),
            maxlen=tf.to_int32(unrolling_num),
            dtype=tf.float32)
        att_mask = tf.expand_dims(att_mask, -1)
        values_mask = values * att_mask

        context = tf.reduce_sum(values_mask, axis=1)
        cnt_mask = tf.reduce_sum(att_mask, axis=1)
        context = tf.div(context, cnt_mask)
        context.set_shape([None, values_depth])

        return context