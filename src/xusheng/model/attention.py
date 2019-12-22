import abc
import six
import tensorflow as tf
from tensorflow.python.ops import array_ops, math_ops

from kangqi.util.tf.cosine_sim import cosine_sim


# ------------------------------- Attention Mechanism by self (old version)-------------------------- #


@six.add_metaclass(abc.ABCMeta)
class AttentionLayer_old(object):

    def __init__(self, att_params):
        self.att_params = att_params

    @abc.abstractmethod
    def _score_fn(self, keys, query):
        """Computes the attention score"""
        raise NotImplementedError

    def build(self, query, values, values_length):
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

        # Fully connected layers to transform both keys and query
        # into a tensor with `num_units` units
        att_keys = tf.contrib.layers.fully_connected(
            inputs=values,
            num_outputs=self.att_params["num_units"],
            activation_fn=None,
            scope="att_keys")
        att_query = tf.contrib.layers.fully_connected(
            inputs=query,
            num_outputs=self.att_params["num_units"],
            activation_fn=None,
            scope="att_query")

        scores = self._score_fn(att_keys, att_query)

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


class AttentionLayerDot_old(AttentionLayer_old):
    """An attention layer that calculates attention scores using
    a dot product.
    returns:
        shape is [B, T]
    """
    def _score_fn(self, keys, query):
        return tf.reduce_sum(keys * tf.expand_dims(query, 1), [2])


class AttentionLayerBahdanau_old(AttentionLayer_old):
    """An attention layer that calculates attention scores using
    a parameterized multiplication.
    returns:
        shape is [B, T]
    """
    def _score_fn(self, keys, query):
        v_att = tf.get_variable(
            "v_att", shape=[self.att_params["num_units"]], dtype=tf.float32)
        return tf.reduce_sum(v_att * tf.tanh(keys + tf.expand_dims(query, 1)), [2])


class AttentionLayerAvg_old(object):
    def __init__(self):
        pass

    @staticmethod
    def build(values, values_length):
        """
        values: The elements to compute attention over. In abstractive_summay, this is
        typically the sequence of encoder outputs.
        A tensor of shape `[B, T, input_dim]`.
        """
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

# ------------------------------- Attention Mechanism using tensorflow api -------------------------- #


class AttentionLayer(object):
    """
    memory: The memory to query; usually the output of an RNN encoder.
    This tensor should be shaped [batch_size, max_time, ...].
    memory_sequence_length (optional): Sequence lengths for the batch entries in memory.
    If provided, the memory tensor rows are masked with zeros for values past the respective sequence lengths.
    num_units: The depth of the query mechanism.
    """
    def __init__(self, memory, memory_sequence_length, num_units):
        self.memory = memory
        self.memory_sequence_length = memory_sequence_length
        self.num_units = num_units
        self.attention_mechanism = None
        self.attention()

    @abc.abstractmethod
    def attention(self):
        pass

    def compute_attention(self, query):
        """Computes the attention and alignments for a given attention_mechanism."""

        alignments = self.attention_mechanism(query, previous_alignments=None)

        # Reshape from [batch_size, memory_time] to [batch_size, 1, memory_time]
        expanded_alignments = array_ops.expand_dims(alignments, 1)
        # Context is the inner product of alignments and values along the
        # memory time dimension.
        # alignments shape is
        #   [batch_size, 1, memory_time]
        # attention_mechanism.values shape is
        #   [batch_size, memory_time, attention_mechanism.num_units]
        # the batched matmul is over memory_time, so the output shape is
        #   [batch_size, 1, attention_mechanism.num_units].
        # we then squeeze out the singleton dim.
        context = math_ops.matmul(expanded_alignments, self.attention_mechanism.values)
        context = array_ops.squeeze(context, [1])

        return context, alignments


class AttentionLayerLuong(AttentionLayer):
    def attention(self):
        self.attention_mechanism = \
            tf.contrib.seq2seq.LuongAttention(self.num_units, self.memory,
                                              memory_sequence_length=self.memory_sequence_length)


class AttentionLayerBahdanau(AttentionLayer):
    def attention(self):
        self.attention_mechanism = \
            tf.contrib.seq2seq.BahdanauAttention(self.num_units, self.memory,
                                                 memory_sequence_length=self.memory_sequence_length)

# ----------------------- Cross Attention Mechanism using Attention Matrix -------------------------- #


class CrossAttentionLayer(object):

    def __init__(self, left_max_len, right_max_len, hidden_dim):
        """
        :param left_max_len: max length of left tensor      (T1)
        :param right_max_len: max length of right tensor    (T2)
        :param hidden_dim: a number
        """
        self.left_max_len = left_max_len
        self.right_max_len = right_max_len
        self.hidden_dim = hidden_dim

    def compute_attention(self, left_tensor, left_len, right_tensor, right_len):
        """
        :param left_tensor: [B, T1, dim1] 
        :param right_tensor: [B, T2, dim2]
        :param left_len: [B, ] real length of left tensor
        :param right_len: [B, ] real length of right tensor
        :return: [B, ] similarity score, [B, T1, T2] attention matrix
        """

        # Fully connected layers to transform both left and right tensor
        # into a tensor with `hidden_dim` units
        # [B, T1, dim]
        att_left = tf.contrib.layers.fully_connected(
            inputs=left_tensor,
            num_outputs=self.hidden_dim,
            activation_fn=None,
            scope="att_keys")
        # [B, T2, dim]
        att_right = tf.contrib.layers.fully_connected(
            inputs=right_tensor,
            num_outputs=self.hidden_dim,
            activation_fn=None,
            scope="att_query")
        # [B, T1, 1, dim]
        att_left = tf.expand_dims(att_left, axis=2)
        # [B, T1, T2, dim]
        att_left = tf.tile(att_left, multiples=[1, 1, self.right_max_len, 1])
        # [B, T2, 1, dim]
        att_right = tf.expand_dims(att_right, axis=2)
        # [B, T2, T1, dim]
        att_right = tf.tile(att_right, multiples=[1, 1, self.left_max_len, 1])
        # [B, T1, T2, dim]
        att_right = tf.transpose(att_right, perm=[0, 2, 1, 3])

        v_att = tf.get_variable(
            name="v_att",
            shape=[self.hidden_dim],
            dtype=tf.float32
        )

        # [B, T1, T2]
        att_matrix = tf.reduce_sum(v_att * tf.tanh(att_left + att_right), axis=3)

        # [B, T1]
        att_val_left = tf.reduce_sum(att_matrix, axis=2)

        # [B, T2]
        att_val_right = tf.reduce_sum(att_matrix, axis=1)

        """
        Kangqi on 180211:
        A bit mistake here. att_matrix haven't removed padding elements (att_maxtrix[i][j]) yet,
        but those elements make contribution to att_val_left/right.
        The masking process below cannot remove such information.
        """

        # Replace all scores for padded inputs with tf.float32.min
        left_mask = tf.sequence_mask(
            lengths=tf.to_int32(left_len),
            maxlen=tf.to_int32(self.left_max_len),
            dtype=tf.float32)       # [B, T1]
        left_val = att_val_left * left_mask + ((1.0 - left_mask) * tf.float32.min)

        right_mask = tf.sequence_mask(
            lengths=tf.to_int32(right_len),
            maxlen=tf.to_int32(self.right_max_len),
            dtype=tf.float32)       # [B, T2]
        right_val = att_val_right * right_mask + ((1.0 - right_mask) * tf.float32.min)

        # Normalize the scores
        left_normalized = tf.nn.softmax(left_val, name="left_normalized")
        right_normalized = tf.nn.softmax(right_val, name="right_normalized")

        # Calculate the weighted average of the attention inputs
        # according to the attention values
        # [B, T1, 1] * [B, T1, dim] --> [B, T1, dim] --> [B, dim]
        left_weighted = tf.expand_dims(left_normalized, axis=2) * left_tensor
        left_weighted = tf.reduce_sum(left_weighted, axis=1)

        # [B, dim]
        right_weighted = tf.expand_dims(right_normalized, axis=2) * right_tensor
        right_weighted = tf.reduce_sum(right_weighted, axis=1)

        # Kangqi edit: cosine similarity is much better
        # score = tf.contrib.layers.fully_connected(
        #     inputs=tf.concat([left_weighted, right_weighted], axis=1),
        #     num_outputs=1,
        #     activation_fn=None,
        #     scope="output")
        score = cosine_sim(lf_input=left_weighted, rt_input=right_weighted)

        # Kangqi edit: return more items.
        # return score, att_matrix

        # Kangqi edit: we need masked att_matrix, padding 0 on useless rows / columns
        left_cube_mask = tf.stack([left_mask] * self.right_max_len, axis=-1)    # [B, T1, T2]
        right_cube_mask = tf.stack([right_mask] * self.left_max_len, axis=1)    # [B, T1, T2]
        masked_att_matrix = att_matrix * left_cube_mask * right_cube_mask       # [B, T1, T2]

        return left_weighted, right_weighted, masked_att_matrix, score
