import tensorflow as tf


def hinge_loss(scores, row, col, margin):
    """
    Calculate hinge loss for pos:neg = 1:PN-1 scores
    :param scores: A tensor of shape [B, 1]
    :param row: row = batch_size / PN (col)
    :param col: col = PN, pos : neg = 1 : PN-1
    :param margin: pos score should larger than highest neg score by at least one margin
    :return: hinge loss (one float)
    """
    score_mat = tf.reshape(scores, [-1, col])
    score_pos = tf.slice(score_mat, [0, 0], [row, 1])
    score_neg = tf.slice(score_mat, [0, 1], [row, col-1])
    loss_matrix = tf.maximum(0., margin - score_pos + score_neg)
    loss = tf.reduce_mean(tf.reduce_max(loss_matrix, axis=1))
    return loss

def get_norm(tensor):
    """
    Get norm of a tensor
    :param tensor: input tensor with shape [B, dim]
    :return: norm tensor with shape [B, ]
    """
    return tf.sqrt(tf.reduce_sum(tf.square(tensor), axis=1, keep_dims=True))


def cosine_sim(x, y):
    """
    Get cosine similarity between x and y
    :param x: tensor with shape [B, dim]
    :param y: tensor with shape [B, dim]
    :return: tensor with shape [B, ]
    """
    x_norm = get_norm(x)
    y_norm = get_norm(y)
    norm = x_norm * y_norm
    dot_product = tf.reduce_sum(x * y, axis=1, keep_dims=True)
    return dot_product / norm


def softmax_sequence_loss(logits, targets, sequence_length):
    """Calculates the per-example cross-entropy loss for a sequence of logits and
    masks out all losses passed the sequence length.

    Args:
        logits: Logits of shape `[B, T, vocab_size]`
        targets: Target classes of shape `[B, T]`
        sequence_length: An int32 tensor of shape `[B]` corresponding
        to the length of each input

    Returns:
        A tensor of shape [B, T] that contains the loss per example, per time step.
    """
    with tf.name_scope("cross_entropy_sequence_loss"):
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=targets)

        # Mask out the losses we don't care about
        loss_mask = tf.sequence_mask(
            tf.to_int32(sequence_length), tf.to_int32(tf.shape(targets)[1]))
        loss_mask = tf.to_float(loss_mask)
        loss = loss * loss_mask

        return tf.reduce_sum(loss) / tf.reduce_sum(loss_mask)


def sigmoid_sequence_loss(logits, targets, sequence_length):
    """Calculates the per-example cross-entropy loss for a sequence of logits and
    masks out all losses passed the sequence length.

    Args:
        logits: Logits of shape `[B, T]`
        targets: Target classes of shape `[B, T]`
        sequence_length: An int32 tensor of shape `[B]` corresponding
        to the length of each input

    Returns:
        A tensor of shape [B, T] that contains the loss per example, per time step.
    """
    with tf.name_scope("sigmoid_sequence_loss"):
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=targets, logits=logits)

        # Mask out the losses we don't care about
        loss_mask = tf.sequence_mask(
            tf.to_int32(sequence_length), tf.to_int32(tf.shape(targets)[1]))
        loss_mask = tf.to_float(loss_mask)
        loss = loss * loss_mask

        return tf.reduce_sum(loss) / tf.reduce_sum(loss_mask)