"""
Single-task model for named entity recognition

"""
import tensorflow as tf
import numpy as np
import abc
import six

from attention import AttentionLayerBahdanau
# import contrib


@six.add_metaclass(abc.ABCMeta)
class NER(object):
    def __init__(self, batch_size,
                 vocab_size,
                 word_dim, lstm_dim,
                 max_seq_len, num_classes,
                 l2_reg_lambda=0.0,
                 lr=0.001,
                 gradient_clip=5,
                 init_embedding=None,
                 layer_size=1):

        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.word_dim = word_dim
        self.lstm_dim = lstm_dim
        self.max_seq_len = max_seq_len
        self.num_classes = num_classes
        self.l2_reg_lambda = l2_reg_lambda
        self.lr = lr
        self.gradient_clip = gradient_clip
        self.layer_size = layer_size

        if init_embedding is None:
            self.init_embedding = np.zeros([vocab_size, word_dim], dtype=np.float32)
        else:
            self.init_embedding = init_embedding

        # placeholders
        self.x = tf.placeholder(tf.int32, [None, None])
        self.y = tf.placeholder(tf.int32, [None, None])
        self.seq_len = tf.placeholder(tf.int32, [None])
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        self._build_graph()
        self.saver = tf.train.Saver()

    @abc.abstractmethod
    def _build_graph(self):
        raise NotImplementedError

    def train_step(self, sess, x_batch, y_batch, seq_len_batch, dropout_keep_prob):
        feed_dict = {
            self.x: x_batch,
            self.y: y_batch,
            self.seq_len: seq_len_batch,
            self.dropout_keep_prob: dropout_keep_prob
        }
        _, step, loss = sess.run(
            [self.train_op, self.global_step, self.loss],
            feed_dict)

        return step, loss

    def decode(self, sess, x, seq_len):
        feed_dict = {
            self.x: x,
            self.seq_len: seq_len,
            self.dropout_keep_prob: 1.0
        }

        logits, transition_params = sess.run(
            [self.logits, self.transition_params], feed_dict)

        y_pred = []
        for logits_, seq_len_ in zip(logits, seq_len):
            logits_ = logits_[:seq_len_]
            viterbi_sequence, _ = tf.contrib.crf.viterbi_decode(
                logits_, transition_params)
            y_pred.append(viterbi_sequence)

        return y_pred

    def save(self, sess, dir_path):
        import os
        if not(os.path.isdir(dir_path)):
            os.mkdir(dir_path)
        fp = dir_path + "/best_model"
        return self.saver.save(sess, fp)

    def load(self, sess, fp):
        self.saver.restore(sess, fp)


class NER_basic(NER):
    def _build_graph(self):
        with tf.variable_scope("embedding"):
            self.embedding = tf.Variable(self.init_embedding, dtype=tf.float32, name="embedding")

        with tf.variable_scope("softmax"):
            self.W = tf.get_variable(
                shape=[self.lstm_dim * 2, self.num_classes],
                initializer=tf.truncated_normal_initializer(stddev=0.01),
                dtype=tf.float32,
                name="weights",
                regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg_lambda))
            self.b = tf.Variable(tf.zeros([self.num_classes], dtype=tf.float32, name="bias"))

        with tf.variable_scope("forward"):
            seq_len = tf.cast(self.seq_len, tf.int64)
            output = tf.nn.embedding_lookup(self.embedding, self.x)  # [batch_size, seq_len, word_dim]
            for num in range(self.layer_size):
                self.fw_cell = tf.nn.rnn_cell.BasicLSTMCell(self.lstm_dim)
                self.bw_cell = tf.nn.rnn_cell.BasicLSTMCell(self.lstm_dim)

                (forward_output, backward_output), _ = tf.nn.bidirectional_dynamic_rnn(
                    self.fw_cell,
                    self.bw_cell,
                    output,
                    time_major=False,
                    dtype=tf.float32,
                    sequence_length=seq_len,
                    scope='layer_' + str(num)
                )
                output = tf.concat(axis=2, values=[forward_output, backward_output])
                output = tf.nn.dropout(output, keep_prob=self.dropout_keep_prob)

            size = tf.shape(output)[0]
            output = tf.reshape(output, [-1, 2 * self.lstm_dim])
            matricized_unary_scores = tf.matmul(output, self.W) + self.b
            self.logits = tf.reshape(matricized_unary_scores, [size, -1, self.num_classes])

        with tf.variable_scope("loss"):
            # CRF log likelihood
            log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(
                self.logits, self.y, self.seq_len)

            self.loss = tf.reduce_mean(-log_likelihood)

        with tf.variable_scope("train_ops"):
            self.optimizer = tf.train.AdamOptimizer(self.lr)

            self.global_step = tf.Variable(0, name="global_step", trainable=False)

            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), self.gradient_clip)
            self.train_op = self.optimizer.apply_gradients(zip(grads, tvars),
                                                           global_step=self.global_step)


class NER_self_attention(NER):
    def _build_graph(self):
        with tf.variable_scope("embedding"):
            self.embedding = tf.Variable(self.init_embedding, dtype=tf.float32, name="embedding")

        with tf.variable_scope("softmax"):
            self.W = tf.get_variable(
                shape=[self.lstm_dim * 4, self.num_classes],
                initializer=tf.truncated_normal_initializer(stddev=0.01),
                dtype=tf.float32,
                name="weights",
                regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg_lambda))
            self.b = tf.Variable(tf.zeros([self.num_classes], dtype=tf.float32, name="bias"))

        with tf.variable_scope("forward"):
            seq_len = tf.cast(self.seq_len, tf.int64)
            output = tf.nn.embedding_lookup(self.embedding, self.x)  # [batch_size, seq_len, word_dim]
            for num in range(self.layer_size):
                self.fw_cell = tf.nn.rnn_cell.BasicLSTMCell(self.lstm_dim)
                self.bw_cell = tf.nn.rnn_cell.BasicLSTMCell(self.lstm_dim)

                (forward_output, backward_output), _ = tf.nn.bidirectional_dynamic_rnn(
                    self.fw_cell,
                    self.bw_cell,
                    output,
                    time_major=False,
                    dtype=tf.float32,
                    sequence_length=seq_len,
                    scope='layer_' + str(num)
                )
                output = tf.concat(axis=2, values=[forward_output, backward_output])
                # output = tf.nn.dropout(output, keep_prob=self.dropout_keep_prob)

            output_final_sliced = []
            att = AttentionLayerBahdanau(keys_input_dim=self.lstm_dim * 2,
                                         query_input_dim=self.lstm_dim * 2,
                                         att_dim=self.lstm_dim * 2)
            output_sliced =  [tf.squeeze(output_, [1]) for output_ in tf.split(output, self.max_seq_len, axis=1)]
            for output_ in output_sliced:
                context = att._build(output_, output, self.seq_len)
                output_final_sliced.append(tf.concat([output_, context], axis=1))

            output_final = tf.stack(output_final_sliced, axis=1)

            size = tf.shape(output_final)[0]
            output_final = tf.reshape(output_final, [-1, 4 * self.lstm_dim])
            matricized_unary_scores = tf.matmul(output_final, self.W) + self.b
            self.logits = tf.reshape(matricized_unary_scores, [size, -1, self.num_classes])

        with tf.variable_scope("loss"):
            # CRF log likelihood
            log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(
                self.logits, self.y, self.seq_len)

            self.loss = tf.reduce_mean(-log_likelihood)

        with tf.variable_scope("train_ops"):
            self.optimizer = tf.train.AdamOptimizer(self.lr)

            self.global_step = tf.Variable(0, name="global_step", trainable=False)

            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), self.gradient_clip)
            self.train_op = self.optimizer.apply_gradients(zip(grads, tvars),
                                                           global_step=self.global_step)


def eval_seq_crf_with_o(y_pred_, y_true_, tag_dict, method='precision'):
    """
    Evaluation for ATIS dataset, including "Outside"
    under specific conditions (3-class)
    :param y_pred_: [B, T, ]
    :param y_true_: [B, T, ]
    :param method: precision/ recall
    :return: f1 score
    """
    # LogInfo.logs("y_pred: %s", '\n'.join([str(x) for x in y_pred_]))
    # LogInfo.logs("y_true: %s", '\n'.join([str(x) for x in y_true_]))
    if method == 'precision':
        y_pred = np.array(y_pred_)
        y_true = np.array(y_true_)
    elif method == 'recall':
        y_pred = np.array(y_true_)
        y_true = np.array(y_pred_)

    names = set()
    for tag in tag_dict:
        if tag == 'O':
            names.add('O')
        else:
            names.add(tag[2:])

    correct = dict()
    act_cnt = dict()
    for name in names:
        correct[name] = 0
        act_cnt[name] = 0

    for line_pred, line_true in zip(y_pred, y_true):
        i = 0
        cnt = len(line_pred)
        while i < cnt:
            tag_num = line_pred[i]
            tag = tag_dict[tag_num]
            if tag_num <= 40:
                # tags with "B" without "I", including "O"
                if tag_num == 0:
                    kind = 'O'
                else:
                    kind = tag[2:]
                act_cnt[kind] += 1
                if line_true[i] == line_pred[i]:
                    correct[kind] += 1
                i += 1
                continue
            else:
                kind = tag[2:]
                sign = tag[0]
            if sign == 'B':
                j = i + 1
                while j < cnt:
                    next_tag = tag_dict[line_pred[j]]
                    if next_tag[2:] == kind and next_tag[0] == 'I':
                        j += 1
                    else:
                        break
            else:
                i += 1
                continue

            act_cnt[kind] += 1

            act_label = ' '.join([str(x) for x in line_true[i:j]])
            proposed_label = ' '.join([str(x) for x in line_pred[i:j]])
            if act_label == proposed_label and (j == cnt or line_true[j] != line_true[i]+1):
                correct[kind] += 1
            i = j

    ret = dict()
    keys = act_cnt.keys()
    correct_total = 0
    cnt_total = 0
    for key in keys:
        if act_cnt[key] == 0:
            ret[key] = 0.0
        else:
            ret[key] = correct[key] * 1.0 / act_cnt[key]
        correct_total += correct[key]
        cnt_total += act_cnt[key]
        if cnt_total == 0:
            overall = 0.0
        else:
            overall = correct_total * 1.0 / cnt_total
    return overall


def eval_seq_crf_no_o(y_pred_, y_true_, tag_dict, method='precision'):
    """
    Evaluation for ATIS dataset, including "Outside"
    under specific conditions (3-class)
    :param y_pred_: [B, T, ]
    :param y_true_: [B, T, ]
    :param method: precision/ recall
    :return: f1 score
    """
    # LogInfo.logs("y_pred: %s", '\n'.join([str(x) for x in y_pred_]))
    # LogInfo.logs("y_true: %s", '\n'.join([str(x) for x in y_true_]))
    if method == 'precision':
        y_pred = np.array(y_pred_)
        y_true = np.array(y_true_)
    elif method == 'recall':
        y_pred = np.array(y_true_)
        y_true = np.array(y_pred_)

    names = set()
    for tag in tag_dict:
        if tag == 'O':
            continue
        else:
            names.add(tag[2:])

    correct = dict()
    act_cnt = dict()
    for name in names:
        correct[name] = 0
        act_cnt[name] = 0

    for line_pred, line_true in zip(y_pred, y_true):
        i = 0
        cnt = len(line_pred)
        while i < cnt:
            tag_num = line_pred[i]
            tag = tag_dict[tag_num]
            if tag_num <= 40:
                # tags with "B" without "I", including "O"
                if tag_num == 0:
                    i += 1
                    continue
                else:
                    kind = tag[2:]
                act_cnt[kind] += 1
                if line_true[i] == line_pred[i]:
                    correct[kind] += 1
                i += 1
                continue
            else:
                kind = tag[2:]
                sign = tag[0]
            if sign == 'B':
                j = i + 1
                while j < cnt:
                    next_tag = tag_dict[line_pred[j]]
                    if next_tag[2:] == kind and next_tag[0] == 'I':
                        j += 1
                    else:
                        break
            else:
                i += 1
                continue

            act_cnt[kind] += 1

            act_label = ' '.join([str(x) for x in line_true[i:j]])
            proposed_label = ' '.join([str(x) for x in line_pred[i:j]])
            if act_label == proposed_label and (j == cnt or line_true[j] != line_true[i]+1):
                correct[kind] += 1
            i = j

    ret = dict()
    keys = act_cnt.keys()
    correct_total = 0
    cnt_total = 0
    for key in keys:
        if act_cnt[key] == 0:
            ret[key] = 0.0
        else:
            ret[key] = correct[key] * 1.0 / act_cnt[key]
        correct_total += correct[key]
        cnt_total += act_cnt[key]
        if cnt_total == 0:
            overall = 0.0
        else:
            overall = correct_total * 1.0 / cnt_total
    return overall