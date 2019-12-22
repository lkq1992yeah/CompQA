import tensorflow as tf
import numpy as np

from kangqi.util.LogUtil import LogInfo

# utilities when using tensorflow


def noise_weight_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)


def noise_bias_variable(shape, name):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)


def zero_weight_variable(shape, name):
    return tf.Variable(tf.zeros(shape), dtype=tf.float32,
                       name=name)


def zero_bias_variable(shape, name):
    return tf.Variable(tf.zeros(shape), dtype=tf.float32,
                       name=name)


def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1],
                     padding='VALID')  # VALID: n - window + 1; SAME: n
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k1=2, k2=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k1, k2, 1],
                          strides=[1, k1, k2, 1],
                          padding='VALID')


def get_rnn_cell(cell_class,
                 num_units,
                 num_layers=1,
                 keep_prob=1.0,
                 dropout_input_keep_prob=None,
                 dropout_output_keep_prob=None,
                 reuse=None):
    if dropout_input_keep_prob is None:
        dropout_input_keep_prob = keep_prob
    if dropout_output_keep_prob is None:
        dropout_output_keep_prob = keep_prob

    cells = []
    for _ in range(num_layers):
        # cell = tf.contrib.rnn.core_rnn_cell.BasicLSTMCell(num_units, state_is_tuple=True, reuse=reuse)
        cell = None                     # Kangqi added below
        if cell_class == 'RNN':
            cell = tf.contrib.rnn.BasicRNNCell(num_units=num_units, reuse=reuse)
        elif cell_class == 'GRU':
            cell = tf.contrib.rnn.GRUCell(num_units=num_units, reuse=reuse)
        elif cell_class == 'LSTM':
            cell = tf.contrib.rnn.BasicLSTMCell(num_units=num_units, state_is_tuple=True, reuse=reuse)

        if keep_prob < 1.0:
            cell = tf.contrib.rnn.DropoutWrapper(cell=cell,
                                                 input_keep_prob=dropout_input_keep_prob,
                                                 output_keep_prob=dropout_output_keep_prob)
        cells.append(cell)

    if len(cells) > 1:
        final_cell = tf.contrib.rnn.core_rnn_cell.MultiRNNCell(cells, state_is_tuple=True)
    else:
        final_cell = cells[0]

    return final_cell


def get_optimizer(name, learning_rate):
    if name == "Adam":
        optimizer = tf.train.AdamOptimizer(learning_rate)
    elif name == "Adadelta":
        optimizer = tf.train.AdadeltaOptimizer(learning_rate)
    elif name == "Adagrad":
        optimizer = tf.train.AdagradOptimizer(learning_rate)
    elif name == "RMSProp":
        optimizer = tf.train.RMSPropOptimizer(learning_rate)
    elif name == "GradientDescent":
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    else:
        raise NameError("Optimizer name error.")

    return optimizer


def print_detail_2d(x):
    rows = len(x)
    cols = len(x[0])
    LogInfo.logs("shape: (%d, %d)", rows, cols)
    for i in range(rows):
        s = "["
        for j in range(cols):
            s += "%.1f " % (x[i][j])
        s = s[:-1] + "]"
        LogInfo.logs(s)


# check a vector is zero-like (roughly)
def is_zero(vec):
    if vec[0] == 0 and vec[1] == 0 and vec[2] == 0:
        return True
    else:
        return False


# get dimension-wised variance for a single matrix
def get_single_variance(x):
    x = np.array(x)
    row_num = x.shape[0]
    col_num = x.shape[1]
    sum_vec = np.zeros(x.shape[2])
    cnt = 0
    for j in range(col_num):
        column = list()
        for i in range(row_num):
            cell = x[i][j]
            if not(is_zero(cell)):
                column.append(cell)
        if len(column) == 0:
            continue
        var_vec = np.var(column, axis=0)
        sum_vec += var_vec
        cnt += 1
    sum_vec /= cnt
    return sum_vec


# get dimension-wised variance for a batch of matrix
def get_variance(x):
    ret = list()
    for single in x:
        ret.append(get_single_variance(single))
    return ret


# scale to sum of squares equals 1
def normalize(x):
    l2norm = np.sqrt((x * x).sum(axis=1))
    for i in range(len(x)):
        if l2norm[i] != 0:
            x[i] = x[i] / l2norm[i]
    return x


# normalize to (0,1 ) distribution
def normalize_kangqi(x):
    mean = np.mean(x, axis=0)
    delta = np.sqrt(np.var(x, axis=0))
    x = x - mean
    row = x.shape[0]
    col = x.shape[1]
    for i in range(row):
        for j in range(col):
            if delta[j]!=0:
                x[i][j] /= delta[j]
    return x


# combine normalize & scaling together
def normalize_mix(x):
    xx = normalize_kangqi(x)
    return normalize(xx)


if __name__ == '__main__':
    xx = [[[1,2,3], [0,0,0]],
          [[3,6,9], [0,0,0]]]
    LogInfo.logs(get_single_variance(xx))
    y = list()
    y.append(xx)
    y.append(xx)
    LogInfo.logs(get_variance(y))






