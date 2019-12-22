# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

# copied from Mengxue's code (q2_initialization.py)
def xavier_weight_init():

  def _xavier_initializer(shape, **kwargs):
    """Defines an initializer for the Xavier distribution.
    This function will be used as a variable scope initializer.
    https://www.tensorflow.org/versions/r0.7/how_tos/variable_scope/index.html#initializers-in-variable-scope
    Args:
      shape: Tuple or 1-d array that species dimensions of requested tensor.
    Returns:
      out: tf.Tensor of specified shape sampled from Xavier distribution.
    """
    m = shape[0]
    n = shape[1] if len(shape) > 1 else shape[0]
    bound = np.sqrt(6) / np.sqrt(m + n)
    out = tf.random_uniform(shape, minval=-bound, maxval=bound)
    return out

  # Returns defined initializer function.
  return _xavier_initializer

def conv2d_xavier_weight_init():
    def _conv2d_xavier_initializer(shape, **kwargs):
        assert len(shape) == 4
        m = shape[2]
        n = shape[3]
        bound = np.sqrt(6) / np.sqrt(m+n)
        out = tf.random_uniform(shape, minval=-bound, maxval=bound)
        return out
    return _conv2d_xavier_initializer

def zero_weight_init():
  def _zero_initializer(shape, **kwargs):
    out = tf.zeros(shape)
    return out
  return _zero_initializer


def zero_variable(shape, name=None):
	initial = tf.zeros(shape)
	return tf.Variable(initial, name=name)

def weight_variable(shape, name=None):
	initial = tf.truncated_normal(shape, stddev = 0.1)
	return tf.Variable(initial, name=name)

def bias_variable(shape, name=None):
	initial = tf.constant(0.1, shape = shape)
	return tf.Variable(initial, name=name)

# padding: VALID / SAME
def conv2d(x, W, padding='VALID'):
	return tf.nn.conv2d(
            x, W,
            strides=[1, 1, 1, 1],
            padding=padding)

def maxpool2d(x, region_shape, padding='SAME'):
    r, w = region_shape
    size = [1, r, w, 1]
    return tf.nn.max_pool(
            x, ksize=size, strides=size,
            padding=padding)

# x, y: 2-dim matrices with the same shape (batch, len)
# return: (batch, 1) vector of cosine similarity
def cosine2d(x, y):
    lx = tf.sqrt(tf.reduce_sum(tf.multiply(x, x), axis=1, keep_dims=True))
    ly = tf.sqrt(tf.reduce_sum(tf.multiply(y, y), axis=1, keep_dims=True))
    dxy = tf.reduce_sum(tf.multiply(x, y), axis=1, keep_dims=True)
    return dxy / lx / ly


if __name__ == '__main__':
    with tf.variable_scope('out', initializer=xavier_weight_init()):
        a = tf.get_variable(name='a', shape=(3,3))
        # a = tf.Variable(tf.constant(0.5, shape=(3,3)), 'a')
        with tf.variable_scope('in', initializer=zero_weight_init()):
            b = tf.get_variable(name='b', shape=(4,4))

    print a
    print b
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    a_val, b_val = sess.run([a, b])
    print a_val
    print b_val
