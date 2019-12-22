"""
Author: Kangqi Luo
Goal: stacked conv layer with an overall attention overall all layers
"""

import tensorflow as tf


class ResidualBlock(tf.layers.Layer):

    def __init__(self, n_outputs, kernel_size, dropout=0.2, residual=True,
                 trainable=True, name=None, dtype=tf.float32, **kwargs):
        super(ResidualBlock, self).__init__(
            trainable=trainable, dtype=dtype,
            name=name, **kwargs
        )
        self.n_outputs = n_outputs
        self.residual = residual
        self.conv = tf.layers.Conv1D(filters=n_outputs, kernel_size=kernel_size,
                                     padding='same', activation=None)
        self.dropout_layer = tf.layers.Dropout(dropout)
        self.down_sample = None

    def build(self, input_shape):
        channel_dim = 2
        if input_shape[channel_dim] != self.n_outputs:
            self.down_sample = tf.layers.Dense(self.n_outputs, activation=None)

    def call(self, inputs, training=True):
        x = self.conv.__call__(inputs=inputs)   # haven't apply non-linearity
        x = tf.contrib.layers.layer_norm(x)     # try layer norm first, instead of batch norm
        x = tf.nn.relu(x)       # non-linearity after normalization
        x = self.dropout_layer(x, training=training)
        if self.residual:
            if self.down_sample is not None:
                inputs = self.down_sample(inputs)
            x = tf.nn.relu(x + inputs)
        return x


class StackConvNet(tf.layers.Layer):

    def __init__(self, n_layers, n_outputs, kernel_size=3, dropout=0.2, residual=True,
                 trainable=True, name=None, dtype=tf.float32, **kwargs):
        super(StackConvNet, self).__init__(
            trainable=trainable, dtype=dtype,
            name=name, **kwargs
        )
        self.layers = []
        for layer_idx in range(n_layers):
            self.layers.append(
                ResidualBlock(n_outputs=n_outputs, kernel_size=kernel_size,
                              dropout=dropout, residual=residual,
                              name='res_block_{}'.format(layer_idx))
            )

    def call(self, inputs, training=True):
        # inputs: (B, T, d)
        output_list = []
        outputs = inputs
        for layer in self.layers:
            outputs = layer(outputs, training=training)     # (B, T, n_outputs)
            output_list.append(outputs)
        return output_list      # list: length=n_layer, each element (B, T, n_outputs)
