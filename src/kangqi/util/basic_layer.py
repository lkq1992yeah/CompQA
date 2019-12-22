from theano import *
import theano.tensor as T
import numpy as np

from theano.tensor.nnet import conv2d, softmax, sigmoid
from theano.tensor.signal import pool

from LogUtil import *

rng = np.random.RandomState(34567)


# ==== Below: attachments used for faster SGD functions ==== #
# The input is a layer (with params stored in layer.params list)

# adaptive learning rate: save local_rates and last_grads
def attach_adaptive(layer):
	attached = True
	try: layer.local_rates
	except AttributeError: attached = False
	if attached: return

	layer.local_rates = []
	layer.last_grads = []
	for param in layer.params:
		shape = param.get_value().shape
		local_rate = shared(np.ones(shape, dtype = config.floatX), borrow = True)
		last_grad = shared(np.zeros(shape, dtype = config.floatX), borrow = True)
		layer.local_rates.append(local_rate)
		layer.last_grads.append(last_grad)
	for p, l_r, l_g in zip(layer.params, layer.local_rates, layer.last_grads):
		assert p.get_value().shape == l_r.get_value().shape == l_g.get_value().shape
	LogInfo.logs('Attached adaptive learning for %d groups of parameters.', len(layer.params))

# rmsprop: save mean_square_grads
def attach_rmsprop(layer):
	attached = True
	try: layer.mean_squared_grads
	except AttributeError: attached = False
	if attached: return

	layer.mean_squared_grads = []
	for param in layer.params:
		shape = param.get_value().shape
		mean_squared_grad = shared(np.zeros(shape, dtype = config.floatX), borrow = True)
		layer.mean_squared_grads.append(mean_squared_grad)
	for p, msg in zip(layer.params, layer.mean_squared_grads):
		assert p.get_value().shape == msg.get_value().shape
	LogInfo.logs('Attached rmsprop for %d groups of parameters.', len(layer.params))



# ==== Below: Update functions for different sgd strategies ==== #

def updates_sgd(model, grads, learning_rate):
	updates = [
		(p, p - learning_rate * g) for p, g in zip(model.params, grads)
	]
	LogInfo.logs('SGD updates defined.')
	return updates

def updates_rmsprop(model, grads, learning_rate = 0.001, rho = 0.9, epsilon = 1e-6):
	attach_rmsprop(model)
	updates_mean_square = [
		(
			m_s_g,
			T.cast( rho * m_s_g + (1.0 - rho) * (g ** 2), config.floatX )
		) for m_s_g, g in zip(model.mean_squared_grads, grads)
	]
	updates_param = [
		(
			p,
			T.cast( p - learning_rate * g / T.sqrt(m_s_g + epsilon), config.floatX )
		) for p, m_s_g, g in zip(model.params, model.mean_squared_grads, grads)
	]
	LogInfo.logs('Rmsprop updates defined.')
	return updates_mean_square + updates_param

# def updates_adaptive(model, grads, learning_rate, min_local_rate = 1e-6, max_local_rate = 500):
# 	attach_adaptive(model)	# first add shared variables storing necessary data
# 	updates_param = [
# 		(p, p - learning_rate * l_r * g) for p, l_r, g in zip(model.params, model.local_rates, grads)
# 	]
# 	updates_local_rate = [(l_r,
# 		T.maximum(min_local_rate,
# 			T.miminum(max_local_rate,
# 				l_r * T.switch(
# 					T.gt(l_g * g, 0), 1.2, 0.5
# 				)
# 			)
# 		)) for l_r, l_g, g in zip(model.local_rates, model.last_grads, grads)
# 	]
# 	updates_last_grad = [(l_g, g) for l_g, g in zip(model.last_grads, grads)]
# 	LogInfo.logs('Adaptive learning rate updates defined.')
# 	return updates_param + updates_local_rate + updates_last_grad





# ==== Below: basic layers ==== #

class PoolLayer(object):

	def __init__(self, pool_size, pooling_op):
		self.pool_size = pool_size
		self.pooling_op = pooling_op

	def apply(self, input_data):
		pool_out = pool.pool_2d(
			input = input_data,
			ds = self.pool_size,
			ignore_border = True,
			mode = self.pooling_op
		)
		return pool_out

class ConvLayer(object):

	def __init__(self, input_shape, filter_shape, pool_size, pooling_op, activation):
		assert input_shape[1] == filter_shape[1]

		fan_in = np.prod(filter_shape[1:])
		fan_out = filter_shape[0] * np.prod(filter_shape[2:]) / np.prod(pool_size)
		W_bound = np.sqrt(6. / (fan_in + fan_out))

		self.W = shared(
			np.asarray(
				rng.uniform(low = -W_bound, high = W_bound, size = filter_shape),
				dtype = config.floatX
			),
			borrow = True
		)
		self.b = shared(
			np.zeros((filter_shape[0],), dtype = config.floatX),
			borrow = True
		)

		self.input_shape = input_shape
		self.filter_shape = filter_shape
		self.pool_size = pool_size
		self.pooling_op = pooling_op
		self.activation = activation

		self.params = [self.W, self.b]
		self.param_num = np.prod(filter_shape) + filter_shape[0]
		self.l1_reg = T.sum(self.W) + T.sum(self.b)
		self.l2_reg = T.sum(self.W ** 2) + T.sum(self.b ** 2)
		LogInfo.logs('Conv Layer: input=%s, W=%s, Pool=(%s, %s), Activation=%s, param_num=%d', 
			input_shape, filter_shape, pool_size, pooling_op, activation, self.param_num)


	def apply(self, input_data):
		conv_out = conv2d(
			input = input_data,
			filters = self.W,
			input_shape = self.input_shape,
			filter_shape = self.filter_shape,
			border_mode = 'valid'
		)
		# conv_row_len = input_shape[2] - filter_shape[2] + 1
		pool_out = pool.pool_2d(
			input = conv_out,
			ds = self.pool_size,
			ignore_border = True,
			mode = self.pooling_op
		)
		lin_output = pool_out + self.b.dimshuffle('x', 0, 'x', 'x')		# D * Filter * out_h * out_w
		if self.activation is None:
			return lin_output
		else:
			tensor_output = self.activation(lin_output)
			return tensor_output
		


class HiddenLayer(object):

	def __init__(self, input_dim, output_dim, activation):
		W_bound = np.sqrt(6. / (input_dim + output_dim))
		self.W = shared(
			np.asarray(
				rng.uniform(low = -W_bound, high = W_bound, size = (input_dim, output_dim)),
				dtype = config.floatX
			),
			borrow = True
		)
		self.b = shared(
			np.zeros((output_dim,), dtype = config.floatX),
			borrow = True
		)

		self.activation = activation
		self.params = [self.W, self.b]
		self.param_num = input_dim * output_dim + output_dim
		self.l1_reg = T.sum(self.W) + T.sum(self.b)
		self.l2_reg = T.sum(self.W ** 2) + T.sum(self.b ** 2)
		LogInfo.logs('Hidden Layer: W=(%d, %d), Activation=%s, param_num=%d', input_dim, output_dim, activation, self.param_num)
	
	def apply(self, input_data):
		lin_output = T.dot(input_data, self.W) + self.b
		if self.activation is None:
			return lin_output
		else:
			tensor_output = self.activation(lin_output)
			return tensor_output



def test_program():
	examples = 400
	max_sent_len = 20
	input_dim = 50
	filter_dim = 20
	window = 3
	pooling_op = 'max'
	activation = T.tanh
	learning_rate = 0.01

	input_shape = (examples, input_dim, max_sent_len, 1)
	input_data = rng.randn(examples, input_dim, max_sent_len, 1).astype(config.floatX)
	label = rng.randint(size = examples, low = 0, high = 2).astype(config.floatX)
	training_steps = 10000

	LogInfo.logs('Input shape: %s', input_data.shape)
	LogInfo.logs('Label shape: %s', label.shape)
	LogInfo.logs('Data loaded.')

	X = shared(input_data, name = 'X', borrow = True)
	Y = shared(label, name = 'Y', borrow = True)

	filter_shape = (filter_dim, input_dim, window, 1)
	LogInfo.logs('input shape: %s', input_shape)
	LogInfo.logs('filter shape: %s', filter_shape)
	conv_layer = holistic_conv_layer(X, input_shape, filter_shape, pooling_op, activation)
	conv_output = conv_layer['output'].flatten(2)

	output_layer = full_connected_layer(conv_output, filter_dim, 1, sigmoid)
	prob = output_layer['output'].flatten(1)



	xent = -(Y * T.log(prob) + (1 - Y) * T.log(1 - prob))
	cost = xent.mean()

	params = conv_layer['params']
	grads = T.grad(cost, params)

	updates = [
		(param_i, param_i - learning_rate * grad_i)
		for param_i, grad_i in zip(params, grads)
	]
	train = function(
		inputs = [],
		outputs = xent,
		updates = updates
	)
	LogInfo.logs('Building Complete')

	show = function(
		inputs = [],
		outputs = prob
	)
	opt = show()
	LogInfo.logs('Show example output prob shape: %s', opt.shape)

	LogInfo.begin_track('Training start, steps = %d ... ', training_steps)
	for i in xrange(training_steps):
		err = train()
		if i % 100 == 0:
			LogInfo.logs('iter = %d, cross-entropy = %.6f', i, err.mean())
	LogInfo.end_track()




if __name__ == '__main__':
	test_program()
