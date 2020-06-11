import numpy as np


def f(w, x, y):
	return (w*x - y)** 2


def gradent(w, x, y):
	return 2*(w*x-y) * x


def unconstrained_optimization(w, x, y, lr=1e-3):
	fx = f(w, x, y)
	while True:
		# print("w = %f, fx = %f" % (w, fx))
		w_new = w - lr * gradent(w, x, y)
		fx_new = f(w_new, x, y)
		print("f(%f) = %f, f(%f) = %f" % (w, fx, w_new, fx_new))
		if fx - fx_new < 1e-9:
			break
		w = w_new
		fx = fx_new


def task_1():
	'''
	min (w*x - y)**2
	x = 1.5, y = 0.5
	w0 = 0.8

	gradient = 2*(w*x-y) * x
		     = 4.5*w - 1.5
	iteration: w_k+1 = w_k - alpha_k * gradient
	'''
	w0, x, y = 0.8, 1.5, 0.5
	unconstrained_optimization(w0, x, y)


#############################################################
def affine_forward(x, w):
	'''
	Inputs:
	- x: Input data, (n: 4, d: 3)
	- w: Weights, (D: 3, M)

	Return:
	- out: (n: 4, M)
	- cache(use in backward): Tuple of: x, w
	'''
	out = x.dot(w)
	cache = (x, w)
	return out, cache


def affine_backward(dout, cache):
	"""
	Inputs:
	- dout: Upstream derivative, of shape (N, M)
	- cache: Tuple of:
		- x: Input data, of shape (N: 4, d: 3)

	Return:
	- dx: Gradient with respect to x, of shape (N, D)
	- dw: Gradient with respect to w, of shape (D, M)
	"""
	x, w = cache
	dx = dout.dot(w.T)
	dw = x.T.dot(dout)
	return dx, dw


def loss_func(x, y):
	loss = np.sum((x - y) ** 2)
	dloss = 2 * (x - y)
	return loss, dloss


class TwoLayerNet(object):

	def __init__(self, input_dim=3, hidden_dim=2, output_dim=1, weight_scale=1e-3):
		self.params = {}
		self.params['W1'] = weight_scale * np.random.randn(input_dim, hidden_dim)
		self.params['W2'] = weight_scale * np.random.randn(hidden_dim+1, output_dim)

	def loss(self, X, y=None):
		W1, W2 = self.params['W1'], self.params['W2']
		X = np.insert(X, 0, values=1, axis=1)  ## rua
		out1, cache1 = affine_forward(X, W1)
		out1 = np.insert(out1, 0, values=1, axis=1)  # rua
		out2, cache2 = affine_forward(out1, W2)
		output = out2
		if y is None:  # test time: just forward
			return output

		# train time: forward & backward
		loss, dloss = loss_func(output, y)
		grads = {}
		dout2, grads['W2'] = affine_backward(dloss, cache2)
		dout2 = dout2[:, 1:]
		dout1, grads['W1'] = affine_backward(dout2, cache1)
		return loss, grads

	def output_params(self):
		print('w1:\n', self.params['W1'])
		print('w2:\n', self.params['W2'])

def task_2(num_iterations=2000, learning_rate=1e-3):
	X = np.array([
		[0, 0],
		[0, 1],
		[1, 0],
		[1, 1],
	])
	y = np.array([[0], [1], [1], [0]])
	model = TwoLayerNet()

	for t in range(num_iterations):
		loss, grads = model.loss(X, y)
		for p, w in model.params.items():
			dw = grads[p]
			w -= learning_rate * dw  # sgd
		if (t % (num_iterations // 30) == 0):
			print('loss = %f' % (loss))
	# model.output_params()
	print(model.loss(X))


def main():
	# task_1()
	task_2()


if __name__ == '__main__':
	main()

