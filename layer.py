import numpy as np
from abc import ABC

np.random.seed(48)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(0, x)


def relu_prime(output_error, input_data):
    derivative = np.array(output_error, copy=True)
    derivative[input_data <= 0] = 0
    assert (derivative.shape == input_data.shape)
    return derivative


def sigmoid_prime(output_error, input_data):
    s = sigmoid(input_data)
    derivative = output_error * s * (1 - s)
    assert (derivative.shape == input_data.shape)
    return derivative


class Layer(ABC):
    def __init__(self):
        self.input_data = None
        self.output = None

    def get_weights(self):
        return np.zeros(1)

    def forward(self, inputs):
        raise NotImplementedError

    def backward(self, output_error, **kwargs):
        raise NotImplementedError


class LinearLayer(Layer):

    def __init__(self, input_size, output_size):
        super().__init__()
        self.weights = np.random.randn(output_size, input_size) * 0.1
        self.bias = np.zeros((output_size, 1))
        self.prev_del_weights = np.zeros((output_size, input_size))
        self.prev_del_bias = np.zeros((output_size, 1))

    def get_weights(self):
        return self.weights.copy()

    def forward(self, input_data):
        self.input_data = input_data.copy()
        dot = self.weights.dot(self.input_data)
        self.output = dot + self.bias
        assert (self.output.shape == (self.weights.shape[0], self.input_data.shape[1]))
        return self.output

    def backward(self, output_error, **kwargs):
        learn_rate = kwargs['learn_rate']
        momentum = kwargs['momentum']
        l2_lambda = kwargs['l2_lambda']
        m = self.input_data.shape[1]

        weights_error = 1/m * output_error.dot(self.input_data.T) + (l2_lambda/m) * self.weights
        bias_error = 1/m * output_error.sum(axis=1, keepdims=True)
        input_error = self.weights.T.dot(output_error)

        assert (input_error.shape == self.input_data.shape)
        assert (weights_error.shape == self.weights.shape)
        assert (bias_error.shape == self.bias.shape)

        del_weights = learn_rate * weights_error + momentum * self.prev_del_weights
        del_bias = learn_rate * bias_error + momentum * self.prev_del_bias

        self.weights -= del_weights
        self.bias -= del_bias

        self.prev_del_weights = del_weights
        self.prev_del_bias = del_bias
        return input_error


class ActivationLayer(Layer):

    def __init__(self, activation='sigmoid'):
        super().__init__()
        self.output = None
        if activation == 'sigmoid':
            self.function = sigmoid
            self.derivative = sigmoid_prime
        elif activation == 'relu':
            self.function = relu
            self.derivative = relu_prime

    def forward(self, input_data):
        self.input_data = input_data.copy()
        self.output = self.function(self.input_data)
        assert (self.output.shape == self.input_data.shape)
        return self.output

    def backward(self, output_error, **kwargs):
        return self.derivative(output_error, self.input_data)
