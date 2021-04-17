import numpy as np
from abc import ABC

# np.random.seed(21)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(0, x)


def relu_prime(output_error, input_data):
    Z = input_data
    dZ = np.array(output_error, copy=True)
    # When z <= 0, you should set dz to 0 as well.
    dZ[Z <= 0] = 0
    assert (dZ.shape == Z.shape)
    return dZ


def sigmoid_prime(output_error, input_data):
    Z = input_data
    s = 1 / (1 + np.exp(-Z))
    dZ = output_error * s * (1 - s)
    assert (dZ.shape == Z.shape)
    return dZ


class Layer(ABC):
    # TODO implementare callable style
    def __init__(self):
        self.input_data = None
        self.output = None

    def forward(self, inputs):
        raise NotImplementedError

    def backward(self, output_error, **kwargs):
        raise NotImplementedError


class LinearLayer(Layer):

    def __init__(self, input_size, output_size):
        super().__init__()
        # self.weights = np.random.normal(scale=1 / input_size ** .5, size=(output_size, input_size))
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.ones((output_size, 1))

    def forward(self, input_data):
        self.input_data = input_data.copy()
        dot = self.weights.dot(self.input_data)
        self.output = dot + self.bias
        assert (self.output.shape == (self.weights.shape[0], self.input_data.shape[1]))
        return self.output

    def backward(self, output_error, **kwargs):
        learn_rate = kwargs['learn_rate']
        m = self.input_data.shape[1]

        weights_error = 1/m * output_error.dot(self.input_data.T)
        bias_error = 1/m * output_error.sum(axis=1, keepdims=True)
        input_error = self.weights.T.dot(output_error)

        assert (input_error.shape == self.input_data.shape)
        assert (weights_error.shape == self.weights.shape)
        assert (bias_error.shape == self.bias.shape)

        self.weights -= learn_rate * weights_error
        self.bias -= learn_rate * bias_error
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