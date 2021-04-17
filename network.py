import numpy as np


def mse(y, y_pred):
    ms = 1/2 * np.mean((y - y_pred)**2)
    assert (ms.shape == ())
    print(ms)
    return ms


def mse_prime(y, y_pred):
    # TODO revise
    return y_pred - y


def cross_entropy(y, y_pred):
    m = y.shape[1]
    ce = -1/m * np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
    ce = np.squeeze(ce)
    assert (ce.shape == ())
    return ce


def cross_entropy_prime(y, y_pred):
    return - (np.divide(y, y_pred) - np.divide(1 - y, 1 - y_pred))


class Network:

    def __init__(self, loss=cross_entropy, loss_prime=cross_entropy_prime, epochs=900, learn_rate=0.005):
        self.loss = loss
        self.loss_prime = loss_prime
        self.epochs = epochs
        self.learn_rate = learn_rate
        self.layers = []

    def add_hidden_layer(self, hidden_layer):
        self.layers.append(hidden_layer)

    def fit(self, x_train, y_train):
        losses = []
        last_loss = None

        for e in range(self.epochs):
            # forward pass
            output = x_train.copy()
            for lay in self.layers:
                output = lay.forward(output)

            loss = self.loss(y_train, output)
            losses.append(loss)

            if e % (self.epochs / 10) == 0:
                if last_loss and last_loss < loss:
                    print("Train loss: ", loss, "  WARNING - Loss Increasing")
                else:
                    print("Train loss: ", loss)
                last_loss = loss

            # backward pass
            output_error = self.loss_prime(y_train, output)
            for lay in reversed(self.layers):
                output_error = lay.backward(output_error, learn_rate=self.learn_rate)

        return losses

    def predict(self, x_test):
        hidden_output = x_test
        for lay in self.layers:
            hidden_output = lay.forward(hidden_output)
        return hidden_output, self.loss
