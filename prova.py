import numpy as np
import network as nn
import layer
import matplotlib.pyplot as plt


def f(a):
    return np.power(a, 2)


def x_or(a):
    return 1 * np.logical_xor(a[0], a[1])


x_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).reshape(2, 4) # np.linspace(0, 100, 1000).reshape(1, -1)
y_train = x_or(x_train).reshape(1, -1)
x_test = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).reshape(2, 4) # np.linspace(0, 100, 50).reshape(1, -1)
y_test = x_or(x_test).reshape(1, -1)

hidden = 2
regression_network = nn.Network(learn_rate=0.09)
# input layer
regression_network.add_hidden_layer(layer.LinearLayer(hidden, hidden))
regression_network.add_hidden_layer(layer.ActivationLayer())
for _ in range(0):
    regression_network.add_hidden_layer(layer.LinearLayer(hidden, hidden))
    regression_network.add_hidden_layer(layer.ActivationLayer(activation='relu'))
# output layer
regression_network.add_hidden_layer(layer.LinearLayer(hidden, 1))
regression_network.add_hidden_layer(layer.ActivationLayer())

losses = regression_network.fit(x_train, y_train)
predictions, model_loss = regression_network.predict(x_test)
# y_test = y_test.reshape(predictions.shape)
print("Test loss: ", model_loss(y_test, predictions))
print(y_test, predictions)
x = np.linspace(0, regression_network.epochs, regression_network.epochs)
plt.plot(x, losses)
plt.show()
