import numpy as np
import network as nn
import layer
import matplotlib.pyplot as plt


def f(a):
    return np.power(a, 2)


def x_or(a):
    return 1 * np.logical_xor(a[0], a[1])


x_train = np.linspace(0, 5, 1000).reshape(1, -1) # np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).reshape(2, 4)
y_train = f(x_train).reshape(1, -1)
x_test = np.linspace(0, 5, 10).reshape(1, -1) # np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).reshape(2, 4)
y_test = f(x_test).reshape(1, -1)

hidden = 2
regression_network = nn.Network(loss=nn.mse, loss_prime=nn.mse_prime, learn_rate=0.005, epochs=500)
# input layer
regression_network.add_layer(layer.LinearLayer(1, hidden))
regression_network.add_layer(layer.ActivationLayer())
for _ in range(1):
    regression_network.add_layer(layer.LinearLayer(hidden, hidden))
    regression_network.add_layer(layer.ActivationLayer(activation='relu'))
# output layer
regression_network.add_layer(layer.LinearLayer(hidden, 1))
# regression_network.add_hidden_layer(layer.ActivationLayer())

losses = regression_network.fit(x_train, y_train)
predictions, model_loss = regression_network.predict(x_test)
# y_test = y_test.reshape(predictions.shape)
print("Test loss: ", model_loss(y_test, predictions))
print(y_test, predictions)
x = np.linspace(0, regression_network.epochs, regression_network.epochs)
plt.plot(x, losses)
plt.show()
