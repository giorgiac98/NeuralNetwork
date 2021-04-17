import numpy as np
import network as nn
import layer
import matplotlib.pyplot as plt
import data_load as dl
import pandas as pd

features, features_test, targets, targets_test = dl.load_admission_dataset()

n_features, n_records = features.shape
hidden = n_features
regression_network = nn.Network(epochs=1500, learn_rate=0.09)
# input layer
regression_network.add_hidden_layer(layer.LinearLayer(n_features, hidden))
regression_network.add_hidden_layer(layer.ActivationLayer())
for _ in range(0):
    regression_network.add_hidden_layer(layer.LinearLayer(hidden, hidden))
    regression_network.add_hidden_layer(layer.ActivationLayer())
# output layer
regression_network.add_hidden_layer(layer.LinearLayer(hidden, 1))
regression_network.add_hidden_layer(layer.ActivationLayer())

losses = regression_network.fit(features, targets)
predictions, model_loss = regression_network.predict(features_test)
print("Test loss: ", model_loss(targets_test, predictions))
print(targets_test, predictions)
x = np.linspace(0, regression_network.epochs, regression_network.epochs)
plt.plot(x, losses)
plt.show()
