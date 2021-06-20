import numpy as np
import network as nn
import layer
import matplotlib.pyplot as plt
import data_load as dl

features, features_test, targets, targets_test = dl.load_mnist()

n_features, n_records = features.shape
hidden = n_features
model = nn.Network(loss=nn.mse, loss_prime=nn.mse_prime, epochs=100, learn_rate=0.001)
# input layer
model.add_layer(layer.LinearLayer(n_features, hidden))
model.add_layer(layer.ActivationLayer(activation='relu'))
for _ in range(2):
    model.add_layer(layer.LinearLayer(hidden, hidden))
    model.add_layer(layer.ActivationLayer(activation='relu'))
# output layer
model.add_layer(layer.LinearLayer(hidden, 1))
# model.add_layer(layer.ActivationLayer())

losses = model.fit(features, targets, batch_size=128, momentum=0.8)
predictions, model_loss = model.predict(features_test)
print("Test loss: ", model_loss(targets_test, predictions))
print(targets_test, predictions)
x = np.linspace(0, model.epochs, model.epochs)
plt.plot(x, losses)
plt.show()
