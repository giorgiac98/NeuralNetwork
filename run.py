import numpy as np
import network as nn
import layer
import matplotlib.pyplot as plt
import data_load as dl

features, features_test, targets, targets_test = dl.load_prepared_data()

n_features, n_records = features.shape
hidden = n_features
model = nn.Network(epochs=80, learn_rate=0.001)
# input layer
model.add_layer(layer.LinearLayer(n_features, 512))
model.add_layer(layer.ActivationLayer(activation='relu'))
model.add_layer(layer.LinearLayer(512, 256))
model.add_layer(layer.ActivationLayer(activation='relu'))
model.add_layer(layer.LinearLayer(256, 128))
model.add_layer(layer.ActivationLayer(activation='relu'))
# output layer
model.add_layer(layer.LinearLayer(128, 1))
model.add_layer(layer.ActivationLayer())

losses = model.fit(features, targets, batch_size=100, momentum=0.9)
predictions, model_loss = model.predict(features_test)
print("Test loss: ", model_loss(targets_test, predictions))
print(targets_test, predictions)
x = np.linspace(0, model.epochs, model.epochs)
plt.plot(x, losses)
plt.show()
