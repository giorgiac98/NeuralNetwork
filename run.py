import numpy as np
import network as nn
import layer
import matplotlib.pyplot as plt
import data_load as dl


features, features_test, targets, targets_test = dl.load_prepared_data(0.8)

n_features, n_records = features.shape
hidden = n_features * 20
model = nn.Network(epochs=100, learn_rate=0.001, l2_lambda=1e-5)
# input layer
model.add_layer(layer.LinearLayer(n_features, hidden))
model.add_layer(layer.ActivationLayer(activation='relu'))
model.add_layer(layer.LinearLayer(hidden, 128))
model.add_layer(layer.ActivationLayer(activation='relu'))
# output layer
model.add_layer(layer.LinearLayer(128, 1))
model.add_layer(layer.ActivationLayer())

losses = model.fit(features, targets, batch_size=128, momentum=0.9)
predictions, test_loss = model.predict(features_test, targets_test)
acc = nn.accuracy_score(targets_test, predictions)
print(f'Test loss: {test_loss}')
print(f'Test accuracy: {acc}')
x = np.linspace(0, model.epochs, model.epochs)
plt.plot(x, losses)
plt.title(f'Accuracy: {acc:.2f} Test Loss: {test_loss:.2f}')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.show()
