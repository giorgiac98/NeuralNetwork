import numpy as np
import network as nn
import layer
import matplotlib.pyplot as plt
import data_load as dl

features, features_test, targets, targets_test = dl.load_prepared_data(0.8)

n_features, n_records = features.shape

configs = [{'learn_rate': 0.001, 'l2_lambda': 0, 'batch_size': None, 'momentum': 0},
           {'learn_rate': 0.0008, 'l2_lambda': 1e-5, 'batch_size': 64, 'momentum': 0},
           {'learn_rate': 0.001, 'l2_lambda': 0, 'batch_size': 128, 'momentum': 0.9},
           {'learn_rate': 0.0005, 'l2_lambda': 1e-5, 'batch_size': 128, 'momentum': 0.9}]

fig, axis = plt.subplots(2, 2)

for i, c in enumerate(configs):
    print(f'Configuration: {c}')
    model = nn.Network(epochs=300, learn_rate=c['learn_rate'], l2_lambda=c['l2_lambda'])
    # input layer
    model.add_layer(layer.LinearLayer(n_features, 256))
    model.add_layer(layer.ActivationLayer(activation='relu'))
    model.add_layer(layer.LinearLayer(256, 128))
    model.add_layer(layer.ActivationLayer(activation='relu'))
    # output layer
    model.add_layer(layer.LinearLayer(128, 1))
    model.add_layer(layer.ActivationLayer())

    losses = model.fit(features, targets, batch_size=c['batch_size'], momentum=c['momentum'])
    predictions, test_loss = model.predict(features_test, targets_test)
    print(f'Test loss: {test_loss}')
    acc = nn.accuracy_score(targets_test, predictions)
    print(f'Test accuracy: {acc}')
    x = np.linspace(0, model.epochs, model.epochs)
    ax = axis[i // 2, i % 2]
    ax.plot(x, losses)
    ax.set_title(f'Accuracy: {acc:.2f}\nConfiguration: learn_rate: {c["learn_rate"]}, l2_lambda: {c["l2_lambda"]},\n'
                 f'batch_size: {c["batch_size"]}, momentum: {c["momentum"]}')
    ax.set(xlabel='epochs', ylabel='loss')

fig.tight_layout()
plt.show()
