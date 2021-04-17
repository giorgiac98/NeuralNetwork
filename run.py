import numpy as np
import network as nn
import layer
import pandas as pd
import matplotlib.pyplot as plt

# data = pd.read_csv('X.csv', sep=' ', header=None)
# target = pd.read_csv('y.csv', header=None)

# df = pd.read_fwf('housing.data', header=None)
# target = df.iloc[:,-1:]
# data = df.iloc[:,:-1]

df = pd.read_csv('data_banknote_authentication.txt', header=None)
target = df.iloc[:,-1:]
data = df.iloc[:,:-1]
for c in data.columns:
    data[c] = (data[c] - data[c].mean())/data[c].std()

# admissions = pd.read_csv('binary.csv')

# Make dummy variables for rank
# data = pd.concat([admissions, pd.get_dummies(admissions['rank'], prefix='rank')], axis=1)
# data = data.drop('rank', axis=1)

# Standarize features
# for field in ['gre', 'gpa']:
#    mean, std = data[field].mean(), data[field].std()
#    data.loc[:,field] = (data[field]-mean)/std

# Split off random 10% of the data for testing
# np.random.seed(21)
# sample = np.random.choice(data.index, size=int(len(data)*0.9), replace=False)
# data, test_data = data.iloc[sample], data.drop(sample)

# Split into features and targets
# features, targets = data.drop('admit', axis=1).to_numpy().T, data['admit'].to_numpy().reshape(1, -1)
# features_test, targets_test = test_data.drop('admit', axis=1).to_numpy().T, test_data['admit'].to_numpy().reshape(1, -1)


# Split off random 10% of the data for testing
sample = np.random.choice(data.index, size=int(len(data)*0.9), replace=False)
features, features_test = data.iloc[sample].to_numpy().T, data.drop(sample).to_numpy().T
targets, targets_test = target.iloc[sample].to_numpy().reshape(1,-1), target.drop(sample).to_numpy().reshape(1,-1)

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
