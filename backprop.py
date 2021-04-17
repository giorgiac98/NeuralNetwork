import numpy as np
from data_prep import features, targets, features_test, targets_test

np.random.seed(21)


def sigmoid(x):
    """
    Calculate sigmoid
    """
    return 1 / (1 + np.exp(-x))


# Hyperparameters
n_hidden = 8  # number of hidden units
epochs = 900
learnrate = 0.01

n_records, n_features = features.shape
last_loss = None
# Initialize weights
weights_input_hidden = np.random.normal(scale=1 / n_features ** .5,
                                        size=(n_features, n_hidden))
weights_hidden_hidden = np.random.normal(scale=1 / n_features ** .5,
                                         size=(n_hidden, n_hidden))
weights_hidden_output = np.random.normal(scale=1 / n_features ** .5,
                                         size=n_hidden)

for e in range(epochs):
    del_w_input_hidden = np.zeros(weights_input_hidden.shape)
    del_w_hidden_hidden = np.zeros(weights_hidden_hidden.shape)
    del_w_hidden_output = np.zeros(weights_hidden_output.shape)
    for x, y in zip(features, targets):
        # Forward pass ##
        # Calculate the output
        hidden1_output = sigmoid(np.dot(x, weights_input_hidden))
        hidden2_output = sigmoid(np.dot(weights_hidden_hidden, hidden1_output))
        output = np.dot(weights_hidden_output, hidden2_output)

        # Backward pass ##
        # Calculate the network's prediction error
        error = abs(y - output)

        # Calculate error term for the output unit
        output_error_term = error * weights_hidden_output * hidden2_output * (1 - hidden2_output)

        # propagate errors to hidden layer
        # Calculate the error term for the hidden layer
        hidden2_error = np.dot(weights_hidden_output, output_error_term)
        hidden2_error_term = hidden2_error * hidden2_output * (1 - hidden2_output)

        hidden_error = np.dot(weights_hidden_hidden, hidden2_error_term)
        hidden_error_term = hidden_error * hidden1_output * (1 - hidden1_output)

        # Update the change in weights
        del_w_hidden_output += output_error_term * hidden2_output
        del_w_hidden_hidden += hidden2_error_term * hidden1_output
        del_w_input_hidden += hidden_error_term * x[:, None]

    # Update weights
    weights_input_hidden += learnrate * del_w_input_hidden / n_records
    weights_hidden_hidden += learnrate * del_w_hidden_hidden / n_records
    weights_hidden_output += learnrate * del_w_hidden_output / n_records

    # Printing out the mean square error on the training set
    if e % (epochs / 10) == 0:
        hidden1 = sigmoid(np.dot(features, weights_input_hidden))
        hidden2 = sigmoid(np.dot(hidden1, weights_hidden_hidden))
        out = np.dot(hidden2, weights_hidden_output)
        loss = np.mean((out - targets) ** 2)

        if last_loss and last_loss < loss:
            print("Train loss: ", loss, "  WARNING - Loss Increasing")
        else:
            print("Train loss: ", loss)
        last_loss = loss

# Calculate accuracy on test data
hidden1 = sigmoid(np.dot(features_test, weights_input_hidden))
hidden2 = sigmoid(np.dot(hidden1, weights_hidden_hidden))
out = np.dot(hidden2, weights_hidden_output)
loss = np.mean((out - targets_test) ** 2)
print("Test loss: ", loss)
