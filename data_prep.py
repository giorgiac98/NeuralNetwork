import numpy as np
import pandas as pd

data = pd.read_csv('X.csv', sep=' ', header=None)
target = pd.read_csv('y.csv', header=None)

# Split off random 10% of the data for testing
sample = np.random.choice(data.index, size=int(len(data)*0.9), replace=False)
features, features_test = data.iloc[sample].to_numpy(), data.drop(sample).to_numpy()
targets, targets_test = target.iloc[sample].to_numpy().reshape(-1), target.drop(sample).to_numpy().reshape(-1)
