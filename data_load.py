import numpy as np
import pandas as pd


def load_regression_dataset(train_size=0.9):
    data = pd.read_csv('data/X.csv', sep=' ', header=None)
    target = pd.read_csv('data/y.csv', header=None)
    for c in data.columns:
        data[c] = (data[c] - data[c].mean()) / data[c].std()
    return split_data(data, target, train_size)


def load_housing_dataset(train_size=0.9):
    df = pd.read_fwf('data/housing.data', header=None)
    target = df.iloc[:, -1:]
    data = df.iloc[:, :-1]
    for c in data.columns:
        data[c] = (data[c] - data[c].mean()) / data[c].std()
    return split_data(data, target, train_size)


def load_banknote_dataset(train_size=0.9):
    df = pd.read_csv('data/data_banknote_authentication.txt', header=None)
    target = df.iloc[:, -1:]
    data = df.iloc[:, :-1]
    for c in data.columns:
        data[c] = (data[c] - data[c].mean()) / data[c].std()
    return split_data(data, target, train_size)


def load_admission_dataset(train_size=0.9):
    admissions = pd.read_csv('data/binary.csv')
    data = pd.concat([admissions, pd.get_dummies(admissions['rank'], prefix='rank')], axis=1)
    target = data['admit']
    data = data.drop(['rank', 'admit'], axis=1)
    for field in ['gre', 'gpa']:
        mean, std = data[field].mean(), data[field].std()
        data.loc[:, field] = (data[field]-mean)/std
    return split_data(data, target, train_size)


def split_data(data, target, train_size):
    sample = np.random.choice(data.index, size=int(len(data) * train_size), replace=False)
    features = data.iloc[sample].to_numpy().T
    features_test = data.drop(sample).to_numpy().T
    targets = target.iloc[sample].to_numpy().reshape(1, -1)
    targets_test = target.drop(sample).to_numpy().reshape(1, -1)
    return features, features_test, targets, targets_test
