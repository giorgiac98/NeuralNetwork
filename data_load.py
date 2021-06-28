import numpy as np
import pandas as pd

# from sklearn.impute import SimpleImputer
# from sklearn.preprocessing import LabelEncoder
# from sklearn.utils import resample, shuffle
# from sklearn.feature_selection import SelectKBest
# from sklearn.feature_selection import f_classif


def split_data(data, target, train_size):
    sample = np.random.choice(data.index, size=int(len(data) * train_size), replace=False)
    features = data.iloc[sample].to_numpy().T
    features_test = data.drop(sample).to_numpy().T
    targets = target.iloc[sample].to_numpy().reshape(1, -1)
    targets_test = target.drop(sample).to_numpy().reshape(1, -1)
    return features, features_test, targets, targets_test


''' 
def data_preparation(save=True):

    df = pd.read_csv('data/weatherAUS.csv')

    imputer1 = SimpleImputer(missing_values=np.nan, strategy='mean')
    imputer2 = SimpleImputer(missing_values=np.nan, strategy='median')
    df[['Evaporation']] = imputer2.fit_transform(df[['Evaporation']])
    df[['Sunshine']] = imputer1.fit_transform(df[['Sunshine']])
    df[['Cloud3pm']] = imputer2.fit_transform(df[['Cloud3pm']])
    df[['Cloud9am']] = imputer1.fit_transform(df[['Cloud9am']])
    # mean and median are same for Pressure9am and Pressure3pm
    df[['Pressure9am']] = imputer1.fit_transform(df[['Pressure9am']])
    df[['Pressure3pm']] = imputer1.fit_transform(df[['Pressure3pm']])

    df = df.dropna()

    for c in df.columns:
        if df[c].dtype == 'object':    # Since we are encoding object datatype to integer/float
            lbl = LabelEncoder()
            lbl.fit(list(df[c].values))
            df[c] = lbl.transform(df[c].values)

    zero = df[df['RainTomorrow'] == 0]   #zero values in outcome column
    one = df[df['RainTomorrow'] == 1]  # one values in outcome column
    df_majority_unsampled = resample(zero, replace=False, n_samples=10000)
    df_minority_unsampled = resample(one, replace=False, n_samples=10000)

    # concatenate
    df = pd.concat([df_minority_unsampled, df_majority_unsampled])

    df = shuffle(df)

    zero = df[df['RainToday'] == 0]  # zero values in outcome column
    one = df[df['RainToday'] == 1]  # one values in outcome column
    df_majority_unsampled = resample(zero, replace=False, n_samples=10000)
    df_minority_unsampled = resample(one, replace=True, n_samples=10000)
    # concatenate
    df = pd.concat([df_minority_unsampled, df_majority_unsampled])

    df = shuffle(df)

    print(df.RainToday.value_counts(), df.RainTomorrow.value_counts())

    X = df.drop(columns='RainTomorrow')
    y = df['RainTomorrow']

    fs = SelectKBest(score_func=f_classif, k=15)
    # apply feature selection
    X_selected = fs.fit_transform(X, y)
    cols = fs.get_support(indices=True)
    X_new = X.iloc[:, cols]

    data = X_new.apply(lambda v: (v - v.mean())/v.std(), axis=1)
    data['RainTomorrow'] = y
    print(data.head())
    if save:
        data.to_csv('data/prep_weatherAUS.csv', index=False)
    return data
'''


def load_prepared_data(train_size=0.9):
    df = pd.read_csv('data/prep_weatherAUS.csv')
    data = df.drop(columns='RainTomorrow')
    target = df['RainTomorrow']
    return split_data(data, target, train_size)


# data_preparation()
