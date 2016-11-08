import numpy as np
import pandas as pd
from pandas.util.testing import debug
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression

data_train = pd.read_csv('data/regression_dataset_training.csv', index_col=0)
data_test = pd.read_csv('data/regression_dataset_testing.csv', index_col=0)

x_train, y_train = data_train.drop('vote', axis=1), data_train.vote
x_test = data_test


def baseline(submit=False):
    # data split
    xtr, xte, ytr, yte = train_test_split(
        x_train, y_train, test_size=0.2, random_state=42
    )

    # training
    model = LinearRegression()
    model.fit(xtr, ytr)

    # testing
    mse = np.mean((model.predict(xte) - yte)**2)
    print('MSE (baseline) : {}'.format(mse))

    if submit:
        evaluate(x_test, model, 'linear_base')


def evaluate(x, model, name, round=False, negative=False):
    """
    Evaluate predictions using input X and MODEL.  Optionally round values to
    0 decimals.  Always remove negative values when predicting count problems.

    Save CSV for submission.

    """
    y = model.predict(x)
    if round:
        y = np.around(y)

    if not negative:
        y[y < 0] = 0

    y = pd.Series(y, index=x.index, name='vote')
    y.to_csv('{}_submission.csv'.format(name), header=True)
