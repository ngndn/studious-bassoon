import numpy as np
import pandas as pd
from pandas.util.testing import debug
from sklearn.model_selection import train_test_split, LeaveOneOut
from sklearn.linear_model import LinearRegression
from BaseLineReg import BaseLineReg

data_train = pd.read_csv('data/regression_dataset_training.csv', index_col=0)
data_test = pd.read_csv('data/regression_dataset_testing.csv', index_col=0)
data_test_solution = pd.read_csv('data/regression_dataset_testing_solution.csv',
                                 index_col=0)

x_train, y_train = data_train.drop('vote', axis=1), data_train.vote
x_test = data_test


def run_LOOCV(x, y, model):
    """
    Run the LOOCV with x (observations), y (outcomes) and a specific model.
    Model must have fit method for training and predict method for testing.

    :param x: observations (matrix)
    :param y: outcomes (column vector)
    :param model: learning model that has fit and predict method
    :return:
    """

    # convert to numpy matrix for compatible with LeaveOneOut()
    x = x.as_matrix()
    y = y.as_matrix()

    # LeaveOneOut object
    loo = LeaveOneOut()

    # TODO: replace with the general measurement function
    mse = 0.0

    for val_train_index, val_test_index in loo.split(x):
        x_train_val, x_test_val = x[val_train_index, :], x[val_test_index, :]
        y_train_val, y_test_val = y[val_train_index], y[val_test_index]

        model.fit(x_train_val, y_train_val)
        y_pred = model.predict(x_test_val)

        mse += (y_pred - y_test_val) ** 2

    mse /= len(x)

    return mse[0]


def baseline_with_LOOCV(x_train, y_train):
    """
    Baseline method for regression task. This method predict the average of y
    for all observations. Use LOOCV for model assessment.

    :param x_train:
    :param y_train:
    :return: Double (Mean Square Error)
    """
    baseline = BaseLineReg()

    return run_LOOCV(x_train, y_train, baseline)


def linear_with_LOOCV(x_train, y_train):
    """
    Linear model for regression task. Use LinearRegression from scikit-learn.
    Use LOOCV for model assessment

    :param x_train:
    :param y_train:
    :return: Double (Mean Square Error)
    """
    linear = LinearRegression()

    return run_LOOCV(x_train, y_train, linear)


def linear(submit=False):
    # data split
    xtr, xte, ytr, yte = train_test_split(
        x_train, y_train, test_size=0.2, random_state=42
    )

    # training
    model = LinearRegression()
    model.fit(xtr, ytr)

    # testing
    mse = np.mean((model.predict(xte) - yte)**2)
    print('MSE (linear) : {}'.format(mse))

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
