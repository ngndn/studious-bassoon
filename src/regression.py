import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, LeaveOneOut, train_test_split

from BaseLineRegression import BaseLineRegression
from LinearRegressionSelfmade import LinearRegressionSelfmade

# Prepare data
data_train = pd.read_csv(
    '../data/regression_dataset_training.csv',
    index_col=0
)
x_train, y_train = data_train.drop('vote', axis=1), data_train.vote

x_test = pd.read_csv('../data/regression_dataset_testing.csv', index_col=0)
y_test = pd.read_csv(
    '../data/regression_dataset_testing_solution.csv', index_col=0
)
y_test = y_test.vote


def _score(x, y, model, cv=10):
    """
    Return model MSE scores for x (observations), y (outcomes) and a specified
    model.  Model must have methods `fit` and `predict`.

    :param x: observations (matrix)
    :param y: outcomes (column vector)
    :param model: learning model that has fit and predict method
    :return:

    """
    x = x.as_matrix()
    y = y.as_matrix()

    # Cross-validator
    if isinstance(cv, int):
        cv = KFold(n_splits=10)
    elif isinstance(cv, str) and cv == 'loo':
        cv = LeaveOneOut()

    scores = []
    for train_index, test_index in cv.split(x):
        x_train, x_test = x[train_index, :], x[test_index, :]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        scores.append(((y_pred - y_test) ** 2).mean())

    return scores


def baseline(x_train, y_train):
    """Baseline method for regression task."""
    baseline = BaseLineRegression()
    return _score(x_train, y_train, baseline)


def linear_regression(x_train, y_train):
    """Linear model for regression task."""
    linear = LinearRegression()
    return _score(x_train, y_train, linear)


def linear_regression_selfmade(x_train, y_train):
    """A self-made linear model for regression task."""
    linear = LinearRegressionSelfmade()
    return _score(x_train, y_train, linear)


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
    y.to_csv('../{}_submission.csv'.format(name), header=True)


def run(model, name=None, submit=False):
    # data split
    xtr, xte, ytr, yte = train_test_split(
        x_train, y_train, test_size=0.2, random_state=42
    )

    # validation, training
    scores = _score(xtr, ytr, model)
    print('MSE (validation, train) : {}'.format(np.mean(scores)))

    # validation, testing
    model.fit(xtr, ytr)
    mse = np.mean((model.predict(xte) - yte) ** 2)
    print('MSE (validation, test)  : {}'.format(mse))

    # testing
    model.fit(x_train, y_train)
    mse = np.mean((model.predict(x_test) - y_test) ** 2)
    print('MSE (testing)           : {}'.format(mse))

    if submit:
        if name is None:
            name = 'linear'

        evaluate(x_test, model, name)
