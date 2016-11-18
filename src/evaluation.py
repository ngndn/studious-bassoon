import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, LeaveOneOut

# Prepare data
data_train = pd.read_csv(
    '../data/regression_dataset_training.csv',
    index_col=0
)
x_train = data_train.drop('vote', axis=1).as_matrix()
y_train = data_train.vote.as_matrix()

x_test = pd.read_csv('../data/regression_dataset_testing.csv', index_col=0)
y_test = pd.read_csv(
    '../data/regression_dataset_testing_solution.csv', index_col=0
)
x_test = x_test.as_matrix()
y_test = y_test.vote.as_matrix()


def _score(x, y, model, cv=10):
    """
    Return model MSE scores for x (observations), y (outcomes) and a specified
    model.  Model must have methods `fit` and `predict`.

    :param x: observations (matrix)
    :param y: outcomes (column vector)
    :param model: learning model that has fit and predict method
    :return:

    """
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

    return np.mean(scores)


class Baseline(object):

    def __init__(self):
        self._model = None

    def fit(self, x, y):
        self._model = np.mean(y)

    def predict(self, x):
        return np.full(x.shape[0], self._model)


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


def run(models, name=None, submit=False):
    """
    Run model evaluation for MODELS and test the best fit.  Additionally, save
    CSV of test predictions for submission to Kaggle.

    Parameters
    ----------
    models : [model]
    name : str
    submit : bool

    """
    topm = None
    for model in models:
        print('Validating  : {}...'.format(model.__class__))
        mse = _score(x_train, y_train, model)
        print('MSE (train) : {}\n'.format(mse))
        if topm is None:
            topm = (model, mse)
            continue

        if topm[1] > mse:
            topm = (model, mse)

    # Testing fit
    model = topm[0]
    model.fit(x_train, y_train)
    mse = np.mean((model.predict(x_test) - y_test) ** 2)
    print('Best fit    : {}...'.format(model.__class__))
    print('MSE (test)  : {}'.format(mse))

    # Kaggel:
    if submit:
        if name is None:
            name = 'linear'

        evaluate(x_test, model, name)
