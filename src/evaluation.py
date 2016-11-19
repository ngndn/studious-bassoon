import numpy as np
import pandas as pd

from sklearn.feature_selection import SelectKBest, f_regression, chi2
from sklearn.model_selection import KFold, LeaveOneOut

from regression import PolynomialRegression

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


class Baseline(object):

    def __init__(self):
        self._model = None

    def __repr__(self):
        return '{}'.format(self.__class__.__name__)

    def fit(self, x, y):
        self._model = np.mean(y)

    def predict(self, x):
        return np.full(x.shape[0], self._model)


def _score(x, y, model, score_func, cv=10):
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
        scores.append(score_func(y_test, y_pred))

    return np.mean(scores)


def log_bernoulli_loss(true, pred):
    return -np.mean((true * np.log(pred)) + ((1 - true) * np.log(1 - pred)))


def mean_squared_error(true, pred):
    return np.mean((pred - true) ** 2)


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


def run(models, data, score_func, name=None, submit=False):
    """
    Run model evaluation for MODELS and test the best fit.  Additionally, save
    CSV of test predictions for submission to Kaggle.

    Parameters
    ----------
    models : [model]
        Family of similar models.
    data : [x_train, y_train, x_test, y_test]
    name : str
    submit : bool

    """
    topm = None
    for model in models:
        print('Validating  : {}'.format(model))
        mse = _score(data[0], data[1], model, score_func)
        print('MSE (train) : {}\n'.format(mse))
        if topm is None:
            topm = (model, mse)
            continue

        if topm[1] > mse:
            topm = (model, mse)

    # Testing fit
    model = topm[0]
    model.fit(*data[:2])
    mse = score_func(data[3], model.predict(data[2]))
    print('Best fit    : {}'.format(model))
    print('MSE (test)  : {}'.format(mse))

    # Kaggel:
    if submit:
        if name is None:
            name = 'linear'

        evaluate(data[2], model, name)


def main():
    global x_train, y_train, x_test, y_test

    # Feature selection for regression on source data
    print('Selecting features for regression...\n')
    fs = SelectKBest(score_func=f_regression, k=5).fit(x_train, y_train)
    for score, feature in sorted(zip(fs.scores_, data_train.columns))[-5:]:
        print('{} ({:0.2f})'.format(feature, score))

    # Select features
    xtr = fs.transform(x_train)
    xte = fs.transform(x_test)

    # Regression
    print('\nScoring models...\n')
    models = [Baseline(), PolynomialRegression(1), PolynomialRegression(2)]
    data = [xtr, y_train, xte, y_test]
    run(models, data, score_func=mean_squared_error)

    # Classification
    pass


if __name__ == '__main__':
    main()
