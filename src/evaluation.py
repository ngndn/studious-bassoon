import operator as op
import sys

import numpy as np
import pandas as pd

from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_regression, chi2
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import KFold, LeaveOneOut

from classification import KNN
from regression import (
    LinearRegression,
    LinearRegressionCustom,
    PolynomialRegression
)


class Baseline(object):

    def __init__(self):
        self._model = None

    def __repr__(self):
        return '{}'.format(self.__class__.__name__)

    def fit(self, x, y):
        self._model = np.mean(y)

    def predict(self, x):
        return np.full(x.shape[0], self._model)


def _load(task_name):
    if task_name == 'regression':
        outcome = 'vote'
    elif task_name == 'classification':
        outcome = 'rating'
    else:
        raise NotImplementedError

    data_train = pd.read_csv(
        '../data/{}_dataset_training.csv'.format(task_name),
        index_col=0
    )
    x_train = data_train.drop(outcome, axis=1).as_matrix()
    y_train = getattr(data_train, outcome).as_matrix()
    x_test = pd.read_csv(
        '../data/{}_dataset_testing.csv'.format(task_name),
        index_col=0
    )
    x_test = x_test.as_matrix()
    y_test = pd.read_csv(
        '../data/{}_dataset_testing_solution.csv'.format(task_name),
        index_col=0
    )
    y_test = getattr(y_test, outcome).as_matrix()
    return data_train, x_train, y_train, x_test, y_test


def _score(x, y, model, score_func, cv=10, round=False):
    """
    Return model MSE scores for x (observations), y (outcomes) and a specified
    model.  Model must have methods `fit` and `predict`.

    Parameters
    ----------
    x : numpy.matrix
    y : numpy.matrix
    model : classifier / predictor
    score_func : callable scoring function
    cv : int
        Number of folds for cross validation
    round : bool
        Whether or not the score_func expects labels, or probabilities for each
        prediction.


    Returns
    -------
    score : float

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
        y_pred = np.round(y_pred) if round else y_pred
        scores.append(score_func(y_test, y_pred))

    return np.mean(scores)


def log_bernoulli_loss(true, pred):
    return -np.mean(true * np.log(pred) + (1 - true) * np.log(1 - pred))


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


def run(models, data, score_func, name=None, submit=False, round=False,
        optimization=None):
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
    round : bool
    optimization : str

    """
    if optimization == 'maximum':
        score_comp = op.lt
    elif optimization == 'minimum':
        score_comp = op.gt
    else:
        # raise NotImplementedError
        # Default to minimum
        score_comp = op.gt

    topm = None
    for model in models:
        print('Validating    : {}'.format(model))
        score = _score(data[0], data[1], model, score_func, round=round)
        print('Score (train) : {}\n'.format(score))
        if topm is None:
            topm = (model, score)
            continue

        if score_comp(topm[1], score):
            topm = (model, score)

    # Testing fit
    model = topm[0]
    model.fit(*data[:2])
    y_pred = model.predict(data[2])
    y_pred = np.round(y_pred) if round else y_pred
    score = score_func(data[3], y_pred)
    print('Best fit      : {}'.format(model))
    print('Score (test)  : {}'.format(score))

    # Kaggel:
    if submit:
        if name is None:
            name = 'linear'

        evaluate(data[2], model, name)


def run_regression():
    # Prepare data
    data_train, x_train, y_train, x_test, y_test = _load('regression')

    # Feature selection for regression on source data
    print('Selecting features for regression...\n')
    k = 7
    fs = SelectKBest(score_func=f_regression, k=k).fit(x_train, y_train)
    for score, feature in sorted(zip(fs.scores_, data_train.columns))[-k:]:
        print('{} ({:0.2f})'.format(feature, score))

    # Select features
    xtr = fs.transform(x_train)
    xte = fs.transform(x_test)

    # Add bias term
    # xtr = np.append(np.ones((xtr.shape[0], 1)), xtr, axis=1)
    # xte = np.append(np.ones((xte.shape[0], 1)), xte, axis=1)

    # Run model comparison
    print('\nScoring models...\n')
    models = [
        Baseline(),
        LinearRegression(),
        # LinearRegressionCustom(),
        PolynomialRegression(1),
        PolynomialRegression(2)
    ]
    data = [xtr, y_train, xte, y_test]
    run(models, data, score_func=mean_squared_error)


def run_classification():
    # Prepare data
    data_train, x_train, y_train, x_test, y_test = _load('classification')

    # Feature selection for classification on source data
    print('Selecting features for classification...\n')
    k = 7
    fs = SelectKBest(score_func=chi2, k=k).fit(x_train, y_train)
    for score, feature in sorted(zip(fs.scores_, data_train.columns))[-k:]:
        print('{} ({:0.2f})'.format(feature, score))

    # Select features
    xtr = fs.transform(x_train)
    xte = fs.transform(x_test)

    # Run model comparison
    print('\nScoring models...\n')
    models = [
        Baseline(),
        # KNN(5),
        KNeighborsClassifier(),
        RandomForestClassifier(n_estimators=10),
        BaggingClassifier(
            KNeighborsClassifier(),
            max_samples=0.5,
            max_features=0.5
        )
    ]
    data = [xtr, y_train, xte, y_test]
    run(
        models,
        data,
        score_func=accuracy_score,
        round=True,
        optimization='maximum'
    )


if __name__ == '__main__':
    if len(sys.argv) > 1:
        if sys.argv[1] == 'reg':
            run_regression()
        elif sys.argv[1] == 'cls':
            run_classification()

    else:
        run_regression()
        run_classification()
