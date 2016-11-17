import numpy as np


def gradient_descent(xtr, ytr, alpha, tfunc):
    """
    Computing gradient descent to get the optimal value for parameter of a
    predefined target function.

    :param xtr: observations matrix
    :param ytr: outcome vector
    :param alpha: step size in gradient descend
    :param tfunc: a target function
    :return: an optimal vector theta
    """
    x = xtr.as_matrix()

    # reshape so that y is a column vector,
    # it's still fine if y already a column vector
    y = ytr.as_matrix().reshape(-1, 1)

    # Initialize the theta, and the previous
    theta = np.random.rand(xtr.shape[1], 1)
    prev_theta = np.zeros([xtr.shape[1], 1])

    # stopping point
    epsilon = 1e-6

    # Length of data x, for make the update step smaller based on data
    m = len(x)

    while abs(np.mean(prev_theta - theta)) > epsilon:
        # Deep copy
        prev_theta = theta.copy()

        different = y - tfunc(x, theta)
        update = x.transpose() @ different

        # Update theta
        theta += alpha/m * update

    return theta


def linear(x, theta):
    """
    Linear regression function. Compute outcomes of a linear model.

    :param x: matrix of observations
    :param theta: weights of the linear model
    :return: outcome y
    """
    return x @ theta


def sigmoid(x, theta):
    """
    Sigmoid function. Compute outcome of sigmoid function.

    :param x: matrix of observations
    :param theta: weights
    :return: vector of probabilities based on sigmoid function
    """
    return 1 / (1 + np.exp(-(x @ theta)))
