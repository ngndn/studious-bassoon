import numpy as np


def gradient_descent(xtr, ytr, alpha, tfunc, max_iter=10000):
    """
    Computing gradient descent to get the optimal value for parameter of a
    predefined target function.

    :param xtr: observations matrix
    :param ytr: outcome vector
    :param alpha: step size in gradient descend
    :param tfunc: a target function
    :return: an optimal vector theta
    """
    # x = xtr.as_matrix()
    x = xtr

    # reshape so that y is a column vector,
    # it's still fine if y already a column vector
    # y = ytr.as_matrix().reshape(-1, 1)
    y = ytr.reshape(-1, 1)

    # Initialize the theta, and the previous
    theta = np.random.rand(xtr.shape[1], 1)
    prev_theta = np.zeros([xtr.shape[1], 1])

    # TODO: examine the epsilon, too small -> too slow, too large --> very
    # inaccurate stopping point
    epsilon = 1e-5

    # Length of data x, for make the update step smaller based on data
    m = len(x)

    xt = x.transpose()

    i = 0
    while abs(np.mean(prev_theta - theta)) > epsilon:
        if i >= max_iter:
            break
        else:
            i += 1

        # Deep copy
        prev_theta = theta.copy()

        loss = tfunc(x, theta) - y
        gradient = xt @ loss / m

        # Update theta
        theta -= alpha * gradient

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
