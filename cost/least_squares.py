import numpy as np


def least_squares_cost(
        y: np.ndarray,
        y_hat: np.ndarray
):
    """ Compute the least-squares cost

    :param y: numpy array of actual target values
    :param y_hat: numpy array of predicted target values
    :return: least squares cost value
    """
    error = sum((y - y_hat) ** 2)
    cost = (1 / (2 * len(y))) * error

    return cost