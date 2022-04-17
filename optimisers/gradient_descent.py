from dataclasses import dataclass
from typing import List, Type

import numpy as np

from cost.base import BaseCost
from learners.base import BaseLearner


@dataclass
class GradientDescentHistory:
    theta_final: np.ndarray
    theta_history: List[np.ndarray]
    cost_history: List[int]


def gradient_descent(
        X: np.ndarray,
        y: np.ndarray,
        cost: Type[BaseCost],
        learner: Type[BaseLearner],
        theta: np.ndarray,
        alpha: float = 0.01,
        n_iter: int = 1500
) -> GradientDescentHistory:
    """ Minimise the given cost function using gradient descent

    :param X: feature matrix (x0 to be 1s for intercept term)
              (m x n+1)
    :param y: target vector
              (m x 1)
    :param cost: BaseCost object with cost and cost_derivative methods
    :param learner: BaseLearner object with predict_adhoc method
    :param theta: vector of initial parameters
                  (n+1 x 1)
    :param alpha: learning rate
    :param n_iter: total number of gradient descent iterations
    :return: History object with final theta, params as well as theta and cost history
    """

    m = len(y)
    theta_history, cost_history = [], []

    for _ in range(n_iter):
        y_hat = learner().predict_adhoc(X, theta)
        _cost = cost(y, y_hat)
        theta = theta - (alpha / m) * _cost.cost_derivative(X)

        theta_history.append(theta)
        cost_history.append(_cost.cost())

    return GradientDescentHistory(theta, theta_history, cost_history)
