import numpy as np

from cost.log_loss import LogLossCost
from learners.base import BaseLearner
from optimisers.gradient_descent import gradient_descent


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


class LogisticRegression(BaseLearner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def pad_X(X: np.ndarray) -> np.ndarray:
        """ Pads X with 1s for theta 0 (intercept)
        :param X: feature matrix (m x n)
        :return: padded feature matrix
        """
        m = X.shape[0]
        return np.concatenate([np.ones((m, 1)), X], axis=1)

    def learn(self, X, y, padded=False):
        """ Train on X and y and store learned parameters

        :param X: feature matrix (m x n) or (mxn+1 if padded is true)
        :param y: target values
        :param padded: has the array already been padded with 1s for theta0
        :return:
        """
        if not padded:
            X = self.pad_X(X)

        results = gradient_descent(
            X,
            y,
            cost=LogLossCost,
            learner=LogisticRegression,
            theta=np.zeros(X.shape[1]),
            alpha=self.alpha,
            n_iter=self.n_iter
        )
        self.theta = results.theta_final
        self.theta_history = results.theta_history
        self.cost_history = results.cost_history

    def predict(
            self,
            X: np.ndarray,
            padded: bool = False
    ):
        return self.predict_adhoc(X, self.theta, padded)

    def predict_adhoc(
            self,
            X: np.ndarray,
            theta: np.ndarray,
            padded: bool = True
    ) -> np.ndarray:
        """ Make a one off prediction not based on any learned parameters

        :param X: feature matrix (m x n) or (mxn+1 if padded is true)
        :param padded: has the array already been padded with 1s for theta0
        :param theta: vector of initial parameters (n+1 x 1)
        :return:
        """
        if not padded:
            X = self.pad_X(X)
        return sigmoid(np.dot(X, theta))
