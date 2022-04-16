import numpy as np

from learners.base import BaseLearner

import numpy as np


class LinearRegression(BaseLearner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def learn(self):
        pass

    def predict(self):
        pass

    @staticmethod
    def predict_adhoc(
            X: np.ndarray,
            theta: np.ndarray
    ) -> np.ndarray:
        """ Make a one off prediction not based on any learned parameters

        :param X: feature matrix (x0 to be 1s for intercept term)
                  (m x n+1)
        :param theta: vector of initial parameters
                      (n+1 x 1)
        :return:
        """
        return np.dot(X, theta)

