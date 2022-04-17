from abc import ABC, abstractmethod

import numpy as np


class BaseLearner(ABC):
    """ Base class which all learner objects must inherit
    """
    def __init__(self, alpha=0.01, n_iter=1500):
        self._setup()
        self.alpha = alpha
        self.n_iter = n_iter

    @abstractmethod
    def learn(self, X, y):
        pass

    @abstractmethod
    def predict(self, X: np.ndarray):
        pass

    @staticmethod
    @abstractmethod
    def predict_adhoc(
            X: np.ndarray,
            theta: np.ndarray
    ):
        pass

    def _setup(self):
        self.theta = None
        self.theta_history = None
        self.cost_history = None
