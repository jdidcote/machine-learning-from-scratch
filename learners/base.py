from abc import ABC, abstractmethod

import numpy as np


class BaseLearner(ABC):
    """ Base class which all learner objects must inherit
    """
    def __init__(self):
        self._setup()

    @abstractmethod
    def learn(self, X, y):
        pass

    @abstractmethod
    def predict(self):
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
