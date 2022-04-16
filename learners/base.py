from abc import ABC, abstractmethod

import numpy as np

class BaseLearner(ABC):
    """ Base class which all learner objects must inherit
    """
    def __init__(self):
        pass

    @abstractmethod
    def learn(self):
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
