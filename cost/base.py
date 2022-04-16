from abc import ABC, abstractmethod

import numpy as np


class BaseCost(ABC):
    """ Base cost class which all cust functions must inherit
    """

    def __init__(self, y: np.ndarray, y_hat: np.ndarray):
        assert len(y) == len(y_hat)
        self.y = y
        self.y_hat = y_hat
        self.m = len(y)

    @abstractmethod
    def cost(self):
        pass
