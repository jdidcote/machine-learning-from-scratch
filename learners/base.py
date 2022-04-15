from abc import ABC, abstractmethod


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
