from abc import ABC, abstractmethod


class BaseLearner(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def learn(self):
        pass

    @abstractmethod
    def predict(self):
        pass
