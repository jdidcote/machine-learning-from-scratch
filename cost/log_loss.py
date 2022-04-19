import numpy as np

from cost.base import BaseCost


class LogLossCost(BaseCost):
    """ Least squares cost function for linear regression
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def cost(self) -> float:

        error = np.sum(
            -self.y.dot(np.log(self.y_hat)) - (1 - self.y).dot(np.log(1 - self.y_hat))
        )
        cost = (1 / self.m) * error
        return cost

    def cost_derivative(self, X: np.ndarray) -> float:
        """ Compute the least squares cost function derivative

        :param X: n by m feature matrix
        :return:
        """

        return (1 / self.m) * (self.y_hat - self.y).dot(X)


if __name__ == '__main__':
    X = np.array([[1, 2], [2, 1], [3, 2]])
    y = np.array([1, 0, 1])
    y_hat = np.array([0.25, 0.6, 0.8])
    _ = LogLossCost(y, y_hat)
    print(_.cost())
