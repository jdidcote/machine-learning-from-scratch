import numpy as np

from cost.base import BaseCost


class LeastSquaresCost(BaseCost):
    """ Least squares cost function for linear regression
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def cost(self) -> float:
        error = sum((self.y_hat - self.y) ** 2)
        cost = (1 / (2 * self.m)) * error
        return cost

    def cost_derivative(self, X: np.ndarray) -> float:
        """ Compute the least squares cost function derivative

        :param X: n by m feature matrix
        :return:
        """

        return (self.y_hat - self.y).dot(X)


if __name__ == '__main__':
    y = np.array([1, 2, 3])
    y_hat = np.array([1, 2, 3])
    print(LeastSquaresCost(y, y_hat).cost())
