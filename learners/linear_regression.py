from learners.base import BaseLearner


class LinearRegression(BaseLearner):
    def __init__(self, *args, **kwargs):
        super().__init__(super().__init__(*args, **kwargs))

    def learn(self):
        pass

    def predict(self):
        pass


if __name__ == '__main__':
    LinearRegression()