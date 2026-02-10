import numpy as np

class LinearRegressionGD:
    def __init__(self, learning_rate=0.01, l1_lambda=0.0, l2_lambda=0.0):
        self.learning_rate = learning_rate
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda
        self.w = None
        self.loss_history = []

    def predict(self, X):
        return X @ self.w

    def compute_loss(self, X, y):
        m = X.shape[0]
        y_pred = self.predict(X)
        data_loss = (1 / (2 * m)) * np.sum((y_pred - y) ** 2)
        l2_loss = (self.l2_lambda / (2 * m)) * np.sum(self.w[1:] ** 2)
        l1_loss = (self.l1_lambda / m) * np.sum(np.abs(self.w[1:]))
        return data_loss + l2_loss + l1_loss

    def compute_gradients(self, X, y):
        m = X.shape[0]
        y_pred = self.predict(X)
        error = y_pred - y
        dw = (1 / m) * (X.T @ error)

        dw[1:] += (self.l2_lambda / m) * self.w[1:]
        dw[1:] += (self.l1_lambda / m) * np.sign(self.w[1:])

        return dw

    def fit(self, X, y, epochs):
        self.w = np.zeros(X.shape[1])

        for _ in range(epochs):
            dw = self.compute_gradients(X, y)
            self.w -= self.learning_rate * dw
            self.loss_history.append(self.compute_loss(X, y))

    def get_params(self):
        return self.w
