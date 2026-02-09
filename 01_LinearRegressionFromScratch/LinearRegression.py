import numpy as np

class LinearRegressionGD:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.w = None
        self.loss_history = []

    def predict(self, X):
        return X @ self.w

    def compute_loss(self, X, Y):
        m = X.shape[0]
        Y_pred = self.predict(X)
        loss = (1 / (2 * m)) * np.sum((Y_pred - Y) ** 2)
        return loss

    def compute_gradients(self, X, Y):
        m = X.shape[0]
        Y_pred = self.predict(X)
        error = Y_pred - Y
        dw = (1 / m) * (X.T @ error)
        return dw

    def update_parameters(self, dw):
        self.w -= self.learning_rate * dw

    def fit(self, X, Y, epochs):
        self.w = np.zeros(X.shape[1])

        for _ in range(epochs):
            dw = self.compute_gradients(X, Y)
            self.update_parameters(dw)
            loss = self.compute_loss(X, Y)
            self.loss_history.append(loss)

    def get_params(self):
        return self.w
