import numpy as np

class LogisticRegressionGD:
    def __init__(self, learning_rate=0.01, l2_lambda=0.0):
        self.learning_rate = learning_rate
        self.l2_lambda = l2_lambda
        self.w = None
        self.loss_history = []

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def predict_proba(self, X):
        return self.sigmoid(X @ self.w)

    def predict(self, X, threshold=0.5):
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)

    def compute_loss(self, X, y):
        m = X.shape[0]
        y_hat = self.predict_proba(X)
        epsilon = 1e-15
        y_hat = np.clip(y_hat, epsilon, 1 - epsilon)

        data_loss = - (1/m) * np.sum(
            y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)
        )

        l2_loss = (self.l2_lambda / (2*m)) * np.sum(self.w[1:] ** 2)

        return data_loss + l2_loss

    def compute_gradients(self, X, y):
        m = X.shape[0]
        y_hat = self.predict_proba(X)
        error = y_hat - y

        dw = (1/m) * (X.T @ error)
        dw[1:] += (self.l2_lambda / m) * self.w[1:]

        return dw

    def fit(self, X, y, epochs=1000):
        self.w = np.zeros(X.shape[1])

        for _ in range(epochs):
            dw = self.compute_gradients(X, y)
            self.w -= self.learning_rate * dw
            self.loss_history.append(self.compute_loss(X, y))

    def get_params(self):
        return self.w
