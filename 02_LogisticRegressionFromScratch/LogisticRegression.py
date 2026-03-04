import numpy as np


class LogisticRegressionGD:

    def __init__(self, learning_rate=0.01, l2_lambda=0.0, multiclass=False):
        self.learning_rate = learning_rate
        self.l2_lambda = l2_lambda
        self.multiclass = multiclass
        self.w = None
        self.loss_history = []
        self.num_classes = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def softmax(self, z):
        z = z - np.max(z, axis=1, keepdims=True)
        exp = np.exp(z)
        return exp / np.sum(exp, axis=1, keepdims=True)

    def one_hot(self, y, k):
        m = len(y)
        Y = np.zeros((m, k))
        Y[np.arange(m), y] = 1
        return Y

    def predict_proba(self, X):
        if self.multiclass:
            return self.softmax(X @ self.w)
        return self.sigmoid(X @ self.w)

    def predict(self, X, threshold=0.5):
        probs = self.predict_proba(X)

        if self.multiclass:
            return np.argmax(probs, axis=1)

        return (probs >= threshold).astype(int)

    def compute_loss(self, X, y):

        m = X.shape[0]
        probs = self.predict_proba(X)
        epsilon = 1e-15
        probs = np.clip(probs, epsilon, 1 - epsilon)

        if self.multiclass:
            Y = self.one_hot(y, self.num_classes)
            data_loss = -(1/m) * np.sum(Y * np.log(probs))
        else:
            data_loss = -(1/m) * np.sum(
                y * np.log(probs) + (1 - y) * np.log(1 - probs)
            )

        l2_loss = (self.l2_lambda / (2*m)) * np.sum(self.w[1:] ** 2)

        return data_loss + l2_loss

    def compute_gradients(self, X, y):

        m = X.shape[0]
        probs = self.predict_proba(X)

        if self.multiclass:
            Y = self.one_hot(y, self.num_classes)
            error = probs - Y
        else:
            error = probs - y

        dw = (1/m) * (X.T @ error)
        dw[1:] += (self.l2_lambda / m) * self.w[1:]

        return dw

    def fit(self, X, y, epochs=1000):

        n = X.shape[1]

        if self.multiclass:
            self.num_classes = len(np.unique(y))
            self.w = np.zeros((n, self.num_classes))
        else:
            self.w = np.zeros(n)

        self.loss_history = []

        for _ in range(epochs):
            dw = self.compute_gradients(X, y)
            self.w -= self.learning_rate * dw
            self.loss_history.append(self.compute_loss(X, y))

    def accuracy(self, y_true, y_pred):
        return np.mean(y_true == y_pred)

    def precision(self, y_true, y_pred):
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        return tp / (tp + fp + 1e-15)

    def recall(self, y_true, y_pred):
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        return tp / (tp + fn + 1e-15)

    def f1(self, y_true, y_pred):
        p = self.precision(y_true, y_pred)
        r = self.recall(y_true, y_pred)
        return 2 * p * r / (p + r + 1e-15)

    def roc_curve(self, y_true, probs):

        thresholds = np.linspace(0, 1, 100)
        tpr = []
        fpr = []

        for t in thresholds:

            preds = (probs >= t).astype(int)

            tp = np.sum((y_true == 1) & (preds == 1))
            fp = np.sum((y_true == 0) & (preds == 1))
            fn = np.sum((y_true == 1) & (preds == 0))
            tn = np.sum((y_true == 0) & (preds == 0))

            tpr.append(tp / (tp + fn + 1e-15))
            fpr.append(fp / (fp + tn + 1e-15))

        return np.array(fpr), np.array(tpr), thresholds

    def auc(self, fpr, tpr):
        order = np.argsort(fpr)
        return np.trapz(tpr[order], fpr[order])

    def evaluate(self, X, y, threshold=0.5):

        probs = self.predict_proba(X)

        if self.multiclass:
            preds = np.argmax(probs, axis=1)
            acc = self.accuracy(y, preds)
            return {"accuracy": acc}

        preds = (probs >= threshold).astype(int)

        acc = self.accuracy(y, preds)
        prec = self.precision(y, preds)
        rec = self.recall(y, preds)
        f1 = self.f1(y, preds)

        fpr, tpr, _ = self.roc_curve(y, probs)
        auc_score = self.auc(fpr, tpr)

        return {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "auc": auc_score
        }

    def hyperparameter_search(self, X_train, y_train, X_val, y_val,
                              learning_rates, l2_values, thresholds,
                              epochs=1000):

        results = []

        for lr in learning_rates:
            for l2 in l2_values:

                model = LogisticRegressionGD(
                    learning_rate=lr,
                    l2_lambda=l2,
                    multiclass=self.multiclass
                )

                model.fit(X_train, y_train, epochs)

                for t in thresholds:

                    metrics = model.evaluate(X_val, y_val, t)

                    results.append({
                        "lr": lr,
                        "l2": l2,
                        "threshold": t,
                        **metrics
                    })

        return results

    def get_params(self):
        return self.w