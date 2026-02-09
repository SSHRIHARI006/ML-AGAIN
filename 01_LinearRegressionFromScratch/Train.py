import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from LinearRegression import LinearRegressionGD

df = pd.read_csv("data.csv")

y = df["y"].to_numpy()
X = df[["x1", "x2", "x3"]].to_numpy()

mu = X.mean(axis=0)
sigma = X.std(axis=0)
X_scaled = (X - mu) / sigma
X_scaled = np.hstack((np.ones((X_scaled.shape[0], 1)), X_scaled))

X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y, test_size=0.25, random_state=42
)

def polynomial_features(X, degree):
    X_poly = X[:, [0]]
    for d in range(1, degree + 1):
        X_poly = np.hstack((X_poly, X[:, 1:] ** d))
    return X_poly

degrees = [1, 3, 8]

print("Degree | Train Loss | Val Loss")
print("--------------------------------")

for d in degrees:
    lr = 0.01 if d == 1 else 0.005 if d == 3 else 0.001
    Xtr = polynomial_features(X_train, d)
    Xva = polynomial_features(X_val, d)

    model = LinearRegressionGD(learning_rate=lr)
    model.fit(Xtr, y_train, epochs=5000)

    train_loss = model.compute_loss(Xtr, y_train)
    val_loss = model.compute_loss(Xva, y_val)

    print(d, train_loss, val_loss)
