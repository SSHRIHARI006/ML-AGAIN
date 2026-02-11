import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from LinearRegression import LinearRegressionGD

df = pd.read_csv('electricity_consumption_based_weather_dataset.csv')

df["date"] = pd.to_datetime(df["date"])
df["day_of_year"] = df["date"].dt.dayofyear
df["weekday"] = df["date"].dt.weekday

df["sin_day"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
df["cos_day"] = np.cos(2 * np.pi * df["day_of_year"] / 365)

X = df[["AWND", "PRCP", "TMAX", "TMIN", "weekday", "sin_day", "cos_day"]]
y = df["daily_consumption"].to_numpy()

X = X.fillna(X.mean()).to_numpy()

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

mu = X_train.mean(axis=0)
sigma = X_train.std(axis=0)

X_train = (X_train - mu) / sigma
X_val = (X_val - mu) / sigma

def poly2(X):
    n = X.shape[1]
    features = [X]
    for i in range(n):
        for j in range(i, n):
            features.append((X[:, i] * X[:, j]).reshape(-1, 1))
    return np.hstack(features)

X_train = poly2(X_train)
X_val = poly2(X_val)

X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
X_val = np.hstack((np.ones((X_val.shape[0], 1)), X_val))

def evaluate(model, X_train, y_train, X_val, y_val):
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)

    train_rmse = np.sqrt(np.mean((y_train - y_train_pred) ** 2))
    val_rmse = np.sqrt(np.mean((y_val - y_val_pred) ** 2))

    ss_res = np.sum((y_val - y_val_pred) ** 2)
    ss_tot = np.sum((y_val - y_val.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot

    return train_rmse, val_rmse, r2

configs = {
    "Normal": (0.0, 0.0),
    "L2": (0.0, 0.001),
    "L1": (0.001, 0.0),
    "ElasticNet": (0.001, 0.001)
}

print("Model | Train RMSE | Val RMSE | R2")
print("--------------------------------------")

for name, (l1, l2) in configs.items():
    model = LinearRegressionGD(
        learning_rate=0.005,
        l1_lambda=l1,
        l2_lambda=l2
    )

    model.fit(X_train, y_train, epochs=5000)

    train_rmse, val_rmse, r2 = evaluate(
        model, X_train, y_train, X_val, y_val
    )

    print(name, train_rmse, val_rmse, r2)
