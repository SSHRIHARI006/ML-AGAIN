import sys
sys.path.append('..')

import pandas as pd
import numpy as np
from LogisticRegression import LogisticRegressionGD
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report


df = pd.read_csv("IRIS.csv")

X = df[['sepal_length','sepal_width','petal_length','petal_width']].to_numpy()
y = df['species'].astype('category').cat.codes.to_numpy()


X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)


mu = X_train.mean(axis=0)
sigma = X_train.std(axis=0)

X_train = (X_train - mu) / sigma
X_val = (X_val - mu) / sigma


X_train = np.hstack((np.ones((X_train.shape[0],1)), X_train))
X_val = np.hstack((np.ones((X_val.shape[0],1)), X_val))


model = LogisticRegressionGD(
    learning_rate=0.1,
    l2_lambda=0.001,
    multiclass=True
)


model.fit(X_train, y_train, epochs=2000)


metrics = model.evaluate(X_val, y_val)
preds = model.predict(X_val)


print("\n==============================")
print("LOGISTIC REGRESSION RESULTS")
print("==============================\n")

for k, v in metrics.items():
    print(f"{k}: {v}")

print("\nFinal Training Loss:", model.loss_history[-1])

print("\nLearned Weights Shape:", model.get_params().shape)


print("\nConfusion Matrix")
print(confusion_matrix(y_val, preds))


print("\nClassification Report")
print(classification_report(y_val, preds))