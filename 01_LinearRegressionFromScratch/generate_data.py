import numpy as np
import pandas as pd

np.random.seed(42)

n = 100        
d = 3          

X = np.random.uniform(1, 100, (n, d))

X = np.hstack((np.ones((n, 1)), X))

true_w = np.array([10, 5, -3, 2])

sigma = 2
noise = np.random.normal(0, sigma, n)

y = X @ true_w + noise

df = pd.DataFrame(
    X[:, 1:],  
    columns=[f"x{i+1}" for i in range(d)]
)
df["y"] = y

df.to_csv("data.csv", index=False)
