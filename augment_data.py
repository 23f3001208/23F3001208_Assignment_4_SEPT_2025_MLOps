import pandas as pd
import numpy as np

df = pd.read_csv("data/iris.csv")

features = ["sepal_length","sepal_width","petal_length","petal_width"]
augmented = df.copy()
augmented[features] += np.random.normal(0, 0.1, size=augmented[features].shape)

augmented.to_csv("data/iris_augmented.csv", index=False)
print("Augmented Data Saved to data/iris_augmented.csv")