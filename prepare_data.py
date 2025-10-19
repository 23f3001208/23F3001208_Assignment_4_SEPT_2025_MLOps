import os
import pandas as pd
from sklearn.datasets import load_iris

os.makedirs("data", exist_ok=True)

iris = load_iris(as_frame=True)
df = iris.frame.copy()
df["species"] = df["target"].map(lambda t: iris.target_names[t])
df = df.drop(columns=["target"])
df.to_csv("data/iris.csv", index=False)
print("Wrote data/iris.csv with", len(df), "rows")
