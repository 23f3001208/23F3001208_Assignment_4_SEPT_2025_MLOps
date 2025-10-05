import os
import pandas as pd
import numpy as np
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import datetime
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, default="data/iris.csv",
                    help="Path to CSV data")
args = parser.parse_args()
DATA_PATH = args.data

TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
ARTIFACT_DIR = f"artifacts/{TIMESTAMP}"
MODEL_PATH = os.path.join(ARTIFACT_DIR, "model.joblib")
METRICS_PATH = os.path.join(ARTIFACT_DIR, "metrics.csv")

os.makedirs(ARTIFACT_DIR, exist_ok=True)

data = pd.read_csv(DATA_PATH)
train, test = train_test_split(data, test_size=0.4, stratify=data["species"], random_state=42)

X_train = train[["sepal_length","sepal_width","petal_length","petal_width"]]
y_train = train["species"]
X_test = test[["sepal_length","sepal_width","petal_length","petal_width"]]
y_test = test["species"]

# Train Model
model = DecisionTreeClassifier(max_depth=3, random_state=1)
model.fit(X_train, y_train)

# Evaluate
preds = model.predict(X_test)
acc = metrics.accuracy_score(y_test, preds)
precision = metrics.precision_score(y_test, preds, average="macro")
recall = metrics.recall_score(y_test, preds, average="macro")
f1 = metrics.f1_score(y_test, preds, average="macro")

metrics_df = pd.DataFrame([{
    "accuracy": acc,
    "precision": precision,
    "recall": recall,
    "f1_score": f1
}])
metrics_df.to_csv(METRICS_PATH, index=False)

print(f"Accuracy: {acc:.3f}")
print(f"Metrics Saved to {METRICS_PATH}")

joblib.dump(model, MODEL_PATH)
print(f"Model Saved to {MODEL_PATH}")
