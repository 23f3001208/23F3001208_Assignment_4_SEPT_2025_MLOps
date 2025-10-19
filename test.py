import unittest
import os
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class TestIrisModel(unittest.TestCase):
    def setUp(self):
        artifacts = sorted(os.listdir("artifacts"))
        self.latest = os.path.join("artifacts", artifacts[-1]) if artifacts else None
        self.model_path = os.path.join(self.latest, "model.joblib")
        self.metrics_path = os.path.join(self.latest, "metrics.csv")
        self.data_path = "data/iris.csv"

        self.assertTrue(os.path.exists(self.model_path), "Model file not found")
        self.assertTrue(os.path.exists(self.metrics_path), "Metrics file not found")
        self.assertTrue(os.path.exists(self.data_path), "Data file not found")

    def test_model_metrics_sanity(self):
        """Verify model metrics and minimum accuracy threshold"""
        metrics = pd.read_csv(self.metrics_path)
        acc = float(metrics["accuracy"].iloc[-1])
        self.assertGreaterEqual(acc, 0.6, f"Accuracy too low: {acc}")

    def test_model_prediction_shape(self):
        """Check predictions on sample inputs"""
        model = joblib.load(self.model_path)
        df = pd.read_csv(self.data_path).head(5)
        X = df[["sepal_length","sepal_width","petal_length","petal_width"]]
        preds = model.predict(X)
        self.assertEqual(len(preds), len(X), "Prediction length mismatch")

if __name__ == "__main__":
    unittest.main()
