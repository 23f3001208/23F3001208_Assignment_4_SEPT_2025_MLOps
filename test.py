import os
import unittest
import pandas as pd
import joblib
import subprocess
import glob

class TestIrisPipeline(unittest.TestCase):

    def test_data_validation(self):
        """Ensure data exists and has correct structure"""
        data_path = "data/iris.csv"
        self.assertTrue(os.path.exists(data_path), "iris.csv not found")
        df = pd.read_csv(data_path)

        expected_cols = {"sepal_length", "sepal_width", "petal_length", "petal_width", "species"}
        self.assertTrue(expected_cols.issubset(df.columns), "Missing required columns")
        self.assertFalse(df.isnull().values.any(), "Data contains null values")
        self.assertGreater(len(df), 0, "Data is empty")

    def test_model_training_and_evaluation(self):
        """Train model and check accuracy >= 0.7"""
        subprocess.run(["python3", "train.py"], check=True)

        # Find latest artifact directory
        artifact_dirs = sorted(glob.glob("artifacts/*"), key=os.path.getmtime)
        latest_dir = artifact_dirs[-1]
        model_path = os.path.join(latest_dir, "model.joblib")
        metrics_path = os.path.join(latest_dir, "metrics.csv")

        self.assertTrue(os.path.exists(model_path), "Model not found")
        self.assertTrue(os.path.exists(metrics_path), "Metrics not found")

        metrics_df = pd.read_csv(metrics_path)
        acc = metrics_df["accuracy"].iloc[0]
        print(f"Model Accuracy: {acc:.3f}")
        self.assertGreaterEqual(acc, 0.7, "Accuracy below acceptable threshold")

if __name__ == "__main__":
    unittest.main()

