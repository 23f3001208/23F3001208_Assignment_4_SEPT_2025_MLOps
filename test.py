import unittest
import os
import subprocess
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score
import numpy as np

class TestIrisModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Ensure data exists
        if not os.path.exists("data/iris.csv"):
            print("⚙️  Running prepare_data.py ...")
            subprocess.run(["python", "prepare_data.py"], check=True)
        
        # Ensure model and artifacts exist
        if not os.path.exists("artifacts"):
            print("⚙️  Training model ...")
            subprocess.run(["python", "train.py"], check=True)

        # Get latest artifacts directory
        artifacts = sorted(os.listdir("artifacts"))
        if not artifacts:
            raise FileNotFoundError("No artifact directories found after training.")
        latest_dir = os.path.join("artifacts", artifacts[-1])

        cls.model_path = os.path.join(latest_dir, "model.joblib")
        cls.metrics_path = os.path.join(latest_dir, "metrics.csv")

        cls.model = joblib.load(cls.model_path)
        cls.metrics = pd.read_csv(cls.metrics_path)

    def test_model_metrics_sanity(self):
        """Verify model metrics and minimum accuracy threshold"""
        acc = float(self.metrics.loc[0, "accuracy"])
        print(f"🔹 Accuracy = {acc}")
        self.assertGreaterEqual(acc, 0.9, "Model accuracy below 0.9 threshold")

    def test_model_prediction_shape(self):
        """Check predictions on sample inputs"""
        sample = np.array([[5.1, 3.5, 1.4, 0.2],
                           [6.7, 3.0, 5.2, 2.3]])
        preds = self.model.predict(sample)
        print(f"🔹 Predictions: {preds}")
        self.assertEqual(preds.shape[0], 2, "Prediction output shape mismatch")

if __name__ == "__main__":
    unittest.main(verbosity=2)

