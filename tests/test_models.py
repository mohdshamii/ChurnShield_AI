"""
Tests for Machine Learning Engine and Predictor
"""

import unittest
import numpy as np
from src.models import ModelManager

class TestModelManager(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mm = ModelManager()

    def test_models_are_loaded(self):
        self.assertIsNotNone(self.mm.feature_names)
        self.assertGreater(len(self.mm.feature_names), 0)
        self.assertTrue(
            self.mm.ensemble_model is not None or
            self.mm.xgb_booster is not None or
            self.mm.rf_model is not None
        )

    def test_single_prediction_bounds(self):
        sample = {
            'tenure': 2,
            'MonthlyCharges': 95.0,
            'Contract': 'Month-to-month',
            'InternetService': 'Fiber optic',
            'PaymentMethod': 'Electronic check',
            'TechSupport': 'No',
            'OnlineSecurity': 'No'
        }
        df_row = self.mm.data_loader.prepare_single_input(sample)
        prob = self.mm.predict_churn(df_row, model_type="ensemble")
        self.assertIsInstance(prob, float)
        self.assertGreaterEqual(prob, 0.0)
        self.assertLessEqual(prob, 1.0)
        # High risk expected for Fiber + M2M + ECheck
        self.assertGreater(prob, 0.5)

    def test_batch_prediction(self):
        raw = self.mm.data_loader.load_raw_data()
        sample = raw.head(15).copy()
        X_batch, _, _ = self.mm.data_loader.prepare_batch_df(sample)
        probs = self.mm.batch_predict(X_batch, model_type="ensemble")
        self.assertEqual(len(probs), 15)
        for p in probs:
            self.assertGreaterEqual(p, 0.0)
            self.assertLessEqual(p, 1.0)

    def test_metrics_json_integrity(self):
        metrics = self.mm.metrics
        self.assertIsInstance(metrics, dict)
        for expected_model in ['Logistic Regression', 'Random Forest', 'XGBoost', 'Voting Ensemble']:
            self.assertIn(expected_model, metrics)
            m = metrics[expected_model]
            self.assertIn('accuracy', m)
            self.assertIn('roc_auc', m)
            self.assertIn('recall', m)
            self.assertIn('confusion_matrix', m)
            self.assertGreater(m['roc_auc'], 0.80)

if __name__ == '__main__':
    unittest.main()
