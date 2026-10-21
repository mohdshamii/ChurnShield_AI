"""
Tests for Data Loader and Feature Engineering Pipeline
"""

import unittest
import pandas as pd
import numpy as np
from src.data_loader import DataLoader

class TestDataLoader(unittest.TestCase):
    def setUp(self):
        self.loader = DataLoader()

    def test_clean_data_handles_spaces_in_total_charges(self):
        raw_df = pd.DataFrame({
            'customerID': ['001-A', '002-B'],
            'tenure': [0, 12],
            'MonthlyCharges': [50.0, 70.0],
            'TotalCharges': [' ', '840.0'],
            'SeniorCitizen': [0, 1]
        })
        cleaned = self.loader.clean_data(raw_df, drop_customer_id=True)
        self.assertNotIn('customerID', cleaned.columns)
        self.assertEqual(cleaned['TotalCharges'].iloc[0], 0.0)
        self.assertEqual(cleaned['TotalCharges'].iloc[1], 840.0)
        self.assertFalse(cleaned['TotalCharges'].isna().any())

    def test_engineered_features(self):
        df = pd.DataFrame({
            'tenure': [10, 2],
            'MonthlyCharges': [50.0, 90.0],
            'TotalCharges': [500.0, 180.0],
            'Contract': ['One year', 'Month-to-month'],
            'PaymentMethod': ['Bank transfer', 'Electronic check'],
            'OnlineSecurity': ['Yes', 'No'],
            'OnlineBackup': ['Yes', 'No'],
            'DeviceProtection': ['No', 'No'],
            'TechSupport': ['Yes', 'No']
        })
        eng = self.loader.add_engineered_features(df)
        self.assertIn('TenureToChargeRatio', eng.columns)
        self.assertIn('TotalValueScore', eng.columns)
        self.assertIn('ServiceDensity', eng.columns)
        self.assertIn('PaymentRisk', eng.columns)
        self.assertEqual(eng['PaymentRisk'].iloc[1], 1)
        self.assertEqual(eng['PaymentRisk'].iloc[0], 0)

    def test_prepare_single_input_schema(self):
        input_data = {
            'tenure': 6,
            'MonthlyCharges': 75.5,
            'Contract': 'Month-to-month',
            'InternetService': 'Fiber optic',
            'PaymentMethod': 'Electronic check'
        }
        df_row = self.loader.prepare_single_input(input_data)
        self.assertEqual(df_row.shape[0], 1)
        self.assertEqual(df_row.shape[1], len(self.loader.feature_names))
        self.assertEqual(list(df_row.columns), self.loader.feature_names)
        self.assertEqual(df_row['Contract_Month-to-month'].iloc[0], 1.0)
        self.assertEqual(df_row['InternetService_Fiber optic'].iloc[0], 1.0)

    def test_prepare_batch_df(self):
        raw = self.loader.load_raw_data()
        sample = raw.head(10).copy()
        X_batch, ids, feats = self.loader.prepare_batch_df(sample)
        self.assertEqual(X_batch.shape[0], 10)
        self.assertEqual(X_batch.shape[1], len(self.loader.feature_names))
        self.assertEqual(len(ids), 10)

if __name__ == '__main__':
    unittest.main()
