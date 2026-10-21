"""
Tests for Flask Web Dashboard, REST API Endpoints, and Persistence
"""

import json
import unittest
from flask_app import create_app

class TestFlaskApp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = create_app()
        cls.app.config['TESTING'] = True
        cls.client = cls.app.test_client()

    def test_dashboard_route_200(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Executive Intelligence Dashboard", response.data)
        self.assertIn(b"Analyzed Customers", response.data)

    def test_predict_view_get_200(self):
        response = self.client.get('/predict')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Real-Time Customer Predictor", response.data)

    def test_models_view_200(self):
        response = self.client.get('/models')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Production Model Benchmark Matrix", response.data)
        self.assertIn(b"Voting Ensemble", response.data)

    def test_retention_view_200(self):
        response = self.client.get('/retention')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Interactive Retention Campaign Financial Simulator", response.data)

    def test_history_view_200(self):
        response = self.client.get('/history')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Audit History", response.data)

    def test_download_sample_csv_200(self):
        response = self.client.get('/download-sample-csv')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, 'text/csv')

    def test_api_predict_json(self):
        payload = {
            'customer_id': 'TEST-USER-001',
            'tenure': 3,
            'MonthlyCharges': 89.0,
            'TotalCharges': 267.0,
            'Contract': 'Month-to-month',
            'InternetService': 'Fiber optic',
            'PaymentMethod': 'Electronic check',
            'TechSupport': 'No',
            'OnlineSecurity': 'No'
        }
        response = self.client.post(
            '/api/predict',
            data=json.dumps(payload),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'success')
        self.assertIn('churn_probability', data)
        self.assertIn('financial_clv_roi', data)
        self.assertIn('feature_impacts', data)
        self.assertGreater(data['churn_percentage'], 50.0)

    def test_api_what_if_simulator(self):
        payload = {
            'baseline': {
                'tenure': 2,
                'MonthlyCharges': 90.0,
                'Contract': 'Month-to-month',
                'TechSupport': 'No',
                'OnlineSecurity': 'No',
                'PaymentMethod': 'Electronic check'
            },
            'scenario': {
                'tenure': 2,
                'MonthlyCharges': 90.0,
                'Contract': 'Two year',
                'TechSupport': 'Yes',
                'OnlineSecurity': 'Yes',
                'PaymentMethod': 'Credit card (automatic)'
            }
        }
        response = self.client.post(
            '/api/what-if',
            data=json.dumps(payload),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertTrue(data['success'])
        self.assertLess(data['scenario_prob'], data['baseline_prob'])
        self.assertTrue(data['risk_mitigated'])
        self.assertGreater(data['percent_risk_reduction'], 30.0)

    def test_save_and_delete_prediction_flow(self):
        # 1. Save
        save_payload = {
            'customer_id': 'TEST-AUDIT-SAVE',
            'churn_prob': 74.5,
            'risk_level': 'HIGH',
            'notes': 'Automated unit test record',
            'raw_input': {'tenure': 1, 'MonthlyCharges': 80}
        }
        save_res = self.client.post(
            '/api/save-prediction',
            data=json.dumps(save_payload),
            content_type='application/json'
        )
        self.assertEqual(save_res.status_code, 200)
        save_data = json.loads(save_res.data)
        self.assertTrue(save_data['success'])
        pred_id = save_data['prediction_id']

        # 2. Delete
        del_res = self.client.post(f'/api/delete-prediction/{pred_id}')
        self.assertEqual(del_res.status_code, 200)
        del_data = json.loads(del_res.data)
        self.assertTrue(del_data['success'])

if __name__ == '__main__':
    unittest.main()
