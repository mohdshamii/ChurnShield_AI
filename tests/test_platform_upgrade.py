"""
Unit and Integration Tests for ChurnShield AI Enterprise Platform Upgrade
Tests all new dashboard routes, services, REST APIs, monitoring, and explainability.
"""

import json
import unittest
import pandas as pd
from flask_app import create_app
from src.analytics import AnalyticsEngine
from src.monitoring import ModelMonitor
from src.profiler import DatasetProfiler
from src.reporting import ReportGenerator
from src.database import DatabaseManager
from src.data_loader import DataLoader
from src.models import ModelManager
from src.explainability import ModelExplainer


class TestEnterprisePlatformUpgrade(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = create_app()
        cls.app.config['TESTING'] = True
        cls.client = cls.app.test_client()
        cls.data_loader = DataLoader()
        cls.model_manager = ModelManager()
        cls.explainer = ModelExplainer(cls.model_manager)
        cls.db = DatabaseManager()
        cls.analytics = AnalyticsEngine()
        cls.monitor = ModelMonitor(cls.data_loader, cls.db)

    # ---------------------------------------------------------
    # 1. New Route Status Code & Content Verification
    # ---------------------------------------------------------
    def test_analytics_route_200(self):
        res = self.client.get('/analytics')
        self.assertEqual(res.status_code, 200)
        self.assertIn(b"Customer Risk Analytics", res.data)
        self.assertIn(b"K-Means Strategic Customer Segmentation", res.data)

    def test_customer_360_route_200(self):
        res = self.client.get('/customer/CUST-7043')
        self.assertEqual(res.status_code, 200)
        self.assertIn(b"Customer 360 Intelligence", res.data)
        self.assertIn(b"Local SHAP Feature Attribution", res.data)

    def test_explainability_route_200(self):
        res = self.client.get('/explainability')
        self.assertEqual(res.status_code, 200)
        self.assertIn(b"Explainable AI", res.data)
        self.assertIn(b"Global Feature Importance", res.data)

    def test_revenue_route_200(self):
        res = self.client.get('/revenue')
        self.assertEqual(res.status_code, 200)
        self.assertIn(b"Revenue Intelligence", res.data)
        self.assertIn(b"Interactive Campaign Retention ROI Simulator", res.data)

    def test_automl_route_200(self):
        res = self.client.get('/automl')
        self.assertEqual(res.status_code, 200)
        self.assertIn(b"AutoML Experiment Studio", res.data)

    def test_monitoring_route_200(self):
        res = self.client.get('/monitoring')
        self.assertEqual(res.status_code, 200)
        self.assertIn(b"Model Monitoring", res.data)
        self.assertIn(b"Population Stability Index", res.data)

    def test_data_intelligence_route_200(self):
        res = self.client.get('/data-intelligence')
        self.assertEqual(res.status_code, 200)
        self.assertIn(b"Data Intelligence", res.data)
        self.assertIn(b"Data Quality Score", res.data)

    def test_api_docs_route_200(self):
        res = self.client.get('/api-docs')
        self.assertEqual(res.status_code, 200)
        self.assertIn(b"REST API Specification", res.data)

    def test_download_report_200(self):
        res = self.client.get('/download-report')
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.mimetype, 'text/html')
        self.assertIn(b"RETENTION INTELLIGENCE AUDIT", res.data)

    # ---------------------------------------------------------
    # 2. REST API Endpoints
    # ---------------------------------------------------------
    def test_api_models_endpoint(self):
        res = self.client.get('/api/models')
        self.assertEqual(res.status_code, 200)
        data = json.loads(res.data)
        self.assertEqual(data["status"], "success")
        self.assertIn("models", data)

    def test_api_drift_endpoint(self):
        res = self.client.get('/api/monitoring/drift')
        self.assertEqual(res.status_code, 200)
        data = json.loads(res.data)
        self.assertIn("overall_health_score", data)
        self.assertIn("feature_drift", data)

    def test_create_campaign_post_flow(self):
        payload = {
            "campaign_name": "Test Integration Campaign",
            "target_risk": "HIGH",
            "offer_name": "Test 15% Incentive",
            "discount_pct": "15",
            "budget": "4000",
            "customers_targeted": "80"
        }
        res = self.client.post('/retention', data=payload, follow_redirects=True)
        self.assertEqual(res.status_code, 200)
        self.assertIn(b"Test Integration Campaign", res.data)

    # ---------------------------------------------------------
    # 3. Core Engine Upgrades & Services
    # ---------------------------------------------------------
    def test_analytics_segmentation(self):
        raw_df = self.data_loader.load_raw_data()
        clean_df = self.data_loader.clean_data(raw_df)
        result = self.analytics.segment_customers(clean_df, n_clusters=4)
        self.assertEqual(result["cluster_count"], 4)
        self.assertGreater(len(result["profiles"]), 0)

    def test_analytics_survival_curve(self):
        raw_df = self.data_loader.load_raw_data()
        clean_df = self.data_loader.clean_data(raw_df)
        surv = self.analytics.estimate_survival_curve(clean_df)
        self.assertIn("months", surv)
        self.assertIn("survival_rates", surv)
        self.assertEqual(surv["survival_rates"][0], 100.0)

    def test_analytics_revenue_forecast(self):
        raw_df = self.data_loader.load_raw_data()
        clean_df = self.data_loader.clean_data(raw_df)
        fc = self.analytics.forecast_churn_and_revenue(clean_df)
        self.assertIn("forecast_30_days", fc)
        self.assertIn("forecast_60_days", fc)
        self.assertIn("forecast_90_days", fc)

    def test_monitoring_psi_calculation(self):
        drift = self.monitor.compute_data_drift()
        self.assertIn("overall_health_score", drift)
        self.assertIn("feature_drift", drift)
        self.assertGreaterEqual(drift["overall_health_score"], 0)
        self.assertLessEqual(drift["overall_health_score"], 100)

    def test_dataset_profiler(self):
        raw_df = self.data_loader.load_raw_data()
        profile = DatasetProfiler.profile_dataframe(raw_df.head(100))
        self.assertEqual(profile["total_rows"], 100)
        self.assertIn("quality_score", profile)
        self.assertIn("numeric_stats", profile)
        self.assertIn("recommendations", profile)

    def test_threshold_optimizer(self):
        sim = self.model_manager.optimize_threshold()
        self.assertIn("optimal_f1_threshold", sim)
        self.assertIn("optimal_cost_threshold", sim)
        self.assertIn("curve_points", sim)

    def test_counterfactual_simulator(self):
        cust = {"tenure": 3, "MonthlyCharges": 95.0, "Contract": "Month-to-month", "TechSupport": "No"}
        cf = self.explainer.simulate_counterfactuals(cust)
        self.assertGreaterEqual(len(cf), 3)
        self.assertIn("risk_reduction_pct", cf[0])


if __name__ == '__main__':
    unittest.main()

