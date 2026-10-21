"""
Tests for Retention Strategy and ROI Financial Engine
"""

import unittest
from src.retention_strategy import RetentionStrategyEngine

class TestRetentionStrategy(unittest.TestCase):
    def test_assess_risk_thresholds(self):
        high = RetentionStrategyEngine.assess_risk(0.75)
        self.assertEqual(high['level'], 'HIGH')
        self.assertEqual(high['color'], '#ff3d57')

        med = RetentionStrategyEngine.assess_risk(0.50)
        self.assertEqual(med['level'], 'MEDIUM')
        self.assertEqual(med['color'], '#ffaa00')

        low = RetentionStrategyEngine.assess_risk(0.20)
        self.assertEqual(low['level'], 'LOW')
        self.assertEqual(low['color'], '#00ffaa')

    def test_calculate_clv_and_roi(self):
        res = RetentionStrategyEngine.calculate_clv_and_roi(
            monthly_charges=80.0,
            tenure_months=12.0,
            churn_prob=0.70,
            intervention_cost=60.0
        )
        self.assertEqual(res['annual_revenue'], 960.0)
        self.assertEqual(res['revenue_at_risk'], 672.0)
        self.assertGreater(res['gross_value_saved'], res['intervention_cost'])
        self.assertGreater(res['retention_roi_percent'], 0.0)

    def test_generate_recommendations(self):
        customer_data = {
            'MonthlyCharges': 95.0,
            'Contract': 'Month-to-month',
            'PaymentMethod': 'Electronic check',
            'TechSupport': 'No',
            'OnlineSecurity': 'No'
        }
        recs = RetentionStrategyEngine.generate_recommendations(
            customer_data,
            churn_prob=0.82,
            risk_catalysts=[{'factor': 'Month-to-Month Contract', 'impact': 0.12}]
        )
        self.assertEqual(recs['risk_meta']['level'], 'HIGH')
        self.assertGreaterEqual(len(recs['strategies']), 3)
        self.assertIn('email_template', recs)
        self.assertIn('sms_template', recs)
        self.assertGreaterEqual(len(recs['timeline']), 3)

if __name__ == '__main__':
    unittest.main()
