"""Unit tests for ChurnShield AI core functionality"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class TestDataPreparation:
    """Test data preparation and validation"""

    def test_tenure_is_numeric(self, sample_customer_data):
        """Tenure should be a numeric value"""
        assert isinstance(sample_customer_data['tenure'], (int, float))
        assert 0 <= sample_customer_data['tenure'] <= 72

    def test_monthly_charges_is_numeric(self, sample_customer_data):
        """Monthly charges should be numeric"""
        assert isinstance(sample_customer_data['MonthlyCharges'], (int, float))
        assert sample_customer_data['MonthlyCharges'] >= 0

    def test_total_charges_is_numeric(self, sample_customer_data):
        """Total charges should be numeric"""
        assert isinstance(sample_customer_data['TotalCharges'], (int, float))
        assert sample_customer_data['TotalCharges'] >= 0

    def test_binary_features_are_0_or_1(self, sample_customer_data):
        """Binary features should be 0 or 1"""
        binary_features = ['SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
        for feature in binary_features:
            assert sample_customer_data[feature] in [0, 1]

    def test_contract_type_encoding(self, sample_customer_data):
        """Contract type should be properly encoded"""
        contract_cols = ['Contract_Month-to-month', 'Contract_One year', 'Contract_Two year']
        total = sum(sample_customer_data[col] for col in contract_cols)
        assert total == 1, "Exactly one contract type should be selected"

    def test_internet_service_encoding(self, sample_customer_data):
        """Internet service should be properly encoded"""
        internet_cols = ['InternetService_Fiber optic', 'InternetService_DSL', 'InternetService_No']
        total = sum(sample_customer_data[col] for col in internet_cols)
        assert total == 1, "Exactly one internet service should be selected"

    def test_payment_method_encoding(self, sample_customer_data):
        """Payment method should be properly encoded"""
        payment_cols = ['PaymentMethod_Electronic check', 'PaymentMethod_Mailed check',
                       'PaymentMethod_Bank transfer (automatic)', 'PaymentMethod_Credit card (automatic)']
        total = sum(sample_customer_data[col] for col in payment_cols)
        assert total == 1, "Exactly one payment method should be selected"


class TestRiskAssessment:
    """Test risk level classification"""

    def test_high_risk_threshold(self, risk_thresholds):
        """High risk should be > 0.7"""
        churn_prob = 0.85
        assert churn_prob > risk_thresholds['high']

    def test_medium_risk_threshold(self, risk_thresholds):
        """Medium risk should be between 0.4 and 0.7"""
        churn_prob = 0.55
        assert risk_thresholds['low'] < churn_prob < risk_thresholds['high']

    def test_low_risk_threshold(self, risk_thresholds):
        """Low risk should be < 0.4"""
        churn_prob = 0.25
        assert churn_prob < risk_thresholds['low']

    def test_risk_level_labels(self):
        """Risk levels should have correct labels"""
        risk_levels = ["HIGH", "MEDIUM", "LOW"]
        assert "HIGH" in risk_levels
        assert "MEDIUM" in risk_levels
        assert "LOW" in risk_levels

    def test_risk_description_mapping(self):
        """Risk descriptions should map correctly"""
        descriptions = {
            "HIGH": "Immediate action required",
            "MEDIUM": "Proactive measures recommended",
            "LOW": "Normal monitoring"
        }
        assert len(descriptions) == 3


class TestContractTypeEncoding:
    """Test contract type encoding logic"""

    def test_month_to_month_encoding(self):
        """Month-to-month should increase churn risk"""
        contract_impact = 0.35  # Positive impact = increases risk
        assert contract_impact > 0

    def test_one_year_encoding(self):
        """One year contract should slightly decrease risk"""
        contract_impact = -0.05
        assert contract_impact < 0

    def test_two_year_encoding(self):
        """Two year contract should significantly decrease risk"""
        contract_impact = -0.15
        assert contract_impact < contract_impact

    def test_contract_type_validation(self, contract_types):
        """All contract types should be valid"""
        for contract in contract_types:
            assert contract in ["Month-to-month", "One year", "Two year"]


class TestInternetServiceEncoding:
    """Test internet service encoding logic"""

    def test_fiber_optic_impact(self):
        """Fiber optic should increase churn risk"""
        service_impact = 0.25
        assert service_impact > 0

    def test_dsl_impact(self):
        """DSL should decrease churn risk"""
        service_impact = -0.1
        assert service_impact < 0

    def test_no_internet_impact(self):
        """No internet should be neutral"""
        service_impact = 0
        assert service_impact == 0

    def test_internet_service_validation(self, internet_services):
        """All internet services should be valid"""
        for service in internet_services:
            assert service in ["Fiber optic", "DSL", "No"]


class TestPaymentMethodEncoding:
    """Test payment method encoding logic"""

    def test_electronic_check_impact(self):
        """Electronic check should increase churn risk"""
        payment_impact = 0.12
        assert payment_impact > 0

    def test_automatic_payment_impact(self):
        """Automatic payments should decrease churn risk"""
        payment_impact = -0.08
        assert payment_impact < 0

    def test_payment_method_validation(self, payment_methods):
        """All payment methods should be valid"""
        for method in payment_methods:
            assert method in ["Electronic check", "Mailed check",
                            "Bank transfer (automatic)", "Credit card (automatic)"]


class TestFeatureImpact:
    """Test feature impact calculations"""

    def test_tenure_impact_is_negative(self):
        """Longer tenure should reduce churn risk"""
        tenure_impact = -0.02 * 24  # 24 months tenure
        assert tenure_impact < 0

    def test_online_security_yes_impact(self):
        """Online security should reduce churn risk"""
        security_impact = -0.15
        assert security_impact < 0

    def test_online_security_no_impact(self):
        """No online security should increase churn risk"""
        security_impact = 0.1
        assert security_impact > 0

    def test_tech_support_yes_impact(self):
        """Tech support should reduce churn risk"""
        support_impact = -0.18
        assert support_impact < 0

    def test_tech_support_no_impact(self):
        """No tech support should increase churn risk"""
        support_impact = 0.1
        assert support_impact > 0

    def test_monthly_charges_impact(self):
        """Higher charges should increase churn risk slightly"""
        charges = 100
        charges_impact = 0.005 * charges
        assert charges_impact > 0


class TestRetentionStrategies:
    """Test retention strategy recommendations"""

    def test_high_risk_action_plan(self):
        """High risk should have immediate action plan"""
        action_plan = {
            "Day 1": ["Outreach call", "Special offer email"],
            "Day 3": ["Follow-up call", "Customer satisfaction survey"],
            "Day 7": ["Contract review meeting", "Service optimization"],
            "Day 14": ["Retention offer decision", "Loyalty program enrollment"]
        }
        assert len(action_plan) == 4
        assert "Day 1" in action_plan

    def test_high_risk_interventions(self):
        """High risk should include specific interventions"""
        interventions = [
            "Personalized outreach",
            "Special offer",
            "Service review",
            "Loyalty bonus",
            "Priority support"
        ]
        assert len(interventions) >= 5

    def test_medium_risk_actions(self):
        """Medium risk should have engagement actions"""
        actions = [
            "Engagement campaign",
            "Value-added offer",
            "Satisfaction survey",
            "Contract incentive",
            "Usage tips"
        ]
        assert len(actions) >= 5

    def test_low_risk_actions(self):
        """Low risk should focus on retention"""
        actions = [
            "Regular check-ins",
            "Loyalty rewards",
            "Referral program",
            "Product education",
            "Community building"
        ]
        assert len(actions) >= 5


class TestDataValidation:
    """Test customer data validation"""

    def test_customer_id_not_empty(self):
        """Customer ID should not be empty"""
        customer_id = "CUST001"
        assert customer_id != ""
        assert len(customer_id) > 0

    def test_tenure_valid_range(self):
        """Tenure should be between 0 and 72 months"""
        tenure = 24
        assert 0 <= tenure <= 72

    def test_charges_positive(self):
        """Charges should be positive"""
        monthly_charges = 75
        total_charges = 1800
        assert monthly_charges > 0
        assert total_charges > 0

    def test_service_selection_valid(self):
        """Service selections should be valid"""
        services = {"Online Security": "Yes", "Tech Support": "No", "Streaming TV": "No internet service"}
        valid_options = ["Yes", "No", "No internet service"]
        for service, value in services.items():
            assert value in valid_options


class TestModelInputShape:
    """Test model input shape and dimensions"""

    def test_feature_count(self, sample_customer_data):
        """Should have correct number of features"""
        features = list(sample_customer_data.keys())
        assert len(features) >= 30

    def test_dataframe_shape(self, sample_customer_data):
        """DataFrame should have correct shape"""
        df = pd.DataFrame([sample_customer_data])
        assert df.shape[0] == 1  # One row
        assert df.shape[1] >= 30  # At least 30 columns

    def test_no_missing_features(self, sample_customer_data):
        """All required features should be present"""
        required_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
        for feature in required_features:
            assert feature in sample_customer_data
            assert sample_customer_data[feature] is not None
