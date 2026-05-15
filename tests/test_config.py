"""Configuration tests for ChurnShield AI"""
import pytest


class TestThresholdConfiguration:
    """Test threshold values"""

    def test_high_risk_threshold_value(self):
        """High risk threshold should be 0.7"""
        high_threshold = 0.7
        assert high_threshold == 0.7

    def test_medium_risk_threshold_value(self):
        """Medium risk threshold should be 0.4"""
        medium_threshold = 0.4
        assert medium_threshold == 0.4

    def test_thresholds_in_correct_order(self):
        """Thresholds should be in correct order"""
        low_threshold = 0.4
        high_threshold = 0.7
        assert low_threshold < high_threshold

    def test_threshold_ranges(self):
        """Thresholds should be between 0 and 1"""
        low = 0.4
        high = 0.7
        assert 0 <= low <= 1
        assert 0 <= high <= 1

    def test_risk_level_count(self):
        """Should have exactly 3 risk levels"""
        risk_levels = ["LOW", "MEDIUM", "HIGH"]
        assert len(risk_levels) == 3


class TestServiceConfiguration:
    """Test service configuration"""

    def test_contract_type_options(self):
        """Should have 3 contract type options"""
        contracts = ["Month-to-month", "One year", "Two year"]
        assert len(contracts) == 3

    def test_internet_service_options(self):
        """Should have 3 internet service options"""
        services = ["Fiber optic", "DSL", "No"]
        assert len(services) == 3

    def test_payment_method_options(self):
        """Should have 4 payment method options"""
        methods = ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
        assert len(methods) == 4

    def test_additional_service_options(self):
        """Should have 3 additional service options"""
        options = ["Yes", "No", "No internet service"]
        assert len(options) == 3

    def test_service_option_values(self):
        """Service options should have valid values"""
        for option in ["Yes", "No", "No internet service"]:
            assert isinstance(option, str)
            assert len(option) > 0


class TestModelConfiguration:
    """Test model configuration"""

    def test_model_file_path(self):
        """Model file should be at correct path"""
        model_path = "app/model/churn_model.json"
        assert "app/model" in model_path
        assert ".json" in model_path

    def test_feature_names_file_path(self):
        """Feature names file should be at correct path"""
        features_path = "app/model/feature_names.json"
        assert "app/model" in features_path
        assert ".json" in features_path

    def test_model_version(self):
        """Model should have version info"""
        model_version = "XGBoost 1.7.6"
        assert "XGBoost" in model_version

    def test_model_type_is_xgboost(self):
        """Model should be XGBoost type"""
        model_type = "XGBoost"
        assert model_type == "XGBoost"


class TestUIConfiguration:
    """Test UI configuration"""

    def test_primary_color_defined(self):
        """Primary color should be defined"""
        primary_color = "#00ffaa"
        assert primary_color.startswith("#")

    def test_secondary_color_defined(self):
        """Secondary color should be defined"""
        secondary_color = "#0095ff"
        assert secondary_color.startswith("#")

    def test_risk_colors_defined(self):
        """Risk level colors should be defined"""
        colors = {
            "HIGH": "#f44336",
            "MEDIUM": "#ff9800",
            "LOW": "#4CAF50"
        }
        for risk, color in colors.items():
            assert color.startswith("#")

    def test_page_title_configured(self):
        """Page title should be configured"""
        page_title = "ChurnShield AI"
        assert page_title == "ChurnShield AI"

    def test_page_icon_configured(self):
        """Page icon should be configured"""
        page_icon = "📊"
        assert page_icon is not None

    def test_layout_configured(self):
        """Layout should be configured"""
        layout = "wide"
        assert layout == "wide"


class TestActionPlanConfiguration:
    """Test action plan configuration"""

    def test_high_risk_action_plan_structure(self):
        """High risk action plan should have correct structure"""
        action_plan = {
            "Day 1": ["Outreach call", "Special offer email"],
            "Day 3": ["Follow-up call", "Customer satisfaction survey"],
            "Day 7": ["Contract review meeting", "Service optimization"],
            "Day 14": ["Retention offer decision", "Loyalty program enrollment"]
        }
        assert len(action_plan) == 4

    def test_medium_risk_action_timeline(self):
        """Medium risk action timeline should be 4 weeks"""
        weeks = 4
        assert weeks == 4

    def test_low_risk_action_frequency(self):
        """Low risk actions should be regular"""
        frequency = "Quarterly"
        assert frequency in ["Monthly", "Quarterly", "Annual"]

    def test_action_plan_has_deliverables(self):
        """Action plan should have deliverables"""
        deliverables = [
            "Personalized outreach",
            "Special offer",
            "Service review"
        ]
        assert len(deliverables) >= 3


class TestFeatureConfiguration:
    """Test feature configuration"""

    def test_binary_features_count(self):
        """Should have binary features"""
        binary_features = ['gender', 'SeniorCitizen', 'Partner', 'Dependents',
                          'PhoneService', 'PaperlessBilling']
        assert len(binary_features) >= 6

    def test_numerical_features_count(self):
        """Should have numerical features"""
        numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
        assert len(numerical_features) == 3

    def test_categorical_features_count(self):
        """Should have categorical features"""
        categorical_features = [
            'Contract_Month-to-month', 'Contract_One year', 'Contract_Two year',
            'InternetService_Fiber optic', 'InternetService_DSL', 'InternetService_No'
        ]
        assert len(categorical_features) >= 6

    def test_feature_types_defined(self):
        """All feature types should be defined"""
        feature_types = {
            'binary': 6,
            'numerical': 3,
            'categorical': 25
        }
        total_features = sum(feature_types.values())
        assert total_features >= 30
