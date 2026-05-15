"""Pytest configuration and fixtures for ChurnShield AI tests"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


# Pytest markers for test organization
def pytest_configure(config):
    config.addinivalue_line("markers", "unit: unit tests")
    config.addinivalue_line("markers", "integration: integration tests")
    config.addinivalue_line("markers", "config: configuration tests")
    config.addinivalue_line("markers", "slow: slow running tests")


@pytest.fixture
def sample_customer_data():
    """Sample customer data for testing"""
    return {
        'tenure': 12,
        'MonthlyCharges': 70,
        'TotalCharges': 840,
        'gender': 1,
        'SeniorCitizen': 0,
        'Partner': 1,
        'Dependents': 0,
        'PhoneService': 1,
        'PaperlessBilling': 1,
        'Contract_Month-to-month': 1,
        'Contract_One year': 0,
        'Contract_Two year': 0,
        'InternetService_Fiber optic': 1,
        'InternetService_DSL': 0,
        'InternetService_No': 0,
        'OnlineSecurity_Yes': 0,
        'OnlineSecurity_No internet service': 0,
        'OnlineBackup_Yes': 0,
        'OnlineBackup_No internet service': 0,
        'DeviceProtection_Yes': 0,
        'DeviceProtection_No internet service': 0,
        'TechSupport_Yes': 0,
        'TechSupport_No internet service': 0,
        'StreamingTV_Yes': 0,
        'StreamingTV_No internet service': 0,
        'StreamingMovies_Yes': 0,
        'StreamingMovies_No internet service': 0,
        'PaymentMethod_Electronic check': 0,
        'PaymentMethod_Mailed check': 0,
        'PaymentMethod_Bank transfer (automatic)': 1,
        'PaymentMethod_Credit card (automatic)': 0,
    }


@pytest.fixture
def high_risk_customer():
    """High risk customer profile"""
    return {
        'tenure': 2,
        'MonthlyCharges': 100,
        'TotalCharges': 200,
        'gender': 0,
        'SeniorCitizen': 0,
        'Partner': 0,
        'Dependents': 0,
        'PhoneService': 0,
        'PaperlessBilling': 0,
        'Contract_Month-to-month': 1,
        'Contract_One year': 0,
        'Contract_Two year': 0,
        'InternetService_Fiber optic': 1,
        'InternetService_DSL': 0,
        'InternetService_No': 0,
        'OnlineSecurity_Yes': 0,
        'OnlineSecurity_No internet service': 1,
        'OnlineBackup_Yes': 0,
        'OnlineBackup_No internet service': 1,
        'DeviceProtection_Yes': 0,
        'DeviceProtection_No internet service': 1,
        'TechSupport_Yes': 0,
        'TechSupport_No internet service': 1,
        'StreamingTV_Yes': 0,
        'StreamingTV_No internet service': 0,
        'StreamingMovies_Yes': 0,
        'StreamingMovies_No internet service': 0,
        'PaymentMethod_Electronic check': 1,
        'PaymentMethod_Mailed check': 0,
        'PaymentMethod_Bank transfer (automatic)': 0,
        'PaymentMethod_Credit card (automatic)': 0,
    }


@pytest.fixture
def low_risk_customer():
    """Low risk customer profile"""
    return {
        'tenure': 60,
        'MonthlyCharges': 50,
        'TotalCharges': 3000,
        'gender': 1,
        'SeniorCitizen': 0,
        'Partner': 1,
        'Dependents': 1,
        'PhoneService': 1,
        'PaperlessBilling': 1,
        'Contract_Month-to-month': 0,
        'Contract_One year': 0,
        'Contract_Two year': 1,
        'InternetService_Fiber optic': 0,
        'InternetService_DSL': 1,
        'InternetService_No': 0,
        'OnlineSecurity_Yes': 1,
        'OnlineSecurity_No internet service': 0,
        'OnlineBackup_Yes': 1,
        'OnlineBackup_No internet service': 0,
        'DeviceProtection_Yes': 1,
        'DeviceProtection_No internet service': 0,
        'TechSupport_Yes': 1,
        'TechSupport_No internet service': 0,
        'StreamingTV_Yes': 1,
        'StreamingTV_No internet service': 0,
        'StreamingMovies_Yes': 1,
        'StreamingMovies_No internet service': 0,
        'PaymentMethod_Electronic check': 0,
        'PaymentMethod_Mailed check': 0,
        'PaymentMethod_Bank transfer (automatic)': 1,
        'PaymentMethod_Credit card (automatic)': 0,
    }


@pytest.fixture
def risk_thresholds():
    """Risk assessment thresholds"""
    return {
        'high': 0.7,
        'medium': 0.4,
        'low': 0.4
    }


@pytest.fixture
def contract_types():
    """Valid contract types"""
    return ["Month-to-month", "One year", "Two year"]


@pytest.fixture
def internet_services():
    """Valid internet service types"""
    return ["Fiber optic", "DSL", "No"]


@pytest.fixture
def payment_methods():
    """Valid payment methods"""
    return ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]


@pytest.fixture
def service_options():
    """Valid service options"""
    return ["Yes", "No", "No internet service"]
