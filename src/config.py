"""
ChurnShield AI - Central Configuration Module
Defines project paths, model hyperparameters, feature definitions,
database configurations, and business rules.
"""

import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = os.path.join(BASE_DIR, "data.csv")
DB_PATH = os.path.join(BASE_DIR, "churn_prediction.db")
MODEL_DIR = os.path.join(BASE_DIR, "app", "model")

os.makedirs(MODEL_DIR, exist_ok=True)

# Saved Artifact Paths
XGB_MODEL_PATH = os.path.join(MODEL_DIR, "churn_model.json")
RF_MODEL_PATH = os.path.join(MODEL_DIR, "rf_model.joblib")
LR_MODEL_PATH = os.path.join(MODEL_DIR, "lr_model.joblib")
ENSEMBLE_PATH = os.path.join(MODEL_DIR, "ensemble_model.joblib")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.joblib")
FEATURE_NAMES_PATH = os.path.join(MODEL_DIR, "feature_names.json")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")

# Model Training Config
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

# Risk Thresholds
RISK_THRESHOLD_HIGH = 0.65
RISK_THRESHOLD_MEDIUM = 0.35

# Business / Financial ROI Defaults
AVERAGE_CAC = 350.0            # Customer Acquisition Cost ($)
AVERAGE_MONTHLY_DISCOUNT = 0.20 # 20% promotional discount
INTERVENTION_SUCCESS_RATE = 0.60 # Estimated success of targeted retention
