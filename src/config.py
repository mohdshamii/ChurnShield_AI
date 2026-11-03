"""
ChurnShield AI - Central Configuration Module
Defines project paths, model hyperparameters, feature definitions,
database configurations, business rules, and monitoring thresholds.
"""

import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = os.path.join(BASE_DIR, "data.csv")
DB_PATH = os.path.join(BASE_DIR, "churn_prediction.db")
MODEL_DIR = os.path.join(BASE_DIR, "app", "model")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# Saved Artifact Paths
XGB_MODEL_PATH = os.path.join(MODEL_DIR, "churn_model.json")
RF_MODEL_PATH = os.path.join(MODEL_DIR, "rf_model.joblib")
LR_MODEL_PATH = os.path.join(MODEL_DIR, "lr_model.joblib")
HGB_MODEL_PATH = os.path.join(MODEL_DIR, "hgb_model.joblib")
ENSEMBLE_PATH = os.path.join(MODEL_DIR, "ensemble_model.joblib")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.joblib")
FEATURE_NAMES_PATH = os.path.join(MODEL_DIR, "feature_names.json")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")
EXPERIMENTS_PATH = os.path.join(MODEL_DIR, "experiments.json")
DRIFT_BASELINE_PATH = os.path.join(MODEL_DIR, "drift_baseline.json")

# Model Training Config
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

# Risk Thresholds
RISK_THRESHOLD_HIGH = 0.65
RISK_THRESHOLD_MEDIUM = 0.35

# Business / Financial ROI Defaults
AVERAGE_CAC = 350.0              # Customer Acquisition Cost ($)
AVERAGE_MONTHLY_DISCOUNT = 0.20   # 20% promotional discount
INTERVENTION_SUCCESS_RATE = 0.60  # Estimated success of targeted retention
DEFAULT_DISCOUNT_PERIOD_MONTHS = 6

# Model Monitoring & Drift Thresholds
PSI_WARNING_THRESHOLD = 0.10     # Slight drift detected
PSI_CRITICAL_THRESHOLD = 0.25    # Significant drift, retraining recommended
KS_PVALUE_THRESHOLD = 0.05       # Significant difference threshold

# API & Security
SECRET_KEY = os.environ.get("SECRET_KEY", "churnshield-ai-production-secret-key-2026")
MAX_UPLOAD_SIZE_MB = 32
ALLOWED_UPLOAD_EXTENSIONS = {".csv", ".xlsx", ".xls"}
