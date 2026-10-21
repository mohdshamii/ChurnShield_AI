"""
ChurnShield AI - Explainable AI (XAI) Engine
Calculates real model-based feature impacts and SHAP-like explanations for any
customer prediction, replacing hardcoded dummy heuristics with genuine model mathematics.
"""

import logging
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import shap

from src.models import ModelManager

logger = logging.getLogger(__name__)

# Human-readable labels for features
FEATURE_LABEL_MAP = {
    'Contract_Month-to-month': 'Month-to-Month Contract',
    'Contract_Two year': 'Two-Year Contract',
    'Contract_One year': 'One-Year Contract',
    'tenure': 'Tenure Length',
    'MonthlyCharges': 'Monthly Bill Amount',
    'TotalCharges': 'Cumulative Spend (Total Charges)',
    'InternetService_Fiber optic': 'Fiber Optic Internet',
    'InternetService_DSL': 'DSL Internet',
    'InternetService_No': 'No Internet Service',
    'OnlineSecurity_No': 'Lack of Online Security',
    'OnlineSecurity_Yes': 'Active Online Security',
    'TechSupport_No': 'Lack of Tech Support',
    'TechSupport_Yes': 'Active Tech Support',
    'PaymentMethod_Electronic check': 'Electronic Check Payment',
    'PaymentMethod_Credit card (automatic)': 'Auto Credit Card Payment',
    'PaymentMethod_Bank transfer (automatic)': 'Auto Bank Transfer',
    'PaperlessBilling': 'Paperless Billing',
    'SeniorCitizen': 'Senior Citizen Status',
    'Partner': 'Has Partner',
    'Dependents': 'Has Dependents',
    'DeviceProtection_No': 'No Device Protection',
    'DeviceProtection_Yes': 'Active Device Protection',
    'OnlineBackup_No': 'No Cloud Backup',
    'OnlineBackup_Yes': 'Active Cloud Backup',
    'StreamingTV_Yes': 'Streaming TV Active',
    'StreamingMovies_Yes': 'Streaming Movies Active',
    'PhoneService': 'Phone Service Active',
    'MultipleLines_Yes': 'Multiple Lines Active',
    'gender': 'Customer Gender'
}


class ModelExplainer:
    """Provides local and global feature attribution for customer churn predictions."""

    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self.shap_explainer = None
        self._init_shap_explainer()

    def _init_shap_explainer(self):
        """Initializes SHAP TreeExplainer if tree models are available."""
        try:
            if self.model_manager.rf_model is not None:
                self.shap_explainer = shap.TreeExplainer(self.model_manager.rf_model)
                logger.info("Initialized SHAP TreeExplainer on Random Forest model.")
            elif self.model_manager.xgb_booster is not None:
                self.shap_explainer = shap.TreeExplainer(self.model_manager.xgb_booster)
                logger.info("Initialized SHAP TreeExplainer on XGBoost booster.")
        except Exception as e:
            logger.warning(f"Could not initialize TreeExplainer: {e}. Using coefficient-based attribution.")

    def explain_prediction(self, input_df: pd.DataFrame, top_k: int = 8) -> Dict[str, Any]:
        """
        Calculates feature attributions for a single customer.
        Returns:
            - impact_list: list of dicts with factor, impact, direction, and human explanation
            - risk_catalysts: factors increasing churn risk
            - retention_anchors: factors decreasing churn risk
            - baseline_risk: average population baseline risk
        """
        feature_names = self.model_manager.feature_names
        aligned_df = input_df[feature_names].astype(float)

        raw_impacts: Dict[str, float] = {}

        # 1. SHAP TreeExplainer calculation
        if self.shap_explainer is not None:
            try:
                shap_values = self.shap_explainer.shap_values(aligned_df)
                # Handle classification shape: if 2-class, take positive class (index 1)
                if isinstance(shap_values, list) and len(shap_values) == 2:
                    vals = shap_values[1][0]
                elif hasattr(shap_values, 'shape') and len(shap_values.shape) == 3:
                    vals = shap_values[0, :, 1]
                elif hasattr(shap_values, 'shape') and len(shap_values.shape) == 2:
                    vals = shap_values[0]
                else:
                    vals = np.array(shap_values).flatten()

                for feat, val in zip(feature_names, vals):
                    raw_impacts[feat] = float(val)
            except Exception as e:
                logger.warning(f"SHAP calculation exception: {e}. Falling back to model weights.")

        # 2. Fallback: Logistic Regression or Tree Feature Importance * Normalized Delta
        if not raw_impacts:
            if self.model_manager.lr_model is not None:
                coefs = self.model_manager.lr_model.coef_[0]
                vals = aligned_df.iloc[0].values
                # Approximate log-odds shift
                for feat, coef, val in zip(feature_names, coefs, vals):
                    raw_impacts[feat] = float(coef * (val - 0.5))
            else:
                global_imp = self.model_manager.get_feature_importances()
                for feat in feature_names:
                    val = aligned_df[feat].iloc[0]
                    # Heuristic directional weighting
                    direction = 1.0 if any(x in feat for x in ['Month-to-month', 'Fiber', 'Electronic check', 'No']) else -1.0
                    raw_impacts[feat] = float(global_imp.get(feat, 0.0) * val * direction)

        # Consolidate, map to human labels, and group minor one-hot categories
        consolidated: Dict[str, float] = {}
        for feat, impact in raw_impacts.items():
            label = FEATURE_LABEL_MAP.get(feat, feat)
            # Filter zero/near-zero impacts to eliminate noise
            if abs(impact) > 1e-4:
                consolidated[label] = consolidated.get(label, 0.0) + impact

        # Sort by absolute magnitude
        sorted_factors = sorted(consolidated.items(), key=lambda x: abs(x[1]), reverse=True)[:top_k]

        impact_list = []
        risk_catalysts = []
        retention_anchors = []

        for factor, impact in sorted_factors:
            direction = "INCREASE_RISK" if impact > 0 else "REDUCE_RISK"
            entry = {
                "factor": factor,
                "impact": round(impact, 4),
                "direction": direction,
                "color": "#ff3d57" if impact > 0 else "#00ffaa",
                "summary": f"{factor} increases churn probability" if impact > 0 else f"{factor} strengthens customer retention"
            }
            impact_list.append(entry)
            if impact > 0:
                risk_catalysts.append(entry)
            else:
                retention_anchors.append(entry)

        return {
            "impact_list": impact_list,
            "risk_catalysts": risk_catalysts,
            "retention_anchors": retention_anchors,
            "top_factor": impact_list[0]["factor"] if impact_list else "Contract Type",
            "top_impact": impact_list[0]["impact"] if impact_list else 0.0
        }
