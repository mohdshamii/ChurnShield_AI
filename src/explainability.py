"""
ChurnShield AI - Explainable AI (XAI) & SHAP Engine
Provides local and global feature attribution for customer churn predictions:
- SHAP TreeExplainer & LinearExplainer integration
- Global feature importance & summary distribution
- Individual customer prediction waterfalls (positive & negative churn drivers)
- Counterfactual scenario simulator & actionable business explanations
"""

import logging
from typing import Dict, Any, List, Tuple, Optional
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
    'PaymentMethod_Mailed check': 'Mailed Paper Check',
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
            - top_factor & top_impact
        """
        feature_names = self.model_manager.feature_names
        aligned_df = input_df[feature_names].astype(float)

        raw_impacts: Dict[str, float] = {}

        # 1. SHAP TreeExplainer calculation
        if self.shap_explainer is not None:
            try:
                shap_values = self.shap_explainer.shap_values(aligned_df)
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
                for feat, coef, val in zip(feature_names, coefs, vals):
                    raw_impacts[feat] = float(coef * (val - 0.5))
            else:
                global_imp = self.model_manager.get_feature_importances()
                for feat in feature_names:
                    val = aligned_df[feat].iloc[0]
                    direction = 1.0 if any(x in feat for x in ['Month-to-month', 'Fiber', 'Electronic check', 'No']) else -1.0
                    raw_impacts[feat] = float(global_imp.get(feat, 0.0) * val * direction)

        # Consolidate, map to human labels, and group minor one-hot categories
        consolidated: Dict[str, float] = {}
        for feat, impact in raw_impacts.items():
            label = FEATURE_LABEL_MAP.get(feat, feat)
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
            "top_impact": impact_list[0]["impact"] if impact_list else 0.0,
            "disclaimer": "Note: SHAP values quantify statistical feature contributions to the model prediction. They identify risk correlates rather than strictly proven causal mechanisms."
        }

    def get_global_shap_summary(self, sample_size: int = 150) -> Dict[str, Any]:
        """
        Calculates global SHAP importance distributions across a representative sample of customers.
        """
        try:
            raw_df = self.model_manager.data_loader.load_raw_data()
            clean_df = self.model_manager.data_loader.clean_data(raw_df)
            X, _, feature_names = self.model_manager.data_loader.preprocess_for_training(raw_df)

            sample_X = X.sample(n=min(sample_size, len(X)), random_state=42)

            if self.shap_explainer is not None:
                shap_values = self.shap_explainer.shap_values(sample_X)
                if isinstance(shap_values, list) and len(shap_values) == 2:
                    vals = shap_values[1]
                elif hasattr(shap_values, 'shape') and len(shap_values.shape) == 3:
                    vals = shap_values[:, :, 1]
                else:
                    vals = np.array(shap_values)

                mean_abs = np.mean(np.abs(vals), axis=0)
            else:
                imp_dict = self.model_manager.get_feature_importances()
                mean_abs = np.array([imp_dict.get(f, 0.01) for f in feature_names])

            summary_items = []
            for feat, score in zip(feature_names, mean_abs):
                label = FEATURE_LABEL_MAP.get(feat, feat)
                summary_items.append({
                    "raw_feature": feat,
                    "label": label,
                    "importance": round(float(score), 4)
                })

            summary_items = sorted(summary_items, key=lambda x: x["importance"], reverse=True)[:15]

            return {
                "features": [item["label"] for item in summary_items],
                "scores": [item["importance"] for item in summary_items],
                "details": summary_items
            }
        except Exception as e:
            logger.error(f"Error generating global SHAP summary: {e}")
            global_imp = self.model_manager.get_feature_importances()
            top10 = list(global_imp.items())[:12]
            return {
                "features": [FEATURE_LABEL_MAP.get(k, k) for k, _ in top10],
                "scores": [v for _, v in top10],
                "details": []
            }

    def simulate_counterfactuals(
        self,
        customer_dict: Dict[str, Any],
        scenarios: Optional[List[Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Runs counterfactual what-if scenarios on a customer profile
        to quantify projected risk mitigation from strategic interventions.
        """
        baseline_df = self.model_manager.data_loader.prepare_single_input(customer_dict)
        baseline_prob = self.model_manager.predict_churn(baseline_df, model_type="ensemble")

        if scenarios is None:
            scenarios = [
                {
                    "title": "Upgrade to 1-Year Contract",
                    "description": "Transition customer from Month-to-Month to a 1-Year Commitment",
                    "changes": {"Contract": "One year"}
                },
                {
                    "title": "Add Online Security & Tech Support Bundle",
                    "description": "Provide free or discounted premium security & technical support",
                    "changes": {"OnlineSecurity": "Yes", "TechSupport": "Yes"}
                },
                {
                    "title": "Switch to Auto Credit Card / Bank Payment",
                    "description": "Convert payment method from manual electronic check to automatic credit card",
                    "changes": {"PaymentMethod": "Credit card (automatic)"}
                },
                {
                    "title": "Comprehensive VIP Retention Package",
                    "description": "2-Year Contract + Security & Tech Support + Auto Billing Discount",
                    "changes": {
                        "Contract": "Two year",
                        "OnlineSecurity": "Yes",
                        "TechSupport": "Yes",
                        "PaymentMethod": "Bank transfer (automatic)"
                    }
                }
            ]

        results = []
        for scen in scenarios:
            modified = dict(customer_dict)
            modified.update(scen["changes"])
            mod_df = self.model_manager.data_loader.prepare_single_input(modified)
            scen_prob = self.model_manager.predict_churn(mod_df, model_type="ensemble")
            risk_delta = round((baseline_prob - scen_prob) * 100, 1)

            results.append({
                "title": scen["title"],
                "description": scen["description"],
                "baseline_prob": round(baseline_prob * 100, 1),
                "scenario_prob": round(scen_prob * 100, 1),
                "risk_reduction_pct": max(risk_delta, 0.0),
                "risk_mitigated": scen_prob < baseline_prob,
                "changes": scen["changes"]
            })

        return results
