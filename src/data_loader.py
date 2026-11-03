"""
ChurnShield AI - Data Loader & Customer Intelligence Feature Engineering Engine
Handles data ingestion, validation, cleaning, dynamic schema detection,
advanced feature engineering, and transformation for model training and real-time inference.
"""

import json
import logging
import os
from typing import Tuple, Dict, Any, List, Optional
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src import config

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# Canonical categorical and binary columns in standard Telco schema
BINARY_COLS = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
MULTI_COLS = [
    'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
    'Contract', 'PaymentMethod'
]
NUMERIC_COLS = ['tenure', 'MonthlyCharges', 'TotalCharges']

# Canonical 40-feature schema from app/model/feature_names.json for exact model compatibility
BASE_FEATURE_NAMES = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService',
    'PaperlessBilling', 'MonthlyCharges', 'TotalCharges', 'MultipleLines_No',
    'MultipleLines_No phone service', 'MultipleLines_Yes', 'InternetService_DSL',
    'InternetService_Fiber optic', 'InternetService_No', 'OnlineSecurity_No',
    'OnlineSecurity_No internet service', 'OnlineSecurity_Yes', 'OnlineBackup_No',
    'OnlineBackup_No internet service', 'OnlineBackup_Yes', 'DeviceProtection_No',
    'DeviceProtection_No internet service', 'DeviceProtection_Yes', 'TechSupport_No',
    'TechSupport_No internet service', 'TechSupport_Yes', 'StreamingTV_No',
    'StreamingTV_No internet service', 'StreamingTV_Yes', 'StreamingMovies_No',
    'StreamingMovies_No internet service', 'StreamingMovies_Yes', 'Contract_Month-to-month',
    'Contract_One year', 'Contract_Two year', 'PaymentMethod_Bank transfer (automatic)',
    'PaymentMethod_Credit card (automatic)', 'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check'
]


class DataLoader:
    """Production data loading, cleaning, dynamic schema detection, and preprocessing pipeline."""

    def __init__(self, data_path: Optional[str] = None):
        self.data_path = data_path or config.DATA_PATH
        self.feature_names = self._load_or_default_features()
        self.scaler: Optional[StandardScaler] = None

    def _load_or_default_features(self) -> List[str]:
        if os.path.exists(config.FEATURE_NAMES_PATH):
            try:
                with open(config.FEATURE_NAMES_PATH, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to read {config.FEATURE_NAMES_PATH}: {e}. Falling back to default.")
        return BASE_FEATURE_NAMES.copy()

    def load_raw_data(self) -> pd.DataFrame:
        """Load raw CSV dataset from data_path."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Dataset file not found at: {self.data_path}")
        df = pd.read_csv(self.data_path)
        logger.info(f"Loaded raw data from {self.data_path}: {df.shape[0]} rows, {df.shape[1]} columns.")
        return df

    @staticmethod
    def detect_schema_categories(df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Dynamically detects present customer intelligence feature categories.
        Never fabricates features; inspects actual dataset columns.
        """
        cols = set(df.columns)
        categories: Dict[str, List[str]] = {
            "demographics": [c for c in cols if c.lower() in [
                'gender', 'seniorcitizen', 'partner', 'dependents', 'age', 'location', 'country', 'city', 'state'
            ]],
            "account": [c for c in cols if c.lower() in [
                'tenure', 'contract', 'customerid', 'registrationdate', 'signupdate', 'subscriptionduration'
            ]],
            "financial": [c for c in cols if c.lower() in [
                'monthlycharges', 'totalcharges', 'paymentmethod', 'paperlessbilling', 'arpu',
                'clv', 'paymentfailures', 'outstandingbalance', 'billingcycle'
            ]],
            "services": [c for c in cols if c.lower() in [
                'phoneservice', 'multiplelines', 'internetservice', 'onlinesecurity', 'onlinebackup',
                'deviceprotection', 'techsupport', 'streamingtv', 'streamingmovies'
            ]],
            "behavioral": [c for c in cols if c.lower() in [
                'loginfrequency', 'sessionduration', 'inactivitydays', 'engagementscore',
                'servicedowngrades', 'productadoption', 'usagetrend'
            ]],
            "support": [c for c in cols if c.lower() in [
                'complaintcount', 'supporttickets', 'avgresolutiontime', 'escalationfrequency', 'csat'
            ]],
            "engineered": [c for c in cols if c in [
                'TenureToChargeRatio', 'TotalValueScore', 'ServiceDensity', 'PaymentRisk',
                'HighCostLongTenure', 'ActiveServiceCount'
            ]]
        }
        return categories

    @staticmethod
    def clean_data(df: pd.DataFrame, drop_customer_id: bool = True) -> pd.DataFrame:
        """
        Cleans data types, treats whitespace strings in TotalCharges,
        and safely imputes missing values.
        """
        df_clean = df.copy()

        # TotalCharges has empty spaces ' ' for new customers where tenure == 0
        if 'TotalCharges' in df_clean.columns:
            df_clean['TotalCharges'] = pd.to_numeric(
                df_clean['TotalCharges'].astype(str).str.strip(), errors='coerce'
            )
            # For customers with tenure == 0, TotalCharges is legitimately 0
            if 'tenure' in df_clean.columns:
                df_clean.loc[df_clean['TotalCharges'].isna() & (df_clean['tenure'] == 0), 'TotalCharges'] = 0.0
                df_clean['TotalCharges'] = df_clean['TotalCharges'].fillna(
                    df_clean['MonthlyCharges'] * df_clean['tenure'] if 'MonthlyCharges' in df_clean.columns else 0.0
                )
            else:
                df_clean['TotalCharges'] = df_clean['TotalCharges'].fillna(0.0)

        # SeniorCitizen to int
        if 'SeniorCitizen' in df_clean.columns:
            df_clean['SeniorCitizen'] = df_clean['SeniorCitizen'].fillna(0).astype(int)

        if drop_customer_id and 'customerID' in df_clean.columns:
            df_clean = df_clean.drop(columns=['customerID'])

        return df_clean

    @staticmethod
    def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds high-signal domain features for customer retention & churn modeling.
        Computes features only when prerequisite columns exist.
        """
        df_eng = df.copy()

        # 1. Tenure to monthly charge ratio
        if 'tenure' in df_eng.columns and 'MonthlyCharges' in df_eng.columns:
            df_eng['TenureToChargeRatio'] = df_eng['tenure'] / (df_eng['MonthlyCharges'] + 1e-6)

        # 2. Total Value Realization Score
        if 'tenure' in df_eng.columns and 'MonthlyCharges' in df_eng.columns and 'TotalCharges' in df_eng.columns:
            df_eng['TotalValueScore'] = (df_eng['tenure'] * df_eng['MonthlyCharges']) / (df_eng['TotalCharges'] + 1e-6)

        # 3. Service Density (Count of active security/tech/backup subscriptions)
        service_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport']
        active_counts = []
        for _, row in df_eng.iterrows():
            cnt = sum(1 for col in service_cols if col in row and str(row[col]).strip().lower() == 'yes')
            active_counts.append(cnt)
        df_eng['ActiveServiceCount'] = active_counts

        if 'tenure' in df_eng.columns:
            df_eng['ServiceDensity'] = df_eng['ActiveServiceCount'] / (df_eng['tenure'] + 1.0)

        # 4. Payment Risk (Month-to-month + Electronic check)
        is_m2m = df_eng['Contract'].astype(str).str.contains('Month-to-month', case=False, na=False) if 'Contract' in df_eng.columns else pd.Series(False, index=df_eng.index)
        is_echeck = df_eng['PaymentMethod'].astype(str).str.contains('Electronic check', case=False, na=False) if 'PaymentMethod' in df_eng.columns else pd.Series(False, index=df_eng.index)
        df_eng['PaymentRisk'] = (is_m2m & is_echeck).astype(int)

        # 5. High Cost Long Tenure Segment Indicator
        if 'MonthlyCharges' in df_eng.columns and 'tenure' in df_eng.columns:
            m_p75 = df_eng['MonthlyCharges'].quantile(0.75) if len(df_eng) > 1 else 70.0
            t_med = df_eng['tenure'].median() if len(df_eng) > 1 else 24.0
            df_eng['HighCostLongTenure'] = (
                (df_eng['MonthlyCharges'] > m_p75) & (df_eng['tenure'] > t_med)
            ).astype(int)

        # 6. Behavioral indicators if support columns exist
        if 'SupportTickets' in df_eng.columns and 'tenure' in df_eng.columns:
            df_eng['TicketRate'] = df_eng['SupportTickets'] / (df_eng['tenure'] + 1.0)

        return df_eng

    def validate_dataset(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Performs thorough validation and data quality checks on an arbitrary dataframe.
        Returns a structured validation and quality report.
        """
        issues = []
        warnings = []
        n_rows, n_cols = df.shape

        if n_rows == 0:
            return {"valid": False, "score": 0, "issues": ["Dataset is completely empty."]}

        # Check for missing values
        null_counts = df.isnull().sum()
        cols_with_nulls = null_counts[null_counts > 0].to_dict()
        total_missing = int(null_counts.sum())
        missing_pct = round((total_missing / (n_rows * n_cols)) * 100, 2)

        if missing_pct > 15:
            issues.append(f"High missingness rate: {missing_pct}% of total cells are null.")
        elif missing_pct > 0:
            warnings.append(f"Missing values present: {len(cols_with_nulls)} columns contain missing cells ({missing_pct}% total).")

        # Check for duplicate rows
        duplicates = int(df.duplicated().sum())
        if duplicates > 0:
            warnings.append(f"Detected {duplicates} duplicate rows ({round((duplicates/n_rows)*100, 2)}%).")

        # Check required/expected modeling columns
        missing_core_cols = [c for c in ['tenure', 'MonthlyCharges', 'Contract'] if c not in df.columns]
        if missing_core_cols:
            issues.append(f"Missing core features required for predictive scoring: {missing_core_cols}")

        # Quality score calculation (0 - 100)
        quality_score = 100
        quality_score -= min(missing_pct * 2, 40)
        quality_score -= min((duplicates / n_rows) * 100, 20)
        if missing_core_cols:
            quality_score -= 30
        quality_score = max(int(round(quality_score)), 0)

        schema_categories = self.detect_schema_categories(df)

        return {
            "valid": len(issues) == 0,
            "quality_score": quality_score,
            "row_count": n_rows,
            "column_count": n_cols,
            "missing_cells": total_missing,
            "missing_pct": missing_pct,
            "cols_with_nulls": cols_with_nulls,
            "duplicate_count": duplicates,
            "issues": issues,
            "warnings": warnings,
            "detected_categories": schema_categories
        }

    def preprocess_for_training(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """
        Processes dataset for model training:
        1. Cleans data
        2. Encodes target 'Churn'
        3. Encodes binary and categorical columns matching production schema
        4. Aligns columns with self.feature_names
        """
        df_clean = self.clean_data(df, drop_customer_id=True)

        if 'Churn' not in df_clean.columns:
            raise ValueError("Target column 'Churn' not found in dataset for training.")

        y = df_clean['Churn'].map({'Yes': 1, 'No': 0, 1: 1, 0: 0, '1': 1, '0': 0})
        # If any rows lacked valid Churn label, drop them
        valid_mask = y.notna()
        df_clean = df_clean[valid_mask].reset_index(drop=True)
        y = y[valid_mask].astype(int).reset_index(drop=True)

        X_raw = df_clean.drop(columns=['Churn'])

        # Encode binary variables
        X_encoded = X_raw.copy()
        for col in BINARY_COLS:
            if col in X_encoded.columns:
                X_encoded[col] = X_encoded[col].map({'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0, 1: 1, 0: 0}).fillna(0).astype(int)

        # One-hot encode multi-categorical variables
        present_multi_cols = [col for col in MULTI_COLS if col in X_encoded.columns]
        X_encoded = pd.get_dummies(X_encoded, columns=present_multi_cols, drop_first=False)

        # Align with canonical feature names
        for feat in self.feature_names:
            if feat not in X_encoded.columns:
                X_encoded[feat] = 0.0

        X_final = X_encoded[self.feature_names].astype(float)
        return X_final, y, self.feature_names

    def get_train_test_split(
        self,
        test_size: float = config.TEST_SIZE,
        random_state: int = config.RANDOM_STATE
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Returns stratified train-test splits adhering to ML best practices.
        Splits data BEFORE applying standardization to prevent leakage.
        """
        df_raw = self.load_raw_data()
        X, y, _ = self.preprocess_for_training(df_raw)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )

        logger.info(f"Train/Test split complete: Train={X_train.shape[0]}, Test={X_test.shape[0]}, Features={X_train.shape[1]}")
        return X_train, X_test, y_train, y_test

    def prepare_single_input(self, input_dict: Dict[str, Any]) -> pd.DataFrame:
        """
        Transforms a raw customer dictionary (from web form or API)
        into a 1-row DataFrame aligned with model features.
        """
        d = dict(input_dict)

        # Numeric fields with safe casting
        tenure = float(d.get('tenure', 0))
        monthly_charges = float(d.get('MonthlyCharges', d.get('monthly_charges', 0.0)))
        total_charges = float(d.get('TotalCharges', d.get('total_charges', tenure * monthly_charges)))

        gender_val = 1 if str(d.get('gender', 'Male')).strip().lower() == 'male' else 0
        senior_val = 1 if d.get('SeniorCitizen') in [1, '1', True, 'Yes', 'yes'] else 0
        partner_val = 1 if d.get('Partner') in [1, '1', True, 'Yes', 'yes'] else 0
        dependents_val = 1 if d.get('Dependents') in [1, '1', True, 'Yes', 'yes'] else 0
        phone_service_val = 1 if d.get('PhoneService') in [1, '1', True, 'Yes', 'yes'] else 0
        paperless_val = 1 if d.get('PaperlessBilling') in [1, '1', True, 'Yes', 'yes'] else 0

        contract = str(d.get('Contract', d.get('contract', 'Month-to-month'))).strip()
        internet = str(d.get('InternetService', d.get('internet_service', 'Fiber optic'))).strip()
        multiple_lines = str(d.get('MultipleLines', d.get('multiple_lines', 'No'))).strip()
        online_security = str(d.get('OnlineSecurity', d.get('online_security', 'No'))).strip()
        online_backup = str(d.get('OnlineBackup', d.get('online_backup', 'No'))).strip()
        device_protection = str(d.get('DeviceProtection', d.get('device_protection', 'No'))).strip()
        tech_support = str(d.get('TechSupport', d.get('tech_support', 'No'))).strip()
        streaming_tv = str(d.get('StreamingTV', d.get('streaming_tv', 'No'))).strip()
        streaming_movies = str(d.get('StreamingMovies', d.get('streaming_movies', 'No'))).strip()
        payment_method = str(d.get('PaymentMethod', d.get('payment_method', 'Electronic check'))).strip()

        row: Dict[str, float] = {
            'gender': float(gender_val),
            'SeniorCitizen': float(senior_val),
            'Partner': float(partner_val),
            'Dependents': float(dependents_val),
            'tenure': float(tenure),
            'PhoneService': float(phone_service_val),
            'PaperlessBilling': float(paperless_val),
            'MonthlyCharges': float(monthly_charges),
            'TotalCharges': float(total_charges),
            # Multiple lines
            'MultipleLines_No': 1.0 if multiple_lines == 'No' else 0.0,
            'MultipleLines_No phone service': 1.0 if multiple_lines == 'No phone service' else 0.0,
            'MultipleLines_Yes': 1.0 if multiple_lines == 'Yes' else 0.0,
            # Internet Service
            'InternetService_DSL': 1.0 if internet == 'DSL' else 0.0,
            'InternetService_Fiber optic': 1.0 if internet == 'Fiber optic' else 0.0,
            'InternetService_No': 1.0 if internet == 'No' else 0.0,
            # Online Security
            'OnlineSecurity_No': 1.0 if online_security == 'No' else 0.0,
            'OnlineSecurity_No internet service': 1.0 if online_security == 'No internet service' or internet == 'No' else 0.0,
            'OnlineSecurity_Yes': 1.0 if online_security == 'Yes' else 0.0,
            # Online Backup
            'OnlineBackup_No': 1.0 if online_backup == 'No' else 0.0,
            'OnlineBackup_No internet service': 1.0 if online_backup == 'No internet service' or internet == 'No' else 0.0,
            'OnlineBackup_Yes': 1.0 if online_backup == 'Yes' else 0.0,
            # Device Protection
            'DeviceProtection_No': 1.0 if device_protection == 'No' else 0.0,
            'DeviceProtection_No internet service': 1.0 if device_protection == 'No internet service' or internet == 'No' else 0.0,
            'DeviceProtection_Yes': 1.0 if device_protection == 'Yes' else 0.0,
            # Tech Support
            'TechSupport_No': 1.0 if tech_support == 'No' else 0.0,
            'TechSupport_No internet service': 1.0 if tech_support == 'No internet service' or internet == 'No' else 0.0,
            'TechSupport_Yes': 1.0 if tech_support == 'Yes' else 0.0,
            # Streaming TV
            'StreamingTV_No': 1.0 if streaming_tv == 'No' else 0.0,
            'StreamingTV_No internet service': 1.0 if streaming_tv == 'No internet service' or internet == 'No' else 0.0,
            'StreamingTV_Yes': 1.0 if streaming_tv == 'Yes' else 0.0,
            # Streaming Movies
            'StreamingMovies_No': 1.0 if streaming_movies == 'No' else 0.0,
            'StreamingMovies_No internet service': 1.0 if streaming_movies == 'No internet service' or internet == 'No' else 0.0,
            'StreamingMovies_Yes': 1.0 if streaming_movies == 'Yes' else 0.0,
            # Contract
            'Contract_Month-to-month': 1.0 if contract == 'Month-to-month' else 0.0,
            'Contract_One year': 1.0 if contract == 'One year' else 0.0,
            'Contract_Two year': 1.0 if contract == 'Two year' else 0.0,
            # Payment Method
            'PaymentMethod_Bank transfer (automatic)': 1.0 if 'Bank transfer' in payment_method else 0.0,
            'PaymentMethod_Credit card (automatic)': 1.0 if 'Credit card' in payment_method else 0.0,
            'PaymentMethod_Electronic check': 1.0 if 'Electronic check' in payment_method else 0.0,
            'PaymentMethod_Mailed check': 1.0 if 'Mailed check' in payment_method else 0.0,
        }

        # Align with canonical self.feature_names
        for feat in self.feature_names:
            if feat not in row:
                row[feat] = 0.0

        df_row = pd.DataFrame([row])[self.feature_names]
        return df_row

    def prepare_batch_df(self, batch_df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str]]:
        """
        Prepares an arbitrary uploaded customer CSV DataFrame for batch prediction.
        Preserves customerID if present for reporting.
        """
        customer_ids = batch_df['customerID'].tolist() if 'customerID' in batch_df.columns else [
            f"CUST-{i+1:04d}" for i in range(len(batch_df))
        ]
        df_clean = self.clean_data(batch_df, drop_customer_id=False)

        # Encode binary variables
        X_encoded = df_clean.copy()
        for col in BINARY_COLS:
            if col in X_encoded.columns:
                X_encoded[col] = X_encoded[col].map({'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0, 1: 1, 0: 0}).fillna(0).astype(int)

        present_multi_cols = [col for col in MULTI_COLS if col in X_encoded.columns]
        X_encoded = pd.get_dummies(X_encoded, columns=present_multi_cols, drop_first=False)

        # Align columns
        for feat in self.feature_names:
            if feat not in X_encoded.columns:
                X_encoded[feat] = 0.0

        X_final = X_encoded[self.feature_names].astype(float)
        return X_final, customer_ids, self.feature_names
