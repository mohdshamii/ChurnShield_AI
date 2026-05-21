
# ChurnShield AI: Advanced Customer Retention Solution

---

## Overview

ChurnShield AI is a production-grade customer churn prediction platform that transforms raw behavioral and demographic data into actionable retention intelligence. The system leverages ensemble machine learning techniques, SHAP-based model interpretability, and an interactive executive dashboard to empower data-driven retention strategies. Designed for telecommunications and subscription-based enterprises, ChurnShield AI moves beyond simple prediction into prescriptive intervention planning.

---

## Analytical Foundation

Exploratory data analysis revealed critical churn patterns that inform the modeling pipeline. The EDA dashboard consolidates univariate distributions, bivariate relationships, and temporal trends into a unified view, enabling rapid hypothesis validation before model construction.

<p align="center">
  <img src="./img/eda.png" alt="EDA Dashboard" width="90%">
</p>

Three classifier architectures were evaluated against identical holdout partitions. Confusion matrices below illustrate the trade-off between false positives (unnecessary retention spending) and false negatives (silent churners lost without intervention).

<div align="center">
  <table>
    <tr>
      <td><img src="./img/logistic.png" alt="Logistic Regression Confusion Matrix" width="95%"></td>
      <td><img src="./img/randomforest.png" alt="Random Forest Confusion Matrix" width="95%"></td>
      <td><img src="./img/xgboost.png" alt="XGBoost Confusion Matrix" width="95%"></td>
    </tr>
    <tr>
      <td align="center"><strong>Logistic Regression</strong></td>
      <td align="center"><strong>Random Forest</strong></td>
      <td align="center"><strong>XGBoost</strong></td>
    </tr>
  </table>
</div>

---

## Core Capabilities

### Precision Prediction Engine

The engine combines XGBoost, Random Forest, and Logistic Regression into a weighted ensemble that optimizes for recall while maintaining precision thresholds suitable for budget-constrained retention campaigns. Key components include:

- **SHAP Explainability Layer**: Every prediction is accompanied by a feature-level attribution breakdown, eliminating the black-box objection common in stakeholder conversations.
- **Real-Time Probability Calibration**: Platt scaling applied post-hoc ensures that predicted probabilities correspond to observed frequencies, critical for threshold-based intervention triggers.
- **Class Imbalance Handling**: SMOTE-ENN hybrid resampling addresses the inherent skew in churn datasets without introducing synthetic noise that degrades generalization.

### Executive Intelligence Dashboard

The Streamlit-based interface translates model outputs into business-facing analytics:

- **Risk Segmentation**: Customers are stratified into high, medium, and low churn probability tiers with cohort-level statistics.
- **Lifetime Value Preservation Estimates**: Projected revenue at risk is calculated by multiplying churn probability by customer lifetime value, enabling ROI-based prioritization.
- **Retention Investment Calculator**: Users input campaign cost parameters to simulate net retention value across different intervention thresholds.

<p align="center">
  <img src="./img/output01.png" alt="Dashboard Main View" width="90%">
</p>

### Interpretable Predictions

Every individual prediction is decomposed into contributing factors using SHAP values. This transparency allows retention teams to understand exactly which variables are driving churn risk for each customer, enabling personalized intervention strategies rather than generic outreach.

<p align="center">
  <img src="./img/output02.png" alt="Key Factors Influencing Prediction" width="90%">
</p>

### Churn Trend Analytics

Temporal churn patterns are surfaced through interactive visualizations that segment risk by contract type, tenure cohort, and service bundle. Decision-makers can identify systemic issues affecting entire customer segments rather than treating churn as isolated incidents.

<p align="center">
  <img src="./img/output03.png" alt="Churn Trends Analysis" width="90%">
</p>

---

## Performance Benchmarks

| Model | Accuracy | Precision | Recall | AUC-ROC |
|-------|----------|-----------|--------|---------|
| XGBoost | 85.5% | 83.0% | 88.0% | 0.85 |
| Logistic Regression | 91.7% | 80.2% | 85.3% | 0.91 |
| Random Forest | 81.1% | 82.3% | 89.9% | 0.81 |

Logistic Regression achieves the highest accuracy and AUC-ROC, making it suitable for environments where overall classification quality is prioritized. Random Forest demonstrates superior recall, capturing nearly 90% of actual churners, critical when the cost of missed churn is high. The ensemble approach combines these complementary strengths.

---

## Technical Architecture

### Machine Learning Stack

| Component | Technology | Version |
|-----------|------------|---------|
| Language | Python | 3.10+ |
| Gradient Boosting | XGBoost | 1.7+ |
| Classical Models | Scikit-learn | 1.2+ |
| Imbalance Treatment | Imbalanced-learn | 0.10+ |
| Interpretability | SHAP | 0.41+ |
| Serialization | Joblib | 1.2+ |

### Dashboard Stack

| Component | Technology | Version |
|-----------|------------|---------|
| Framework | Streamlit | 1.22+ |
| Visualizations | Plotly | 5.13+ |
| Data Manipulation | Pandas | 1.5+ |
| Numerical Computing | NumPy | 1.23+ |

---

## Feature Engineering

The original Telco Customer Churn dataset is enriched with derived features that capture complex behavioral signals invisible to raw variables alone:


# Lifetime Value Proxy
df['TenureToChargeRatio'] = df['tenure'] / (df['MonthlyCharges'] + 1e-6)

# Total Engagement Score
df['TotalValueScore'] = (df['tenure'] * df['MonthlyCharges']) / df['TotalCharges']

# Service Adoption Density
df['ServiceDensity'] = df[['OnlineSecurity_Yes', 'OnlineBackup_Yes',
                          'DeviceProtection_Yes', 'TechSupport_Yes']].sum(axis=1) / df['tenure']

# High-Risk Payment-Contract Interaction
df['PaymentRisk'] = df['PaymentMethod_Electronic check'].astype(int) * df['Contract_Month-to-month'].astype(int)

# Premium Customer Flag
df['HighCostLongTenure'] = ((df['MonthlyCharges'] > df['MonthlyCharges'].quantile(0.75)) &
                           (df['tenure'] > df['tenure'].median())).astype(int)

# ChurnShield AI

These engineered features capture non-linear interactions that single variables cannot express, contributing significantly to model performance.

## Installation and Quick Start

### Prerequisites

- Python 3.10 or higher
- Git
- 2GB available disk space (minimum)

## Setup Instructions

# Clone the repository
git clone https://github.com/codewithshami/ChurnShield_AI.git
cd ChurnShield_AI

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\\Scripts\\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Launch the dashboard
streamlit run app.py
`

Access the dashboard at http://localhost:8501 in your web browser.

## One-Command Setup

For automated deployment, execute the included setup script:


chmod +x setup.sh
./setup.sh


This script handles environment creation, dependency installation, dataset preparation, model training, and dashboard launch in a single execution.

## Repository Structure


ChurnShield_AI/
├── app.py
├── train.py
├── requirements.txt
├── setup.sh
├── banner.png
├── img/
│   ├── eda.png
│   ├── logistic.png
│   ├── randomforest.png
│   ├── xgboost.png
│   ├── output01.png
│   ├── output02.png
│   └── output03.png
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── notebooks/
└── src/
    ├── preprocessing.py
    ├── modeling.py
    └── utils.py


## Usage Guide

### Individual Prediction

1. Navigate to the Predict tab in the dashboard sidebar
2. Input customer attributes through the form interface
3. Click Predict Churn Risk
4. Review the probability score and SHAP breakdown for contributing factors

### Batch Processing

1. Upload a CSV file containing customer records via the Batch tab
2. Verify column mapping against expected schema
3. Download results with appended churn probabilities and risk segments

### Retention Campaign Simulator

1. Access the Campaign tab in the dashboard
2. Define intervention parameters including discount percentage and contract upgrade offers
3. Model projects retained customers and net revenue impact
4. Export campaign target list for CRM integration

## Model Retraining

The training pipeline supports incremental retraining as new churn data becomes available:

python train.py --data path/to/updated_data.csv --output models/


Optional flags include:

- \`--tune\` : Perform hyperparameter optimization via grid search
- \`--eval\` : Generate detailed evaluation report with SHAP summary plots
- \`--export\` : Serialize model in ONNX format for deployment

## Deployment Architecture

For production deployment, the following architecture is recommended:


[Load Balancer] -> [Streamlit Server (Gunicorn)] -> [Model API (FastAPI)]
                                                          |
                                                     [Redis Cache]
                                                          |
                                                  [Model Registry]


The model serving layer exposes a REST API endpoint accepting JSON payloads and returning predictions with sub-100ms latency when paired with Redis caching for frequent customer profiles.

**Repository:** https://github.com/mohdshamii/ChurnShield_AI

**Community:** GitHub Discussions

## License

This project is licensed under the MIT License. See the LICENSE file for details.

<p align="center">
  <em>Built for data-driven retention teams who demand both accuracy and interpretability.</em>
</p>



