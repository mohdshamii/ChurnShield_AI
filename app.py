"""
ChurnShield AI - Streamlit Dashboard (Upgraded Production Edition)
Features Real Multi-Model Ensemble, Genuine SHAP Feature Attribution,
Dynamic Financial Retention Playbooks, and Real SQLite Persistence.
"""

import json
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Core modular backend
from src.data_loader import DataLoader
from src.models import ModelManager
from src.explainability import ModelExplainer
from src.retention_strategy import RetentionStrategyEngine
from src.database import DatabaseManager

# Page configuration
st.set_page_config(
    page_title="ChurnShield AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize singletons with Streamlit resource caching
@st.cache_resource
def get_backend():
    dl = DataLoader()
    mm = ModelManager()
    exp = ModelExplainer(mm)
    db = DatabaseManager()
    return dl, mm, exp, db

data_loader, model_manager, explainer, db_manager = get_backend()

# Custom CSS styling (preserves original cyberpunk/neon aesthetic)
st.markdown("""
    <style>
        :root {
            --primary: #00ffaa;
            --secondary: #0095ff;
            --danger: #ff3d57;
            --warning: #ffaa00;
            --dark: #0f1118;
            --light: #f8f9fa;
            --neon-glow: 0 0 10px rgba(0, 255, 200, 0.8);
        }
        
        .main {
            background: linear-gradient(-45deg, #0f2027, #203a43, #2c5364, #3a7bd5);
            background-size: 400% 400%;
            animation: gradientBG 15s ease infinite;
            color: white;
        }
        
        @keyframes gradientBG {
            0% {background-position: 0% 50%;}
            50% {background-position: 100% 50%;}
            100% {background-position: 0% 50%;}
        }
        
        .card {
            background: rgba(42, 46, 61, 0.65);
            backdrop-filter: blur(8px);
            -webkit-backdrop-filter: blur(8px);
            border-radius: 16px;
            padding: 24px;
            margin-bottom: 24px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
            border: 1px solid rgba(0, 255, 200, 0.1);
            transition: all 0.3s ease;
        }
        
        .card:hover {
            border-color: rgba(0, 255, 200, 0.4);
            box-shadow: 0 12px 40px rgba(0, 255, 200, 0.2);
        }
        
        .card-title {
            font-size: 1.3em;
            font-weight: 700;
            margin-bottom: 14px;
            color: var(--primary);
            text-shadow: var(--neon-glow);
        }
        
        .risk-high {
            color: #ff3d57;
            font-weight: 800;
            font-size: 1.8em;
            text-shadow: 0 0 10px rgba(255, 61, 87, 0.5);
        }
        
        .risk-medium {
            color: #ffaa00;
            font-weight: 800;
            font-size: 1.8em;
            text-shadow: 0 0 10px rgba(255, 170, 0, 0.5);
        }
        
        .risk-low {
            color: #00ffaa;
            font-weight: 800;
            font-size: 1.8em;
            text-shadow: 0 0 10px rgba(0, 255, 200, 0.5);
        }
        
        .sticky-footer {
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            background: rgba(15,17,24,0.95);
            color: #ccc;
            font-size: 0.85em;
            padding: 10px 0;
            text-align: center;
            z-index: 1000;
            border-top: 1px solid rgba(0, 255, 200, 0.2);
        }
    </style>
""", unsafe_allow_html=True)

# Header
col1, col2 = st.columns([1, 4])
with col1:
    st.image("https://cdn-icons-png.flaticon.com/512/2331/2331895.png", width=100)
with col2:
    st.title("ChurnShield AI - Customer Retention Solution")
    st.markdown("Predict customer churn risk with AI-powered insights, real SHAP explainability, and actionable financial playbooks.")

# Sidebar - Customer Profile Configuration
with st.sidebar:
    st.header("📋 Customer Profile")
    st.markdown("---")

    customer_id = st.text_input("Customer ID / Name", "Mohd Shami")
    join_date = st.date_input("Join Date", datetime.now() - timedelta(days=365))

    st.subheader("Demographics")
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        gender = st.radio("Gender", ("Male", "Female"))
    with col_d2:
        senior_citizen = st.checkbox("Senior Citizen")

    partner = st.checkbox("Has Partner")
    dependents = st.checkbox("Has Dependents")

    st.subheader("Account Details")
    tenure = st.slider("Tenure (months)", 0, 72, 6)
    monthly_charges = st.slider("Monthly Charges ($)", 18, 140, 85)
    total_charges = st.slider("Total Charges ($)", 0, 9000, int(tenure * monthly_charges))

    col_b1, col_b2 = st.columns(2)
    with col_b1:
        paperless_billing = st.checkbox("Paperless Billing", value=True)
    with col_b2:
        phone_service = st.checkbox("Phone Service", value=True)

    multiple_lines = st.selectbox("Multiple Lines", ("No", "Yes", "No phone service"))

    st.subheader("Service Details")
    contract = st.selectbox("Contract Type", ("Month-to-month", "One year", "Two year"))
    internet_service = st.selectbox("Internet Service", ("Fiber optic", "DSL", "No"))

    st.markdown("**Additional Services**")
    online_security = st.selectbox("Online Security", ("No", "Yes", "No internet service"))
    online_backup = st.selectbox("Online Backup", ("No", "Yes", "No internet service"))
    device_protection = st.selectbox("Device Protection", ("No", "Yes", "No internet service"))
    tech_support = st.selectbox("Tech Support", ("No", "Yes", "No internet service"))
    streaming_tv = st.selectbox("Streaming TV", ("Yes", "No", "No internet service"))
    streaming_movies = st.selectbox("Streaming Movies", ("Yes", "No", "No internet service"))

    st.subheader("Payment Details")
    payment_method = st.selectbox("Payment Method", (
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ))

    st.markdown("---")
    customer_notes = st.text_area("Retention Specialist Notes", "Evaluated via Streamlit dashboard.")
    
    # Save to SQLite Database
    if st.button("💾 Save Profile to Database"):
        customer_dict = {
            'gender': gender, 'SeniorCitizen': 1 if senior_citizen else 0,
            'Partner': partner, 'Dependents': dependents,
            'tenure': tenure, 'PhoneService': phone_service,
            'PaperlessBilling': paperless_billing,
            'MonthlyCharges': monthly_charges, 'TotalCharges': total_charges,
            'Contract': contract, 'InternetService': internet_service,
            'OnlineSecurity': online_security, 'OnlineBackup': online_backup,
            'DeviceProtection': device_protection, 'TechSupport': tech_support,
            'StreamingTV': streaming_tv, 'StreamingMovies': streaming_movies,
            'PaymentMethod': payment_method, 'MultipleLines': multiple_lines
        }
        input_df = data_loader.prepare_single_input(customer_dict)
        prob = model_manager.predict_churn(input_df, model_type="ensemble")
        r_meta = RetentionStrategyEngine.assess_risk(prob)
        saved_id = db_manager.save_prediction(customer_id, customer_dict, prob, r_meta['level'], customer_notes)
        st.success(f"Profile saved to SQLite database! (Record ID #{saved_id})")

# Construct dictionary and run model inference
customer_data = {
    'gender': gender, 'SeniorCitizen': 1 if senior_citizen else 0,
    'Partner': partner, 'Dependents': dependents,
    'tenure': tenure, 'PhoneService': phone_service,
    'PaperlessBilling': paperless_billing,
    'MonthlyCharges': monthly_charges, 'TotalCharges': total_charges,
    'Contract': contract, 'InternetService': internet_service,
    'OnlineSecurity': online_security, 'OnlineBackup': online_backup,
    'DeviceProtection': device_protection, 'TechSupport': tech_support,
    'StreamingTV': streaming_tv, 'StreamingMovies': streaming_movies,
    'PaymentMethod': payment_method, 'MultipleLines': multiple_lines
}

input_df = data_loader.prepare_single_input(customer_data)
churn_prob = model_manager.predict_churn(input_df, model_type="ensemble")
explanation = explainer.explain_prediction(input_df)
risk_assessment = RetentionStrategyEngine.assess_risk(churn_prob)
clv_metrics = RetentionStrategyEngine.calculate_clv_and_roi(monthly_charges, tenure, churn_prob)
retention_plan = RetentionStrategyEngine.generate_recommendations(
    customer_data, churn_prob, explanation['risk_catalysts']
)

# Main Application Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Prediction & SHAP",
    "📈 Customer Analytics",
    "🛡️ Retention Playbook",
    "📋 Database History & Notes"
])

# -------------------------------------------------------------
# TAB 1: Prediction & SHAP Explainability
# -------------------------------------------------------------
with tab1:
    col_g1, col_g2 = st.columns([2, 1])

    with col_g1:
        gauge_fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=churn_prob * 100,
            number={'suffix': "%", 'font': {'size': 44, 'color': 'white'}},
            title={'text': "Ensemble Churn Probability", 'font': {'size': 20, 'color': 'white'}},
            gauge={
                'axis': {'range': [0, 100], 'tickcolor': "white"},
                'bar': {'color': risk_assessment['color']},
                'steps': [
                    {'range': [0, 35], 'color': '#00ffaa'},
                    {'range': [35, 65], 'color': '#ffaa00'},
                    {'range': [65, 100], 'color': '#ff3d57'}
                ],
                'threshold': {
                    'line': {'color': "white", 'width': 4},
                    'thickness': 0.8,
                    'value': churn_prob * 100
                }
            }
        ))
        gauge_fig.update_layout(
            height=300,
            margin=dict(l=30, r=30, b=20, t=50),
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white")
        )
        st.plotly_chart(gauge_fig, use_container_width=True)

    with col_g2:
        risk_class = "risk-high" if risk_assessment['level'] == 'HIGH' else ("risk-medium" if risk_assessment['level'] == 'MEDIUM' else "risk-low")
        st.markdown(f"""
        <div class="card">
            <div class="card-title">Risk Assessment</div>
            <p class="{risk_class}">{risk_assessment['level']} RISK</p>
            <p style="color: #aaa; margin-bottom: 10px;">{risk_assessment['urgency']}</p>
            <p>Annual Revenue At Risk: <strong style="color: #ff3d57;">${clv_metrics['revenue_at_risk']:,.0f}</strong></p>
            <p>Projected Lifetime Value: <strong style="color: #00ffaa;">${clv_metrics['projected_clv']:,.0f}</strong></p>
            <p>Retention ROI Potential: <strong style="color: #0095ff;">+{clv_metrics['retention_roi_percent']}%</strong></p>
        </div>
        """, unsafe_allow_html=True)

    # Real Model Feature Attribution (SHAP)
    st.markdown("### 🔍 Model Feature Attribution (SHAP Values)")
    st.markdown("Features driving the prediction for this specific customer (positive indicates churn driver, negative indicates loyalty anchor):")

    impact_df = pd.DataFrame(explanation['impact_list'])
    if not impact_df.empty:
        fig_bars = px.bar(
            impact_df,
            x='impact',
            y='factor',
            orientation='h',
            color='direction',
            color_discrete_map={'INCREASE_RISK': '#ff3d57', 'REDUCE_RISK': '#00ffaa'},
            labels={'impact': 'SHAP Contribution Score', 'factor': 'Feature'},
            title="Local Feature Contribution"
        )
        fig_bars.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color="white"),
            yaxis=dict(autorange="reversed")
        )
        st.plotly_chart(fig_bars, use_container_width=True)

# -------------------------------------------------------------
# TAB 2: Customer Analytics
# -------------------------------------------------------------
with tab2:
    st.header("📈 Portfolio & Model Analytics")

    col_a1, col_a2 = st.columns(2)
    with col_a1:
        st.markdown("### 🎯 Global Feature Importance")
        feat_imp = model_manager.get_feature_importances()
        top_10 = pd.DataFrame(list(feat_imp.items())[:10], columns=['Feature', 'Importance'])
        fig_imp = px.bar(
            top_10, x='Importance', y='Feature', orientation='h',
            color='Importance', color_continuous_scale='Blues'
        )
        fig_imp.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color="white"),
            yaxis=dict(autorange="reversed")
        )
        st.plotly_chart(fig_imp, use_container_width=True)

    with col_a2:
        st.markdown("### 📉 Empirical Churn Rate by Contract")
        raw_df = data_loader.load_raw_data()
        contract_summary = raw_df.groupby('Contract')['Churn'].apply(lambda x: (x == 'Yes').mean() * 100).reset_index()
        fig_cont = px.bar(
            contract_summary, x='Contract', y='Churn',
            color='Churn', color_continuous_scale='Reds',
            labels={'Churn': 'Churn Rate (%)'}
        )
        fig_cont.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color="white")
        )
        st.plotly_chart(fig_cont, use_container_width=True)

# -------------------------------------------------------------
# TAB 3: Retention Playbook
# -------------------------------------------------------------
with tab3:
    st.header("🛡️ Targeted Retention Playbook")

    st.markdown(f"#### Recommended Strategy for {risk_assessment['level']} Risk Customer")
    for s in retention_plan['strategies']:
        st.info(f"**{s['title']}** ({s['tag']})\n\n{s['detail']}")

    st.markdown("### 📅 14-Day Action Timeline")
    for step in retention_plan['timeline']:
        st.write(f"- 📌 **{step['day']}**: {step['action']}")

    st.markdown("### ✉️ Outreach Copy Templates")
    with st.expander("View Pre-Drafted Email Template"):
        st.code(retention_plan['email_template'], language="text")
    with st.expander("View Pre-Drafted SMS Template"):
        st.code(retention_plan['sms_template'], language="text")

# -------------------------------------------------------------
# TAB 4: Database History & Notes
# -------------------------------------------------------------
with tab4:
    st.header("📋 SQLite Prediction Records")
    records = db_manager.get_recent_predictions(limit=25)
    if records:
        rec_df = pd.DataFrame(records)
        st.dataframe(rec_df[['id', 'customer_id', 'churn_prob', 'risk_level', 'tenure', 'monthly_charges', 'notes', 'timestamp']], use_container_width=True)
    else:
        st.info("No records found in database. Click 'Save Profile to Database' in the sidebar to persist records.")

st.markdown("""
    <div class="sticky-footer">
        <span>ChurnShield AI • Soft Voting Ensemble (XGBoost + Random Forest + Logistic Regression) • Database: SQLite</span>
    </div>
""", unsafe_allow_html=True)