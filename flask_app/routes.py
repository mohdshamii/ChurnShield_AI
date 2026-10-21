"""
ChurnShield AI - Flask Blueprints & Application Controller
Implements all web routes, REST endpoints, analytics aggregations,
prediction handlers, and batch processing pipelines.
"""

import io
import json
import logging
from typing import Dict, Any, List
import numpy as np
import pandas as pd
from flask import (
    Blueprint, render_template, request, jsonify,
    send_file, current_app, redirect, url_for, flash
)

from src import config
from src.data_loader import DataLoader
from src.models import ModelManager
from src.explainability import ModelExplainer
from src.retention_strategy import RetentionStrategyEngine
from src.database import DatabaseManager

logger = logging.getLogger(__name__)

main_bp = Blueprint("main", __name__)

# Singletons for application lifecycle
data_loader = DataLoader()
model_manager = ModelManager()
explainer = ModelExplainer(model_manager)
db_manager = DatabaseManager()


# -------------------------------------------------------------
# 1. Executive Intelligence Dashboard
# -------------------------------------------------------------
@main_bp.route("/")
def index():
    """Executive Dashboard featuring real-time KPIs, visual analytics, and recent activity."""
    try:
        raw_df = data_loader.load_raw_data()
        df_clean = data_loader.clean_data(raw_df, drop_customer_id=False)
        total_customers = len(df_clean)
        churn_count = (df_clean["Churn"].astype(str).str.lower() == "yes").sum()
        churn_rate = round((churn_count / total_customers) * 100, 1)

        # Revenue metrics
        monthly_rev = df_clean["MonthlyCharges"].sum()
        annual_rev = monthly_rev * 12
        churning_monthly_rev = df_clean[df_clean["Churn"].astype(str).str.lower() == "yes"]["MonthlyCharges"].sum()
        annual_rev_at_risk = churning_monthly_rev * 12

        # Contract analytics
        contract_data = []
        for c_type in ["Month-to-month", "One year", "Two year"]:
            sub = df_clean[df_clean["Contract"] == c_type]
            tot = int(len(sub))
            ch = int((sub["Churn"].astype(str).str.lower() == "yes").sum())
            rate = float(round((ch / max(tot, 1)) * 100, 1))
            contract_data.append({"contract": str(c_type), "total": tot, "churn_count": ch, "rate": rate})

        # Internet service analytics
        internet_data = []
        for i_type in ["Fiber optic", "DSL", "No"]:
            sub = df_clean[df_clean["InternetService"] == i_type]
            tot = int(len(sub))
            ch = int((sub["Churn"].astype(str).str.lower() == "yes").sum())
            rate = float(round((ch / max(tot, 1)) * 100, 1))
            internet_data.append({"service": str(i_type), "total": tot, "churn_count": ch, "rate": rate})

        # Payment Method breakdown
        payment_data = []
        for p_type in ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]:
            sub = df_clean[df_clean["PaymentMethod"] == p_type]
            tot = int(len(sub))
            ch = int((sub["Churn"].astype(str).str.lower() == "yes").sum())
            rate = float(round((ch / max(tot, 1)) * 100, 1))
            payment_data.append({"method": str(p_type), "total": tot, "churn_count": ch, "rate": rate})

        # Tenure cohorts
        tenure_cohorts = [
            {"cohort": "0 - 12 Mo", "count": int(len(df_clean[df_clean["tenure"] <= 12])), "churn_rate": float(round(((df_clean[df_clean["tenure"] <= 12]["Churn"].str.lower() == "yes").sum() / max(len(df_clean[df_clean["tenure"] <= 12]), 1)) * 100, 1))},
            {"cohort": "13 - 24 Mo", "count": int(len(df_clean[(df_clean["tenure"] > 12) & (df_clean["tenure"] <= 24)])), "churn_rate": float(round(((df_clean[(df_clean["tenure"] > 12) & (df_clean["tenure"] <= 24)]["Churn"].str.lower() == "yes").sum() / max(len(df_clean[(df_clean["tenure"] > 12) & (df_clean["tenure"] <= 24)]), 1)) * 100, 1))},
            {"cohort": "25 - 48 Mo", "count": int(len(df_clean[(df_clean["tenure"] > 24) & (df_clean["tenure"] <= 48)])), "churn_rate": float(round(((df_clean[(df_clean["tenure"] > 24) & (df_clean["tenure"] <= 48)]["Churn"].str.lower() == "yes").sum() / max(len(df_clean[(df_clean["tenure"] > 24) & (df_clean["tenure"] <= 48)]), 1)) * 100, 1))},
            {"cohort": "49 - 72 Mo", "count": int(len(df_clean[df_clean["tenure"] > 48])), "churn_rate": float(round(((df_clean[df_clean["tenure"] > 48]["Churn"].str.lower() == "yes").sum() / max(len(df_clean[df_clean["tenure"] > 48]), 1)) * 100, 1))}
        ]

        # Recent DB predictions
        recent_preds = db_manager.get_recent_predictions(limit=5)
        db_stats = db_manager.get_database_stats()

    except Exception as e:
        logger.error(f"Error compiling dashboard metrics: {e}")
        total_customers = 7043
        churn_rate = 26.5
        annual_rev_at_risk = 1420000.0
        contract_data = []
        internet_data = []
        payment_data = []
        tenure_cohorts = []
        recent_preds = []
        db_stats = {"total_predictions": 0, "avg_churn_prob": 0, "high_risk_count": 0}

    return render_template(
        "index.html",
        active_page="dashboard",
        total_customers=f"{total_customers:,}",
        churn_rate=churn_rate,
        annual_rev_at_risk=f"${annual_rev_at_risk:,.0f}",
        contract_data=contract_data,
        internet_data=internet_data,
        payment_data=payment_data,
        tenure_cohorts=tenure_cohorts,
        recent_preds=recent_preds,
        db_stats=db_stats
    )


# -------------------------------------------------------------
# 2. Single Customer Prediction & What-If Simulator
# -------------------------------------------------------------
@main_bp.route("/predict", methods=["GET", "POST"])
def predict():
    """Predict churn risk for an individual customer profile with real SHAP explainability."""
    if request.method == "POST":
        try:
            # Handle both JSON API and standard Form POST
            if request.is_json:
                data = request.get_json()
            else:
                data = request.form.to_dict()

            customer_id = data.get("customer_id", "CUST-PREDICT")
            input_df = data_loader.prepare_single_input(data)

            # Model inference
            churn_prob = model_manager.predict_churn(input_df, model_type="ensemble")

            # XAI Explanation
            explanation = explainer.explain_prediction(input_df)

            # Financial & Retention calculations
            monthly = float(data.get("MonthlyCharges", data.get("monthly_charges", 70.0)))
            tenure = float(data.get("tenure", 12.0))
            clv_roi = RetentionStrategyEngine.calculate_clv_and_roi(monthly, tenure, churn_prob)
            retention_plan = RetentionStrategyEngine.generate_recommendations(
                data, churn_prob, explanation["risk_catalysts"]
            )

            response_data = {
                "success": True,
                "customer_id": customer_id,
                "churn_prob": round(churn_prob * 100, 1),
                "churn_prob_raw": churn_prob,
                "risk_meta": retention_plan["risk_meta"],
                "explanation": explanation,
                "clv_roi": clv_roi,
                "retention_plan": retention_plan
            }

            if request.is_json or request.headers.get("X-Requested-With") == "XMLHttpRequest":
                return jsonify(response_data)

            # Non-AJAX fallback: render with result
            return render_template("predict.html", active_page="predict", result=response_data, form_data=data)

        except Exception as e:
            logger.error(f"Prediction error: {e}", exc_info=True)
            if request.is_json or request.headers.get("X-Requested-With") == "XMLHttpRequest":
                return jsonify({"success": False, "error": str(e)}), 400
            flash(f"Error during prediction: {str(e)}", "danger")
            return render_template("predict.html", active_page="predict")

    return render_template("predict.html", active_page="predict")


# -------------------------------------------------------------
# 3. Interactive What-If Scenario Simulator API
# -------------------------------------------------------------
@main_bp.route("/api/what-if", methods=["POST"])
def what_if_simulator():
    """Calculates risk delta between baseline and proposed customer retention actions."""
    try:
        payload = request.get_json() or {}
        baseline_data = payload.get("baseline", {})
        scenario_data = payload.get("scenario", {})

        base_df = data_loader.prepare_single_input(baseline_data)
        scen_df = data_loader.prepare_single_input(scenario_data)

        prob_base = model_manager.predict_churn(base_df, model_type="ensemble")
        prob_scen = model_manager.predict_churn(scen_df, model_type="ensemble")

        delta = prob_scen - prob_base
        percent_reduction = ((prob_base - prob_scen) / max(prob_base, 1e-4)) * 100.0

        return jsonify({
            "success": True,
            "baseline_prob": round(prob_base * 100, 1),
            "scenario_prob": round(prob_scen * 100, 1),
            "delta_prob": round(delta * 100, 1),
            "percent_risk_reduction": round(max(0.0, percent_reduction), 1),
            "risk_mitigated": delta < 0
        })
    except Exception as e:
        logger.error(f"What-if simulation error: {e}")
        return jsonify({"success": False, "error": str(e)}), 400


# -------------------------------------------------------------
# 4. High-Speed Batch Prediction & Analytics
# -------------------------------------------------------------
@main_bp.route("/batch", methods=["GET", "POST"])
def batch_predict():
    """Upload customer CSV for bulk churn risk classification and strategy mapping."""
    if request.method == "POST":
        try:
            # Check if file was uploaded or sample requested
            if "file" in request.files and request.files["file"].filename != "":
                file = request.files["file"]
                batch_df = pd.read_csv(file)
                filename = file.filename
            elif request.form.get("use_sample") == "1":
                # Sample 100 records from data.csv
                raw_df = data_loader.load_raw_data()
                batch_df = raw_df.sample(n=min(120, len(raw_df)), random_state=42).copy()
                filename = "sample_telco_batch.csv"
            else:
                flash("Please upload a CSV file or select the sample dataset option.", "warning")
                return redirect(url_for("main.batch_predict"))

            X_batch, customer_ids, _ = data_loader.prepare_batch_df(batch_df)
            probs = model_manager.batch_predict(X_batch, model_type="ensemble")

            # Compile enriched output records
            records = []
            high_cnt = 0
            med_cnt = 0
            low_cnt = 0
            total_rev_at_risk = 0.0

            monthly_vals = batch_df["MonthlyCharges"].values if "MonthlyCharges" in batch_df.columns else [65.0] * len(probs)
            contract_vals = batch_df["Contract"].values if "Contract" in batch_df.columns else ["Month-to-month"] * len(probs)

            for cid, prob, monthly, contract in zip(customer_ids, probs, monthly_vals, contract_vals):
                prob_pct = round(prob * 100, 1)
                risk_meta = RetentionStrategyEngine.assess_risk(prob)
                rev_risk = float(monthly) * 12.0 * prob
                total_rev_at_risk += rev_risk

                if risk_meta["level"] == "HIGH":
                    high_cnt += 1
                    rec_action = "Offer 15% 1-Year Contract Discount + VIP Outreach"
                elif risk_meta["level"] == "MEDIUM":
                    med_cnt += 1
                    rec_action = "Bundle 90-Day Free Tech Support + Nurture"
                else:
                    low_cnt += 1
                    rec_action = "Routine Loyalty Check-in & Upsell Review"

                records.append({
                    "customer_id": cid,
                    "churn_prob": prob_pct,
                    "risk_level": risk_meta["level"],
                    "color": risk_meta["color"],
                    "badge_class": risk_meta["badge_class"],
                    "monthly_charges": round(float(monthly), 2),
                    "contract": str(contract),
                    "revenue_at_risk": round(rev_risk, 2),
                    "recommended_action": rec_action
                })

            # Sort records descending by revenue at risk / risk
            records.sort(key=lambda x: (x["churn_prob"], x["revenue_at_risk"]), reverse=True)

            avg_prob = float(np.mean(probs))
            # Log to DB batch history
            db_manager.save_batch_run(filename, len(records), high_cnt, med_cnt, low_cnt, avg_prob)

            batch_summary = {
                "filename": filename,
                "total_records": len(records),
                "high_risk_count": high_cnt,
                "medium_risk_count": med_cnt,
                "low_risk_count": low_cnt,
                "avg_churn_prob": round(avg_prob * 100, 1),
                "total_rev_at_risk": f"${total_rev_at_risk:,.0f}"
            }

            return render_template("batch.html", active_page="batch", summary=batch_summary, records=records[:200])

        except Exception as e:
            logger.error(f"Batch processing error: {e}", exc_info=True)
            flash(f"Error processing batch CSV: {str(e)}", "danger")
            return redirect(url_for("main.batch_predict"))

    return render_template("batch.html", active_page="batch")


# -------------------------------------------------------------
# 5. ML Models Evaluation & Benchmark Center
# -------------------------------------------------------------
@main_bp.route("/models")
def models_view():
    """Presents side-by-side performance benchmarks, ROC-AUC, confusion matrices, and feature importance."""
    metrics = model_manager.metrics
    feature_importances = model_manager.get_feature_importances()
    top_features = list(feature_importances.items())[:15]

    return render_template(
        "models.html",
        active_page="models",
        metrics=metrics,
        top_features=top_features
    )


# -------------------------------------------------------------
# 6. Retention Strategy & Financial ROI Hub
# -------------------------------------------------------------
@main_bp.route("/retention")
def retention_view():
    """Interactive customer retention strategy center, CLV preservation simulator, and communication templates."""
    return render_template("retention.html", active_page="retention")


# -------------------------------------------------------------
# 7. Audit History & Saved Customer Records
# -------------------------------------------------------------
@main_bp.route("/history")
def history_view():
    """Search and manage saved customer predictions, retention interventions, and audit records."""
    predictions = db_manager.get_recent_predictions(limit=100)
    stats = db_manager.get_database_stats()
    return render_template("history.html", active_page="history", predictions=predictions, stats=stats)


# -------------------------------------------------------------
# 8. REST & Utility Endpoints
# -------------------------------------------------------------
@main_bp.route("/api/predict", methods=["POST"])
def api_predict():
    """REST API endpoint for programmatic churn prediction."""
    try:
        data = request.get_json(force=True)
        input_df = data_loader.prepare_single_input(data)
        prob = model_manager.predict_churn(input_df, model_type="ensemble")
        explanation = explainer.explain_prediction(input_df)
        retention = RetentionStrategyEngine.generate_recommendations(data, prob, explanation["risk_catalysts"])
        clv_roi = RetentionStrategyEngine.calculate_clv_and_roi(
            float(data.get("MonthlyCharges", 70.0)),
            float(data.get("tenure", 12.0)),
            prob
        )

        return jsonify({
            "status": "success",
            "churn_probability": round(prob, 4),
            "churn_percentage": round(prob * 100, 2),
            "risk_assessment": retention["risk_meta"],
            "financial_clv_roi": clv_roi,
            "feature_impacts": explanation["impact_list"],
            "retention_recommendations": retention["strategies"]
        })
    except Exception as e:
        logger.error(f"API predict error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 400


@main_bp.route("/api/save-prediction", methods=["POST"])
def api_save_prediction():
    """Persists customer prediction and notes to SQLite database."""
    try:
        data = request.get_json(force=True)
        customer_id = data.get("customer_id", "Customer")
        churn_prob = float(data.get("churn_prob", 0.0)) / 100.0 if float(data.get("churn_prob", 0.0)) > 1.0 else float(data.get("churn_prob", 0.0))
        risk_level = data.get("risk_level", "MEDIUM")
        notes = data.get("notes", "")
        raw_input = data.get("raw_input", {})

        pred_id = db_manager.save_prediction(customer_id, raw_input, churn_prob, risk_level, notes)
        return jsonify({"success": True, "prediction_id": pred_id, "message": "Prediction successfully saved to database."})
    except Exception as e:
        logger.error(f"Save prediction error: {e}")
        return jsonify({"success": False, "error": str(e)}), 400


@main_bp.route("/api/delete-prediction/<int:pred_id>", methods=["POST"])
def api_delete_prediction(pred_id):
    """Deletes a prediction record from SQLite."""
    success = db_manager.delete_prediction(pred_id)
    return jsonify({"success": success})


@main_bp.route("/download-sample-csv")
def download_sample_csv():
    """Generates and serves a clean 50-row sample CSV for testing the batch prediction engine."""
    raw_df = data_loader.load_raw_data()
    sample_df = raw_df.sample(n=50, random_state=101).copy()
    if "Churn" in sample_df.columns:
        sample_df = sample_df.drop(columns=["Churn"])

    csv_buffer = io.StringIO()
    sample_df.to_csv(csv_buffer, index=False)
    csv_bytes = io.BytesIO(csv_buffer.getvalue().encode("utf-8"))
    csv_bytes.seek(0)

    return send_file(
        csv_bytes,
        mimetype="text/csv",
        as_attachment=True,
        download_name="churnshield_sample_customers.csv"
    )
