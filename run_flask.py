"""
ChurnShield AI - Flask Web Application Runner
Launches the industry-standard Flask intelligence dashboard and REST API.
"""

import os
from flask_app import create_app

app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n========================================================")
    print(f"[START] ChurnShield AI - Flask Intelligence Web Dashboard")
    print(f"[SERVER] Serving on: http://127.0.0.1:{port}")
    print(f"[MODELS] ML Engine: XGBoost + Random Forest + Logistic Regression + Soft Voting Ensemble")
    print(f"[DATABASE] Persistence: SQLite (churn_prediction.db)")
    print(f"========================================================\n")
    app.run(host="0.0.0.0", port=port, debug=False)
