"""
ChurnShield AI - Flask Web Application Runner
Launches the industry-standard Flask intelligence dashboard and REST API.
"""

import os
from flask_app import create_app

app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    host = os.environ.get("HOST", "127.0.0.1")
    debug = os.environ.get("FLASK_DEBUG", "True").lower() in ("true", "1", "t")

    print(f"\n========================================================")
    print(f"[START] ChurnShield AI - Flask Intelligence Web Dashboard")
    print(f"[SERVER] Serving on: http://{host}:{port}")
    print(f"[MODELS] ML Engine: XGBoost + Random Forest + Logistic Regression + Soft Voting Ensemble")
    print(f"[DATABASE] Persistence: SQLite (churn_prediction.db)")
    print(f"[DEBUG] Auto-Reload: {debug}")
    print(f"========================================================\n")
    app.run(host=host, port=port, debug=debug)
