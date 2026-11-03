"""
ChurnShield AI - Primary Flask Web Application Entrypoint
Serves the Flask Intelligence Dashboard, Jinja2 Templates, and REST API.
"""

import os
from flask_app import create_app

# Primary Flask application instance
app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    host = os.environ.get("HOST", "127.0.0.1")
    # Default debug to True for responsive local development and auto-reloading
    debug = os.environ.get("FLASK_DEBUG", "True").lower() in ("true", "1", "t")

    print("\n========================================================")
    print(" [START] ChurnShield AI - Flask Intelligence Web Dashboard")
    print(f" [SERVER] Serving on: http://{host}:{port}")
    print(" [MODELS] ML Engine: XGBoost + Random Forest + Logistic Regression + Soft Voting Ensemble")
    print(" [DATABASE] Persistence: SQLite (churn_prediction.db)")
    print(f" [DEBUG] Auto-Reload: {debug}")
    print("========================================================\n")

    app.run(host=host, port=port, debug=debug)
