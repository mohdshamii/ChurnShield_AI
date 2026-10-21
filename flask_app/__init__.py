"""
ChurnShield AI - Flask Application Package
"""

import os
from flask import Flask
from src import config

def create_app():
    """Application factory for ChurnShield AI Flask web application."""
    app = Flask(__name__,
                template_folder="templates",
                static_folder="static")
    
    app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "churnshield-ai-production-secret-key-2026")
    app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32MB max batch upload

    # Register blueprints / routes
    from flask_app.routes import main_bp
    app.register_blueprint(main_bp)

    return app
