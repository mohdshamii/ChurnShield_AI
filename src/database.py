"""
ChurnShield AI - SQLite Persistence & Data Repository
Handles storage of single predictions, batch runs, customer notes,
audit histories, campaigns, experiments, and model monitoring logs.
"""

import json
import logging
import os
import sqlite3
from datetime import datetime
from typing import Dict, Any, List, Optional
import pandas as pd

from src import config

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages SQLite database connections and operations for ChurnShield AI."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or config.DB_PATH
        self.init_db()

    def get_connection(self) -> sqlite3.Connection:
        """Returns a configured SQLite connection with row factory."""
        conn = sqlite3.connect(self.db_path, timeout=15.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        return conn

    def init_db(self) -> None:
        """Initializes required tables if they don't already exist."""
        with self.get_connection() as conn:
            cur = conn.cursor()

            # 1. Users table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    password TEXT NOT NULL,
                    role TEXT DEFAULT 'Analyst',
                    email TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            # 2. Predictions table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER DEFAULT 1,
                    customer_id TEXT DEFAULT 'Unknown',
                    prediction_data TEXT NOT NULL,
                    churn_prob REAL NOT NULL,
                    risk_level TEXT NOT NULL,
                    notes TEXT DEFAULT '',
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            # Schema Migration Check for predictions table
            cur.execute("PRAGMA table_info(predictions);")
            existing_cols = [col[1] for col in cur.fetchall()]
            if "customer_id" not in existing_cols:
                cur.execute("ALTER TABLE predictions ADD COLUMN customer_id TEXT DEFAULT 'Unknown';")
            if "risk_level" not in existing_cols:
                cur.execute("ALTER TABLE predictions ADD COLUMN risk_level TEXT DEFAULT 'MEDIUM';")

            # 3. Batch history table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS batch_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT NOT NULL,
                    total_records INTEGER NOT NULL,
                    high_risk_count INTEGER NOT NULL,
                    medium_risk_count INTEGER NOT NULL,
                    low_risk_count INTEGER NOT NULL,
                    avg_churn_prob REAL NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            # 4. Retention interventions table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS retention_interventions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    prediction_id INTEGER,
                    customer_id TEXT NOT NULL,
                    strategy_applied TEXT NOT NULL,
                    discount_cost REAL DEFAULT 0.0,
                    status TEXT DEFAULT 'Scheduled',
                    scheduled_date TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (prediction_id) REFERENCES predictions(id)
                );
            """)

            # 5. Marketing & Retention Campaigns table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS campaigns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    target_risk_level TEXT NOT NULL,
                    offer_name TEXT NOT NULL,
                    discount_pct REAL DEFAULT 15.0,
                    status TEXT DEFAULT 'Active',
                    customers_targeted INTEGER DEFAULT 0,
                    retained_count INTEGER DEFAULT 0,
                    budget REAL DEFAULT 5000.0,
                    revenue_saved REAL DEFAULT 0.0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            # 6. AutoML Experiments Registry
            cur.execute("""
                CREATE TABLE IF NOT EXISTS experiments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_name TEXT NOT NULL,
                    model_type TEXT NOT NULL,
                    accuracy REAL,
                    roc_auc REAL,
                    f1_score REAL,
                    recall REAL,
                    hyperparams_json TEXT,
                    status TEXT DEFAULT 'Completed',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            # 7. Model Monitoring & Data Drift Logs
            cur.execute("""
                CREATE TABLE IF NOT EXISTS drift_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metric_name TEXT NOT NULL,
                    feature_name TEXT NOT NULL,
                    drift_value REAL NOT NULL,
                    threshold REAL NOT NULL,
                    status TEXT NOT NULL,
                    details TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            # Seed default campaigns if table is empty
            cur.execute("SELECT COUNT(*) FROM campaigns;")
            if cur.fetchone()[0] == 0:
                cur.execute("""
                    INSERT INTO campaigns (name, target_risk_level, offer_name, discount_pct, status, customers_targeted, retained_count, budget, revenue_saved)
                    VALUES 
                    ('Q4 Fiber Retention Blitz', 'HIGH', 'Free Speed Boost + 20% Off 1-Year', 20.0, 'Active', 142, 88, 7100.0, 75600.0),
                    ('Month-to-Month Annual Upgrade', 'MEDIUM', 'One-Month Free on Annual Contract', 10.0, 'Active', 310, 195, 9300.0, 140400.0),
                    ('Senior Loyalty Tech Support Bundle', 'HIGH', 'Free 24/7 Priority Tech Support', 15.0, 'Active', 95, 62, 3800.0, 48200.0)
                """)

            conn.commit()
            logger.info("Database schema initialized and verified.")

    def save_prediction(
        self,
        customer_id: str,
        input_data: Dict[str, Any],
        churn_prob: float,
        risk_level: str,
        notes: str = "",
        user_id: int = 1
    ) -> int:
        """Saves a single customer prediction record."""
        with self.get_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO predictions (user_id, customer_id, prediction_data, churn_prob, risk_level, notes, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                user_id,
                customer_id,
                json.dumps(input_data),
                float(churn_prob),
                risk_level,
                notes,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ))
            conn.commit()
            return cur.lastrowid

    def get_recent_predictions(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieves most recent prediction logs."""
        with self.get_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT id, customer_id, prediction_data, churn_prob, risk_level, notes, timestamp
                FROM predictions
                ORDER BY id DESC
                LIMIT ?
            """, (limit,))
            rows = cur.fetchall()

            results = []
            for r in rows:
                try:
                    data = json.loads(r["prediction_data"])
                except Exception:
                    data = {}
                results.append({
                    "id": r["id"],
                    "customer_id": r["customer_id"],
                    "churn_prob": round(float(r["churn_prob"]) * 100, 1),
                    "risk_level": r["risk_level"],
                    "notes": r["notes"],
                    "timestamp": r["timestamp"],
                    "tenure": data.get("tenure", "N/A"),
                    "monthly_charges": data.get("MonthlyCharges", data.get("monthly_charges", "N/A")),
                    "contract": data.get("Contract", data.get("contract", "N/A"))
                })
            return results

    def get_customer_history(self, customer_id: str) -> List[Dict[str, Any]]:
        """Retrieves all historical evaluations for a specific customer ID."""
        with self.get_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT id, customer_id, prediction_data, churn_prob, risk_level, notes, timestamp
                FROM predictions
                WHERE customer_id = ?
                ORDER BY id DESC
            """, (customer_id,))
            rows = cur.fetchall()

            results = []
            for r in rows:
                try:
                    data = json.loads(r["prediction_data"])
                except Exception:
                    data = {}
                results.append({
                    "id": r["id"],
                    "customer_id": r["customer_id"],
                    "churn_prob": round(float(r["churn_prob"]) * 100, 1),
                    "risk_level": r["risk_level"],
                    "notes": r["notes"],
                    "timestamp": r["timestamp"],
                    "raw_data": data
                })
            return results

    def get_customer_360_record(self, customer_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves latest profile and summary for a customer."""
        records = self.get_customer_history(customer_id)
        if not records:
            return None
        latest = records[0]
        return {
            "customer_id": customer_id,
            "latest_prediction": latest,
            "total_evaluations": len(records),
            "history": records
        }

    def add_note_to_prediction(self, prediction_id: int, notes: str) -> bool:
        """Appends or updates a note on an existing prediction record."""
        with self.get_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
                UPDATE predictions SET notes = ? WHERE id = ?
            """, (notes, prediction_id))
            conn.commit()
            return cur.rowcount > 0

    def delete_prediction(self, prediction_id: int) -> bool:
        """Deletes a prediction record by ID."""
        with self.get_connection() as conn:
            cur = conn.cursor()
            cur.execute("DELETE FROM predictions WHERE id = ?", (prediction_id,))
            conn.commit()
            return cur.rowcount > 0

    def save_batch_run(
        self,
        filename: str,
        total_records: int,
        high_risk_count: int,
        medium_risk_count: int,
        low_risk_count: int,
        avg_prob: float
    ) -> int:
        """Logs a completed batch prediction execution."""
        with self.get_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO batch_history (
                    filename, total_records, high_risk_count,
                    medium_risk_count, low_risk_count, avg_churn_prob, timestamp
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                filename, total_records, high_risk_count,
                medium_risk_count, low_risk_count, float(avg_prob),
                datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ))
            conn.commit()
            return cur.lastrowid

    def get_batch_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Retrieves recent batch execution logs."""
        with self.get_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT id, filename, total_records, high_risk_count, medium_risk_count, low_risk_count, avg_churn_prob, timestamp
                FROM batch_history
                ORDER BY id DESC
                LIMIT ?
            """, (limit,))
            return [dict(r) for r in cur.fetchall()]

    # -------------------------------------------------------------
    # Campaigns Management
    # -------------------------------------------------------------
    def get_campaigns(self) -> List[Dict[str, Any]]:
        """Retrieves all retention campaigns."""
        with self.get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT * FROM campaigns ORDER BY id DESC")
            return [dict(r) for r in cur.fetchall()]

    def create_campaign(
        self,
        name: str,
        target_risk_level: str,
        offer_name: str,
        discount_pct: float,
        budget: float = 5000.0,
        customers_targeted: int = 0
    ) -> int:
        """Creates a new retention campaign."""
        with self.get_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO campaigns (name, target_risk_level, offer_name, discount_pct, budget, customers_targeted, status)
                VALUES (?, ?, ?, ?, ?, ?, 'Active')
            """, (name, target_risk_level, offer_name, float(discount_pct), float(budget), int(customers_targeted)))
            conn.commit()
            return cur.lastrowid

    def update_campaign_status(self, campaign_id: int, status: str) -> bool:
        """Updates campaign execution status."""
        with self.get_connection() as conn:
            cur = conn.cursor()
            cur.execute("UPDATE campaigns SET status = ? WHERE id = ?", (status, campaign_id))
            conn.commit()
            return cur.rowcount > 0

    # -------------------------------------------------------------
    # AutoML Experiments Registry
    # -------------------------------------------------------------
    def log_experiment(
        self,
        experiment_name: str,
        model_type: str,
        accuracy: float,
        roc_auc: float,
        f1_score: float,
        recall: float,
        hyperparams: Dict[str, Any]
    ) -> int:
        """Logs an AutoML training run into the experiments registry."""
        with self.get_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO experiments (experiment_name, model_type, accuracy, roc_auc, f1_score, recall, hyperparams_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                experiment_name, model_type, float(accuracy), float(roc_auc),
                float(f1_score), float(recall), json.dumps(hyperparams)
            ))
            conn.commit()
            return cur.lastrowid

    def get_experiments(self, limit: int = 25) -> List[Dict[str, Any]]:
        """Retrieves recent experiments from registry."""
        with self.get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT * FROM experiments ORDER BY id DESC LIMIT ?", (limit,))
            return [dict(r) for r in cur.fetchall()]

    # -------------------------------------------------------------
    # Model Monitoring & Drift Logging
    # -------------------------------------------------------------
    def log_drift_check(
        self,
        metric_name: str,
        feature_name: str,
        drift_value: float,
        threshold: float,
        status: str,
        details: str = ""
    ) -> int:
        """Logs a data drift or prediction drift check."""
        with self.get_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO drift_logs (metric_name, feature_name, drift_value, threshold, status, details)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (metric_name, feature_name, float(drift_value), float(threshold), status, details))
            conn.commit()
            return cur.lastrowid

    def get_drift_logs(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieves recent drift logs."""
        with self.get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT * FROM drift_logs ORDER BY id DESC LIMIT ?", (limit,))
            return [dict(r) for r in cur.fetchall()]

    def get_database_stats(self) -> Dict[str, Any]:
        """Calculates global aggregates from SQLite database."""
        with self.get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM predictions;")
            total_predictions = cur.fetchone()[0]

            cur.execute("SELECT COUNT(*) FROM predictions WHERE risk_level = 'HIGH';")
            high_risk_count = cur.fetchone()[0]

            cur.execute("SELECT COUNT(*) FROM predictions WHERE risk_level = 'MEDIUM';")
            medium_risk_count = cur.fetchone()[0]

            cur.execute("SELECT COUNT(*) FROM predictions WHERE risk_level = 'LOW';")
            low_risk_count = cur.fetchone()[0]

            cur.execute("SELECT AVG(churn_prob) FROM predictions;")
            avg_prob_row = cur.fetchone()[0]
            avg_churn_prob = round(float(avg_prob_row) * 100, 1) if avg_prob_row is not None else 0.0

            cur.execute("SELECT COUNT(*) FROM batch_history;")
            total_batch_runs = cur.fetchone()[0]

            cur.execute("SELECT COUNT(*) FROM campaigns WHERE status = 'Active';")
            active_campaigns = cur.fetchone()[0]

            return {
                "total_predictions": total_predictions,
                "high_risk_count": high_risk_count,
                "medium_risk_count": medium_risk_count,
                "low_risk_count": low_risk_count,
                "avg_churn_prob": avg_churn_prob,
                "total_batch_runs": total_batch_runs,
                "active_campaigns": active_campaigns
            }
