"""
ChurnShield AI - SQLite Persistence & Data Repository
Handles storage of single predictions, batch runs, customer notes,
audit histories, and user sessions with parameterization and thread safety.
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
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        # Enable Write-Ahead Logging for better concurrent read/write performance
        conn.execute("PRAGMA journal_mode=WAL;")
        return conn

    def init_db(self) -> None:
        """Initializes required tables if they don't already exist."""
        with self.get_connection() as conn:
            cur = conn.cursor()

            # Users table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    password TEXT NOT NULL,
                    email TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            # Predictions table (expanded with customer_id and risk_level)
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

            # Schema Migration Check for pre-existing predictions table
            cur.execute("PRAGMA table_info(predictions);")
            existing_cols = [col[1] for col in cur.fetchall()]
            if "customer_id" not in existing_cols:
                cur.execute("ALTER TABLE predictions ADD COLUMN customer_id TEXT DEFAULT 'Unknown';")
                logger.info("Migrated predictions table: added customer_id column.")
            if "risk_level" not in existing_cols:
                cur.execute("ALTER TABLE predictions ADD COLUMN risk_level TEXT DEFAULT 'MEDIUM';")
                logger.info("Migrated predictions table: added risk_level column.")

            # Batch history table
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

            # Retention interventions table
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
                filename,
                total_records,
                high_risk_count,
                medium_risk_count,
                low_risk_count,
                float(avg_prob),
                datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ))
            conn.commit()
            return cur.lastrowid

    def get_database_stats(self) -> Dict[str, Any]:
        """Calculates aggregate statistics over the predictions history."""
        with self.get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*), AVG(churn_prob) FROM predictions")
            total_count, avg_prob = cur.fetchone()
            total_count = total_count or 0
            avg_prob = (avg_prob or 0.0) * 100.0

            cur.execute("SELECT COUNT(*) FROM predictions WHERE risk_level = 'HIGH'")
            high_count = cur.fetchone()[0] or 0

            cur.execute("SELECT COUNT(*) FROM predictions WHERE risk_level = 'MEDIUM'")
            med_count = cur.fetchone()[0] or 0

            cur.execute("SELECT COUNT(*) FROM predictions WHERE risk_level = 'LOW'")
            low_count = cur.fetchone()[0] or 0

            return {
                "total_predictions": total_count,
                "avg_churn_prob": round(avg_prob, 1),
                "high_risk_count": high_count,
                "medium_risk_count": med_count,
                "low_risk_count": low_count
            }
