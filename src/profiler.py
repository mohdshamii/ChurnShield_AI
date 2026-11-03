"""
ChurnShield AI - Data Intelligence & Dataset Profiling Engine
Performs automatic exploratory data profiling:
- Missing value audit & column missingness
- Duplicate row detection
- Statistical distributions (mean, std, min, quartiles, max)
- Outlier detection using the Interquartile Range (IQR) technique
- Automated Data Quality Scoring (0 - 100%) and Preprocessing Roadmap
"""

import logging
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DatasetProfiler:
    """Profiles datasets and computes comprehensive data quality metrics."""

    @staticmethod
    def profile_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Executes end-to-end data intelligence audit on any pandas DataFrame.
        """
        n_rows, n_cols = df.shape
        if n_rows == 0:
            return {"error": "Empty dataset cannot be profiled."}

        # 1. Column classifications
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category', 'string']).columns.tolist()

        # 2. Missing value audit
        null_counts = df.isnull().sum()
        total_cells = n_rows * n_cols
        total_missing_cells = int(null_counts.sum())
        missing_rate_pct = round((total_missing_cells / total_cells) * 100, 2)

        missing_details = []
        for col in df.columns:
            cnt = int(null_counts[col])
            if cnt > 0:
                missing_details.append({
                    "column": col,
                    "missing_count": cnt,
                    "missing_pct": round((cnt / n_rows) * 100, 2),
                    "dtype": str(df[col].dtype)
                })

        # 3. Duplicate row check
        duplicate_count = int(df.duplicated().sum())
        duplicate_pct = round((duplicate_count / n_rows) * 100, 2)

        # 4. Outlier detection on numeric features using IQR
        outlier_summary = []
        for col in numeric_cols:
            series = df[col].dropna()
            if len(series) > 10:
                q25 = float(series.quantile(0.25))
                q75 = float(series.quantile(0.75))
                iqr = q75 - q25
                lower_bound = q25 - 1.5 * iqr
                upper_bound = q75 + 1.5 * iqr
                outliers = int(((series < lower_bound) | (series > upper_bound)).sum())
                outlier_summary.append({
                    "column": col,
                    "outlier_count": outliers,
                    "outlier_pct": round((outliers / len(series)) * 100, 2),
                    "q25": round(q25, 2),
                    "median": round(float(series.median()), 2),
                    "q75": round(q75, 2)
                })

        # 5. Numeric statistics
        num_stats = []
        for col in numeric_cols:
            s = df[col].dropna()
            if len(s) > 0:
                num_stats.append({
                    "column": col,
                    "mean": round(float(s.mean()), 2),
                    "std": round(float(s.std()), 2) if len(s) > 1 else 0.0,
                    "min": round(float(s.min()), 2),
                    "median": round(float(s.median()), 2),
                    "max": round(float(s.max()), 2)
                })

        # 6. Overall Data Quality Score (0 - 100%)
        quality_score = 100
        quality_score -= min(int(missing_rate_pct * 3), 40)
        quality_score -= min(int(duplicate_pct * 2), 20)
        high_outlier_cols = sum(1 for o in outlier_summary if o["outlier_pct"] > 5)
        quality_score -= min(high_outlier_cols * 5, 20)
        quality_score = max(quality_score, 0)

        # 7. Actionable Recommendations
        recommendations = []
        if total_missing_cells > 0:
            recommendations.append(f"Impute {total_missing_cells} missing values across {len(missing_details)} columns.")
        if duplicate_count > 0:
            recommendations.append(f"Deduplicate {duplicate_count} identical records prior to model ingestion.")
        if high_outlier_cols > 0:
            recommendations.append(f"Apply robust scaling or log-transforms to {high_outlier_cols} heavy-tailed numeric features.")
        if not recommendations:
            recommendations.append("Dataset adheres to high-quality standards. Ready for production pipeline.")

        return {
            "total_rows": n_rows,
            "total_columns": n_cols,
            "numeric_count": len(numeric_cols),
            "categorical_count": len(categorical_cols),
            "total_missing_cells": total_missing_cells,
            "missing_rate_pct": missing_rate_pct,
            "missing_columns": sorted(missing_details, key=lambda x: x["missing_count"], reverse=True),
            "duplicate_count": duplicate_count,
            "duplicate_pct": duplicate_pct,
            "outliers": outlier_summary,
            "numeric_stats": num_stats,
            "quality_score": quality_score,
            "quality_rating": "EXCELLENT" if quality_score >= 85 else ("GOOD" if quality_score >= 70 else "NEEDS_CLEANING"),
            "recommendations": recommendations
        }

