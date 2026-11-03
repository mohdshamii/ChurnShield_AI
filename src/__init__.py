"""
ChurnShield AI Core Package
Modular, industry-standard customer retention and predictive intelligence framework.
"""

from src.data_loader import DataLoader
from src.models import ModelManager
from src.explainability import ModelExplainer
from src.database import DatabaseManager
from src.retention_strategy import RetentionStrategyEngine
from src.analytics import AnalyticsEngine
from src.monitoring import ModelMonitor
from src.profiler import DatasetProfiler
from src.reporting import ReportGenerator

__version__ = "2.0.0"
__author__ = "Mohd Shami"

__all__ = [
    "DataLoader",
    "ModelManager",
    "ModelExplainer",
    "DatabaseManager",
    "RetentionStrategyEngine",
    "AnalyticsEngine",
    "ModelMonitor",
    "DatasetProfiler",
    "ReportGenerator"
]
