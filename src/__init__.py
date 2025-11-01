"""
CortexX - Enterprise Sales Forecasting Platform
Professional sales forecasting and demand prediction system.
"""

__version__ = "1.0.0"
__author__ = "CortexX Team"
__email__ = "info@cortexx.ai"

from src.data.collection import DataCollector
from src.data.preprocessing import DataPreprocessor
from src.data.exploration import DataExplorer
from src.features.engineering import FeatureEngineer
from src.features.selection import FeatureSelector
from src.models.training import ModelTrainer
from src.models.evaluation import ModelEvaluator
from src.visualization.dashboard import VisualizationEngine

__all__ = [
    "DataCollector",
    "DataPreprocessor", 
    "DataExplorer",
    "FeatureEngineer",
    "FeatureSelector",
    "ModelTrainer",
    "ModelEvaluator",
    "VisualizationEngine",
]