"""
NFL Prediction System - ML Models Module

This module provides the machine learning infrastructure for the NFL prediction system.
"""

from src.models.base import BaseModel, ModelMetadata
from src.models.model_registry import ModelRegistry, ModelVersion
from src.models.xgboost_predictor import XGBoostPredictor

__all__ = [
    "BaseModel",
    "ModelMetadata",
    "ModelRegistry",
    "ModelVersion",
    "XGBoostPredictor",
]

__version__ = "1.0.0"
