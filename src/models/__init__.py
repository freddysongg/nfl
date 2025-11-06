"""
NFL Prediction System - ML Models Module

This module provides the machine learning infrastructure for the NFL prediction system.
"""

from src.models.base import BaseModel, ModelMetadata
from src.models.model_registry import ModelRegistry, ModelVersion

__all__ = [
    "BaseModel",
    "ModelMetadata",
    "ModelRegistry",
    "ModelVersion",
]

__version__ = "1.0.0"
