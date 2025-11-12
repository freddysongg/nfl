"""
Base model class for NFL prediction system.
Provides abstract base class for all ML models with consistent API.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import json
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import polars as pl
import joblib
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ModelMetadata:
    """Metadata for tracking model information"""

    model_id: str
    model_name: str
    model_type: str  # 'regression', 'classification', 'multioutput'
    version: str
    target_variable: str
    features: List[str]
    hyperparameters: Dict[str, Any]

    # Training metadata
    training_date: str = field(default_factory=lambda: datetime.now().isoformat())
    training_samples: int = 0
    training_duration_seconds: float = 0.0

    # Performance metrics
    metrics: Dict[str, float] = field(default_factory=dict)
    cv_scores: Optional[List[float]] = None

    # Model configuration
    feature_engineering_config: Dict[str, Any] = field(default_factory=dict)
    preprocessing_steps: List[str] = field(default_factory=list)

    # MLflow tracking
    mlflow_run_id: Optional[str] = None
    mlflow_experiment_id: Optional[str] = None

    # Status
    status: str = "training"  # 'training', 'completed', 'deployed', 'deprecated'
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary"""
        return asdict(self)

    def to_json(self) -> str:
        """Convert metadata to JSON string"""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelMetadata":
        """Create metadata from dictionary"""
        return cls(**data)

    @classmethod
    def from_json(cls, json_str: str) -> "ModelMetadata":
        """Create metadata from JSON string"""
        data = json.loads(json_str)
        return cls.from_dict(data)


class BaseModel(ABC):
    """
    Abstract base class for all NFL prediction models.

    Provides consistent API for training, prediction, evaluation, and persistence.
    Includes optional MLflow integration for experiment tracking.
    """

    def __init__(
        self,
        model_name: str,
        model_type: str,
        target_variable: str,
        features: List[str],
        hyperparameters: Optional[Dict[str, Any]] = None,
        enable_mlflow: bool = False,
        random_seed: int = 42,
    ):
        """
        Initialize base model.

        Args:
            model_name: Name of the model
            model_type: Type of model ('regression', 'classification', 'multioutput')
            target_variable: Target variable to predict
            features: List of feature column names
            hyperparameters: Model hyperparameters
            enable_mlflow: Whether to enable MLflow tracking
            random_seed: Random seed for reproducibility
        """
        self.model_name = model_name
        self.model_type = model_type
        self.target_variable = target_variable
        self.features = features
        self.hyperparameters = hyperparameters or {}
        self.enable_mlflow = enable_mlflow
        self.random_seed = random_seed

        # Initialize model and metadata
        self.model = None
        self.metadata: Optional[ModelMetadata] = None
        self.is_trained = False

        # Feature importance tracking
        self.feature_importance_: Optional[pd.DataFrame] = None

        logger.info(f"Initialized {model_name} model (type: {model_type})")

    @abstractmethod
    def _build_model(self) -> Any:
        """
        Build the underlying model instance.

        Returns:
            Model instance (sklearn, xgboost, lightgbm, etc.)
        """
        pass

    @abstractmethod
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> "BaseModel":
        """
        Train the model.

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)

        Returns:
            self for method chaining
        """
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Features to predict on

        Returns:
            Predictions array
        """
        pass

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities (for classification models).

        Args:
            X: Features to predict on

        Returns:
            Probability predictions array

        Raises:
            NotImplementedError: If model doesn't support probability predictions
        """
        if not hasattr(self.model, "predict_proba"):
            raise NotImplementedError(
                f"{self.model_name} does not support probability predictions"
            )
        return self.model.predict_proba(X[self.features])

    def evaluate(
        self, X_test: pd.DataFrame, y_test: pd.Series
    ) -> Dict[str, float]:
        """
        Evaluate model performance.

        Args:
            X_test: Test features
            y_test: Test targets

        Returns:
            Dictionary of evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")

        predictions = self.predict(X_test)

        metrics = {}

        if self.model_type == "regression":
            metrics = self._evaluate_regression(y_test, predictions)
        elif self.model_type == "classification":
            metrics = self._evaluate_classification(y_test, predictions)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        logger.info(f"Evaluation metrics: {metrics}")
        return metrics

    def _evaluate_regression(
        self, y_true: pd.Series, y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Calculate regression metrics"""
        return {
            "mse": mean_squared_error(y_true, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
            "mae": mean_absolute_error(y_true, y_pred),
            "r2": r2_score(y_true, y_pred),
        }

    def _evaluate_classification(
        self, y_true: pd.Series, y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Calculate classification metrics"""
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
            "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
            "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        }

    def cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cv: int = 5,
        scoring: Optional[str] = None,
    ) -> Tuple[float, float]:
        """
        Perform cross-validation.

        Args:
            X: Features
            y: Targets
            cv: Number of cross-validation folds
            scoring: Scoring metric (sklearn compatible)

        Returns:
            Tuple of (mean_score, std_score)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before cross-validation")

        if scoring is None:
            scoring = "neg_mean_squared_error" if self.model_type == "regression" else "accuracy"

        scores = cross_val_score(
            self.model, X[self.features], y, cv=cv, scoring=scoring
        )

        logger.info(
            f"Cross-validation ({cv}-fold): mean={scores.mean():.4f}, std={scores.std():.4f}"
        )

        return scores.mean(), scores.std()

    def extract_feature_importance(self) -> pd.DataFrame:
        """
        Extract feature importance from trained model.

        Returns:
            DataFrame with features and their importance scores

        Raises:
            ValueError: If model doesn't support feature importance
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before extracting feature importance")

        importance_attrs = ["feature_importances_", "coef_"]
        importance_values = None

        for attr in importance_attrs:
            if hasattr(self.model, attr):
                importance_values = getattr(self.model, attr)
                break

        if importance_values is None:
            raise ValueError(f"{self.model_name} does not support feature importance")

        # Handle coefficient matrices (multiple classes)
        if importance_values.ndim > 1:
            importance_values = np.abs(importance_values).mean(axis=0)

        self.feature_importance_ = pd.DataFrame({
            "feature": self.features,
            "importance": importance_values,
        }).sort_values("importance", ascending=False)

        return self.feature_importance_

    def get_top_features(self, n: int = 10) -> List[str]:
        """
        Get top N most important features.

        Args:
            n: Number of top features to return

        Returns:
            List of top feature names
        """
        if self.feature_importance_ is None:
            self.extract_feature_importance()

        return self.feature_importance_.head(n)["feature"].tolist()

    @abstractmethod
    def save_model(self, path: Path) -> None:
        """
        Save model to disk.

        Args:
            path: Path to save the model
        """
        pass

    @abstractmethod
    def load_model(self, path: Path) -> "BaseModel":
        """
        Load model from disk.

        Args:
            path: Path to load the model from

        Returns:
            self for method chaining
        """
        pass

    def save_metadata(self, path: Path) -> None:
        """
        Save model metadata to JSON file.

        Args:
            path: Path to save metadata (will append .json)
        """
        if self.metadata is None:
            raise ValueError("No metadata to save")

        metadata_path = path.parent / f"{path.stem}_metadata.json"
        metadata_path.write_text(self.metadata.to_json())
        logger.info(f"Saved metadata to {metadata_path}")

    def load_metadata(self, path: Path) -> None:
        """
        Load model metadata from JSON file.

        Args:
            path: Path to load metadata from (will append .json)
        """
        metadata_path = path.parent / f"{path.stem}_metadata.json"

        if not metadata_path.exists():
            logger.warning(f"Metadata file not found: {metadata_path}")
            return

        metadata_json = metadata_path.read_text()
        self.metadata = ModelMetadata.from_json(metadata_json)
        logger.info(f"Loaded metadata from {metadata_path}")

    def _create_metadata(
        self,
        training_samples: int,
        training_duration: float,
        metrics: Dict[str, float],
        cv_scores: Optional[List[float]] = None,
    ) -> ModelMetadata:
        """
        Create metadata object after training.

        Args:
            training_samples: Number of training samples
            training_duration: Training duration in seconds
            metrics: Performance metrics
            cv_scores: Cross-validation scores (optional)

        Returns:
            ModelMetadata instance
        """
        import uuid

        model_id = str(uuid.uuid4())

        return ModelMetadata(
            model_id=model_id,
            model_name=self.model_name,
            model_type=self.model_type,
            version="1.0.0",
            target_variable=self.target_variable,
            features=self.features,
            hyperparameters=self.hyperparameters,
            training_samples=training_samples,
            training_duration_seconds=training_duration,
            metrics=metrics,
            cv_scores=cv_scores,
            status="completed",
        )

    def _log_to_mlflow(self, metrics: Dict[str, float]) -> None:
        """
        Log model and metrics to MLflow.

        Args:
            metrics: Performance metrics to log
        """
        if not self.enable_mlflow:
            return

        try:
            import mlflow

            with mlflow.start_run():
                # Log parameters
                mlflow.log_params(self.hyperparameters)

                # Log metrics
                mlflow.log_metrics(metrics)

                # Log model
                mlflow.sklearn.log_model(self.model, "model")

                # Store run ID
                if self.metadata:
                    self.metadata.mlflow_run_id = mlflow.active_run().info.run_id

                logger.info(f"Logged to MLflow: run_id={mlflow.active_run().info.run_id}")

        except ImportError:
            logger.warning("MLflow not available, skipping logging")
        except Exception as e:
            logger.error(f"Error logging to MLflow: {e}")

    def __repr__(self) -> str:
        """String representation"""
        status = "trained" if self.is_trained else "untrained"
        return f"{self.model_name}({status}, type={self.model_type}, target={self.target_variable})"
