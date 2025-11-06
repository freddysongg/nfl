"""
Model registry for NFL prediction system.
Provides versioning, storage, and retrieval of trained models.
"""

import json
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime

import joblib
import pandas as pd
from dataclasses import asdict

from src.database import NFLDatabase
from src.models.base import BaseModel, ModelMetadata

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelVersion:
    """Represents a versioned model in the registry"""

    def __init__(
        self,
        model_id: str,
        model_name: str,
        version: str,
        model_type: str,
        target_variable: str,
        model_path: str,
        metadata: Dict[str, Any],
        created_at: str,
        status: str = "active",
    ):
        self.model_id = model_id
        self.model_name = model_name
        self.version = version
        self.model_type = model_type
        self.target_variable = target_variable
        self.model_path = model_path
        self.metadata = metadata
        self.created_at = created_at
        self.status = status

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "model_id": self.model_id,
            "model_name": self.model_name,
            "version": self.version,
            "model_type": self.model_type,
            "target_variable": self.target_variable,
            "model_path": self.model_path,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelVersion":
        """Create from dictionary"""
        return cls(**data)

    def __repr__(self) -> str:
        return f"ModelVersion(name={self.model_name}, version={self.version}, status={self.status})"


class ModelRegistry:
    """
    Model registry for managing trained models.

    Provides CRUD operations for model storage, versioning, and retrieval.
    Integrates with DuckDB for metadata storage and filesystem for model artifacts.
    """

    def __init__(
        self,
        db: Optional[NFLDatabase] = None,
        models_dir: Optional[Path] = None,
    ):
        """
        Initialize model registry.

        Args:
            db: Database connection (creates new if not provided)
            models_dir: Directory to store model artifacts (defaults to ./models/)
        """
        self.db = db or NFLDatabase()
        self.models_dir = models_dir or Path("models")
        self.models_dir.mkdir(exist_ok=True, parents=True)

        logger.info(f"Initialized ModelRegistry with models_dir: {self.models_dir}")

    def register_model(
        self,
        model: BaseModel,
        version: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        description: str = "",
    ) -> str:
        """
        Register a trained model in the registry.

        Args:
            model: Trained model instance
            version: Semantic version (auto-increments if not provided)
            tags: Optional tags for categorization
            description: Model description

        Returns:
            model_id of the registered model

        Raises:
            ValueError: If model is not trained
        """
        if not model.is_trained:
            raise ValueError("Cannot register untrained model")

        if model.metadata is None:
            raise ValueError("Model must have metadata")

        # Auto-increment version if not provided
        if version is None:
            version = self._get_next_version(model.model_name)

        # Update metadata with version
        model.metadata.version = version

        # Generate unique model path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"{model.model_name}_v{version}_{timestamp}.joblib"
        model_path = self.models_dir / model_filename

        # Save model to filesystem
        model.save_model(model_path)
        model.save_metadata(model_path)

        # Store metadata in database
        metadata_dict = model.metadata.to_dict()
        metadata_json = json.dumps(metadata_dict)

        # Extract key metrics for quick access
        primary_metric = self._get_primary_metric(model.metadata.metrics, model.model_type)

        conn = self.db.connect()
        conn.execute(
            """
            INSERT INTO model_versions (
                model_id, model_name, version, model_type, target_variable,
                model_path, features, hyperparameters, metrics, primary_metric,
                training_samples, training_duration_seconds, training_date,
                status, description, tags, metadata_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                model.metadata.model_id,
                model.model_name,
                version,
                model.model_type,
                model.target_variable,
                str(model_path),
                json.dumps(model.features),
                json.dumps(model.hyperparameters),
                json.dumps(model.metadata.metrics),
                primary_metric,
                model.metadata.training_samples,
                model.metadata.training_duration_seconds,
                model.metadata.training_date,
                model.metadata.status,
                description,
                json.dumps(tags or {}),
                metadata_json,
                datetime.now().isoformat(),
            ],
        )

        logger.info(
            f"Registered model: {model.model_name} v{version} (id: {model.metadata.model_id})"
        )

        return model.metadata.model_id

    def load_model_by_id(self, model_id: str) -> BaseModel:
        """
        Load a model by its ID.

        Args:
            model_id: Unique model identifier

        Returns:
            Loaded model instance

        Raises:
            ValueError: If model not found
        """
        conn = self.db.connect()
        result = conn.execute(
            """
            SELECT model_path, model_name, model_type, target_variable, metadata_json
            FROM model_versions
            WHERE model_id = ?
            """,
            [model_id],
        ).fetchone()

        if not result:
            raise ValueError(f"Model not found: {model_id}")

        model_path, model_name, model_type, target_variable, metadata_json = result

        # Load model using joblib
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Note: This loads the raw model artifact
        # In practice, you'd need to reconstruct the BaseModel subclass
        logger.info(f"Loaded model: {model_name} (id: {model_id})")

        return self._load_model_from_path(model_path, metadata_json)

    def load_model_by_name(
        self, model_name: str, version: Optional[str] = None
    ) -> BaseModel:
        """
        Load a model by name and optional version.

        Args:
            model_name: Name of the model
            version: Semantic version (loads latest if not provided)

        Returns:
            Loaded model instance

        Raises:
            ValueError: If model not found
        """
        conn = self.db.connect()

        if version:
            query = """
                SELECT model_id, model_path, metadata_json
                FROM model_versions
                WHERE model_name = ? AND version = ?
            """
            params = [model_name, version]
        else:
            # Get latest version
            query = """
                SELECT model_id, model_path, metadata_json
                FROM model_versions
                WHERE model_name = ?
                ORDER BY created_at DESC
                LIMIT 1
            """
            params = [model_name]

        result = conn.execute(query, params).fetchone()

        if not result:
            raise ValueError(f"Model not found: {model_name} v{version or 'latest'}")

        model_id, model_path, metadata_json = result

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        logger.info(f"Loaded model: {model_name} v{version or 'latest'} (id: {model_id})")

        return self._load_model_from_path(model_path, metadata_json)

    def load_best_model(
        self,
        model_name: str,
        metric: str = "primary_metric",
        higher_is_better: bool = True,
    ) -> BaseModel:
        """
        Load the best performing model by metric.

        Args:
            model_name: Name of the model
            metric: Metric to optimize (default: primary_metric)
            higher_is_better: Whether higher metric values are better

        Returns:
            Best performing model instance

        Raises:
            ValueError: If no models found
        """
        conn = self.db.connect()

        order = "DESC" if higher_is_better else "ASC"
        result = conn.execute(
            f"""
            SELECT model_id, model_path, metadata_json, {metric}
            FROM model_versions
            WHERE model_name = ? AND status = 'active'
            ORDER BY {metric} {order}
            LIMIT 1
            """,
            [model_name],
        ).fetchone()

        if not result:
            raise ValueError(f"No active models found: {model_name}")

        model_id, model_path, metadata_json, metric_value = result

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        logger.info(
            f"Loaded best model: {model_name} (id: {model_id}, {metric}={metric_value:.4f})"
        )

        return self._load_model_from_path(model_path, metadata_json)

    def list_models(
        self,
        model_name: Optional[str] = None,
        model_type: Optional[str] = None,
        status: str = "active",
        limit: int = 100,
    ) -> List[ModelVersion]:
        """
        List models in the registry.

        Args:
            model_name: Filter by model name (optional)
            model_type: Filter by model type (optional)
            status: Filter by status (default: active)
            limit: Maximum number of results

        Returns:
            List of ModelVersion objects
        """
        conn = self.db.connect()

        query = "SELECT * FROM model_versions WHERE 1=1"
        params = []

        if model_name:
            query += " AND model_name = ?"
            params.append(model_name)

        if model_type:
            query += " AND model_type = ?"
            params.append(model_type)

        if status:
            query += " AND status = ?"
            params.append(status)

        query += f" ORDER BY created_at DESC LIMIT {limit}"

        results = conn.execute(query, params).fetchall()
        columns = [desc[0] for desc in conn.description]

        models = []
        for row in results:
            row_dict = dict(zip(columns, row))
            # Parse JSON fields
            row_dict["metadata"] = json.loads(row_dict.get("metadata_json", "{}"))
            models.append(ModelVersion.from_dict(row_dict))

        logger.info(f"Found {len(models)} models matching criteria")
        return models

    def promote_to_production(self, model_id: str) -> None:
        """
        Promote a model to production status.

        Args:
            model_id: Model identifier to promote

        Raises:
            ValueError: If model not found
        """
        conn = self.db.connect()

        # Check if model exists
        result = conn.execute(
            "SELECT model_name FROM model_versions WHERE model_id = ?",
            [model_id],
        ).fetchone()

        if not result:
            raise ValueError(f"Model not found: {model_id}")

        model_name = result[0]

        # Demote other production models
        conn.execute(
            """
            UPDATE model_versions
            SET status = 'active'
            WHERE model_name = ? AND status = 'production'
            """,
            [model_name],
        )

        # Promote this model
        conn.execute(
            """
            UPDATE model_versions
            SET status = 'production', updated_at = ?
            WHERE model_id = ?
            """,
            [datetime.now().isoformat(), model_id],
        )

        logger.info(f"Promoted model {model_id} to production")

    def deprecate_model(self, model_id: str) -> None:
        """
        Deprecate a model (soft delete).

        Args:
            model_id: Model identifier to deprecate
        """
        conn = self.db.connect()

        conn.execute(
            """
            UPDATE model_versions
            SET status = 'deprecated', updated_at = ?
            WHERE model_id = ?
            """,
            [datetime.now().isoformat(), model_id],
        )

        logger.info(f"Deprecated model {model_id}")

    def delete_model(self, model_id: str, delete_files: bool = False) -> None:
        """
        Delete a model from the registry.

        Args:
            model_id: Model identifier to delete
            delete_files: Whether to delete model files from disk

        Raises:
            ValueError: If model not found
        """
        conn = self.db.connect()

        # Get model path before deletion
        result = conn.execute(
            "SELECT model_path FROM model_versions WHERE model_id = ?",
            [model_id],
        ).fetchone()

        if not result:
            raise ValueError(f"Model not found: {model_id}")

        model_path = Path(result[0])

        # Delete from database
        conn.execute("DELETE FROM model_versions WHERE model_id = ?", [model_id])

        # Optionally delete files
        if delete_files and model_path.exists():
            model_path.unlink()
            metadata_path = model_path.parent / f"{model_path.stem}_metadata.json"
            if metadata_path.exists():
                metadata_path.unlink()
            logger.info(f"Deleted model files: {model_path}")

        logger.info(f"Deleted model {model_id} from registry")

    def compare_models(
        self, model_ids: List[str], metrics: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Compare multiple models by metrics.

        Args:
            model_ids: List of model IDs to compare
            metrics: List of metrics to compare (defaults to all)

        Returns:
            DataFrame with comparison results
        """
        conn = self.db.connect()

        placeholders = ",".join(["?" for _ in model_ids])
        query = f"""
            SELECT model_id, model_name, version, model_type,
                   metrics, primary_metric, training_date
            FROM model_versions
            WHERE model_id IN ({placeholders})
        """

        results = conn.execute(query, model_ids).fetchall()
        columns = [desc[0] for desc in conn.description]

        comparison_data = []
        for row in results:
            row_dict = dict(zip(columns, row))
            metrics_dict = json.loads(row_dict["metrics"])
            row_dict.update(metrics_dict)
            comparison_data.append(row_dict)

        df = pd.DataFrame(comparison_data)

        logger.info(f"Compared {len(model_ids)} models")
        return df

    def _get_next_version(self, model_name: str) -> str:
        """Auto-increment version number"""
        conn = self.db.connect()

        result = conn.execute(
            """
            SELECT version FROM model_versions
            WHERE model_name = ?
            ORDER BY created_at DESC
            LIMIT 1
            """,
            [model_name],
        ).fetchone()

        if not result:
            return "1.0.0"

        # Parse semantic version and increment
        current_version = result[0]
        major, minor, patch = map(int, current_version.split("."))
        return f"{major}.{minor}.{patch + 1}"

    def _get_primary_metric(
        self, metrics: Dict[str, float], model_type: str
    ) -> float:
        """Extract primary metric based on model type"""
        if model_type == "regression":
            return metrics.get("rmse", metrics.get("mse", 0.0))
        elif model_type == "classification":
            return metrics.get("f1", metrics.get("accuracy", 0.0))
        return 0.0

    def _load_model_from_path(
        self, model_path: Path, metadata_json: str
    ) -> BaseModel:
        """
        Load model from filesystem.

        Note: This is a placeholder. In practice, you'd need to:
        1. Determine the specific model class (e.g., XGBoostModel, RandomForestModel)
        2. Instantiate that class
        3. Load the underlying model artifact
        4. Restore metadata

        Args:
            model_path: Path to model file
            metadata_json: JSON string of metadata

        Returns:
            Loaded BaseModel instance (or raises NotImplementedError)
        """
        # This is a simplified implementation
        # Real implementation would need to handle different model types
        logger.warning(
            "Model loading is simplified. Implement concrete model loaders for production."
        )

        # For now, just load the joblib artifact
        model_artifact = joblib.load(model_path)

        # Parse metadata
        metadata_dict = json.loads(metadata_json)
        metadata = ModelMetadata.from_dict(metadata_dict)

        # In practice, you'd reconstruct the appropriate BaseModel subclass here
        # For now, return a placeholder
        raise NotImplementedError(
            "Model loading requires concrete model class implementation. "
            "Extend this method to handle your specific model types."
        )

    def __repr__(self) -> str:
        """String representation"""
        return f"ModelRegistry(models_dir={self.models_dir})"
