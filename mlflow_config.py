"""
MLflow configuration and utilities for NFL prediction system.
Provides experiment tracking, model registry, and artifact management.
"""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MLflowConfig:
    """
    MLflow configuration manager.

    Handles MLflow tracking URI, experiments, and artifacts storage.
    """

    def __init__(
        self,
        tracking_uri: Optional[str] = None,
        artifact_location: Optional[str] = None,
        experiment_name: str = "nfl_predictions",
    ):
        """
        Initialize MLflow configuration.

        Args:
            tracking_uri: MLflow tracking server URI (defaults to local ./mlruns)
            artifact_location: Artifact storage location
            experiment_name: Default experiment name
        """
        self.project_root = Path(__file__).parent
        self.mlruns_dir = self.project_root / "mlruns"
        self.artifacts_dir = self.project_root / "mlflow_artifacts"

        # Set tracking URI
        if tracking_uri is None:
            tracking_uri = f"file://{self.mlruns_dir}"

        self.tracking_uri = tracking_uri
        self.artifact_location = artifact_location
        self.experiment_name = experiment_name

        # Create directories
        self.mlruns_dir.mkdir(exist_ok=True)
        self.artifacts_dir.mkdir(exist_ok=True)

        logger.info(f"Initialized MLflow config: tracking_uri={self.tracking_uri}")

    def setup(self) -> None:
        """
        Setup MLflow tracking and experiment.

        Configures MLflow tracking URI and creates default experiment if needed.
        """
        try:
            import mlflow

            # Set tracking URI
            mlflow.set_tracking_uri(self.tracking_uri)
            logger.info(f"Set MLflow tracking URI: {self.tracking_uri}")

            # Create or get experiment
            experiment = mlflow.get_experiment_by_name(self.experiment_name)

            if experiment is None:
                experiment_id = mlflow.create_experiment(
                    self.experiment_name,
                    artifact_location=self.artifact_location,
                )
                logger.info(
                    f"Created MLflow experiment: {self.experiment_name} (id: {experiment_id})"
                )
            else:
                experiment_id = experiment.experiment_id
                logger.info(
                    f"Using existing experiment: {self.experiment_name} (id: {experiment_id})"
                )

            # Set default experiment
            mlflow.set_experiment(self.experiment_name)

        except ImportError:
            logger.warning("MLflow not installed. Install with: pip install mlflow")
        except Exception as e:
            logger.error(f"Error setting up MLflow: {e}")

    def create_experiment(
        self, name: str, tags: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Create a new MLflow experiment.

        Args:
            name: Experiment name
            tags: Optional tags for the experiment

        Returns:
            Experiment ID

        Raises:
            ImportError: If MLflow not installed
        """
        try:
            import mlflow

            experiment = mlflow.get_experiment_by_name(name)

            if experiment is not None:
                logger.info(f"Experiment {name} already exists")
                return experiment.experiment_id

            experiment_id = mlflow.create_experiment(name, tags=tags)
            logger.info(f"Created experiment: {name} (id: {experiment_id})")

            return experiment_id

        except ImportError:
            raise ImportError("MLflow not installed. Install with: pip install mlflow")

    def set_experiment(self, name: str) -> None:
        """
        Set the active MLflow experiment.

        Args:
            name: Experiment name
        """
        try:
            import mlflow

            mlflow.set_experiment(name)
            logger.info(f"Set active experiment: {name}")

        except ImportError:
            logger.warning("MLflow not installed")

    def start_run(
        self,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> Any:
        """
        Start an MLflow run.

        Args:
            run_name: Optional run name
            tags: Optional tags for the run

        Returns:
            MLflow active run context
        """
        try:
            import mlflow

            return mlflow.start_run(run_name=run_name, tags=tags)

        except ImportError:
            raise ImportError("MLflow not installed. Install with: pip install mlflow")

    def log_params(self, params: Dict[str, Any]) -> None:
        """
        Log parameters to MLflow.

        Args:
            params: Dictionary of parameters
        """
        try:
            import mlflow

            mlflow.log_params(params)
            logger.debug(f"Logged {len(params)} parameters to MLflow")

        except ImportError:
            logger.warning("MLflow not installed")

    def log_metrics(
        self, metrics: Dict[str, float], step: Optional[int] = None
    ) -> None:
        """
        Log metrics to MLflow.

        Args:
            metrics: Dictionary of metrics
            step: Optional step number for tracking over time
        """
        try:
            import mlflow

            mlflow.log_metrics(metrics, step=step)
            logger.debug(f"Logged {len(metrics)} metrics to MLflow")

        except ImportError:
            logger.warning("MLflow not installed")

    def log_artifact(self, artifact_path: str) -> None:
        """
        Log an artifact to MLflow.

        Args:
            artifact_path: Path to artifact file
        """
        try:
            import mlflow

            mlflow.log_artifact(artifact_path)
            logger.debug(f"Logged artifact: {artifact_path}")

        except ImportError:
            logger.warning("MLflow not installed")

    def get_run_info(self, run_id: str) -> Dict[str, Any]:
        """
        Get information about an MLflow run.

        Args:
            run_id: MLflow run ID

        Returns:
            Dictionary with run information
        """
        try:
            import mlflow

            client = mlflow.tracking.MlflowClient()
            run = client.get_run(run_id)

            return {
                "run_id": run.info.run_id,
                "experiment_id": run.info.experiment_id,
                "status": run.info.status,
                "start_time": run.info.start_time,
                "end_time": run.info.end_time,
                "params": run.data.params,
                "metrics": run.data.metrics,
                "tags": run.data.tags,
            }

        except ImportError:
            raise ImportError("MLflow not installed. Install with: pip install mlflow")

    def search_runs(
        self,
        experiment_name: Optional[str] = None,
        filter_string: str = "",
        max_results: int = 100,
    ) -> Any:
        """
        Search for MLflow runs.

        Args:
            experiment_name: Experiment name to search in (defaults to current)
            filter_string: MLflow filter string
            max_results: Maximum number of results

        Returns:
            DataFrame of runs
        """
        try:
            import mlflow

            if experiment_name:
                experiment = mlflow.get_experiment_by_name(experiment_name)
                experiment_ids = [experiment.experiment_id] if experiment else []
            else:
                experiment_ids = None

            runs = mlflow.search_runs(
                experiment_ids=experiment_ids,
                filter_string=filter_string,
                max_results=max_results,
            )

            logger.info(f"Found {len(runs)} runs")
            return runs

        except ImportError:
            raise ImportError("MLflow not installed. Install with: pip install mlflow")

    def get_best_run(
        self,
        metric: str,
        experiment_name: Optional[str] = None,
        order_by: str = "DESC",
    ) -> Any:
        """
        Get the best run by metric.

        Args:
            metric: Metric to optimize
            experiment_name: Experiment name (defaults to current)
            order_by: "DESC" for maximize, "ASC" for minimize

        Returns:
            Best run information
        """
        try:
            import mlflow

            if experiment_name:
                experiment = mlflow.get_experiment_by_name(experiment_name)
                experiment_ids = [experiment.experiment_id] if experiment else []
            else:
                experiment_ids = None

            runs = mlflow.search_runs(
                experiment_ids=experiment_ids,
                order_by=[f"metrics.{metric} {order_by}"],
                max_results=1,
            )

            if len(runs) == 0:
                logger.warning(f"No runs found with metric: {metric}")
                return None

            logger.info(f"Best run: {runs.iloc[0]['run_id']} ({metric}={runs.iloc[0][f'metrics.{metric}']})")
            return runs.iloc[0]

        except ImportError:
            raise ImportError("MLflow not installed. Install with: pip install mlflow")

    def launch_ui(self, port: int = 5000, host: str = "127.0.0.1") -> None:
        """
        Launch MLflow UI.

        Note: This will block the current process. Run in a separate terminal.

        Args:
            port: Port to run UI on
            host: Host to bind to
        """
        logger.info(f"To launch MLflow UI, run: mlflow ui --backend-store-uri {self.tracking_uri} --port {port} --host {host}")
        logger.info(f"Then open: http://{host}:{port}")

    def get_artifact_uri(self) -> str:
        """Get the artifact URI for the current run"""
        try:
            import mlflow

            return mlflow.get_artifact_uri()

        except ImportError:
            return str(self.artifacts_dir)

    def __repr__(self) -> str:
        """String representation"""
        return f"MLflowConfig(tracking_uri={self.tracking_uri}, experiment={self.experiment_name})"


# Global config instance
_mlflow_config: Optional[MLflowConfig] = None


def get_mlflow_config(
    tracking_uri: Optional[str] = None,
    experiment_name: str = "nfl_predictions",
) -> MLflowConfig:
    """
    Get or create MLflow configuration singleton.

    Args:
        tracking_uri: MLflow tracking URI (only used on first call)
        experiment_name: Experiment name (only used on first call)

    Returns:
        MLflowConfig instance
    """
    global _mlflow_config

    if _mlflow_config is None:
        _mlflow_config = MLflowConfig(
            tracking_uri=tracking_uri,
            experiment_name=experiment_name,
        )
        _mlflow_config.setup()

    return _mlflow_config


def reset_mlflow_config() -> None:
    """Reset the global MLflow configuration"""
    global _mlflow_config
    _mlflow_config = None


# Experiment naming conventions
class ExperimentNames:
    """Standard experiment names for NFL prediction system"""

    PLAYER_STATS = "nfl_player_stats_prediction"
    TEAM_POINTS = "nfl_team_points_prediction"
    WIN_PROBABILITY = "nfl_win_probability"
    QUARTER_SCORES = "nfl_quarter_scores"
    HYPERPARAMETER_TUNING = "nfl_hyperparameter_tuning"
    FEATURE_ENGINEERING = "nfl_feature_engineering"


# Example usage and testing
def main():
    """Test MLflow configuration"""
    print("MLflow Configuration")
    print("=" * 50)

    config = get_mlflow_config()
    print(f"Tracking URI: {config.tracking_uri}")
    print(f"Experiment: {config.experiment_name}")
    print(f"MLruns directory: {config.mlruns_dir}")
    print(f"Artifacts directory: {config.artifacts_dir}")

    print("\nTo launch MLflow UI:")
    config.launch_ui()


if __name__ == "__main__":
    main()
