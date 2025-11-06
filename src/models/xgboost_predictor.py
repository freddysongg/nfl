"""
XGBoost-based predictor for NFL player and team statistics.

Supports:
- Position-specific models (QB, RB, WR, TE, K, DEF)
- Regression (yards, points, TDs) and classification (win probability)
- Hyperparameter tuning with Optuna
- SHAP explanations for interpretability
- MLflow experiment tracking
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

import numpy as np
import pandas as pd
import polars as pl
import xgboost as xgb
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
import json
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class XGBoostPredictor:
    """
    XGBoost-based predictor for NFL player and team statistics.

    Supports position-specific models with regression and classification tasks,
    hyperparameter tuning, SHAP explanations, and MLflow tracking.
    """

    def __init__(
        self,
        model_dir: str = "models/xgboost",
        experiment_name: str = "xgboost_nfl_predictor",
        use_mlflow: bool = False,
        random_seed: int = 42,
    ):
        """
        Initialize XGBoostPredictor.

        Args:
            model_dir: Directory to save/load models
            experiment_name: MLflow experiment name
            use_mlflow: Whether to enable MLflow tracking
            random_seed: Random seed for reproducibility
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.experiment_name = experiment_name
        self.use_mlflow = use_mlflow
        self.random_seed = random_seed

        # Model registry: {position: {target: model}}
        self.model_registry: Dict[str, Dict[str, xgb.XGBModel]] = {}

        # Feature information: {position: {target: feature_names}}
        self.feature_info: Dict[str, Dict[str, List[str]]] = {}

        # Training metadata: {position: {target: metadata}}
        self.training_metadata: Dict[str, Dict[str, Dict[str, Any]]] = {}

        # MLflow setup
        if self.use_mlflow:
            self._setup_mlflow()

        logger.info(f"Initialized XGBoostPredictor with model_dir={model_dir}")

    def _setup_mlflow(self):
        """Configure MLflow tracking."""
        try:
            import mlflow

            mlflow.set_tracking_uri("sqlite:///mlruns.db")
            mlflow.set_experiment(self.experiment_name)
            logger.info(
                f"MLflow enabled: experiment={self.experiment_name}, "
                f"tracking_uri={mlflow.get_tracking_uri()}"
            )
        except ImportError:
            logger.warning("MLflow not available, tracking disabled")
            self.use_mlflow = False
        except Exception as e:
            logger.error(f"Error setting up MLflow: {e}")
            self.use_mlflow = False

    # =========================================================================
    # Core Training and Prediction Methods
    # =========================================================================

    def _build_model(self, task_type: str, params: Dict[str, Any]) -> xgb.XGBModel:
        """
        Build XGBoost model instance.

        Args:
            task_type: 'regression' or 'classification'
            params: Model hyperparameters

        Returns:
            XGBoost model instance
        """
        if task_type == "regression":
            return xgb.XGBRegressor(**params)
        else:
            return xgb.XGBClassifier(**params)

    def train(
        self,
        position: str,
        target: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: List[str],
        hyperparameter_tune: bool = False,
        n_trials: int = 50,
    ) -> Dict[str, Any]:
        """
        Train position-specific model for a target variable.

        Args:
            position: Player position (QB, RB, WR, TE, K, DEF)
            target: Target variable (passing_yards, rushing_tds, etc.)
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            feature_names: List of feature names
            hyperparameter_tune: Whether to run Optuna tuning
            n_trials: Number of Optuna trials

        Returns:
            Training results dictionary with metrics
        """
        logger.info("=" * 70)
        logger.info(f"Training {position} - {target}")
        logger.info("=" * 70)
        logger.info(
            f"Train samples: {len(X_train)}, Val samples: {len(X_val)}, Features: {len(feature_names)}"
        )

        start_time = time.time()

        # Start MLflow run
        if self.use_mlflow:
            import mlflow

            mlflow.start_run(run_name=f"{position}_{target}")

        try:
            # Determine task type
            task_type = self._determine_task_type(target)
            logger.info(f"Task type: {task_type}")

            # Get base parameters
            base_params = self._get_base_params(task_type)

            # Log basic parameters to MLflow
            if self.use_mlflow:
                import mlflow

                mlflow.log_param("position", position)
                mlflow.log_param("target", target)
                mlflow.log_param("n_features", len(feature_names))
                mlflow.log_param("n_train_samples", len(X_train))
                mlflow.log_param("n_val_samples", len(X_val))
                mlflow.log_param("task_type", task_type)

            # Hyperparameter tuning
            if hyperparameter_tune:
                logger.info(f"Starting hyperparameter tuning ({n_trials} trials)...")
                best_params = self._tune_hyperparameters(
                    X_train, y_train, X_val, y_val, base_params, task_type, n_trials
                )
            else:
                best_params = base_params

            # Log hyperparameters
            if self.use_mlflow:
                import mlflow

                mlflow.log_params(best_params)

            # Train final model
            logger.info("Training final model...")
            model = self._train_model(
                X_train, y_train, X_val, y_val, best_params, task_type
            )

            # Store model
            if position not in self.model_registry:
                self.model_registry[position] = {}
            self.model_registry[position][target] = model

            # Store feature info
            if position not in self.feature_info:
                self.feature_info[position] = {}
            self.feature_info[position][target] = feature_names

            # Evaluate model
            metrics = self._evaluate_model(
                model, X_train, y_train, X_val, y_val, task_type
            )

            # Log metrics
            logger.info("Training Results:")
            for metric_name, metric_value in metrics.items():
                logger.info(f"  {metric_name}: {metric_value:.4f}")

            if self.use_mlflow:
                import mlflow

                mlflow.log_metrics(metrics)

            # Calculate SHAP values
            shap_results = None
            try:
                logger.info("Calculating SHAP values...")
                shap_results = self._calculate_shap_values(
                    model, X_val[:1000], feature_names, position, target
                )
            except Exception as e:
                logger.warning(f"SHAP calculation failed: {e}")

            # Save model
            model_path = self._save_model_internal(model, position, target)
            logger.info(f"Model saved: {model_path}")

            # Log model to MLflow
            if self.use_mlflow:
                import mlflow

                mlflow.xgboost.log_model(model, artifact_path="model")

                # Log SHAP artifacts if available
                if shap_results and "visualizations" in shap_results:
                    for viz_path in shap_results["visualizations"].values():
                        if isinstance(viz_path, (str, Path)) and Path(viz_path).exists():
                            mlflow.log_artifact(str(viz_path))

            # Save training metadata
            training_duration = time.time() - start_time
            self._save_training_metadata(
                position, target, best_params, metrics, feature_names, training_duration
            )

            results = {
                "position": position,
                "target": target,
                "metrics": metrics,
                "best_params": best_params,
                "model_path": str(model_path),
                "training_duration": training_duration,
                "shap_results": shap_results,
            }

            logger.info(
                f"Training completed in {training_duration:.2f}s - {position} {target}"
            )

            return results

        finally:
            # End MLflow run
            if self.use_mlflow:
                import mlflow

                mlflow.end_run()

    def predict(
        self, position: str, target: str, X: np.ndarray
    ) -> np.ndarray:
        """
        Make predictions using trained model.

        Args:
            position: Player position
            target: Target variable
            X: Input features

        Returns:
            Predictions array
        """
        if position not in self.model_registry:
            raise ValueError(f"No models trained for position: {position}")

        if target not in self.model_registry[position]:
            raise ValueError(f"No model trained for {position} {target}")

        model = self.model_registry[position][target]
        predictions = model.predict(X)

        return predictions

    def predict_proba(
        self, position: str, target: str, X: np.ndarray
    ) -> np.ndarray:
        """
        Predict class probabilities (for classification models).

        Args:
            position: Player position
            target: Target variable
            X: Input features

        Returns:
            Probability predictions array

        Raises:
            ValueError: If model is not classification or not trained
        """
        if position not in self.model_registry:
            raise ValueError(f"No models trained for position: {position}")

        if target not in self.model_registry[position]:
            raise ValueError(f"No model trained for {position} {target}")

        model = self.model_registry[position][target]

        if not hasattr(model, "predict_proba"):
            raise ValueError(f"Model {position} {target} does not support predict_proba")

        return model.predict_proba(X)

    # =========================================================================
    # Model Persistence
    # =========================================================================

    def save(
        self, position: str, target: str, filepath: Optional[str] = None
    ) -> str:
        """
        Save trained model to disk.

        Args:
            position: Player position
            target: Target variable
            filepath: Custom filepath (optional)

        Returns:
            Path where model was saved
        """
        if position not in self.model_registry:
            raise ValueError(f"No models trained for position: {position}")

        if target not in self.model_registry[position]:
            raise ValueError(f"No model trained for {position} {target}")

        if filepath is None:
            filepath = self.model_dir / f"{position}_{target}_model.json"
        else:
            filepath = Path(filepath)

        model = self.model_registry[position][target]
        model.save_model(str(filepath))

        # Save feature info
        feature_info_path = filepath.parent / f"{filepath.stem}.features.json"
        with open(feature_info_path, "w") as f:
            json.dump(
                {
                    "feature_names": self.feature_info[position][target],
                    "n_features": len(self.feature_info[position][target]),
                },
                f,
                indent=2,
            )

        logger.info(f"Saved model: {filepath}")
        return str(filepath)

    def load(
        self, position: str, target: str, filepath: Optional[str] = None
    ):
        """
        Load trained model from disk.

        Args:
            position: Player position
            target: Target variable
            filepath: Custom filepath (optional)
        """
        if filepath is None:
            filepath = self.model_dir / f"{position}_{target}_model.json"
        else:
            filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")

        # Determine task type and create model
        task_type = self._determine_task_type(target)

        if task_type == "regression":
            model = xgb.XGBRegressor()
        else:
            model = xgb.XGBClassifier()

        model.load_model(str(filepath))

        # Load feature info
        feature_info_path = filepath.parent / f"{filepath.stem}.features.json"
        if not feature_info_path.exists():
            raise FileNotFoundError(f"Feature info not found: {feature_info_path}")

        with open(feature_info_path, "r") as f:
            feature_data = json.load(f)

        # Store in registry
        if position not in self.model_registry:
            self.model_registry[position] = {}
        self.model_registry[position][target] = model

        if position not in self.feature_info:
            self.feature_info[position] = {}
        self.feature_info[position][target] = feature_data["feature_names"]

        logger.info(
            f"Loaded model: {position} {target} ({len(feature_data['feature_names'])} features)"
        )

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _determine_task_type(self, target: str) -> str:
        """
        Determine if target is regression or classification.

        Args:
            target: Target variable name

        Returns:
            'regression' or 'classification'
        """
        classification_targets = [
            "win_probability",
            "over_under",
            "fg_made",
            "td_scored",
            "first_down_conversion",
        ]

        return "classification" if target in classification_targets else "regression"

    def _get_base_params(self, task_type: str) -> Dict[str, Any]:
        """
        Get base XGBoost parameters for task type.

        Args:
            task_type: 'regression' or 'classification'

        Returns:
            Dictionary of base parameters
        """
        if task_type == "regression":
            return {
                "objective": "reg:squarederror",
                "eval_metric": "rmse",
                "tree_method": "hist",
                "max_depth": 6,
                "learning_rate": 0.05,
                "n_estimators": 200,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "min_child_weight": 3,
                "gamma": 0.1,
                "reg_alpha": 0.1,
                "reg_lambda": 1.0,
                "random_state": self.random_seed,
                "n_jobs": -1,
            }
        else:
            return {
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "tree_method": "hist",
                "max_depth": 5,
                "learning_rate": 0.05,
                "n_estimators": 200,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "min_child_weight": 3,
                "gamma": 0.1,
                "reg_alpha": 0.1,
                "reg_lambda": 1.0,
                "scale_pos_weight": 1.0,
                "random_state": self.random_seed,
                "n_jobs": -1,
            }

    def _train_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        params: Dict[str, Any],
        task_type: str,
    ) -> xgb.XGBModel:
        """
        Train XGBoost model with early stopping.

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            params: Model hyperparameters
            task_type: 'regression' or 'classification'

        Returns:
            Trained XGBoost model
        """
        if task_type == "regression":
            model = xgb.XGBRegressor(**params)
        else:
            model = xgb.XGBClassifier(**params)

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=False,
        )

        return model

    def _evaluate_model(
        self,
        model: xgb.XGBModel,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        task_type: str,
    ) -> Dict[str, float]:
        """
        Evaluate model performance.

        Args:
            model: Trained model
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            task_type: 'regression' or 'classification'

        Returns:
            Dictionary of evaluation metrics
        """
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)

        if task_type == "regression":
            metrics = {
                "train_rmse": np.sqrt(mean_squared_error(y_train, y_train_pred)),
                "train_mae": mean_absolute_error(y_train, y_train_pred),
                "train_r2": r2_score(y_train, y_train_pred),
                "val_rmse": np.sqrt(mean_squared_error(y_val, y_val_pred)),
                "val_mae": mean_absolute_error(y_val, y_val_pred),
                "val_r2": r2_score(y_val, y_val_pred),
            }
        else:
            # Get probability predictions for AUC
            y_val_proba = model.predict_proba(X_val)[:, 1]

            metrics = {
                "train_accuracy": accuracy_score(y_train, y_train_pred),
                "val_accuracy": accuracy_score(y_val, y_val_pred),
                "val_precision": precision_score(y_val, y_val_pred, zero_division=0),
                "val_recall": recall_score(y_val, y_val_pred, zero_division=0),
                "val_roc_auc": roc_auc_score(y_val, y_val_proba),
            }

        return metrics

    def _save_model_internal(
        self, model: xgb.XGBModel, position: str, target: str
    ) -> Path:
        """
        Save model to disk (internal method).

        Args:
            model: Model to save
            position: Player position
            target: Target variable

        Returns:
            Path where model was saved
        """
        model_path = self.model_dir / f"{position}_{target}_model.json"
        model.save_model(str(model_path))
        return model_path

    def _save_training_metadata(
        self,
        position: str,
        target: str,
        params: Dict[str, Any],
        metrics: Dict[str, float],
        feature_names: List[str],
        training_duration: float,
    ):
        """
        Save training metadata to JSON file.

        Args:
            position: Player position
            target: Target variable
            params: Model hyperparameters
            metrics: Performance metrics
            feature_names: List of feature names
            training_duration: Training duration in seconds
        """
        metadata = {
            "position": position,
            "target": target,
            "params": params,
            "metrics": metrics,
            "feature_names": feature_names,
            "n_features": len(feature_names),
            "training_duration": training_duration,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        # Store in memory
        if position not in self.training_metadata:
            self.training_metadata[position] = {}
        self.training_metadata[position][target] = metadata

        # Save to disk
        metadata_path = self.model_dir / f"{position}_{target}_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

    # =========================================================================
    # Hyperparameter Tuning
    # =========================================================================

    def _tune_hyperparameters(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        base_params: Dict[str, Any],
        task_type: str,
        n_trials: int = 50,
    ) -> Dict[str, Any]:
        """
        Tune hyperparameters using Optuna.

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            base_params: Base parameters
            task_type: 'regression' or 'classification'
            n_trials: Number of Optuna trials

        Returns:
            Best hyperparameters
        """
        try:
            import optuna

            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            logger.warning("Optuna not available, using base parameters")
            return base_params

        def objective(trial: "optuna.Trial") -> float:
            """Optuna objective function."""
            params = {
                **base_params,
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.01, 0.3, log=True
                ),
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "gamma": trial.suggest_float("gamma", 0.0, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 2.0),
            }

            if task_type == "classification":
                params["scale_pos_weight"] = trial.suggest_float(
                    "scale_pos_weight", 0.5, 2.0
                )

            # Train model
            model = self._train_model(
                X_train, y_train, X_val, y_val, params, task_type
            )

            # Evaluate
            y_val_pred = model.predict(X_val)

            if task_type == "regression":
                score = np.sqrt(mean_squared_error(y_val, y_val_pred))
                return score  # Minimize RMSE
            else:
                y_val_proba = model.predict_proba(X_val)[:, 1]
                score = roc_auc_score(y_val, y_val_proba)
                return -score  # Minimize (negate AUC)

        # Run optimization
        study = optuna.create_study(
            direction="minimize", sampler=optuna.samplers.TPESampler(seed=self.random_seed)
        )

        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        logger.info(f"Best trial: {study.best_trial.number}")
        logger.info(f"Best value: {study.best_trial.value:.4f}")

        # Log to MLflow
        if self.use_mlflow:
            import mlflow

            mlflow.log_param("optuna_n_trials", n_trials)
            mlflow.log_metric("optuna_best_value", study.best_trial.value)

        # Merge with base params
        best_params = {**base_params, **study.best_trial.params}

        return best_params

    # =========================================================================
    # Feature Importance and SHAP
    # =========================================================================

    def get_feature_importance(
        self, position: str, target: str, importance_type: str = "gain"
    ) -> pd.DataFrame:
        """
        Extract feature importance from trained model.

        Args:
            position: Player position
            target: Prediction target
            importance_type: 'gain', 'weight', or 'cover'

        Returns:
            DataFrame with features and importance scores
        """
        if position not in self.model_registry:
            raise ValueError(f"No models trained for position: {position}")

        if target not in self.model_registry[position]:
            raise ValueError(f"No model trained for {position} {target}")

        model = self.model_registry[position][target]
        importance = model.get_booster().get_score(importance_type=importance_type)

        df = pd.DataFrame(
            [{"feature": k, "importance": v} for k, v in importance.items()]
        ).sort_values("importance", ascending=False)

        return df

    def _calculate_shap_values(
        self,
        model: xgb.XGBModel,
        X_sample: np.ndarray,
        feature_names: List[str],
        position: str,
        target: str,
    ) -> Dict[str, Any]:
        """
        Calculate SHAP values for model interpretability.

        Args:
            model: Trained XGBoost model
            X_sample: Sample data (max 1000 rows)
            feature_names: List of feature names
            position: Player position
            target: Prediction target

        Returns:
            Dictionary with SHAP results
        """
        try:
            import shap
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("SHAP or matplotlib not available, skipping SHAP calculation")
            return {}

        # Limit sample size
        if len(X_sample) > 1000:
            X_sample = X_sample[:1000]

        logger.info(f"Calculating SHAP values for {len(X_sample)} samples...")

        # Create TreeExplainer
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X_sample)

        # Create visualizations directory
        viz_dir = self.model_dir / "shap_visualizations" / position
        viz_dir.mkdir(parents=True, exist_ok=True)

        visualizations = {}

        try:
            # 1. Summary plot (bar)
            plt.figure(figsize=(10, 8))
            shap.summary_plot(
                shap_values.values,
                X_sample,
                feature_names=feature_names,
                plot_type="bar",
                show=False,
            )
            bar_plot_path = viz_dir / f"{target}_summary_bar.png"
            plt.savefig(bar_plot_path, dpi=150, bbox_inches="tight")
            plt.close()
            visualizations["summary_bar"] = str(bar_plot_path)
            logger.info(f"Saved: {bar_plot_path}")
        except Exception as e:
            logger.warning(f"Failed to create summary bar plot: {e}")

        try:
            # 2. Summary plot (beeswarm)
            plt.figure(figsize=(10, 8))
            shap.summary_plot(
                shap_values.values, X_sample, feature_names=feature_names, show=False
            )
            beeswarm_plot_path = viz_dir / f"{target}_summary_beeswarm.png"
            plt.savefig(beeswarm_plot_path, dpi=150, bbox_inches="tight")
            plt.close()
            visualizations["summary_beeswarm"] = str(beeswarm_plot_path)
            logger.info(f"Saved: {beeswarm_plot_path}")
        except Exception as e:
            logger.warning(f"Failed to create beeswarm plot: {e}")

        # Calculate mean absolute SHAP values
        mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
        feature_importance_df = pd.DataFrame(
            {"feature": feature_names, "mean_abs_shap": mean_abs_shap}
        ).sort_values("mean_abs_shap", ascending=False)

        return {
            "mean_abs_shap": feature_importance_df.to_dict("records"),
            "visualizations": visualizations,
        }

    # =========================================================================
    # Position-Specific Utilities
    # =========================================================================

    def get_position_targets(self, position: str) -> List[str]:
        """
        Get available prediction targets for a position.

        Args:
            position: Player position (QB, RB, WR, TE, K, DEF)

        Returns:
            List of available targets for that position
        """
        position_targets = {
            "QB": [
                "passing_yards",
                "passing_tds",
                "interceptions",
                "completions",
                "attempts",
                "rushing_yards",
                "rushing_tds",
                "fantasy_points",
            ],
            "RB": [
                "rushing_yards",
                "rushing_tds",
                "carries",
                "receptions",
                "receiving_yards",
                "receiving_tds",
                "fantasy_points",
            ],
            "WR": [
                "receptions",
                "receiving_yards",
                "receiving_tds",
                "targets",
                "rushing_yards",
                "fantasy_points",
            ],
            "TE": [
                "receptions",
                "receiving_yards",
                "receiving_tds",
                "targets",
                "fantasy_points",
            ],
            "K": [
                "fg_made",
                "fg_att",
                "fg_pct",
                "fg_made_40_49",
                "fg_made_50_plus",
                "pat_made",
                "fantasy_points",
            ],
            "DEF": [
                "sacks",
                "interceptions",
                "fumbles_forced",
                "tackles",
                "tds",
                "points_allowed",
                "fantasy_points",
            ],
        }

        return position_targets.get(position, [])

    def train_all_position_models(
        self,
        position: str,
        X_train: np.ndarray,
        y_train_dict: Dict[str, np.ndarray],
        X_val: np.ndarray,
        y_val_dict: Dict[str, np.ndarray],
        feature_names: List[str],
        hyperparameter_tune: bool = True,
        n_trials: int = 50,
    ) -> Dict[str, Any]:
        """
        Train all models for a specific position.

        Args:
            position: Player position
            X_train: Training features
            y_train_dict: Dictionary mapping target names to training targets
            X_val: Validation features
            y_val_dict: Dictionary mapping target names to validation targets
            feature_names: List of feature names
            hyperparameter_tune: Whether to tune hyperparameters
            n_trials: Number of Optuna trials

        Returns:
            Training results for all targets
        """
        results = {}
        targets = self.get_position_targets(position)

        logger.info(f"\nTraining {len(targets)} models for position: {position}")
        logger.info(
            f"Hyperparameter tuning: {hyperparameter_tune} (n_trials={n_trials})"
        )
        logger.info("=" * 70)

        for target in targets:
            # Check if we have data for this target
            if target not in y_train_dict or target not in y_val_dict:
                logger.warning(f"Skipping {position} {target} - no data available")
                results[target] = {"error": "No data available"}
                continue

            try:
                # Train model
                result = self.train(
                    position,
                    target,
                    X_train,
                    y_train_dict[target],
                    X_val,
                    y_val_dict[target],
                    feature_names,
                    hyperparameter_tune=hyperparameter_tune,
                    n_trials=n_trials,
                )

                results[target] = result

            except Exception as e:
                logger.error(f"Error training {position} {target}: {e}")
                results[target] = {"error": str(e)}

        logger.info("=" * 70)
        logger.info(f"Training complete for {position}")
        success_count = sum(1 for r in results.values() if "error" not in r)
        logger.info(f"Success: {success_count}/{len(results)}")
        logger.info("=" * 70)

        return results

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def get_trained_models(self) -> List[Tuple[str, str]]:
        """
        Get list of trained models.

        Returns:
            List of (position, target) tuples
        """
        models = []
        for position, targets in self.model_registry.items():
            for target in targets.keys():
                models.append((position, target))
        return models

    def get_model_info(self, position: str, target: str) -> Dict[str, Any]:
        """
        Get information about a trained model.

        Args:
            position: Player position
            target: Target variable

        Returns:
            Dictionary with model information
        """
        if position not in self.model_registry:
            raise ValueError(f"No models trained for position: {position}")

        if target not in self.model_registry[position]:
            raise ValueError(f"No model trained for {position} {target}")

        info = {
            "position": position,
            "target": target,
            "n_features": len(self.feature_info[position][target]),
            "feature_names": self.feature_info[position][target],
        }

        # Add metadata if available
        if (
            position in self.training_metadata
            and target in self.training_metadata[position]
        ):
            info["metadata"] = self.training_metadata[position][target]

        return info

    def __repr__(self) -> str:
        """String representation"""
        n_models = sum(len(targets) for targets in self.model_registry.values())
        return f"XGBoostPredictor(models={n_models}, dir={self.model_dir})"
