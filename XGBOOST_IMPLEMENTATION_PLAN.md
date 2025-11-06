# XGBoost Model Implementation Plan

## Overview

This document provides a detailed implementation plan for the XGBoostPredictor class, which will serve as the first concrete ML model for NFL predictions using gradient boosting. The model will support position-specific predictions for multiple target types (regression and classification) with hyperparameter tuning, SHAP explanations, and MLflow tracking.

---

## 1. Technical Specification

### 1.1 XGBoost Parameter Configurations

#### Regression Configuration (Yards, Points, Fantasy Points)
```python
REGRESSION_BASE_PARAMS = {
    'objective': 'reg:squarederror',  # MSE loss
    'eval_metric': ['rmse', 'mae'],
    'tree_method': 'hist',            # Fast histogram-based algorithm
    'max_depth': 6,                   # Default depth
    'learning_rate': 0.05,            # Conservative learning rate
    'n_estimators': 200,              # Will be tuned
    'subsample': 0.8,                 # Row sampling
    'colsample_bytree': 0.8,          # Feature sampling
    'min_child_weight': 3,            # Minimum sum of instance weight
    'gamma': 0.1,                     # Minimum loss reduction for split
    'reg_alpha': 0.1,                 # L1 regularization
    'reg_lambda': 1.0,                # L2 regularization
    'random_state': 42,
    'n_jobs': -1,                     # Use all cores
    'early_stopping_rounds': 20,      # Stop if no improvement
}
```

#### Classification Configuration (Win Probability, Over/Under)
```python
CLASSIFICATION_BASE_PARAMS = {
    'objective': 'binary:logistic',   # Binary classification
    'eval_metric': ['logloss', 'auc'],
    'tree_method': 'hist',
    'max_depth': 5,                   # Slightly shallower for classification
    'learning_rate': 0.05,
    'n_estimators': 200,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 3,
    'gamma': 0.1,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'scale_pos_weight': 1.0,          # Balance positive/negative classes
    'random_state': 42,
    'n_jobs': -1,
    'early_stopping_rounds': 20,
}
```

### 1.2 Position-Specific Model Training Strategy

**Approach: Separate Models Per Position**

- **Rationale:** Each position has vastly different feature spaces and prediction targets
  - QB: Passing stats dominate (completions, yards, TDs, interceptions)
  - RB: Rushing stats + receiving work (carries, rush yards, receptions)
  - WR/TE: Receiving stats (targets, receptions, yards, TDs)
  - K: Field goal attempts and accuracy by distance
  - DEF: Defensive stats (sacks, tackles, interceptions)

- **Model Registry Structure:**
```python
model_registry = {
    'QB': {
        'passing_yards': XGBRegressor(...),
        'passing_tds': XGBRegressor(...),
        'interceptions': XGBRegressor(...),
        'rushing_yards': XGBRegressor(...),
        'fantasy_points': XGBRegressor(...),
    },
    'RB': {
        'rushing_yards': XGBRegressor(...),
        'rushing_tds': XGBRegressor(...),
        'receptions': XGBRegressor(...),
        'receiving_yards': XGBRegressor(...),
        'fantasy_points': XGBRegressor(...),
    },
    'WR': { ... },
    'TE': { ... },
    'K': { ... },
    'DEF': { ... },
}
```

### 1.3 Feature Importance Extraction

```python
def get_feature_importance(self, position: str, target: str,
                          importance_type: str = 'gain') -> pd.DataFrame:
    """
    Extract feature importance from trained model.

    Args:
        position: Player position (QB, RB, WR, etc.)
        target: Prediction target (passing_yards, rushing_yards, etc.)
        importance_type: 'gain' (default), 'weight', or 'cover'
            - gain: Average gain when feature is used in splits
            - weight: Number of times feature appears in trees
            - cover: Average coverage (samples affected by splits)

    Returns:
        DataFrame with features and importance scores
    """
    model = self.model_registry[position][target]
    importance = model.get_booster().get_score(importance_type=importance_type)

    df = pd.DataFrame([
        {'feature': k, 'importance': v}
        for k, v in importance.items()
    ]).sort_values('importance', ascending=False)

    return df
```

### 1.4 SHAP Value Calculation

```python
def calculate_shap_values(self, position: str, target: str,
                         X_sample: np.ndarray) -> shap.Explanation:
    """
    Calculate SHAP values for model interpretability.

    Args:
        position: Player position
        target: Prediction target
        X_sample: Sample data for SHAP calculation (max 1000 rows for speed)

    Returns:
        SHAP Explanation object with values, base_values, and data
    """
    model = self.model_registry[position][target]

    # Use TreeExplainer for XGBoost (fast and exact)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_sample)

    return shap_values
```

---

## 2. Implementation Details

### 2.1 Class Structure: XGBoostPredictor(BaseModel)

```python
# File: src/models/xgboost_predictor.py

import xgboost as xgb
import numpy as np
import pandas as pd
import polars as pl
from typing import Dict, List, Optional, Tuple, Any, Union
import shap
import optuna
import mlflow
from pathlib import Path
import joblib
import json

from src.database import NFLDatabase
from src.config import NFLConfig
from src.models.base_model import BaseModel  # Will be created by another subagent


class XGBoostPredictor(BaseModel):
    """
    XGBoost-based predictor for NFL player and team statistics.

    Supports:
    - Position-specific models (QB, RB, WR, TE, K, DEF)
    - Regression (yards, points) and classification (win probability)
    - Hyperparameter tuning with Optuna
    - SHAP explanations for interpretability
    - MLflow experiment tracking
    """

    def __init__(
        self,
        db: NFLDatabase,
        config: NFLConfig,
        model_dir: str = "models/xgboost",
        experiment_name: str = "xgboost_nfl_predictor"
    ):
        """
        Initialize XGBoostPredictor.

        Args:
            db: Database connection
            config: NFL configuration
            model_dir: Directory to save/load models
            experiment_name: MLflow experiment name
        """
        super().__init__(db, config)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Model registry: {position: {target: model}}
        self.model_registry: Dict[str, Dict[str, xgb.XGBModel]] = {}

        # Feature information: {position: {target: feature_names}}
        self.feature_info: Dict[str, Dict[str, List[str]]] = {}

        # Training metadata
        self.training_metadata: Dict[str, Any] = {}

        # MLflow setup
        self.experiment_name = experiment_name
        mlflow.set_experiment(experiment_name)

    # =====================================================================
    # Core Methods (Required by BaseModel)
    # =====================================================================

    def train(
        self,
        position: str,
        target: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: List[str],
        hyperparameter_tune: bool = True,
        n_trials: int = 50
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
        with mlflow.start_run(run_name=f"{position}_{target}"):
            # Log parameters
            mlflow.log_param("position", position)
            mlflow.log_param("target", target)
            mlflow.log_param("n_features", len(feature_names))
            mlflow.log_param("n_train_samples", len(X_train))
            mlflow.log_param("n_val_samples", len(X_val))

            # Determine task type
            task_type = self._determine_task_type(target)
            mlflow.log_param("task_type", task_type)

            # Get base parameters
            base_params = self._get_base_params(task_type)

            # Hyperparameter tuning
            if hyperparameter_tune:
                print(f"Starting hyperparameter tuning for {position} {target}...")
                best_params = self._tune_hyperparameters(
                    X_train, y_train, X_val, y_val,
                    base_params, task_type, n_trials
                )
                mlflow.log_params(best_params)
            else:
                best_params = base_params

            # Train final model
            print(f"Training final model for {position} {target}...")
            model = self._train_model(
                X_train, y_train, X_val, y_val,
                best_params, task_type
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
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)

            # Calculate and log SHAP values
            print(f"Calculating SHAP values for {position} {target}...")
            shap_results = self._calculate_and_log_shap(
                model, X_val[:1000], feature_names, position, target
            )

            # Save model
            model_path = self._save_model(model, position, target)
            mlflow.log_artifact(model_path)

            # Save training metadata
            self._save_training_metadata(
                position, target, best_params, metrics, feature_names
            )

            results = {
                'position': position,
                'target': target,
                'metrics': metrics,
                'best_params': best_params,
                'model_path': str(model_path),
                'shap_results': shap_results
            }

            return results

    def predict(
        self,
        position: str,
        target: str,
        X: np.ndarray
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
            raise ValueError(
                f"No model trained for {position} {target}"
            )

        model = self.model_registry[position][target]
        predictions = model.predict(X)

        return predictions

    def save_model(
        self,
        position: str,
        target: str,
        filepath: Optional[str] = None
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
        if filepath is None:
            filepath = self.model_dir / f"{position}_{target}_model.json"

        model = self.model_registry[position][target]
        model.save_model(filepath)

        # Save feature info
        feature_info_path = Path(filepath).with_suffix('.features.json')
        with open(feature_info_path, 'w') as f:
            json.dump({
                'feature_names': self.feature_info[position][target],
                'n_features': len(self.feature_info[position][target])
            }, f, indent=2)

        return str(filepath)

    def load_model(
        self,
        position: str,
        target: str,
        filepath: Optional[str] = None
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

        # Determine task type and create model
        task_type = self._determine_task_type(target)

        if task_type == 'regression':
            model = xgb.XGBRegressor()
        else:
            model = xgb.XGBClassifier()

        model.load_model(filepath)

        # Load feature info
        feature_info_path = Path(filepath).with_suffix('.features.json')
        with open(feature_info_path, 'r') as f:
            feature_data = json.load(f)

        # Store in registry
        if position not in self.model_registry:
            self.model_registry[position] = {}
        self.model_registry[position][target] = model

        if position not in self.feature_info:
            self.feature_info[position] = {}
        self.feature_info[position][target] = feature_data['feature_names']

    # =====================================================================
    # Helper Methods
    # =====================================================================

    def _determine_task_type(self, target: str) -> str:
        """Determine if target is regression or classification."""
        classification_targets = [
            'win_probability', 'over_under', 'fg_made',
            'td_scored', 'first_down_conversion'
        ]

        return 'classification' if target in classification_targets else 'regression'

    def _get_base_params(self, task_type: str) -> Dict[str, Any]:
        """Get base XGBoost parameters for task type."""
        if task_type == 'regression':
            return {
                'objective': 'reg:squarederror',
                'eval_metric': 'rmse',
                'tree_method': 'hist',
                'max_depth': 6,
                'learning_rate': 0.05,
                'n_estimators': 200,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_weight': 3,
                'gamma': 0.1,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
                'random_state': 42,
                'n_jobs': -1,
            }
        else:
            return {
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'tree_method': 'hist',
                'max_depth': 5,
                'learning_rate': 0.05,
                'n_estimators': 200,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_weight': 3,
                'gamma': 0.1,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
                'scale_pos_weight': 1.0,
                'random_state': 42,
                'n_jobs': -1,
            }

    def _train_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        params: Dict[str, Any],
        task_type: str
    ) -> xgb.XGBModel:
        """Train XGBoost model with early stopping."""
        if task_type == 'regression':
            model = xgb.XGBRegressor(**params)
        else:
            model = xgb.XGBClassifier(**params)

        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=False
        )

        return model

    def _evaluate_model(
        self,
        model: xgb.XGBModel,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        task_type: str
    ) -> Dict[str, float]:
        """Evaluate model performance."""
        from sklearn.metrics import (
            mean_squared_error, mean_absolute_error, r2_score,
            accuracy_score, precision_score, recall_score, roc_auc_score
        )

        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)

        if task_type == 'regression':
            metrics = {
                'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
                'train_mae': mean_absolute_error(y_train, y_train_pred),
                'train_r2': r2_score(y_train, y_train_pred),
                'val_rmse': np.sqrt(mean_squared_error(y_val, y_val_pred)),
                'val_mae': mean_absolute_error(y_val, y_val_pred),
                'val_r2': r2_score(y_val, y_val_pred),
            }
        else:
            # Get probability predictions for AUC
            y_val_proba = model.predict_proba(X_val)[:, 1]

            metrics = {
                'train_accuracy': accuracy_score(y_train, y_train_pred),
                'val_accuracy': accuracy_score(y_val, y_val_pred),
                'val_precision': precision_score(y_val, y_val_pred, zero_division=0),
                'val_recall': recall_score(y_val, y_val_pred, zero_division=0),
                'val_roc_auc': roc_auc_score(y_val, y_val_proba),
            }

        return metrics

    def _save_model(
        self,
        model: xgb.XGBModel,
        position: str,
        target: str
    ) -> Path:
        """Save model to disk."""
        model_path = self.model_dir / f"{position}_{target}_model.json"
        model.save_model(model_path)
        return model_path

    def _save_training_metadata(
        self,
        position: str,
        target: str,
        params: Dict[str, Any],
        metrics: Dict[str, float],
        feature_names: List[str]
    ):
        """Save training metadata."""
        metadata = {
            'position': position,
            'target': target,
            'params': params,
            'metrics': metrics,
            'feature_names': feature_names,
            'n_features': len(feature_names)
        }

        metadata_path = self.model_dir / f"{position}_{target}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

    # ... (continued in next section)
```

### 2.2 Training Algorithm

The training algorithm follows this workflow:

1. **Data Loading**: Load features from `ml_training_features` table
2. **Train/Val Split**: Split data temporally (no data leakage)
3. **Hyperparameter Tuning** (optional): Run Optuna optimization
4. **Model Training**: Train with early stopping on validation set
5. **Evaluation**: Calculate metrics on train and validation sets
6. **SHAP Calculation**: Generate SHAP values for interpretability
7. **Model Persistence**: Save model, features, and metadata
8. **MLflow Logging**: Log params, metrics, and artifacts

### 2.3 Prediction Method

```python
def predict_batch(
    self,
    features_df: pl.DataFrame
) -> pl.DataFrame:
    """
    Make predictions for a batch of players/games.

    Args:
        features_df: DataFrame with columns:
            - entity_type (player/team)
            - entity_id
            - position
            - prediction_target
            - numerical_features (list)

    Returns:
        DataFrame with predictions and confidence intervals
    """
    results = []

    for row in features_df.iter_rows(named=True):
        position = row['position']
        target = row['prediction_target']
        features = np.array(row['numerical_features']).reshape(1, -1)

        # Make prediction
        prediction = self.predict(position, target, features)[0]

        # Calculate prediction interval (if available)
        prediction_interval = self._calculate_prediction_interval(
            position, target, features
        )

        results.append({
            'entity_id': row['entity_id'],
            'position': position,
            'target': target,
            'prediction': prediction,
            'lower_bound': prediction_interval[0],
            'upper_bound': prediction_interval[1],
        })

    return pl.DataFrame(results)
```

---

## 3. Hyperparameter Tuning

### 3.1 Optuna Objective Function

```python
def _tune_hyperparameters(
    self,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    base_params: Dict[str, Any],
    task_type: str,
    n_trials: int = 50
) -> Dict[str, Any]:
    """
    Tune hyperparameters using Optuna.

    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        base_params: Base parameters to start with
        task_type: 'regression' or 'classification'
        n_trials: Number of Optuna trials

    Returns:
        Best hyperparameters
    """
    def objective(trial: optuna.Trial) -> float:
        """Optuna objective function."""
        params = {
            **base_params,
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0.0, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 2.0),
        }

        if task_type == 'classification':
            params['scale_pos_weight'] = trial.suggest_float(
                'scale_pos_weight', 0.5, 2.0
            )

        # Train model
        model = self._train_model(
            X_train, y_train, X_val, y_val, params, task_type
        )

        # Evaluate on validation set
        y_val_pred = model.predict(X_val)

        if task_type == 'regression':
            from sklearn.metrics import mean_squared_error
            score = np.sqrt(mean_squared_error(y_val, y_val_pred))
            return score  # Minimize RMSE
        else:
            from sklearn.metrics import roc_auc_score
            y_val_proba = model.predict_proba(X_val)[:, 1]
            score = roc_auc_score(y_val, y_val_proba)
            return -score  # Optuna minimizes, so negate AUC

    # Run optimization
    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=42)
    )

    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"Best trial: {study.best_trial.number}")
    print(f"Best value: {study.best_trial.value:.4f}")

    # Log to MLflow
    mlflow.log_param("optuna_n_trials", n_trials)
    mlflow.log_metric("optuna_best_value", study.best_trial.value)

    return study.best_trial.params
```

### 3.2 Search Space Definition

```python
OPTUNA_SEARCH_SPACE = {
    'max_depth': {
        'type': 'int',
        'low': 3,
        'high': 10,
        'description': 'Maximum tree depth. Controls model complexity.'
    },
    'learning_rate': {
        'type': 'float',
        'low': 0.01,
        'high': 0.3,
        'log': True,
        'description': 'Step size shrinkage. Lower = more conservative.'
    },
    'n_estimators': {
        'type': 'int',
        'low': 100,
        'high': 500,
        'description': 'Number of boosting rounds.'
    },
    'subsample': {
        'type': 'float',
        'low': 0.6,
        'high': 1.0,
        'description': 'Subsample ratio of training instances.'
    },
    'colsample_bytree': {
        'type': 'float',
        'low': 0.6,
        'high': 1.0,
        'description': 'Subsample ratio of columns when constructing each tree.'
    },
    'min_child_weight': {
        'type': 'int',
        'low': 1,
        'high': 10,
        'description': 'Minimum sum of instance weight needed in a child.'
    },
    'gamma': {
        'type': 'float',
        'low': 0.0,
        'high': 1.0,
        'description': 'Minimum loss reduction required for split.'
    },
    'reg_alpha': {
        'type': 'float',
        'low': 0.0,
        'high': 1.0,
        'description': 'L1 regularization term on weights.'
    },
    'reg_lambda': {
        'type': 'float',
        'low': 0.0,
        'high': 2.0,
        'description': 'L2 regularization term on weights.'
    },
}
```

### 3.3 Tuning Strategy

- **Number of trials**: 50 (default), configurable up to 200 for important models
- **Sampler**: Tree-structured Parzen Estimator (TPE) for efficient search
- **Pruning**: MedianPruner to stop unpromising trials early
- **Parallel execution**: Support for distributed optimization (future)
- **Best model selection**: Based on validation set performance (RMSE for regression, ROC-AUC for classification)

---

## 4. Position-Specific Models

### 4.1 Position Routing

```python
def get_position_targets(self, position: str) -> List[str]:
    """
    Get available prediction targets for a position.

    Args:
        position: Player position (QB, RB, WR, TE, K, DEF)

    Returns:
        List of available targets for that position
    """
    position_targets = {
        'QB': [
            'passing_yards', 'passing_tds', 'interceptions',
            'completions', 'attempts', 'rushing_yards',
            'rushing_tds', 'fantasy_points'
        ],
        'RB': [
            'rushing_yards', 'rushing_tds', 'carries',
            'receptions', 'receiving_yards', 'receiving_tds',
            'fantasy_points'
        ],
        'WR': [
            'receptions', 'receiving_yards', 'receiving_tds',
            'targets', 'rushing_yards', 'fantasy_points'
        ],
        'TE': [
            'receptions', 'receiving_yards', 'receiving_tds',
            'targets', 'fantasy_points'
        ],
        'K': [
            'fg_made', 'fg_att', 'fg_pct',
            'fg_made_40_49', 'fg_made_50_plus',
            'pat_made', 'fantasy_points'
        ],
        'DEF': [
            'sacks', 'interceptions', 'fumbles_forced',
            'tackles', 'tds', 'points_allowed',
            'fantasy_points'
        ]
    }

    return position_targets.get(position, [])
```

### 4.2 Model Training by Position

```python
def train_all_position_models(
    self,
    position: str,
    season_range: Tuple[int, int] = (2021, 2024),
    hyperparameter_tune: bool = True
) -> Dict[str, Any]:
    """
    Train all models for a specific position.

    Args:
        position: Player position
        season_range: Range of seasons to include (start, end)
        hyperparameter_tune: Whether to tune hyperparameters

    Returns:
        Training results for all targets
    """
    results = {}
    targets = self.get_position_targets(position)

    print(f"Training {len(targets)} models for position: {position}")

    for target in targets:
        print(f"\n{'='*60}")
        print(f"Training {position} - {target}")
        print(f"{'='*60}")

        # Load data for this position/target
        X_train, y_train, X_val, y_val, feature_names = (
            self._load_training_data(position, target, season_range)
        )

        # Train model
        result = self.train(
            position, target,
            X_train, y_train, X_val, y_val,
            feature_names,
            hyperparameter_tune=hyperparameter_tune
        )

        results[target] = result

    return results
```

### 4.3 Shared vs Position-Specific Parameters

**Shared Base Parameters:**
- All positions share the same base XGBoost structure
- `tree_method='hist'` for performance
- `random_state=42` for reproducibility
- `early_stopping_rounds=20`

**Position-Specific Tuning:**
- Each position/target combination gets separately tuned hyperparameters
- Allows QB passing models to have different depth than K field goal models
- Accounts for different feature space sizes across positions

---

## 5. Feature Engineering Integration

### 5.1 Loading Features from Database

```python
def _load_training_data(
    self,
    position: str,
    target: str,
    season_range: Tuple[int, int]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Load and prepare training data from ml_training_features table.

    Args:
        position: Player position
        target: Prediction target
        season_range: (start_season, end_season)

    Returns:
        X_train, y_train, X_val, y_val, feature_names
    """
    # Query ml_training_features table
    query = """
        SELECT
            numerical_features,
            feature_names,
            actual_outcomes,
            season,
            week,
            data_quality_score
        FROM ml_training_features
        WHERE entity_type = 'player'
            AND prediction_target = ?
            AND season BETWEEN ? AND ?
            AND data_quality_score >= 0.7
        ORDER BY season, week
    """

    conn = self.db.connect()
    results = conn.execute(query, [target, season_range[0], season_range[1]])
    df = pl.from_pandas(results.fetchdf())

    # Filter by position (stored in categorical_features)
    # This assumes position is stored in the features

    # Extract features and targets
    X = np.vstack(df['numerical_features'].to_list())

    # Extract target from actual_outcomes JSON
    y = np.array([
        json.loads(outcome)[target]
        for outcome in df['actual_outcomes']
    ])

    # Get feature names
    feature_names = json.loads(df['feature_names'][0])

    # Temporal split: Use last 20% of data for validation
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    print(f"Loaded data: Train={len(X_train)}, Val={len(X_val)}")
    print(f"Features: {len(feature_names)}")

    return X_train, y_train, X_val, y_val, feature_names
```

### 5.2 Feature Preprocessing

```python
def preprocess_features(
    self,
    X: np.ndarray,
    feature_names: List[str],
    fit: bool = False
) -> np.ndarray:
    """
    Preprocess features before training/prediction.

    Args:
        X: Feature matrix
        feature_names: List of feature names
        fit: Whether to fit preprocessing (True for training, False for prediction)

    Returns:
        Preprocessed features
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer

    # Handle missing values
    if fit:
        self.imputer = SimpleImputer(strategy='median')
        X = self.imputer.fit_transform(X)
    else:
        X = self.imputer.transform(X)

    # Optional: Scale features (XGBoost is scale-invariant, but can help)
    # Uncomment if needed:
    # if fit:
    #     self.scaler = StandardScaler()
    #     X = self.scaler.fit_transform(X)
    # else:
    #     X = self.scaler.transform(X)

    return X
```

### 5.3 Feature Selection

```python
def select_top_features(
    self,
    position: str,
    target: str,
    n_features: int = 50
) -> List[str]:
    """
    Select top N most important features.

    Args:
        position: Player position
        target: Prediction target
        n_features: Number of features to keep

    Returns:
        List of top feature names
    """
    importance_df = self.get_feature_importance(position, target, 'gain')
    top_features = importance_df.head(n_features)['feature'].tolist()

    print(f"Selected top {n_features} features for {position} {target}")
    print(f"Top 5: {top_features[:5]}")

    return top_features
```

---

## 6. Evaluation Metrics

### 6.1 Regression Metrics

```python
def calculate_regression_metrics(
    self,
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Calculate comprehensive regression metrics.

    Returns:
        - RMSE: Root Mean Squared Error
        - MAE: Mean Absolute Error
        - R²: Coefficient of determination
        - MAPE: Mean Absolute Percentage Error
        - Max Error: Maximum residual error
    """
    from sklearn.metrics import (
        mean_squared_error, mean_absolute_error, r2_score, max_error
    )

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # MAPE (avoid division by zero)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100

    max_err = max_error(y_true, y_pred)

    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mape': mape,
        'max_error': max_err
    }
```

### 6.2 Classification Metrics

```python
def calculate_classification_metrics(
    self,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray
) -> Dict[str, float]:
    """
    Calculate comprehensive classification metrics.

    Returns:
        - Accuracy
        - Precision
        - Recall
        - F1 Score
        - ROC-AUC
        - Log Loss
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, roc_auc_score, log_loss
    )

    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_true, y_proba),
        'log_loss': log_loss(y_true, y_proba)
    }
```

### 6.3 Position-Specific Metrics

```python
POSITION_METRIC_THRESHOLDS = {
    'QB': {
        'passing_yards': {'excellent_rmse': 50, 'good_rmse': 75},
        'passing_tds': {'excellent_mae': 0.5, 'good_mae': 0.8},
        'interceptions': {'excellent_mae': 0.3, 'good_mae': 0.5},
    },
    'RB': {
        'rushing_yards': {'excellent_rmse': 20, 'good_rmse': 35},
        'rushing_tds': {'excellent_mae': 0.3, 'good_mae': 0.5},
    },
    'WR': {
        'receiving_yards': {'excellent_rmse': 15, 'good_rmse': 25},
        'receptions': {'excellent_mae': 1.0, 'good_mae': 1.5},
    },
    'K': {
        'fg_made': {'excellent_accuracy': 0.85, 'good_accuracy': 0.75},
        'fg_pct': {'excellent_mae': 0.05, 'good_mae': 0.10},
    }
}
```

---

## 7. SHAP Explanations

### 7.1 SHAP Calculation

```python
def _calculate_and_log_shap(
    self,
    model: xgb.XGBModel,
    X_sample: np.ndarray,
    feature_names: List[str],
    position: str,
    target: str
) -> Dict[str, Any]:
    """
    Calculate SHAP values and create visualizations.

    Args:
        model: Trained XGBoost model
        X_sample: Sample data for SHAP (max 1000 rows)
        feature_names: List of feature names
        position: Player position
        target: Prediction target

    Returns:
        Dictionary with SHAP results
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Limit sample size for performance
    if len(X_sample) > 1000:
        X_sample = X_sample[:1000]

    # Create TreeExplainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_sample)

    # Create visualizations
    viz_dir = self.model_dir / "shap_visualizations" / position
    viz_dir.mkdir(parents=True, exist_ok=True)

    # 1. Summary plot (bar)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values.values, X_sample,
        feature_names=feature_names,
        plot_type='bar',
        show=False
    )
    bar_plot_path = viz_dir / f"{target}_summary_bar.png"
    plt.savefig(bar_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    mlflow.log_artifact(bar_plot_path)

    # 2. Summary plot (beeswarm)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values.values, X_sample,
        feature_names=feature_names,
        show=False
    )
    beeswarm_plot_path = viz_dir / f"{target}_summary_beeswarm.png"
    plt.savefig(beeswarm_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    mlflow.log_artifact(beeswarm_plot_path)

    # 3. Dependence plots for top 3 features
    importance = np.abs(shap_values.values).mean(axis=0)
    top_3_indices = np.argsort(importance)[-3:]

    for idx in top_3_indices:
        feature_name = feature_names[idx]
        plt.figure(figsize=(8, 6))
        shap.dependence_plot(
            idx, shap_values.values, X_sample,
            feature_names=feature_names,
            show=False
        )
        dep_plot_path = viz_dir / f"{target}_dependence_{feature_name}.png"
        plt.savefig(dep_plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        mlflow.log_artifact(dep_plot_path)

    # Calculate mean absolute SHAP values
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'mean_abs_shap': mean_abs_shap
    }).sort_values('mean_abs_shap', ascending=False)

    # Save SHAP values
    shap_data_path = viz_dir / f"{target}_shap_values.csv"
    pd.DataFrame(
        shap_values.values,
        columns=feature_names
    ).to_csv(shap_data_path, index=False)
    mlflow.log_artifact(shap_data_path)

    return {
        'mean_abs_shap': feature_importance_df.to_dict('records'),
        'visualizations': {
            'summary_bar': str(bar_plot_path),
            'summary_beeswarm': str(beeswarm_plot_path),
            'dependence_plots': [str(viz_dir / f"{target}_dependence_{feature_names[idx]}.png") for idx in top_3_indices]
        }
    }
```

### 7.2 SHAP Interpretation Guide

```python
def interpret_shap_results(
    self,
    position: str,
    target: str,
    top_n: int = 10
) -> str:
    """
    Generate human-readable interpretation of SHAP results.

    Args:
        position: Player position
        target: Prediction target
        top_n: Number of top features to interpret

    Returns:
        Interpretation text
    """
    importance_df = self.get_feature_importance(position, target, 'gain')
    top_features = importance_df.head(top_n)

    interpretation = f"""
SHAP Feature Importance Interpretation for {position} {target}
{'='*70}

The model's predictions are most influenced by the following features:

"""

    for i, row in top_features.iterrows():
        interpretation += f"{i+1}. {row['feature']}: {row['importance']:.4f}\n"

    interpretation += f"""

How to read SHAP plots:
- Bar plot: Shows average impact of each feature on predictions
- Beeswarm plot: Shows distribution of SHAP values across samples
  - Red points: High feature value
  - Blue points: Low feature value
  - Position on x-axis: Impact on prediction
- Dependence plot: Shows how feature values relate to SHAP values
"""

    return interpretation
```

---

## 8. MLflow Integration

### 8.1 Experiment Tracking Setup

```python
def setup_mlflow(self):
    """Configure MLflow tracking."""
    mlflow.set_tracking_uri("sqlite:///mlruns.db")
    mlflow.set_experiment(self.experiment_name)

    print(f"MLflow experiment: {self.experiment_name}")
    print(f"Tracking URI: {mlflow.get_tracking_uri()}")
```

### 8.2 Logging Strategy

```python
def log_training_run(
    self,
    position: str,
    target: str,
    params: Dict[str, Any],
    metrics: Dict[str, float],
    model_path: str,
    shap_artifacts: List[str]
):
    """
    Log complete training run to MLflow.

    Args:
        position: Player position
        target: Prediction target
        params: Model hyperparameters
        metrics: Performance metrics
        model_path: Path to saved model
        shap_artifacts: List of SHAP visualization paths
    """
    with mlflow.start_run(run_name=f"{position}_{target}"):
        # Log parameters
        mlflow.log_params(params)

        # Log metrics
        mlflow.log_metrics(metrics)

        # Log model
        mlflow.xgboost.log_model(
            self.model_registry[position][target],
            artifact_path="model",
            registered_model_name=f"{position}_{target}_xgboost"
        )

        # Log SHAP artifacts
        for artifact_path in shap_artifacts:
            mlflow.log_artifact(artifact_path)

        # Log tags
        mlflow.set_tags({
            'position': position,
            'target': target,
            'model_type': 'xgboost',
            'framework_version': xgb.__version__
        })
```

### 8.3 Model Registry

```python
def register_best_model(
    self,
    position: str,
    target: str,
    stage: str = "Production"
):
    """
    Register model in MLflow Model Registry.

    Args:
        position: Player position
        target: Prediction target
        stage: Model stage (Staging/Production/Archived)
    """
    model_name = f"{position}_{target}_xgboost"

    client = mlflow.tracking.MlflowClient()

    # Get latest version
    latest_versions = client.get_latest_versions(model_name)

    if latest_versions:
        latest_version = latest_versions[0].version

        # Transition to stage
        client.transition_model_version_stage(
            name=model_name,
            version=latest_version,
            stage=stage
        )

        print(f"Model {model_name} v{latest_version} transitioned to {stage}")
```

---

## 9. Test Cases

### 9.1 Unit Tests

```python
# File: tests/test_xgboost_predictor.py

import pytest
import numpy as np
import pandas as pd
from src.models.xgboost_predictor import XGBoostPredictor
from src.database import NFLDatabase
from src.config import NFLConfig


@pytest.fixture
def predictor():
    """Create predictor instance for testing."""
    db = NFLDatabase("test_nfl_predictions.duckdb")
    config = NFLConfig()
    return XGBoostPredictor(db, config, model_dir="test_models")


@pytest.fixture
def sample_data():
    """Create sample training data."""
    np.random.seed(42)
    n_samples = 100
    n_features = 20

    X = np.random.randn(n_samples, n_features)
    y_regression = np.random.randn(n_samples) * 50 + 200  # yards
    y_classification = np.random.randint(0, 2, n_samples)  # win/loss

    feature_names = [f"feature_{i}" for i in range(n_features)]

    return X, y_regression, y_classification, feature_names


class TestXGBoostPredictor:
    """Test suite for XGBoostPredictor."""

    def test_initialization(self, predictor):
        """Test predictor initialization."""
        assert predictor is not None
        assert predictor.model_registry == {}
        assert predictor.feature_info == {}

    def test_determine_task_type(self, predictor):
        """Test task type determination."""
        assert predictor._determine_task_type('passing_yards') == 'regression'
        assert predictor._determine_task_type('win_probability') == 'classification'
        assert predictor._determine_task_type('fg_made') == 'classification'

    def test_get_base_params(self, predictor):
        """Test base parameter retrieval."""
        reg_params = predictor._get_base_params('regression')
        assert reg_params['objective'] == 'reg:squarederror'

        clf_params = predictor._get_base_params('classification')
        assert clf_params['objective'] == 'binary:logistic'

    def test_train_regression_model(self, predictor, sample_data):
        """Test regression model training."""
        X, y_reg, _, feature_names = sample_data

        # Split data
        split = int(len(X) * 0.8)
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y_reg[:split], y_reg[split:]

        # Train model
        result = predictor.train(
            position='QB',
            target='passing_yards',
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            feature_names=feature_names,
            hyperparameter_tune=False
        )

        # Assertions
        assert 'metrics' in result
        assert 'val_rmse' in result['metrics']
        assert result['metrics']['val_rmse'] > 0
        assert 'QB' in predictor.model_registry
        assert 'passing_yards' in predictor.model_registry['QB']

    def test_train_classification_model(self, predictor, sample_data):
        """Test classification model training."""
        X, _, y_clf, feature_names = sample_data

        split = int(len(X) * 0.8)
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y_clf[:split], y_clf[split:]

        result = predictor.train(
            position='QB',
            target='win_probability',
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            feature_names=feature_names,
            hyperparameter_tune=False
        )

        assert 'metrics' in result
        assert 'val_accuracy' in result['metrics']
        assert 0 <= result['metrics']['val_accuracy'] <= 1

    def test_predict(self, predictor, sample_data):
        """Test prediction functionality."""
        X, y_reg, _, feature_names = sample_data

        # Train model first
        split = int(len(X) * 0.8)
        predictor.train(
            position='QB',
            target='passing_yards',
            X_train=X[:split],
            y_train=y_reg[:split],
            X_val=X[split:],
            y_val=y_reg[split:],
            feature_names=feature_names,
            hyperparameter_tune=False
        )

        # Make predictions
        predictions = predictor.predict('QB', 'passing_yards', X[:10])

        assert len(predictions) == 10
        assert all(isinstance(p, (int, float, np.number)) for p in predictions)

    def test_save_and_load_model(self, predictor, sample_data, tmp_path):
        """Test model persistence."""
        X, y_reg, _, feature_names = sample_data

        # Train model
        split = int(len(X) * 0.8)
        predictor.train(
            position='QB',
            target='passing_yards',
            X_train=X[:split],
            y_train=y_reg[:split],
            X_val=X[split:],
            y_val=y_reg[split:],
            feature_names=feature_names,
            hyperparameter_tune=False
        )

        # Save model
        model_path = tmp_path / "qb_passing_yards_test.json"
        predictor.save_model('QB', 'passing_yards', str(model_path))

        assert model_path.exists()

        # Create new predictor and load model
        new_predictor = XGBoostPredictor(
            predictor.db, predictor.config, model_dir=str(tmp_path)
        )
        new_predictor.load_model('QB', 'passing_yards', str(model_path))

        # Compare predictions
        pred1 = predictor.predict('QB', 'passing_yards', X[:5])
        pred2 = new_predictor.predict('QB', 'passing_yards', X[:5])

        np.testing.assert_array_almost_equal(pred1, pred2)

    def test_hyperparameter_tuning(self, predictor, sample_data):
        """Test hyperparameter tuning with Optuna."""
        X, y_reg, _, feature_names = sample_data

        split = int(len(X) * 0.8)
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y_reg[:split], y_reg[split:]

        # Tune with small number of trials
        result = predictor.train(
            position='QB',
            target='passing_yards',
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            feature_names=feature_names,
            hyperparameter_tune=True,
            n_trials=5  # Small number for testing
        )

        assert 'best_params' in result
        assert 'max_depth' in result['best_params']
        assert 'learning_rate' in result['best_params']

    def test_shap_calculation(self, predictor, sample_data):
        """Test SHAP value calculation."""
        X, y_reg, _, feature_names = sample_data

        # Train model
        split = int(len(X) * 0.8)
        result = predictor.train(
            position='QB',
            target='passing_yards',
            X_train=X[:split],
            y_train=y_reg[:split],
            X_val=X[split:],
            y_val=y_reg[split:],
            feature_names=feature_names,
            hyperparameter_tune=False
        )

        assert 'shap_results' in result
        assert 'mean_abs_shap' in result['shap_results']
        assert len(result['shap_results']['mean_abs_shap']) > 0

    def test_feature_importance(self, predictor, sample_data):
        """Test feature importance extraction."""
        X, y_reg, _, feature_names = sample_data

        # Train model
        split = int(len(X) * 0.8)
        predictor.train(
            position='QB',
            target='passing_yards',
            X_train=X[:split],
            y_train=y_reg[:split],
            X_val=X[split:],
            y_val=y_reg[split:],
            feature_names=feature_names,
            hyperparameter_tune=False
        )

        # Get feature importance
        importance_df = predictor.get_feature_importance('QB', 'passing_yards')

        assert len(importance_df) > 0
        assert 'feature' in importance_df.columns
        assert 'importance' in importance_df.columns

    def test_position_targets(self, predictor):
        """Test position target retrieval."""
        qb_targets = predictor.get_position_targets('QB')
        assert 'passing_yards' in qb_targets
        assert 'passing_tds' in qb_targets

        rb_targets = predictor.get_position_targets('RB')
        assert 'rushing_yards' in rb_targets

        wr_targets = predictor.get_position_targets('WR')
        assert 'receiving_yards' in wr_targets
```

### 9.2 Integration Tests

```python
# File: tests/test_xgboost_integration.py

import pytest
import numpy as np
from src.models.xgboost_predictor import XGBoostPredictor
from src.database import NFLDatabase
from src.config import NFLConfig


@pytest.mark.integration
class TestXGBoostIntegration:
    """Integration tests with real database."""

    def test_end_to_end_workflow(self):
        """Test complete workflow from data loading to prediction."""
        # Initialize
        db = NFLDatabase("test_nfl_predictions.duckdb")
        config = NFLConfig()
        predictor = XGBoostPredictor(db, config)

        # This assumes test data exists in database
        # In real test, would need to populate test data

        # Train model
        result = predictor.train_all_position_models(
            position='QB',
            season_range=(2023, 2024),
            hyperparameter_tune=False
        )

        assert len(result) > 0

        # Make predictions
        # (Would need test feature data)

    def test_model_versioning(self):
        """Test model versioning and registry."""
        db = NFLDatabase("test_nfl_predictions.duckdb")
        config = NFLConfig()
        predictor = XGBoostPredictor(db, config)

        # Train and register model
        # ... training code ...

        # Register in MLflow
        predictor.register_best_model('QB', 'passing_yards', 'Production')

        # Verify registration
        # ... verification code ...
```

---

## 10. Dependencies to Add

Update `pyproject.toml`:

```toml
dependencies = [
    # ... existing dependencies ...
    "xgboost>=2.0.0",
    "scikit-learn>=1.3.0",
    "optuna>=3.5.0",
    "shap>=0.44.0",
    "mlflow>=2.10.0",
    "matplotlib>=3.7.0",
    "seaborn>=0.13.0",
]
```

---

## 11. File Structure

```
src/
├── models/
│   ├── __init__.py
│   ├── base_model.py           # Abstract base class (created by another subagent)
│   └── xgboost_predictor.py    # XGBoost implementation
├── config.py
├── database.py
└── ...

models/
└── xgboost/
    ├── QB_passing_yards_model.json
    ├── QB_passing_yards_metadata.json
    ├── QB_passing_yards_model.features.json
    └── ...

mlruns/                          # MLflow tracking directory
└── ...

tests/
├── test_xgboost_predictor.py
└── test_xgboost_integration.py
```

---

## 12. Example Usage

```python
# Example: Train all QB models
from src.models.xgboost_predictor import XGBoostPredictor
from src.database import NFLDatabase
from src.config import NFLConfig

db = NFLDatabase()
config = NFLConfig()
predictor = XGBoostPredictor(db, config)

# Train all QB models with hyperparameter tuning
results = predictor.train_all_position_models(
    position='QB',
    season_range=(2021, 2024),
    hyperparameter_tune=True
)

# Print results
for target, result in results.items():
    print(f"\n{target}:")
    print(f"  Validation RMSE: {result['metrics']['val_rmse']:.2f}")
    print(f"  Validation R²: {result['metrics']['val_r2']:.3f}")

# Make predictions
import numpy as np

# Load test features (example)
X_test = np.random.randn(10, 20)  # 10 samples, 20 features

predictions = predictor.predict('QB', 'passing_yards', X_test)
print(f"\nPredictions: {predictions}")

# Get feature importance
importance = predictor.get_feature_importance('QB', 'passing_yards')
print(f"\nTop 5 features:\n{importance.head()}")

# Interpret SHAP results
interpretation = predictor.interpret_shap_results('QB', 'passing_yards')
print(interpretation)
```

---

## 13. Implementation Timeline

**Phase 1: Core Implementation (Week 1)**
- Implement XGBoostPredictor class structure
- Implement train/predict/save/load methods
- Add basic evaluation metrics
- Write unit tests

**Phase 2: Hyperparameter Tuning (Week 2)**
- Implement Optuna integration
- Define search spaces
- Test tuning on sample data

**Phase 3: SHAP Integration (Week 3)**
- Implement SHAP calculation
- Create visualizations
- Add interpretation methods

**Phase 4: MLflow Integration (Week 4)**
- Set up experiment tracking
- Implement model registry
- Add artifact logging

**Phase 5: Testing & Documentation (Week 5)**
- Write integration tests
- Performance testing on real data
- Documentation and examples

---

## 14. Performance Considerations

- **Training Time**: ~1-5 minutes per model depending on data size and hyperparameter tuning
- **Memory Usage**: XGBoost is memory-efficient; expect ~500MB-2GB for large datasets
- **SHAP Calculation**: TreeExplainer is fast for XGBoost; ~1-10 seconds for 1000 samples
- **Prediction Speed**: Very fast; ~1000 predictions/second

---

## 15. Future Enhancements

1. **Multi-output Models**: Train single model for multiple correlated targets
2. **Calibration**: Add probability calibration for classification models
3. **Uncertainty Quantification**: Implement prediction intervals using quantile regression
4. **Online Learning**: Support incremental model updates as new data arrives
5. **AutoML**: Integrate with AutoML frameworks for automated feature engineering
6. **Ensemble Methods**: Combine XGBoost with other models (LightGBM, Random Forest)
7. **GPU Support**: Enable GPU acceleration for large-scale training

---

## Summary

This implementation plan provides a comprehensive blueprint for building the XGBoostPredictor class with:

✅ Position-specific models (QB, RB, WR, TE, K, DEF)
✅ Regression and classification support
✅ Hyperparameter tuning with Optuna
✅ SHAP explanations for interpretability
✅ MLflow integration for experiment tracking
✅ Comprehensive testing strategy
✅ Production-ready code structure

The implementation follows best practices for ML model development and integrates seamlessly with the existing NFL prediction pipeline infrastructure.
