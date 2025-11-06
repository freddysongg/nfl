# XGBoost Implementation Guide - Step by Step

This guide provides a concrete, step-by-step approach to implementing the XGBoostPredictor class.

---

## Phase 1: Setup and Dependencies (Day 1)

### Step 1.1: Update Dependencies

Update `/home/user/nfl/pyproject.toml`:

```toml
dependencies = [
    "nflreadpy>=0.1.2",
    "requests>=2.28.0",
    "beautifulsoup4>=4.11.0",
    "duckdb>=1.4.0",
    "polars>=0.20.0",
    "pyarrow>=21.0.0",
    "numpy>=2.3.3",
    "pandas>=2.3.2",
    # NEW: ML dependencies
    "xgboost>=2.0.0",
    "scikit-learn>=1.3.0",
    "optuna>=3.5.0",
    "shap>=0.44.0",
    "mlflow>=2.10.0",
    "matplotlib>=3.7.0",
    "seaborn>=0.13.0",
]
```

Install dependencies:
```bash
uv sync
```

### Step 1.2: Create Directory Structure

```bash
mkdir -p src/models
mkdir -p models/xgboost
mkdir -p tests
mkdir -p mlruns
```

### Step 1.3: Create `__init__.py` Files

```bash
touch src/models/__init__.py
```

---

## Phase 2: Base Model Interface (Day 1)

**Note**: This will be created by another subagent, but here's the expected interface:

### Step 2.1: Create `/home/user/nfl/src/models/base_model.py`

```python
"""
Abstract base class for NFL prediction models.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import numpy as np
from src.database import NFLDatabase
from src.config import NFLConfig


class BaseModel(ABC):
    """
    Abstract base class for all NFL prediction models.

    All model implementations must inherit from this class and implement
    the required methods: train, predict, save_model, load_model.
    """

    def __init__(self, db: NFLDatabase, config: NFLConfig):
        """
        Initialize base model.

        Args:
            db: Database connection
            config: NFL configuration
        """
        self.db = db
        self.config = config

    @abstractmethod
    def train(
        self,
        position: str,
        target: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: List[str],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train model for specific position and target.

        Args:
            position: Player position (QB, RB, WR, TE, K, DEF)
            target: Target variable to predict
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            feature_names: List of feature names
            **kwargs: Additional training parameters

        Returns:
            Dictionary with training results and metrics
        """
        pass

    @abstractmethod
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
            Array of predictions
        """
        pass

    @abstractmethod
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
            filepath: Optional custom filepath

        Returns:
            Path where model was saved
        """
        pass

    @abstractmethod
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
            filepath: Optional custom filepath
        """
        pass
```

---

## Phase 3: Core XGBoostPredictor Implementation (Days 2-3)

### Step 3.1: Create `/home/user/nfl/src/models/xgboost_predictor.py`

Start with the basic structure:

```python
"""
XGBoost-based predictor for NFL player and team statistics.
"""

import xgboost as xgb
import numpy as np
import pandas as pd
import polars as pl
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import json
import joblib

from src.models.base_model import BaseModel
from src.database import NFLDatabase
from src.config import NFLConfig


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

    # Methods will be implemented step by step below...
```

### Step 3.2: Implement Helper Methods

Add these helper methods to the class:

```python
    def _determine_task_type(self, target: str) -> str:
        """
        Determine if target is regression or classification.

        Args:
            target: Target variable name

        Returns:
            'regression' or 'classification'
        """
        classification_targets = [
            'win_probability', 'over_under', 'fg_made',
            'td_scored', 'first_down_conversion'
        ]

        return 'classification' if target in classification_targets else 'regression'

    def _get_base_params(self, task_type: str) -> Dict[str, Any]:
        """
        Get base XGBoost parameters for task type.

        Args:
            task_type: 'regression' or 'classification'

        Returns:
            Dictionary of base parameters
        """
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

### Step 3.3: Implement Training Method

```python
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
        print(f"\n{'='*70}")
        print(f"Training {position} - {target}")
        print(f"{'='*70}")
        print(f"Train samples: {len(X_train)}, Val samples: {len(X_val)}")
        print(f"Features: {len(feature_names)}")

        # Determine task type
        task_type = self._determine_task_type(target)
        print(f"Task type: {task_type}")

        # Get base parameters
        base_params = self._get_base_params(task_type)

        # Hyperparameter tuning (skip for now, add later)
        best_params = base_params

        # Train final model
        print(f"Training final model...")
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

        print(f"\nTraining Results:")
        for metric_name, metric_value in metrics.items():
            print(f"  {metric_name}: {metric_value:.4f}")

        # Save model
        model_path = self._save_model(model, position, target)
        print(f"\nModel saved: {model_path}")

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
        }

        return results

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
```

### Step 3.4: Implement Prediction and Persistence Methods

```python
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

        print(f"Loaded model: {position} {target}")
        print(f"Features: {len(feature_data['feature_names'])}")

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
```

---

## Phase 4: Testing Basic Functionality (Day 3)

### Step 4.1: Create `/home/user/nfl/test_xgboost_basic.py`

```python
"""
Basic test script for XGBoostPredictor.
Tests with synthetic data before using real NFL data.
"""

import numpy as np
from src.models.xgboost_predictor import XGBoostPredictor
from src.database import NFLDatabase
from src.config import NFLConfig


def generate_synthetic_data(n_samples=200, n_features=20):
    """Generate synthetic training data."""
    np.random.seed(42)

    X = np.random.randn(n_samples, n_features)

    # Regression target: simulate passing yards
    y_regression = (
        X[:, 0] * 50 +  # Feature 0 is important
        X[:, 1] * 30 +  # Feature 1 is important
        np.random.randn(n_samples) * 20 +
        200  # Base value
    )

    # Classification target: win probability
    y_classification = (y_regression > 250).astype(int)

    feature_names = [f"feature_{i}" for i in range(n_features)]

    return X, y_regression, y_classification, feature_names


def test_basic_training():
    """Test basic model training."""
    print("Test 1: Basic Training")
    print("="*70)

    # Initialize
    db = NFLDatabase("test_nfl_predictions.duckdb")
    config = NFLConfig()
    predictor = XGBoostPredictor(db, config, model_dir="test_models")

    # Generate data
    X, y_reg, y_clf, feature_names = generate_synthetic_data()

    # Split
    split = int(len(X) * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_train_reg, y_val_reg = y_reg[:split], y_reg[split:]

    # Train regression model
    result = predictor.train(
        position='QB',
        target='passing_yards',
        X_train=X_train,
        y_train=y_train_reg,
        X_val=X_val,
        y_val=y_val_reg,
        feature_names=feature_names,
        hyperparameter_tune=False
    )

    assert result['metrics']['val_rmse'] > 0
    print("\n✅ Training test passed!")

    return predictor, X_val


def test_prediction(predictor, X_test):
    """Test prediction."""
    print("\n\nTest 2: Prediction")
    print("="*70)

    predictions = predictor.predict('QB', 'passing_yards', X_test)

    print(f"Predictions: {predictions[:5]}")
    assert len(predictions) == len(X_test)
    print("\n✅ Prediction test passed!")

    return predictions


def test_save_load(predictor):
    """Test model persistence."""
    print("\n\nTest 3: Save and Load")
    print("="*70)

    # Save
    model_path = predictor.save_model('QB', 'passing_yards')
    print(f"Model saved: {model_path}")

    # Create new predictor and load
    new_predictor = XGBoostPredictor(
        predictor.db, predictor.config, model_dir="test_models"
    )
    new_predictor.load_model('QB', 'passing_yards')

    print("\n✅ Save/load test passed!")

    return new_predictor


def test_feature_importance(predictor):
    """Test feature importance."""
    print("\n\nTest 4: Feature Importance")
    print("="*70)

    importance = predictor.get_feature_importance('QB', 'passing_yards')
    print(f"\nTop 5 features:\n{importance.head()}")

    assert len(importance) > 0
    print("\n✅ Feature importance test passed!")


def main():
    """Run all tests."""
    print("XGBoost Predictor - Basic Tests")
    print("="*70)

    # Test 1: Training
    predictor, X_test = test_basic_training()

    # Test 2: Prediction
    predictions = test_prediction(predictor, X_test)

    # Test 3: Save/Load
    new_predictor = test_save_load(predictor)

    # Test 4: Feature Importance (add method first!)
    # test_feature_importance(predictor)

    print("\n\n" + "="*70)
    print("All basic tests passed! ✅")
    print("="*70)


if __name__ == "__main__":
    main()
```

### Step 4.2: Run Basic Tests

```bash
python test_xgboost_basic.py
```

Expected output:
```
XGBoost Predictor - Basic Tests
======================================================================
Test 1: Basic Training
======================================================================

======================================================================
Training QB - passing_yards
======================================================================
Train samples: 160, Val samples: 40
Features: 20
Task type: regression
Training final model...

Training Results:
  train_rmse: 15.2341
  train_mae: 12.3421
  train_r2: 0.9234
  val_rmse: 18.4532
  val_mae: 14.2341
  val_r2: 0.8932

Model saved: test_models/QB_passing_yards_model.json

✅ Training test passed!


Test 2: Prediction
======================================================================
Predictions: [234.5 256.7 198.3 278.9 212.1]

✅ Prediction test passed!


Test 3: Save and Load
======================================================================
Model saved: test_models/QB_passing_yards_model.json
Loaded model: QB passing_yards
Features: 20

✅ Save/load test passed!

======================================================================
All basic tests passed! ✅
======================================================================
```

---

## Phase 5: Feature Importance and SHAP (Days 4-5)

### Step 5.1: Add Feature Importance Method

Add to `XGBoostPredictor` class:

```python
    def get_feature_importance(
        self,
        position: str,
        target: str,
        importance_type: str = 'gain'
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
        model = self.model_registry[position][target]
        importance = model.get_booster().get_score(importance_type=importance_type)

        df = pd.DataFrame([
            {'feature': k, 'importance': v}
            for k, v in importance.items()
        ]).sort_values('importance', ascending=False)

        return df
```

### Step 5.2: Add SHAP Calculation Method

```python
import shap
import matplotlib.pyplot as plt
import seaborn as sns

# Add to XGBoostPredictor class:

    def calculate_shap_values(
        self,
        position: str,
        target: str,
        X_sample: np.ndarray
    ) -> Dict[str, Any]:
        """
        Calculate SHAP values for model interpretability.

        Args:
            position: Player position
            target: Prediction target
            X_sample: Sample data (max 1000 rows)

        Returns:
            Dictionary with SHAP results
        """
        model = self.model_registry[position][target]
        feature_names = self.feature_info[position][target]

        # Limit sample size
        if len(X_sample) > 1000:
            X_sample = X_sample[:1000]

        print(f"Calculating SHAP values for {len(X_sample)} samples...")

        # Create TreeExplainer
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X_sample)

        # Create visualizations directory
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
        print(f"Saved: {bar_plot_path}")

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
        print(f"Saved: {beeswarm_plot_path}")

        # Calculate mean absolute SHAP values
        mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'mean_abs_shap': mean_abs_shap
        }).sort_values('mean_abs_shap', ascending=False)

        return {
            'shap_values': shap_values,
            'mean_abs_shap': feature_importance_df.to_dict('records'),
            'visualizations': {
                'summary_bar': str(bar_plot_path),
                'summary_beeswarm': str(beeswarm_plot_path),
            }
        }
```

### Step 5.3: Update Training Method to Include SHAP

Update the `train` method to calculate SHAP:

```python
    # In train() method, after model evaluation:

    # Calculate SHAP values
    print(f"Calculating SHAP values...")
    shap_results = self.calculate_shap_values(
        position, target, X_val[:1000]
    )

    results = {
        'position': position,
        'target': target,
        'metrics': metrics,
        'best_params': best_params,
        'model_path': str(model_path),
        'shap_results': shap_results  # Add this
    }
```

---

## Phase 6: Hyperparameter Tuning with Optuna (Days 6-7)

### Step 6.1: Add Optuna Tuning Method

```python
import optuna

# Add to XGBoostPredictor class:

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
            base_params: Base parameters
            task_type: 'regression' or 'classification'
            n_trials: Number of trials

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

            # Evaluate
            y_val_pred = model.predict(X_val)

            if task_type == 'regression':
                from sklearn.metrics import mean_squared_error
                score = np.sqrt(mean_squared_error(y_val, y_val_pred))
                return score  # Minimize RMSE
            else:
                from sklearn.metrics import roc_auc_score
                y_val_proba = model.predict_proba(X_val)[:, 1]
                score = roc_auc_score(y_val, y_val_proba)
                return -score  # Minimize (negate AUC)

        # Run optimization
        study = optuna.create_study(
            direction='minimize',
            sampler=optuna.samplers.TPESampler(seed=42)
        )

        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        print(f"Best trial: {study.best_trial.number}")
        print(f"Best value: {study.best_trial.value:.4f}")

        # Merge with base params
        best_params = {**base_params, **study.best_trial.params}

        return best_params
```

### Step 6.2: Update Training Method to Use Optuna

```python
    # In train() method, replace this line:
    # best_params = base_params

    # With:
    if hyperparameter_tune:
        print(f"Starting hyperparameter tuning ({n_trials} trials)...")
        best_params = self._tune_hyperparameters(
            X_train, y_train, X_val, y_val,
            base_params, task_type, n_trials
        )
    else:
        best_params = base_params
```

---

## Phase 7: MLflow Integration (Day 8)

### Step 7.1: Add MLflow Setup

```python
import mlflow
import mlflow.xgboost

# Add to __init__:

    def __init__(self, ...):
        # ... existing code ...

        # MLflow setup
        self.experiment_name = experiment_name
        self._setup_mlflow()

    def _setup_mlflow(self):
        """Configure MLflow tracking."""
        mlflow.set_tracking_uri("sqlite:///mlruns.db")
        mlflow.set_experiment(self.experiment_name)

        print(f"MLflow experiment: {self.experiment_name}")
        print(f"Tracking URI: {mlflow.get_tracking_uri()}")
```

### Step 7.2: Update Training Method with MLflow Logging

```python
    def train(self, ...):
        # Wrap entire training in MLflow run
        with mlflow.start_run(run_name=f"{position}_{target}"):
            # Log parameters
            mlflow.log_param("position", position)
            mlflow.log_param("target", target)
            mlflow.log_param("n_features", len(feature_names))
            mlflow.log_param("n_train_samples", len(X_train))
            mlflow.log_param("n_val_samples", len(X_val))
            mlflow.log_param("task_type", task_type)

            # ... existing training code ...

            # Log hyperparameters
            mlflow.log_params(best_params)

            # Log metrics
            mlflow.log_metrics(metrics)

            # Log model
            mlflow.xgboost.log_model(
                model,
                artifact_path="model"
            )

            # Log SHAP artifacts
            if 'shap_results' in result:
                for viz_name, viz_path in result['shap_results']['visualizations'].items():
                    mlflow.log_artifact(viz_path)

            # Set tags
            mlflow.set_tags({
                'position': position,
                'target': target,
                'model_type': 'xgboost',
            })

            # ... rest of training code ...
```

---

## Phase 8: Integration with Real Data (Days 9-10)

### Step 8.1: Add Data Loading Method

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

        print(f"Loaded {len(df)} samples from database")

        if len(df) == 0:
            raise ValueError(f"No data found for {position} {target}")

        # Extract features and targets
        X = np.vstack(df['numerical_features'].to_list())

        # Extract target from actual_outcomes JSON
        y = np.array([
            json.loads(outcome)[target]
            for outcome in df['actual_outcomes']
        ])

        # Get feature names
        feature_names = json.loads(df['feature_names'][0])

        # Temporal split: Use last 20% for validation
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        print(f"Train: {len(X_train)}, Val: {len(X_val)}")
        print(f"Features: {len(feature_names)}")

        return X_train, y_train, X_val, y_val, feature_names
```

### Step 8.2: Add Training Convenience Method

```python
    def train_all_position_models(
        self,
        position: str,
        season_range: Tuple[int, int] = (2021, 2024),
        hyperparameter_tune: bool = True,
        n_trials: int = 50
    ) -> Dict[str, Any]:
        """
        Train all models for a specific position.

        Args:
            position: Player position
            season_range: Range of seasons (start, end)
            hyperparameter_tune: Whether to tune hyperparameters
            n_trials: Number of Optuna trials

        Returns:
            Training results for all targets
        """
        results = {}
        targets = self.get_position_targets(position)

        print(f"\nTraining {len(targets)} models for position: {position}")
        print(f"Season range: {season_range[0]}-{season_range[1]}")
        print(f"Hyperparameter tuning: {hyperparameter_tune}")
        print("="*70)

        for target in targets:
            print(f"\n{'='*70}")
            print(f"Training {position} - {target}")
            print(f"{'='*70}")

            try:
                # Load data
                X_train, y_train, X_val, y_val, feature_names = (
                    self._load_training_data(position, target, season_range)
                )

                # Train model
                result = self.train(
                    position, target,
                    X_train, y_train, X_val, y_val,
                    feature_names,
                    hyperparameter_tune=hyperparameter_tune,
                    n_trials=n_trials
                )

                results[target] = result

            except Exception as e:
                print(f"❌ Error training {position} {target}: {e}")
                results[target] = {'error': str(e)}

        print(f"\n{'='*70}")
        print(f"Training complete for {position}")
        print(f"Success: {sum(1 for r in results.values() if 'error' not in r)}/{len(results)}")
        print(f"{'='*70}")

        return results
```

---

## Phase 9: Comprehensive Testing (Day 11)

### Step 9.1: Create Full Test Suite

Create `/home/user/nfl/tests/test_xgboost_predictor.py`:

```python
"""
Comprehensive test suite for XGBoostPredictor.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
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
    y_regression = np.random.randn(n_samples) * 50 + 200
    y_classification = np.random.randint(0, 2, n_samples)

    feature_names = [f"feature_{i}" for i in range(n_features)]

    return X, y_regression, y_classification, feature_names


class TestXGBoostPredictor:
    def test_initialization(self, predictor):
        assert predictor is not None
        assert predictor.model_registry == {}

    def test_determine_task_type(self, predictor):
        assert predictor._determine_task_type('passing_yards') == 'regression'
        assert predictor._determine_task_type('win_probability') == 'classification'

    def test_train_regression(self, predictor, sample_data):
        X, y_reg, _, feature_names = sample_data
        split = int(len(X) * 0.8)

        result = predictor.train(
            'QB', 'passing_yards',
            X[:split], y_reg[:split],
            X[split:], y_reg[split:],
            feature_names,
            hyperparameter_tune=False
        )

        assert 'metrics' in result
        assert 'val_rmse' in result['metrics']

    def test_predict(self, predictor, sample_data):
        X, y_reg, _, feature_names = sample_data
        split = int(len(X) * 0.8)

        predictor.train(
            'QB', 'passing_yards',
            X[:split], y_reg[:split],
            X[split:], y_reg[split:],
            feature_names,
            hyperparameter_tune=False
        )

        predictions = predictor.predict('QB', 'passing_yards', X[:10])
        assert len(predictions) == 10

    def test_save_load(self, predictor, sample_data, tmp_path):
        X, y_reg, _, feature_names = sample_data
        split = int(len(X) * 0.8)

        predictor.train(
            'QB', 'passing_yards',
            X[:split], y_reg[:split],
            X[split:], y_reg[split:],
            feature_names,
            hyperparameter_tune=False
        )

        model_path = tmp_path / "test_model.json"
        predictor.save_model('QB', 'passing_yards', str(model_path))

        assert model_path.exists()

        new_predictor = XGBoostPredictor(
            predictor.db, predictor.config, model_dir=str(tmp_path)
        )
        new_predictor.load_model('QB', 'passing_yards', str(model_path))

        pred1 = predictor.predict('QB', 'passing_yards', X[:5])
        pred2 = new_predictor.predict('QB', 'passing_yards', X[:5])

        np.testing.assert_array_almost_equal(pred1, pred2)
```

### Step 9.2: Run Tests

```bash
pytest tests/test_xgboost_predictor.py -v
```

---

## Phase 10: Documentation and Examples (Day 12)

### Step 10.1: Create Usage Examples

Create `/home/user/nfl/examples/train_xgboost_example.py`:

```python
"""
Example: Training XGBoost models for NFL predictions.
"""

from src.models.xgboost_predictor import XGBoostPredictor
from src.database import NFLDatabase
from src.config import NFLConfig


def main():
    # Initialize
    db = NFLDatabase()
    config = NFLConfig()
    predictor = XGBoostPredictor(db, config)

    # Train all QB models with hyperparameter tuning
    print("Training QB models with hyperparameter tuning...")
    results = predictor.train_all_position_models(
        position='QB',
        season_range=(2021, 2024),
        hyperparameter_tune=True,
        n_trials=20  # Use 20 trials for speed
    )

    # Print results
    print("\n\nTraining Results:")
    print("="*70)
    for target, result in results.items():
        if 'error' not in result:
            print(f"\n{target}:")
            print(f"  Validation RMSE: {result['metrics']['val_rmse']:.2f}")
            print(f"  Validation R²: {result['metrics']['val_r2']:.3f}")
            print(f"  Model path: {result['model_path']}")
        else:
            print(f"\n{target}: ERROR - {result['error']}")


if __name__ == "__main__":
    main()
```

---

## Summary of Implementation Steps

1. **Day 1**: Setup dependencies, create base model interface
2. **Days 2-3**: Implement core XGBoostPredictor (train, predict, save, load)
3. **Day 3**: Test with synthetic data
4. **Days 4-5**: Add feature importance and SHAP
5. **Days 6-7**: Add Optuna hyperparameter tuning
6. **Day 8**: Add MLflow integration
7. **Days 9-10**: Integrate with real NFL data
8. **Day 11**: Comprehensive testing
9. **Day 12**: Documentation and examples

## Expected Files After Implementation

```
src/
├── models/
│   ├── __init__.py
│   ├── base_model.py
│   └── xgboost_predictor.py

models/
└── xgboost/
    ├── QB_passing_yards_model.json
    ├── QB_passing_yards_metadata.json
    ├── QB_passing_yards_model.features.json
    └── shap_visualizations/

tests/
└── test_xgboost_predictor.py

examples/
└── train_xgboost_example.py
```

---

## Final Checklist

- [ ] Dependencies installed (`uv sync`)
- [ ] Base model interface created
- [ ] XGBoostPredictor class implemented
- [ ] Train/predict/save/load methods working
- [ ] Feature importance working
- [ ] SHAP calculation working
- [ ] Optuna tuning working
- [ ] MLflow tracking working
- [ ] Data loading from database working
- [ ] All tests passing
- [ ] Documentation complete
- [ ] Example scripts working
