"""
Unit tests for XGBoostPredictor model.

Tests model training, prediction, persistence, and feature importance
in isolation using synthetic data.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from src.models.xgboost_predictor import XGBoostPredictor


@pytest.fixture
def sample_training_data():
    """Generate synthetic training data for testing."""
    np.random.seed(42)
    n_samples = 100
    n_features = 47

    X_train = np.random.randn(n_samples, n_features)
    X_val = np.random.randn(30, n_features)
    X_test = np.random.randn(20, n_features)

    # Regression targets
    y_train = 200 + 50 * X_train[:, 0] + np.random.randn(n_samples) * 10
    y_val = 200 + 50 * X_val[:, 0] + np.random.randn(30) * 10
    y_test = 200 + 50 * X_test[:, 0] + np.random.randn(20) * 10

    feature_names = [f"feature_{i}" for i in range(n_features)]

    return {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test,
        'feature_names': feature_names
    }


@pytest.fixture
def sample_classification_data():
    """Generate synthetic classification data for testing."""
    np.random.seed(42)
    n_samples = 100
    n_features = 47

    X_train = np.random.randn(n_samples, n_features)
    X_val = np.random.randn(30, n_features)
    X_test = np.random.randn(20, n_features)

    # Binary classification targets
    y_train = (X_train[:, 0] + X_train[:, 1] > 0).astype(int)
    y_val = (X_val[:, 0] + X_val[:, 1] > 0).astype(int)
    y_test = (X_test[:, 0] + X_test[:, 1] > 0).astype(int)

    feature_names = [f"feature_{i}" for i in range(n_features)]

    return {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test,
        'feature_names': feature_names
    }


@pytest.fixture
def temp_model_dir(tmp_path):
    """Create temporary model directory."""
    model_dir = tmp_path / "test_models"
    model_dir.mkdir()
    return str(model_dir)


class TestXGBoostPredictor:
    """Unit tests for XGBoostPredictor"""

    def test_initialization(self, temp_model_dir):
        """Test predictor initialization."""
        predictor = XGBoostPredictor(model_dir=temp_model_dir, use_mlflow=False)
        assert predictor.model_dir.exists()
        assert predictor.use_mlflow == False
        assert predictor.model_registry == {}
        assert predictor.feature_info == {}
        assert predictor.training_metadata == {}
        assert predictor.random_seed == 42

    def test_initialization_custom_seed(self, temp_model_dir):
        """Test predictor initialization with custom random seed."""
        predictor = XGBoostPredictor(
            model_dir=temp_model_dir,
            use_mlflow=False,
            random_seed=123
        )
        assert predictor.random_seed == 123

    def test_train_regression_model(self, sample_training_data, temp_model_dir):
        """Test training a regression model."""
        predictor = XGBoostPredictor(model_dir=temp_model_dir, use_mlflow=False)

        result = predictor.train(
            position="QB",
            target="passing_yards",
            X_train=sample_training_data['X_train'],
            y_train=sample_training_data['y_train'],
            X_val=sample_training_data['X_val'],
            y_val=sample_training_data['y_val'],
            feature_names=sample_training_data['feature_names']
        )

        # Assertions
        assert "QB" in predictor.model_registry
        assert "passing_yards" in predictor.model_registry["QB"]
        assert result['metrics']['train_rmse'] > 0
        assert result['metrics']['val_rmse'] > 0
        assert result['metrics']['train_r2'] >= 0
        assert result['position'] == "QB"
        assert result['target'] == "passing_yards"
        assert 'best_params' in result
        assert 'training_duration' in result

    def test_train_classification_model(self, sample_classification_data, temp_model_dir):
        """Test training a classification model."""
        predictor = XGBoostPredictor(model_dir=temp_model_dir, use_mlflow=False)

        result = predictor.train(
            position="QB",
            target="win_probability",
            X_train=sample_classification_data['X_train'],
            y_train=sample_classification_data['y_train'],
            X_val=sample_classification_data['X_val'],
            y_val=sample_classification_data['y_val'],
            feature_names=sample_classification_data['feature_names']
        )

        # Assertions
        assert "QB" in predictor.model_registry
        assert "win_probability" in predictor.model_registry["QB"]
        assert 'train_accuracy' in result['metrics']
        assert 'val_accuracy' in result['metrics']
        assert 'val_roc_auc' in result['metrics']
        assert 0 <= result['metrics']['val_roc_auc'] <= 1

    def test_predict_before_training_raises_error(self, sample_training_data, temp_model_dir):
        """Test that predict raises error if model not trained."""
        predictor = XGBoostPredictor(model_dir=temp_model_dir, use_mlflow=False)

        with pytest.raises(ValueError, match="No models trained"):
            predictor.predict("QB", "passing_yards", sample_training_data['X_test'])

    def test_predict_after_training(self, sample_training_data, temp_model_dir):
        """Test making predictions after training."""
        predictor = XGBoostPredictor(model_dir=temp_model_dir, use_mlflow=False)

        # Train
        predictor.train(
            position="QB", target="passing_yards",
            X_train=sample_training_data['X_train'],
            y_train=sample_training_data['y_train'],
            X_val=sample_training_data['X_val'],
            y_val=sample_training_data['y_val'],
            feature_names=sample_training_data['feature_names']
        )

        # Predict
        predictions = predictor.predict("QB", "passing_yards", sample_training_data['X_test'])

        assert predictions.shape == (20,)
        assert np.all(np.isfinite(predictions))
        assert predictions.dtype in [np.float32, np.float64]

    def test_predict_proba_classification(self, sample_classification_data, temp_model_dir):
        """Test probability predictions for classification models."""
        predictor = XGBoostPredictor(model_dir=temp_model_dir, use_mlflow=False)

        # Train classification model
        predictor.train(
            position="QB", target="win_probability",
            X_train=sample_classification_data['X_train'],
            y_train=sample_classification_data['y_train'],
            X_val=sample_classification_data['X_val'],
            y_val=sample_classification_data['y_val'],
            feature_names=sample_classification_data['feature_names']
        )

        # Predict probabilities
        probas = predictor.predict_proba("QB", "win_probability", sample_classification_data['X_test'])

        assert probas.shape == (20, 2)  # Binary classification
        assert np.all(probas >= 0) and np.all(probas <= 1)
        assert np.allclose(probas.sum(axis=1), 1.0)  # Probabilities sum to 1

    def test_predict_proba_regression_raises_error(self, sample_training_data, temp_model_dir):
        """Test that predict_proba raises error for regression models."""
        predictor = XGBoostPredictor(model_dir=temp_model_dir, use_mlflow=False)

        # Train regression model
        predictor.train(
            position="QB", target="passing_yards",
            X_train=sample_training_data['X_train'],
            y_train=sample_training_data['y_train'],
            X_val=sample_training_data['X_val'],
            y_val=sample_training_data['y_val'],
            feature_names=sample_training_data['feature_names']
        )

        # Try to get probabilities from regression model
        with pytest.raises(ValueError, match="does not support predict_proba"):
            predictor.predict_proba("QB", "passing_yards", sample_training_data['X_test'])

    def test_save_and_load_model(self, sample_training_data, temp_model_dir):
        """Test model persistence."""
        # Train and save
        predictor1 = XGBoostPredictor(model_dir=temp_model_dir, use_mlflow=False)
        predictor1.train(
            position="QB", target="passing_yards",
            X_train=sample_training_data['X_train'],
            y_train=sample_training_data['y_train'],
            X_val=sample_training_data['X_val'],
            y_val=sample_training_data['y_val'],
            feature_names=sample_training_data['feature_names']
        )
        predictor1.save("QB", "passing_yards")

        # Load and predict
        predictor2 = XGBoostPredictor(model_dir=temp_model_dir, use_mlflow=False)
        predictor2.load("QB", "passing_yards")
        predictions = predictor2.predict("QB", "passing_yards", sample_training_data['X_test'])

        assert predictions.shape == (20,)
        assert np.all(np.isfinite(predictions))
        assert "QB" in predictor2.model_registry
        assert "passing_yards" in predictor2.feature_info["QB"]

    def test_load_nonexistent_model_raises_error(self, temp_model_dir):
        """Test that loading a non-existent model raises error."""
        predictor = XGBoostPredictor(model_dir=temp_model_dir, use_mlflow=False)

        with pytest.raises(FileNotFoundError):
            predictor.load("QB", "passing_yards")

    def test_save_untrained_model_raises_error(self, temp_model_dir):
        """Test that saving an untrained model raises error."""
        predictor = XGBoostPredictor(model_dir=temp_model_dir, use_mlflow=False)

        with pytest.raises(ValueError, match="No models trained"):
            predictor.save("QB", "passing_yards")

    def test_get_feature_importance(self, sample_training_data, temp_model_dir):
        """Test feature importance extraction."""
        predictor = XGBoostPredictor(model_dir=temp_model_dir, use_mlflow=False)

        predictor.train(
            position="QB", target="passing_yards",
            X_train=sample_training_data['X_train'],
            y_train=sample_training_data['y_train'],
            X_val=sample_training_data['X_val'],
            y_val=sample_training_data['y_val'],
            feature_names=sample_training_data['feature_names']
        )

        importance = predictor.get_feature_importance("QB", "passing_yards")

        assert isinstance(importance, pd.DataFrame)
        assert len(importance) > 0
        assert 'feature' in importance.columns
        assert 'importance' in importance.columns
        assert importance['importance'].sum() > 0

    def test_get_feature_importance_untrained_raises_error(self, temp_model_dir):
        """Test that getting feature importance for untrained model raises error."""
        predictor = XGBoostPredictor(model_dir=temp_model_dir, use_mlflow=False)

        with pytest.raises(ValueError, match="No models trained"):
            predictor.get_feature_importance("QB", "passing_yards")

    def test_multiple_positions(self, sample_training_data, temp_model_dir):
        """Test training models for multiple positions."""
        predictor = XGBoostPredictor(model_dir=temp_model_dir, use_mlflow=False)

        # Train QB
        predictor.train(
            position="QB", target="passing_yards",
            X_train=sample_training_data['X_train'],
            y_train=sample_training_data['y_train'],
            X_val=sample_training_data['X_val'],
            y_val=sample_training_data['y_val'],
            feature_names=sample_training_data['feature_names']
        )

        # Train RB
        predictor.train(
            position="RB", target="rushing_yards",
            X_train=sample_training_data['X_train'],
            y_train=sample_training_data['y_train'],
            X_val=sample_training_data['X_val'],
            y_val=sample_training_data['y_val'],
            feature_names=sample_training_data['feature_names']
        )

        assert "QB" in predictor.model_registry
        assert "RB" in predictor.model_registry
        assert "passing_yards" in predictor.model_registry["QB"]
        assert "rushing_yards" in predictor.model_registry["RB"]

        # Verify independent predictions
        qb_pred = predictor.predict("QB", "passing_yards", sample_training_data['X_test'])
        rb_pred = predictor.predict("RB", "rushing_yards", sample_training_data['X_test'])

        assert qb_pred.shape == (20,)
        assert rb_pred.shape == (20,)

    def test_multiple_targets_same_position(self, sample_training_data, temp_model_dir):
        """Test training multiple targets for the same position."""
        predictor = XGBoostPredictor(model_dir=temp_model_dir, use_mlflow=False)

        # Train passing_yards
        predictor.train(
            position="QB", target="passing_yards",
            X_train=sample_training_data['X_train'],
            y_train=sample_training_data['y_train'],
            X_val=sample_training_data['X_val'],
            y_val=sample_training_data['y_val'],
            feature_names=sample_training_data['feature_names']
        )

        # Train passing_tds
        predictor.train(
            position="QB", target="passing_tds",
            X_train=sample_training_data['X_train'],
            y_train=sample_training_data['y_train'] / 50,  # Scale down for TDs
            X_val=sample_training_data['X_val'],
            y_val=sample_training_data['y_val'] / 50,
            feature_names=sample_training_data['feature_names']
        )

        assert "QB" in predictor.model_registry
        assert "passing_yards" in predictor.model_registry["QB"]
        assert "passing_tds" in predictor.model_registry["QB"]
        assert len(predictor.model_registry["QB"]) == 2

    def test_get_position_targets(self, temp_model_dir):
        """Test getting available targets for a position."""
        predictor = XGBoostPredictor(model_dir=temp_model_dir, use_mlflow=False)

        qb_targets = predictor.get_position_targets("QB")
        assert isinstance(qb_targets, list)
        assert len(qb_targets) > 0
        assert "passing_yards" in qb_targets
        assert "passing_tds" in qb_targets

        rb_targets = predictor.get_position_targets("RB")
        assert "rushing_yards" in rb_targets
        assert "rushing_tds" in rb_targets

        # Invalid position returns empty list
        invalid_targets = predictor.get_position_targets("INVALID")
        assert invalid_targets == []

    def test_get_trained_models(self, sample_training_data, temp_model_dir):
        """Test getting list of trained models."""
        predictor = XGBoostPredictor(model_dir=temp_model_dir, use_mlflow=False)

        # Initially empty
        assert predictor.get_trained_models() == []

        # Train two models
        predictor.train(
            position="QB", target="passing_yards",
            X_train=sample_training_data['X_train'],
            y_train=sample_training_data['y_train'],
            X_val=sample_training_data['X_val'],
            y_val=sample_training_data['y_val'],
            feature_names=sample_training_data['feature_names']
        )
        predictor.train(
            position="RB", target="rushing_yards",
            X_train=sample_training_data['X_train'],
            y_train=sample_training_data['y_train'],
            X_val=sample_training_data['X_val'],
            y_val=sample_training_data['y_val'],
            feature_names=sample_training_data['feature_names']
        )

        trained = predictor.get_trained_models()
        assert len(trained) == 2
        assert ("QB", "passing_yards") in trained
        assert ("RB", "rushing_yards") in trained

    def test_get_model_info(self, sample_training_data, temp_model_dir):
        """Test getting model information."""
        predictor = XGBoostPredictor(model_dir=temp_model_dir, use_mlflow=False)

        predictor.train(
            position="QB", target="passing_yards",
            X_train=sample_training_data['X_train'],
            y_train=sample_training_data['y_train'],
            X_val=sample_training_data['X_val'],
            y_val=sample_training_data['y_val'],
            feature_names=sample_training_data['feature_names']
        )

        info = predictor.get_model_info("QB", "passing_yards")

        assert info['position'] == "QB"
        assert info['target'] == "passing_yards"
        assert info['n_features'] == 47
        assert len(info['feature_names']) == 47
        assert 'metadata' in info

    def test_get_model_info_untrained_raises_error(self, temp_model_dir):
        """Test that getting info for untrained model raises error."""
        predictor = XGBoostPredictor(model_dir=temp_model_dir, use_mlflow=False)

        with pytest.raises(ValueError, match="No models trained"):
            predictor.get_model_info("QB", "passing_yards")

    def test_determine_task_type(self, temp_model_dir):
        """Test task type determination."""
        predictor = XGBoostPredictor(model_dir=temp_model_dir, use_mlflow=False)

        assert predictor._determine_task_type("passing_yards") == "regression"
        assert predictor._determine_task_type("rushing_tds") == "regression"
        assert predictor._determine_task_type("win_probability") == "classification"
        assert predictor._determine_task_type("fg_made") == "classification"

    def test_model_persistence_creates_files(self, sample_training_data, temp_model_dir):
        """Test that saving model creates expected files."""
        predictor = XGBoostPredictor(model_dir=temp_model_dir, use_mlflow=False)

        predictor.train(
            position="QB", target="passing_yards",
            X_train=sample_training_data['X_train'],
            y_train=sample_training_data['y_train'],
            X_val=sample_training_data['X_val'],
            y_val=sample_training_data['y_val'],
            feature_names=sample_training_data['feature_names']
        )

        model_path = predictor.save("QB", "passing_yards")

        # Check files exist
        assert Path(model_path).exists()

        # Check for feature info file
        feature_path = Path(model_path).parent / "QB_passing_yards_model.features.json"
        assert feature_path.exists()

        # Check for metadata file
        metadata_path = Path(model_path).parent / "QB_passing_yards_metadata.json"
        assert metadata_path.exists()

    def test_repr(self, temp_model_dir):
        """Test string representation."""
        predictor = XGBoostPredictor(model_dir=temp_model_dir, use_mlflow=False)
        repr_str = repr(predictor)

        assert "XGBoostPredictor" in repr_str
        assert "models=0" in repr_str
        assert temp_model_dir in repr_str
