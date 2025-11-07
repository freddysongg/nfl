"""
Integration tests for ML Pipeline.

Tests the ML workflow integration:
- XGBoost predictor training and prediction
- NFLTrainer data loading and model training
- Model registry integration
- Complete train → predict → evaluate workflow

These tests verify that ML components work together correctly.
"""

import pytest
import numpy as np
import duckdb
from pathlib import Path
import json

from src.models.xgboost_predictor import XGBoostPredictor
from src.models.model_registry import ModelRegistry
from src.training.trainer import NFLTrainer
from src.table_schemas import create_ml_training_features_table


@pytest.fixture
def ml_integration_db(tmp_path):
    """
    Create database with ml_training_features for ML testing.

    Creates 200 samples across 2 seasons (2023-2024) for:
    - 2 QBs (100 samples each)
    - 2 WRs (50 samples each)
    - 2 RBs (50 samples each)
    """
    db_path = tmp_path / "ml_integration.duckdb"
    conn = duckdb.connect(str(db_path))

    # Create ml_training_features table
    create_ml_training_features_table(conn)

    # Insert realistic synthetic data
    _insert_ml_training_data(conn)

    conn.close()

    yield str(db_path)

    # Cleanup
    db_path.unlink(missing_ok=True)


def _insert_ml_training_data(conn):
    """
    Insert realistic ML training data.

    Features:
    - 47 numerical features (as per Stage 4 design)
    - Categorical features (position, experience, etc.)
    - Actual outcomes (targets)
    """
    np.random.seed(42)

    players = [
        ("QB001", "QB", 2),  # Veteran QB
        ("QB002", "QB", 0),  # Rookie QB
        ("WR001", "WR", 3),  # Veteran WR
        ("WR002", "WR", 1),  # Developing WR
        ("RB001", "RB", 2),  # Veteran RB
        ("RB002", "RB", 0),  # Rookie RB
    ]

    feature_id = 1

    # Generate data for 2 seasons
    for season in [2023, 2024]:
        for week in range(1, 18):  # 17 weeks per season
            for player_id, position, experience_years in players:
                # Skip some weeks randomly to simulate realistic patterns
                if np.random.random() < 0.1:  # 10% missing data
                    continue

                # Generate 47 numerical features
                numerical_features = _generate_numerical_features(
                    position, experience_years, week
                )

                # Generate categorical features
                categorical_features = {
                    "position": position,
                    "experience_level": _get_experience_level(experience_years),
                    "home_away": "home" if week % 2 == 0 else "away",
                    "opponent_strength": np.random.choice(["strong", "medium", "weak"]),
                    "weather": "dome",
                }

                # Generate actual outcomes based on position
                actual_outcomes = _generate_outcomes(position, numerical_features)

                # Quality score (most data is high quality)
                quality_score = np.random.uniform(0.7, 1.0)

                # Insert into database
                conn.execute("""
                    INSERT INTO ml_training_features (
                        feature_id, entity_id, season, week,
                        game_date, numerical_features, feature_names,
                        categorical_features, actual_outcomes,
                        data_quality_score, player_experience_level
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, [
                    f"FEAT_{feature_id:06d}",
                    player_id,
                    season,
                    week,
                    f"{season}-09-{10 + week:02d}",  # Approximate game date
                    numerical_features,
                    _get_feature_names(),
                    json.dumps(categorical_features),
                    json.dumps(actual_outcomes),
                    quality_score,
                    _get_experience_level(experience_years),
                ])

                feature_id += 1


def _generate_numerical_features(position: str, experience_years: int, week: int) -> list:
    """Generate 47 realistic numerical features."""
    np.random.seed(hash(f"{position}_{experience_years}_{week}") % 2**32)

    features = []

    # Recent performance (10 features)
    if position == "QB":
        features.extend([
            np.random.uniform(200, 300),  # passing_yards_avg_3
            np.random.uniform(1.5, 3.0),  # passing_tds_avg_3
            np.random.uniform(0.3, 1.0),  # interceptions_avg_3
            np.random.uniform(60, 70),    # completion_pct_3
            np.random.uniform(20, 30),    # rushing_yards_avg_3
            np.random.uniform(250, 350),  # passing_yards_avg_5
            np.random.uniform(1.8, 3.2),  # passing_tds_avg_5
            np.random.uniform(0.4, 1.2),  # interceptions_avg_5
            np.random.uniform(230, 320),  # passing_yards_avg_10
            np.random.uniform(2.0, 3.5),  # passing_tds_avg_10
        ])
    elif position == "WR":
        features.extend([
            np.random.uniform(60, 100),   # receiving_yards_avg_3
            np.random.uniform(0.4, 1.2),  # receiving_tds_avg_3
            np.random.uniform(5, 8),      # receptions_avg_3
            np.random.uniform(7, 10),     # targets_avg_3
            np.random.uniform(65, 75),    # catch_rate_3
            np.random.uniform(65, 105),   # receiving_yards_avg_5
            np.random.uniform(0.5, 1.3),  # receiving_tds_avg_5
            np.random.uniform(5.5, 8.5),  # receptions_avg_5
            np.random.uniform(70, 110),   # receiving_yards_avg_10
            np.random.uniform(0.6, 1.4),  # receiving_tds_avg_10
        ])
    elif position == "RB":
        features.extend([
            np.random.uniform(60, 100),   # rushing_yards_avg_3
            np.random.uniform(0.3, 1.0),  # rushing_tds_avg_3
            np.random.uniform(15, 22),    # carries_avg_3
            np.random.uniform(20, 35),    # receiving_yards_avg_3
            np.random.uniform(3, 5),      # receptions_avg_3
            np.random.uniform(65, 105),   # rushing_yards_avg_5
            np.random.uniform(0.4, 1.1),  # rushing_tds_avg_5
            np.random.uniform(16, 23),    # carries_avg_5
            np.random.uniform(70, 110),   # rushing_yards_avg_10
            np.random.uniform(0.5, 1.2),  # rushing_tds_avg_10
        ])

    # Season-long stats (5 features)
    features.extend([
        np.random.uniform(0.5, 1.5),   # season_td_rate
        np.random.uniform(0.1, 0.5),   # season_turnover_rate
        np.random.uniform(10, 20),     # season_fantasy_avg
        week / 17.0,                    # season_progress
        np.random.uniform(0.7, 1.0),   # consistency_score
    ])

    # Career stats (5 features)
    features.extend([
        float(experience_years),        # years_experience
        np.random.uniform(0.6, 0.95),  # career_games_played_pct
        np.random.uniform(0.4, 1.0),   # career_injury_rate
        np.random.uniform(12, 18),     # career_fantasy_avg
        np.random.uniform(0.6, 0.9),   # career_consistency
    ])

    # Opponent defense (7 features)
    features.extend([
        np.random.uniform(300, 370),   # opp_def_yards_allowed_avg
        np.random.uniform(18, 28),     # opp_def_points_allowed_avg
        np.random.uniform(0.3, 0.8),   # opp_def_turnover_rate
        np.random.uniform(0.4, 0.7),   # opp_def_sack_rate
        np.random.uniform(200, 270),   # opp_def_pass_yards_allowed
        np.random.uniform(100, 140),   # opp_def_rush_yards_allowed
        np.random.uniform(0.5, 0.8),   # opp_def_rating
    ])

    # Team offense (7 features)
    features.extend([
        np.random.uniform(350, 420),   # team_off_yards_avg
        np.random.uniform(22, 32),     # team_off_points_avg
        np.random.uniform(0.5, 0.85),  # team_off_third_down_conv
        np.random.uniform(0.3, 0.6),   # team_off_redzone_conv
        np.random.uniform(250, 300),   # team_off_pass_yards
        np.random.uniform(100, 150),   # team_off_rush_yards
        np.random.uniform(10, 18),     # team_off_plays_per_drive
    ])

    # Situational factors (8 features)
    features.extend([
        1.0 if week % 2 == 0 else 0.0,  # is_home
        np.random.uniform(0.45, 0.65),  # implied_team_total
        np.random.uniform(45, 52),      # game_total_line
        np.random.uniform(-7, 7),       # point_spread
        np.random.uniform(60, 75),      # temperature
        1.0 if np.random.random() < 0.2 else 0.0,  # is_dome
        np.random.uniform(0.3, 0.7),    # Vegas_over_under_hit_rate
        np.random.uniform(0.4, 0.6),    # Vegas_spread_hit_rate
    ])

    # Advanced metrics (5 features)
    features.extend([
        np.random.uniform(0.05, 0.25),  # target_share
        np.random.uniform(40, 60),      # snap_count_pct
        np.random.uniform(0.5, 0.9),    # route_participation
        np.random.uniform(-5, 15),      # epa_per_play
        np.random.uniform(0.4, 0.7),    # success_rate
    ])

    # Ensure exactly 47 features
    assert len(features) == 47, f"Expected 47 features, got {len(features)}"

    return features


def _get_feature_names() -> list:
    """Return 47 feature names matching the numerical features."""
    return [
        # Recent performance (10)
        "primary_stat_avg_3", "primary_tds_avg_3", "turnovers_avg_3",
        "efficiency_metric_3", "secondary_stat_avg_3",
        "primary_stat_avg_5", "primary_tds_avg_5", "turnovers_avg_5",
        "primary_stat_avg_10", "primary_tds_avg_10",
        # Season stats (5)
        "season_td_rate", "season_turnover_rate", "season_fantasy_avg",
        "season_progress", "consistency_score",
        # Career stats (5)
        "years_experience", "career_games_played_pct", "career_injury_rate",
        "career_fantasy_avg", "career_consistency",
        # Opponent defense (7)
        "opp_def_yards_allowed", "opp_def_points_allowed", "opp_def_turnover_rate",
        "opp_def_sack_rate", "opp_def_pass_yards_allowed", "opp_def_rush_yards_allowed",
        "opp_def_rating",
        # Team offense (7)
        "team_off_yards", "team_off_points", "team_off_third_down_conv",
        "team_off_redzone_conv", "team_off_pass_yards", "team_off_rush_yards",
        "team_off_plays_per_drive",
        # Situational (8)
        "is_home", "implied_team_total", "game_total_line", "point_spread",
        "temperature", "is_dome", "vegas_over_hit_rate", "vegas_spread_hit_rate",
        # Advanced (5)
        "target_share", "snap_count_pct", "route_participation",
        "epa_per_play", "success_rate",
    ]


def _generate_outcomes(position: str, features: list) -> dict:
    """Generate realistic target outcomes based on features."""
    # Use features to influence outcomes (add some correlation)
    base_value = features[0]  # Primary stat avg
    noise = np.random.normal(0, base_value * 0.15)  # 15% noise

    if position == "QB":
        return {
            "passing_yards": max(0, int(base_value + noise)),
            "passing_tds": max(0, int(features[1] + np.random.normal(0, 0.5))),
            "passing_interceptions": max(0, int(features[2] + np.random.normal(0, 0.3))),
            "completions": max(0, int(20 + np.random.normal(0, 5))),
            "attempts": max(0, int(32 + np.random.normal(0, 5))),
            "rushing_yards": max(0, int(features[4] + np.random.normal(0, 5))),
            "rushing_tds": max(0, int(np.random.binomial(1, 0.15))),
            "fantasy_points_ppr": max(0, features[12] + np.random.normal(0, 3)),
        }
    elif position == "WR":
        return {
            "receiving_yards": max(0, int(base_value + noise)),
            "receiving_tds": max(0, int(features[1] + np.random.normal(0, 0.4))),
            "receptions": max(0, int(features[2] + np.random.normal(0, 1.5))),
            "targets": max(0, int(features[3] + np.random.normal(0, 2))),
            "catch_rate": min(100, max(0, features[4] + np.random.normal(0, 5))),
            "rushing_yards": max(0, int(np.random.gamma(2, 2))),  # Occasionally high
            "fantasy_points_ppr": max(0, features[12] + np.random.normal(0, 2.5)),
        }
    elif position == "RB":
        return {
            "rushing_yards": max(0, int(base_value + noise)),
            "rushing_tds": max(0, int(features[1] + np.random.normal(0, 0.4))),
            "carries": max(0, int(features[2] + np.random.normal(0, 3))),
            "receptions": max(0, int(features[4] + np.random.normal(0, 1))),
            "receiving_yards": max(0, int(features[3] + np.random.normal(0, 8))),
            "receiving_tds": max(0, int(np.random.binomial(1, 0.1))),
            "fantasy_points_ppr": max(0, features[12] + np.random.normal(0, 2.5)),
        }

    return {}


def _get_experience_level(years: int) -> str:
    """Get experience level from years."""
    if years == 0:
        return "rookie"
    elif years <= 2:
        return "developing"
    else:
        return "veteran"


# =============================================================================
# Integration Tests
# =============================================================================


class TestMLPipelineIntegration:
    """Integration tests for ML pipeline components."""

    def test_trainer_loads_data_from_db(self, ml_integration_db, tmp_path):
        """Test NFLTrainer can load data from database."""
        trainer = NFLTrainer(
            db_path=ml_integration_db,
            model_dir=str(tmp_path / "models"),
            use_mlflow=False
        )

        # Load QB data
        df = trainer.load_training_data(position="QB")

        assert df is not None, "Should load data"
        assert len(df) > 0, "Should have data rows"
        assert "numerical_features" in df.columns, "Should have features"
        assert "actual_outcomes" in df.columns, "Should have outcomes"

    def test_trainer_splits_data_temporally(self, ml_integration_db, tmp_path):
        """Test temporal train/val/test splits."""
        trainer = NFLTrainer(
            db_path=ml_integration_db,
            model_dir=str(tmp_path / "models"),
            use_mlflow=False
        )

        # Load and split data
        df = trainer.load_training_data(position="QB")
        train_df, val_df, test_df = trainer.split_data_temporal(df)

        assert len(train_df) > 0, "Should have training data"
        assert len(val_df) > 0, "Should have validation data"
        assert len(test_df) > 0, "Should have test data"

        # Verify temporal ordering (train < val < test by date)
        if "game_date" in train_df.columns:
            max_train_date = train_df["game_date"].max()
            min_val_date = val_df["game_date"].min()
            min_test_date = test_df["game_date"].min()

            assert max_train_date <= min_val_date, "Train should come before validation"
            assert min_val_date <= min_test_date, "Validation should come before test"

    def test_trainer_prepares_features_and_targets(self, ml_integration_db, tmp_path):
        """Test feature and target preparation."""
        trainer = NFLTrainer(
            db_path=ml_integration_db,
            model_dir=str(tmp_path / "models"),
            use_mlflow=False
        )

        # Load and prepare data
        df = trainer.load_training_data(position="QB")
        X, y, feature_names = trainer.prepare_features_and_targets(
            df, position="QB", target="passing_yards"
        )

        assert X is not None, "Should have features"
        assert y is not None, "Should have targets"
        assert len(X) == len(y), "Features and targets should have same length"
        assert X.shape[1] == 47, "Should have 47 features"
        assert len(feature_names) == 47, "Should have 47 feature names"

    def test_xgboost_training_and_prediction(self, ml_integration_db, tmp_path):
        """Test XGBoost can train and make predictions."""
        # Load data
        trainer = NFLTrainer(
            db_path=ml_integration_db,
            model_dir=str(tmp_path / "models"),
            use_mlflow=False
        )

        df = trainer.load_training_data(position="QB")
        train_df, val_df, test_df = trainer.split_data_temporal(df)

        X_train, y_train, feature_names = trainer.prepare_features_and_targets(
            train_df, "QB", "passing_yards"
        )
        X_test, y_test, _ = trainer.prepare_features_and_targets(
            test_df, "QB", "passing_yards"
        )

        # Train XGBoost model
        predictor = XGBoostPredictor(
            model_dir=str(tmp_path / "models"),
            use_mlflow=False
        )

        predictor.train(
            position="QB",
            target="passing_yards",
            X_train=X_train,
            y_train=y_train,
            feature_names=feature_names,
            n_estimators=10,  # Fast training for tests
        )

        # Make predictions
        predictions = predictor.predict("QB", "passing_yards", X_test)

        assert predictions is not None, "Should make predictions"
        assert len(predictions) == len(y_test), "Should predict for all test samples"
        assert all(p >= 0 for p in predictions), "Passing yards should be non-negative"

    def test_train_predict_evaluate_workflow(self, ml_integration_db, tmp_path):
        """Test complete train → predict → evaluate workflow."""
        trainer = NFLTrainer(
            db_path=ml_integration_db,
            model_dir=str(tmp_path / "models"),
            use_mlflow=False
        )

        # Load and split data
        df = trainer.load_training_data(position="QB")
        train_df, val_df, test_df = trainer.split_data_temporal(df)

        # Prepare features
        X_train, y_train, feature_names = trainer.prepare_features_and_targets(
            train_df, "QB", "passing_yards"
        )
        X_test, y_test, _ = trainer.prepare_features_and_targets(
            test_df, "QB", "passing_yards"
        )

        # Train
        predictor = XGBoostPredictor(
            model_dir=str(tmp_path / "models"),
            use_mlflow=False
        )

        predictor.train(
            position="QB",
            target="passing_yards",
            X_train=X_train,
            y_train=y_train,
            feature_names=feature_names,
            n_estimators=10,
        )

        # Predict
        predictions = predictor.predict("QB", "passing_yards", X_test)

        # Evaluate
        metrics = trainer.evaluate_model("QB", "passing_yards", predictor, X_test, y_test)

        assert "rmse" in metrics, "Should have RMSE metric"
        assert "mae" in metrics, "Should have MAE metric"
        assert "r2" in metrics, "Should have R2 metric"
        assert metrics["rmse"] > 0, "RMSE should be positive"
        assert metrics["mae"] > 0, "MAE should be positive"

    def test_trainer_trains_multiple_models(self, ml_integration_db, tmp_path):
        """Test training multiple models for different targets."""
        trainer = NFLTrainer(
            db_path=ml_integration_db,
            model_dir=str(tmp_path / "models"),
            use_mlflow=False
        )

        # Train QB models (multiple targets)
        results = trainer.train_position_models(
            position="QB",
            targets=["passing_yards", "passing_tds"],
            hyperparameter_tune=False
        )

        assert isinstance(results, dict), "Should return results dict"
        assert len(results) >= 2, "Should train at least 2 models"

        # Verify models were saved
        model_files = list((tmp_path / "models" / "xgboost").glob("*.json"))
        assert len(model_files) >= 2, "Should save model files"

    def test_model_save_and_load(self, ml_integration_db, tmp_path):
        """Test model saving and loading."""
        trainer = NFLTrainer(
            db_path=ml_integration_db,
            model_dir=str(tmp_path / "models"),
            use_mlflow=False
        )

        # Train and save
        df = trainer.load_training_data(position="QB")
        train_df, _, _ = trainer.split_data_temporal(df)

        X_train, y_train, feature_names = trainer.prepare_features_and_targets(
            train_df, "QB", "passing_yards"
        )

        predictor = XGBoostPredictor(
            model_dir=str(tmp_path / "models"),
            use_mlflow=False
        )

        predictor.train(
            position="QB",
            target="passing_yards",
            X_train=X_train,
            y_train=y_train,
            feature_names=feature_names,
            n_estimators=10,
        )

        predictor.save("QB", "passing_yards")

        # Load model
        new_predictor = XGBoostPredictor(
            model_dir=str(tmp_path / "models"),
            use_mlflow=False
        )

        new_predictor.load("QB", "passing_yards")

        # Verify loaded model works
        X_test = X_train[:10]  # Use subset for quick test
        predictions = new_predictor.predict("QB", "passing_yards", X_test)

        assert predictions is not None, "Loaded model should make predictions"
        assert len(predictions) == len(X_test), "Should predict for all samples"


class TestModelRegistry:
    """Integration tests for model registry."""

    def test_registry_initialization(self, ml_integration_db, tmp_path):
        """Test model registry initialization."""
        registry = ModelRegistry(
            db=None,  # Will create its own
            models_dir=tmp_path / "registry_models"
        )

        assert registry.models_dir.exists(), "Models directory should exist"

    def test_list_models_empty(self, ml_integration_db, tmp_path):
        """Test listing models when registry is empty."""
        registry = ModelRegistry(
            db=None,
            models_dir=tmp_path / "registry_models"
        )

        models = registry.list_models()

        # May be empty or have models from previous tests
        assert isinstance(models, list), "Should return list"


class TestFeatureExtraction:
    """Integration tests for feature extraction and processing."""

    def test_numerical_features_extraction(self, ml_integration_db, tmp_path):
        """Test extracting numerical features from database."""
        trainer = NFLTrainer(
            db_path=ml_integration_db,
            model_dir=str(tmp_path / "models"),
            use_mlflow=False
        )

        df = trainer.load_training_data(position="QB")

        # Verify numerical features format
        assert "numerical_features" in df.columns, "Should have numerical_features column"

        # Check first row
        first_features = df["numerical_features"][0]
        assert isinstance(first_features, list), "Features should be list"
        assert len(first_features) == 47, "Should have 47 features"
        assert all(isinstance(f, (int, float)) for f in first_features), \
            "All features should be numeric"

    def test_categorical_features_extraction(self, ml_integration_db, tmp_path):
        """Test extracting categorical features."""
        trainer = NFLTrainer(
            db_path=ml_integration_db,
            model_dir=str(tmp_path / "models"),
            use_mlflow=False
        )

        df = trainer.load_training_data(position="QB")

        assert "categorical_features" in df.columns, "Should have categorical features"

        # Parse first categorical features
        first_cat = df["categorical_features"][0]
        if isinstance(first_cat, str):
            cat_dict = json.loads(first_cat)
            assert "position" in cat_dict, "Should have position"
            assert "experience_level" in cat_dict, "Should have experience level"

    def test_target_extraction(self, ml_integration_db, tmp_path):
        """Test extracting targets from actual outcomes."""
        trainer = NFLTrainer(
            db_path=ml_integration_db,
            model_dir=str(tmp_path / "models"),
            use_mlflow=False
        )

        df = trainer.load_training_data(position="QB")
        X, y, _ = trainer.prepare_features_and_targets(df, "QB", "passing_yards")

        assert y is not None, "Should have targets"
        assert len(y) > 0, "Should have target values"
        assert all(y_val >= 0 for y_val in y), "Passing yards should be non-negative"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
