"""
NFL Model Trainer - Orchestrates training for all prediction models.

Features:
- Loads ML-ready data from DuckDB (ml_training_features table)
- Temporal train/val/test splits (respects time ordering)
- Position-specific model training (QB, RB, WR, TE, K)
- Multi-target training (all targets per position)
- Model evaluation and comparison
- MLflow experiment tracking
- Model registry integration
"""

import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import polars as pl
import duckdb

from src.models.xgboost_predictor import XGBoostPredictor
from src.config import config

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class NFLTrainer:
    """
    Orchestrates training for all NFL prediction models.

    Features:
    - Loads ML-ready data from DuckDB (ml_training_features table)
    - Temporal train/val/test splits (respects time ordering)
    - Position-specific model training (QB, RB, WR, TE, K)
    - Multi-target training (all targets per position)
    - Model evaluation and comparison
    - MLflow experiment tracking
    - Model registry integration
    """

    # Position-target mapping (from XGBoostPredictor and data pipeline)
    POSITION_TARGETS = {
        "QB": [
            "passing_yards",
            "passing_tds",
            "passing_interceptions",
            "completions",
            "attempts",
            "rushing_yards",
            "rushing_tds",
            "fantasy_points_ppr",
        ],
        "RB": [
            "rushing_yards",
            "rushing_tds",
            "carries",
            "receptions",
            "receiving_yards",
            "receiving_tds",
            "fantasy_points_ppr",
        ],
        "WR": [
            "receiving_yards",
            "receiving_tds",
            "receptions",
            "targets",
            "catch_rate",
            "rushing_yards",
            "fantasy_points_ppr",
        ],
        "TE": [
            "receiving_yards",
            "receiving_tds",
            "receptions",
            "targets",
            "catch_rate",
            "fantasy_points_ppr",
        ],
        "K": [
            "fg_made",
            "fg_att",
            "fg_pct",
            "pat_made",
            "pat_att",
            "fantasy_points_standard",
        ],
    }

    def __init__(
        self,
        db_path: str = "nfl_predictions.duckdb",
        model_dir: str = "models",
        use_mlflow: bool = True,
        random_seed: int = 42,
    ):
        """
        Initialize NFLTrainer.

        Args:
            db_path: Path to DuckDB database
            model_dir: Directory to save models
            use_mlflow: Whether to enable MLflow tracking
            random_seed: Random seed for reproducibility
        """
        self.db_path = db_path
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.use_mlflow = use_mlflow
        self.random_seed = random_seed

        # Initialize XGBoost predictor
        self.predictor = XGBoostPredictor(
            model_dir=str(self.model_dir / "xgboost"),
            experiment_name="nfl_training_pipeline",
            use_mlflow=use_mlflow,
            random_seed=random_seed,
        )

        # Training results storage
        self.training_results: Dict[str, Dict[str, Any]] = {}

        logger.info(f"Initialized NFLTrainer with db_path={db_path}, model_dir={model_dir}")

    # =========================================================================
    # Data Loading
    # =========================================================================

    def load_training_data(
        self,
        position: Optional[str] = None,
        season_start: int = 2021,
        season_end: int = 2025,
        min_quality_score: float = 0.5,
    ) -> pl.DataFrame:
        """
        Load ml_training_features from DuckDB.

        Args:
            position: Position to filter by (QB, RB, WR, TE, K). If None, loads all.
            season_start: Starting season (inclusive)
            season_end: Ending season (inclusive)
            min_quality_score: Minimum data quality score threshold

        Returns:
            Polars DataFrame with ML-ready features

        Raises:
            ValueError: If ml_training_features table doesn't exist or is empty
        """
        logger.info("=" * 70)
        logger.info("Loading training data from DuckDB")
        logger.info("=" * 70)
        logger.info(f"Position: {position or 'ALL'}")
        logger.info(f"Seasons: {season_start}-{season_end}")
        logger.info(f"Min quality score: {min_quality_score}")

        try:
            conn = duckdb.connect(self.db_path, read_only=True)

            # Check if table exists
            table_exists = conn.execute(
                """
                SELECT COUNT(*) FROM information_schema.tables
                WHERE table_name = 'ml_training_features'
                """
            ).fetchone()[0]

            if not table_exists:
                raise ValueError(
                    "ml_training_features table does not exist. "
                    "Run Stage 4 of data pipeline first."
                )

            # Build query
            query = """
                SELECT
                    feature_id,
                    entity_id,
                    season,
                    week,
                    game_date,
                    numerical_features,
                    feature_names,
                    categorical_features,
                    actual_outcomes,
                    data_quality_score,
                    player_experience_level
                FROM ml_training_features
                WHERE 1=1
                    AND season BETWEEN ? AND ?
                    AND actual_outcomes IS NOT NULL
                    AND COALESCE(data_quality_score, 1.0) >= ?
            """

            params = [season_start, season_end, min_quality_score]

            # Add position filter if specified
            if position:
                query += """
                    AND json_extract(categorical_features, '$.position.value') = ?
                """
                params.append(position)

            # Order by time for temporal splits
            query += """
                ORDER BY season, week
            """

            logger.info(f"Executing query with params: {params}")
            df = conn.execute(query, params).pl()
            conn.close()

            if df.height == 0:
                raise ValueError(
                    f"No data found for position={position}, "
                    f"seasons={season_start}-{season_end}"
                )

            logger.info(f"Loaded {df.height:,} rows")

            # Extract position from categorical_features for convenience
            df = df.with_columns(
                pl.col("categorical_features")
                .str.json_extract()
                .struct.field("position")
                .struct.field("value")
                .alias("position")
            )

            # Log summary by position
            position_summary = (
                df.group_by("position")
                .agg(pl.count().alias("count"))
                .sort("position")
            )

            logger.info("Position distribution:")
            for row in position_summary.iter_rows(named=True):
                logger.info(f"  {row['position']}: {row['count']:,} samples")

            logger.info("=" * 70)

            return df

        except Exception as e:
            logger.error(f"Error loading training data: {e}")
            raise

    # =========================================================================
    # Data Splitting
    # =========================================================================

    def split_data_temporal(
        self,
        df: pl.DataFrame,
        train_ratio: float = 0.7,
        val_ratio: float = 0.2,
        test_ratio: float = 0.1,
    ) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        """
        Split data temporally to prevent data leakage.

        Data is already sorted by (season, week) from load_training_data.
        Split based on row counts:
        - Train: earliest 70%
        - Val: middle 20%
        - Test: latest 10%

        Args:
            df: Input DataFrame (must be sorted by season, week)
            train_ratio: Ratio of data for training
            val_ratio: Ratio of data for validation
            test_ratio: Ratio of data for testing

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Ratios must sum to 1.0"

        n_total = df.height
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        n_test = n_total - n_train - n_val  # Remaining goes to test

        train_df = df[:n_train]
        val_df = df[n_train : n_train + n_val]
        test_df = df[n_train + n_val :]

        logger.info("Temporal split complete:")
        logger.info(f"  Train: {train_df.height:,} samples ({100*train_ratio:.1f}%)")
        logger.info(f"  Val:   {val_df.height:,} samples ({100*val_ratio:.1f}%)")
        logger.info(f"  Test:  {test_df.height:,} samples ({100*test_ratio:.1f}%)")

        # Log season ranges for each split
        logger.info("Season ranges:")
        if train_df.height > 0:
            train_seasons = (
                train_df.select(pl.col("season")).min().item(),
                train_df.select(pl.col("season")).max().item(),
            )
            logger.info(f"  Train: {train_seasons[0]}-{train_seasons[1]}")

        if val_df.height > 0:
            val_seasons = (
                val_df.select(pl.col("season")).min().item(),
                val_df.select(pl.col("season")).max().item(),
            )
            logger.info(f"  Val:   {val_seasons[0]}-{val_seasons[1]}")

        if test_df.height > 0:
            test_seasons = (
                test_df.select(pl.col("season")).min().item(),
                test_df.select(pl.col("season")).max().item(),
            )
            logger.info(f"  Test:  {test_seasons[0]}-{test_seasons[1]}")

        return train_df, val_df, test_df

    # =========================================================================
    # Feature and Target Extraction
    # =========================================================================

    def prepare_features_and_targets(
        self, df: pl.DataFrame, position: str, target: str
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Extract numerical_features array and target from actual_outcomes JSON.

        Args:
            df: DataFrame with ml_training_features data
            position: Position (QB, RB, WR, TE, K)
            target: Target variable name (e.g., 'passing_yards', 'rushing_tds')

        Returns:
            Tuple of (X, y, feature_names)
            - X: Feature matrix (numpy array) of shape (n_samples, n_features)
            - y: Target values (numpy array) of shape (n_samples,)
            - feature_names: List of feature names

        Raises:
            ValueError: If target is not available in actual_outcomes
        """
        # Convert numerical_features from list to numpy array
        X_list = df.select(pl.col("numerical_features")).to_numpy().tolist()
        X = np.array([row[0] for row in X_list], dtype=np.float32)

        # Extract feature names (they're the same for all rows of same position)
        feature_names_list = df.select(pl.col("feature_names")).to_numpy().tolist()
        feature_names = feature_names_list[0][0] if feature_names_list else []

        # Extract target from actual_outcomes JSON
        y_list = []
        for row in df.iter_rows(named=True):
            outcomes_json = row["actual_outcomes"]
            if outcomes_json:
                try:
                    outcomes = json.loads(outcomes_json)
                    target_value = outcomes.get(target, None)
                    if target_value is None:
                        # Target not found - use 0.0 or raise error
                        logger.warning(
                            f"Target '{target}' not found in outcomes for "
                            f"feature_id={row['feature_id']}, using 0.0"
                        )
                        y_list.append(0.0)
                    else:
                        y_list.append(float(target_value))
                except (json.JSONDecodeError, TypeError) as e:
                    logger.warning(
                        f"Failed to parse actual_outcomes for "
                        f"feature_id={row['feature_id']}: {e}, using 0.0"
                    )
                    y_list.append(0.0)
            else:
                y_list.append(0.0)

        y = np.array(y_list, dtype=np.float32)

        logger.info(f"Extracted features for {position} - {target}:")
        logger.info(f"  X shape: {X.shape}")
        logger.info(f"  y shape: {y.shape}")
        logger.info(f"  Features: {len(feature_names)}")
        logger.info(f"  Target range: [{y.min():.2f}, {y.max():.2f}]")
        logger.info(f"  Target mean: {y.mean():.2f}, std: {y.std():.2f}")

        return X, y, feature_names

    # =========================================================================
    # Training Orchestration
    # =========================================================================

    def train_position_models(
        self,
        position: str,
        hyperparameter_tune: bool = False,
        n_trials: int = 50,
        season_start: int = 2021,
        season_end: int = 2025,
    ) -> Dict[str, Any]:
        """
        Train all target models for a specific position.

        Args:
            position: Position to train (QB, RB, WR, TE, K)
            hyperparameter_tune: Whether to run Optuna hyperparameter tuning
            n_trials: Number of Optuna trials (if hyperparameter_tune=True)
            season_start: Starting season for data
            season_end: Ending season for data

        Returns:
            Dictionary with training results for all targets
        """
        logger.info("=" * 70)
        logger.info(f"Training models for position: {position}")
        logger.info("=" * 70)
        logger.info(f"Hyperparameter tuning: {hyperparameter_tune}")
        if hyperparameter_tune:
            logger.info(f"Optuna trials: {n_trials}")

        # Validate position
        if position not in self.POSITION_TARGETS:
            raise ValueError(
                f"Invalid position: {position}. "
                f"Valid positions: {list(self.POSITION_TARGETS.keys())}"
            )

        # Load data for position
        df = self.load_training_data(
            position=position, season_start=season_start, season_end=season_end
        )

        # Split data temporally
        train_df, val_df, test_df = self.split_data_temporal(df)

        # Get targets for this position
        targets = self.POSITION_TARGETS[position]

        results = {}

        for target in targets:
            logger.info("")
            logger.info("=" * 70)
            logger.info(f"Training: {position} - {target}")
            logger.info("=" * 70)

            try:
                # Extract features and targets
                X_train, y_train, feature_names = self.prepare_features_and_targets(
                    train_df, position, target
                )
                X_val, y_val, _ = self.prepare_features_and_targets(
                    val_df, position, target
                )
                X_test, y_test, _ = self.prepare_features_and_targets(
                    test_df, position, target
                )

                # Check for sufficient data
                if len(y_train) < 10:
                    logger.warning(
                        f"Insufficient training data for {position} {target}: "
                        f"{len(y_train)} samples. Skipping."
                    )
                    results[target] = {
                        "status": "skipped",
                        "reason": "insufficient_data",
                        "n_train": len(y_train),
                    }
                    continue

                # Train model using XGBoostPredictor
                training_result = self.predictor.train(
                    position=position,
                    target=target,
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_val,
                    y_val=y_val,
                    feature_names=feature_names,
                    hyperparameter_tune=hyperparameter_tune,
                    n_trials=n_trials,
                )

                # Evaluate on test set
                test_metrics = self.evaluate_model(
                    position, target, self.predictor, X_test, y_test
                )

                # Combine results
                results[target] = {
                    "status": "success",
                    "training_metrics": training_result["metrics"],
                    "test_metrics": test_metrics,
                    "model_path": training_result["model_path"],
                    "training_duration": training_result["training_duration"],
                    "n_train": len(y_train),
                    "n_val": len(y_val),
                    "n_test": len(y_test),
                }

                logger.info(f"✓ Successfully trained {position} - {target}")

            except Exception as e:
                logger.error(f"✗ Failed to train {position} - {target}: {e}")
                results[target] = {"status": "error", "error": str(e)}

        # Store results
        self.training_results[position] = results

        # Summary
        success_count = sum(
            1 for r in results.values() if r.get("status") == "success"
        )
        logger.info("")
        logger.info("=" * 70)
        logger.info(f"Position {position} training complete")
        logger.info(f"Success: {success_count}/{len(targets)}")
        logger.info("=" * 70)

        return results

    def train_all_positions(
        self,
        positions: Optional[List[str]] = None,
        hyperparameter_tune: bool = False,
        n_trials: int = 50,
        season_start: int = 2021,
        season_end: int = 2025,
    ) -> pd.DataFrame:
        """
        Train models for all positions and all their targets.

        Args:
            positions: List of positions to train. If None, trains all.
            hyperparameter_tune: Whether to run hyperparameter tuning
            n_trials: Number of Optuna trials
            season_start: Starting season for data
            season_end: Ending season for data

        Returns:
            Summary DataFrame with all training results
        """
        if positions is None:
            positions = list(self.POSITION_TARGETS.keys())

        logger.info("=" * 70)
        logger.info("TRAINING ALL POSITIONS")
        logger.info("=" * 70)
        logger.info(f"Positions: {positions}")
        logger.info(f"Hyperparameter tuning: {hyperparameter_tune}")
        logger.info(f"Seasons: {season_start}-{season_end}")
        logger.info("=" * 70)

        all_results = []

        for position in positions:
            try:
                position_results = self.train_position_models(
                    position=position,
                    hyperparameter_tune=hyperparameter_tune,
                    n_trials=n_trials,
                    season_start=season_start,
                    season_end=season_end,
                )

                # Convert to DataFrame rows
                for target, result in position_results.items():
                    row = {
                        "position": position,
                        "target": target,
                        "status": result.get("status", "unknown"),
                    }

                    if result.get("status") == "success":
                        # Add metrics
                        train_metrics = result.get("training_metrics", {})
                        test_metrics = result.get("test_metrics", {})

                        row.update(
                            {
                                "train_rmse": train_metrics.get("train_rmse", None),
                                "train_mae": train_metrics.get("train_mae", None),
                                "train_r2": train_metrics.get("train_r2", None),
                                "val_rmse": train_metrics.get("val_rmse", None),
                                "val_mae": train_metrics.get("val_mae", None),
                                "val_r2": train_metrics.get("val_r2", None),
                                "test_rmse": test_metrics.get("test_rmse", None),
                                "test_mae": test_metrics.get("test_mae", None),
                                "test_r2": test_metrics.get("test_r2", None),
                                "n_train": result.get("n_train", None),
                                "n_val": result.get("n_val", None),
                                "n_test": result.get("n_test", None),
                                "training_duration": result.get(
                                    "training_duration", None
                                ),
                                "model_path": result.get("model_path", None),
                            }
                        )
                    elif result.get("status") == "error":
                        row["error"] = result.get("error", "Unknown error")
                    elif result.get("status") == "skipped":
                        row["reason"] = result.get("reason", "Unknown reason")

                    all_results.append(row)

            except Exception as e:
                logger.error(f"Failed to train position {position}: {e}")
                all_results.append(
                    {
                        "position": position,
                        "target": "ALL",
                        "status": "error",
                        "error": str(e),
                    }
                )

        # Create summary DataFrame
        results_df = pd.DataFrame(all_results)

        logger.info("")
        logger.info("=" * 70)
        logger.info("TRAINING SUMMARY")
        logger.info("=" * 70)

        if len(results_df) > 0:
            # Count by status
            status_counts = results_df["status"].value_counts()
            logger.info("Status summary:")
            for status, count in status_counts.items():
                logger.info(f"  {status}: {count}")

            # Summary by position
            logger.info("")
            logger.info("Position summary:")
            for position in positions:
                pos_df = results_df[results_df["position"] == position]
                success_count = (pos_df["status"] == "success").sum()
                total_count = len(pos_df)
                logger.info(f"  {position}: {success_count}/{total_count} successful")

        logger.info("=" * 70)

        return results_df

    # =========================================================================
    # Evaluation
    # =========================================================================

    def evaluate_model(
        self,
        position: str,
        target: str,
        predictor: XGBoostPredictor,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict[str, float]:
        """
        Evaluate model on test set.

        Args:
            position: Position
            target: Target variable
            predictor: Trained XGBoostPredictor
            X_test: Test features
            y_test: Test targets

        Returns:
            Dictionary of evaluation metrics
        """
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

        # Make predictions
        y_pred = predictor.predict(position, target, X_test)

        # Calculate metrics
        metrics = {
            "test_rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
            "test_mae": mean_absolute_error(y_test, y_pred),
            "test_r2": r2_score(y_test, y_pred),
        }

        logger.info("Test set evaluation:")
        for metric_name, metric_value in metrics.items():
            logger.info(f"  {metric_name}: {metric_value:.4f}")

        return metrics

    # =========================================================================
    # Reporting
    # =========================================================================

    def generate_training_report(
        self, results: pd.DataFrame, output_file: str = "training_report.md"
    ) -> None:
        """
        Generate markdown report with training results.

        Args:
            results: DataFrame with training results
            output_file: Output file path
        """
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            f.write("# NFL Model Training Report\n\n")
            f.write(f"Generated: {pd.Timestamp.now()}\n\n")

            # Overall summary
            f.write("## Overall Summary\n\n")
            total_models = len(results)
            successful = (results["status"] == "success").sum()
            failed = (results["status"] == "error").sum()
            skipped = (results["status"] == "skipped").sum()

            f.write(f"- Total models: {total_models}\n")
            f.write(f"- Successful: {successful}\n")
            f.write(f"- Failed: {failed}\n")
            f.write(f"- Skipped: {skipped}\n\n")

            # Results by position
            f.write("## Results by Position\n\n")

            for position in results["position"].unique():
                pos_df = results[results["position"] == position]
                f.write(f"### {position}\n\n")

                success_df = pos_df[pos_df["status"] == "success"]

                if len(success_df) > 0:
                    f.write("| Target | Train RMSE | Val RMSE | Test RMSE | Test R² | Samples |\n")
                    f.write("|--------|------------|----------|-----------|---------|----------|\n")

                    for _, row in success_df.iterrows():
                        f.write(
                            f"| {row['target']} | "
                            f"{row.get('train_rmse', 0):.3f} | "
                            f"{row.get('val_rmse', 0):.3f} | "
                            f"{row.get('test_rmse', 0):.3f} | "
                            f"{row.get('test_r2', 0):.3f} | "
                            f"{row.get('n_train', 0):,} |\n"
                        )

                    f.write("\n")
                else:
                    f.write("*No successful models trained*\n\n")

                # Failed/skipped models
                failed_df = pos_df[pos_df["status"].isin(["error", "skipped"])]
                if len(failed_df) > 0:
                    f.write("**Issues:**\n\n")
                    for _, row in failed_df.iterrows():
                        status = row["status"]
                        target = row["target"]
                        reason = row.get("error") or row.get("reason", "Unknown")
                        f.write(f"- {target}: {status} - {reason}\n")
                    f.write("\n")

            # Model paths
            f.write("## Model Artifacts\n\n")
            success_df = results[results["status"] == "success"]
            if len(success_df) > 0:
                f.write("| Position | Target | Model Path |\n")
                f.write("|----------|--------|------------|\n")
                for _, row in success_df.iterrows():
                    model_path = row.get("model_path", "N/A")
                    f.write(f"| {row['position']} | {row['target']} | {model_path} |\n")
                f.write("\n")

        logger.info(f"Training report saved to: {output_path}")

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def get_trained_models_summary(self) -> pd.DataFrame:
        """
        Get summary of all trained models.

        Returns:
            DataFrame with model summaries
        """
        models = self.predictor.get_trained_models()

        summary = []
        for position, target in models:
            info = self.predictor.get_model_info(position, target)
            summary.append(
                {
                    "position": position,
                    "target": target,
                    "n_features": info.get("n_features"),
                    "status": "trained",
                }
            )

        return pd.DataFrame(summary)

    def __repr__(self) -> str:
        """String representation"""
        return f"NFLTrainer(db={self.db_path}, models={self.model_dir})"
