#!/usr/bin/env python3
"""
CLI tool for training NFL prediction models.

Usage:
    # Train all models for QB position
    python train_models.py --position QB

    # Train all models for QB with hyperparameter tuning
    python train_models.py --position QB --tune

    # Train all positions
    python train_models.py --all-positions

    # Train specific target
    python train_models.py --position RB --target rushing_yards

    # Custom seasons and database
    python train_models.py --position WR --season-start 2022 --season-end 2024 --db-path custom.duckdb
"""

import argparse
import sys
from pathlib import Path

from src.training.trainer import NFLTrainer


def main():
    """Main entry point for training CLI"""
    parser = argparse.ArgumentParser(
        description="Train NFL prediction models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train all QB models
  python train_models.py --position QB

  # Train all QB models with hyperparameter tuning
  python train_models.py --position QB --tune --n-trials 100

  # Train all positions (all targets for each)
  python train_models.py --all-positions

  # Train specific QB target with tuning
  python train_models.py --position QB --target passing_yards --tune

  # Train with custom data range
  python train_models.py --position RB --season-start 2022 --season-end 2024

  # Generate report only (no MLflow)
  python train_models.py --all-positions --no-mlflow --report training_results.md
        """,
    )

    # Position/target selection
    parser.add_argument(
        "--position",
        type=str,
        choices=["QB", "RB", "WR", "TE", "K"],
        help="Position to train (QB, RB, WR, TE, K)",
    )
    parser.add_argument(
        "--target", type=str, help="Specific target to train (optional)"
    )
    parser.add_argument(
        "--all-positions",
        action="store_true",
        help="Train all positions (QB, RB, WR, TE, K)",
    )

    # Training configuration
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Enable hyperparameter tuning with Optuna",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=50,
        help="Number of Optuna trials for hyperparameter tuning (default: 50)",
    )

    # Data configuration
    parser.add_argument(
        "--db-path",
        type=str,
        default="nfl_predictions.duckdb",
        help="Path to DuckDB database (default: nfl_predictions.duckdb)",
    )
    parser.add_argument(
        "--season-start",
        type=int,
        default=2021,
        help="Starting season for training data (default: 2021)",
    )
    parser.add_argument(
        "--season-end",
        type=int,
        default=2025,
        help="Ending season for training data (default: 2025)",
    )

    # Output configuration
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models",
        help="Directory to save models (default: models)",
    )
    parser.add_argument(
        "--report",
        type=str,
        help="Generate training report and save to file (e.g., training_report.md)",
    )

    # MLflow configuration
    parser.add_argument(
        "--no-mlflow", action="store_true", help="Disable MLflow tracking"
    )

    # Other
    parser.add_argument(
        "--random-seed", type=int, default=42, help="Random seed (default: 42)"
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.all_positions and not args.position:
        parser.error("Must specify either --position or --all-positions")

    if args.target and not args.position:
        parser.error("--target requires --position")

    if args.all_positions and args.position:
        parser.error("Cannot specify both --all-positions and --position")

    # Check if database exists and has data
    db_path = Path(args.db_path)
    if not db_path.exists():
        print(f"Error: Database not found at {args.db_path}")
        print("\nTo create the database:")
        print("  1. Run: uv run python setup_database.py")
        print("  2. Run: uv run python run_pipeline.py")
        print("\nNote: Data download may fail in restricted environments due to")
        print("      GitHub release assets being hosted on Azure Blob Storage.")
        print("      If you encounter 403 Forbidden errors, try running outside")
        print("      the sandboxed environment or contact support.")
        sys.exit(1)

    # Check if database has data
    try:
        from src.database import NFLDatabase
        with NFLDatabase(str(db_path)) as db:
            result = db.execute("SELECT COUNT(*) as count FROM ml_training_features")
            row_count = result.fetchone()[0] if result else 0
            if row_count == 0:
                print(f"Error: Database exists but contains no training data")
                print(f"Database: {args.db_path}")
                print("\nTo load data:")
                print("  Run: uv run python run_pipeline.py")
                print("\nKnown Issue:")
                print("  The data pipeline downloads data from GitHub release assets,")
                print("  which are hosted on Azure Blob Storage. Some environments")
                print("  (sandboxed/restricted networks) may block access, resulting")
                print("  in '403 Forbidden' errors.")
                print("\nWorkarounds:")
                print("  1. Run the pipeline from a non-restricted environment")
                print("  2. Download data manually and import it")
                print("  3. Use a VPN or different network if corporate firewall is blocking")
                sys.exit(1)
    except Exception as e:
        print(f"Warning: Could not verify database contents: {e}")
        print("Proceeding anyway...")

    # Initialize trainer
    print("=" * 70)
    print("NFL MODEL TRAINING")
    print("=" * 70)
    print(f"Database: {args.db_path}")
    print(f"Model directory: {args.model_dir}")
    print(f"Seasons: {args.season_start}-{args.season_end}")
    print(f"Hyperparameter tuning: {args.tune}")
    if args.tune:
        print(f"Optuna trials: {args.n_trials}")
    print(f"MLflow tracking: {not args.no_mlflow}")
    print(f"Random seed: {args.random_seed}")
    print("=" * 70)

    trainer = NFLTrainer(
        db_path=args.db_path,
        model_dir=args.model_dir,
        use_mlflow=not args.no_mlflow,
        random_seed=args.random_seed,
    )

    # Train models
    results = None

    try:
        if args.all_positions:
            # Train all positions
            print("\nTraining all positions...")
            results = trainer.train_all_positions(
                hyperparameter_tune=args.tune,
                n_trials=args.n_trials,
                season_start=args.season_start,
                season_end=args.season_end,
            )

        elif args.position and args.target:
            # Train single target
            print(f"\nTraining {args.position} - {args.target}...")

            # Validate target for position
            if args.target not in trainer.POSITION_TARGETS.get(args.position, []):
                print(
                    f"Error: '{args.target}' is not a valid target for {args.position}"
                )
                print(
                    f"Valid targets: {trainer.POSITION_TARGETS[args.position]}"
                )
                sys.exit(1)

            # Load and split data
            df = trainer.load_training_data(
                position=args.position,
                season_start=args.season_start,
                season_end=args.season_end,
            )
            train_df, val_df, test_df = trainer.split_data_temporal(df)

            # Extract features
            X_train, y_train, feature_names = trainer.prepare_features_and_targets(
                train_df, args.position, args.target
            )
            X_val, y_val, _ = trainer.prepare_features_and_targets(
                val_df, args.position, args.target
            )
            X_test, y_test, _ = trainer.prepare_features_and_targets(
                test_df, args.position, args.target
            )

            # Train
            training_result = trainer.predictor.train(
                position=args.position,
                target=args.target,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                feature_names=feature_names,
                hyperparameter_tune=args.tune,
                n_trials=args.n_trials,
            )

            # Evaluate
            test_metrics = trainer.evaluate_model(
                args.position, args.target, trainer.predictor, X_test, y_test
            )

            print("\n" + "=" * 70)
            print("TRAINING COMPLETE")
            print("=" * 70)
            print(f"Model saved: {training_result['model_path']}")
            print(f"Training duration: {training_result['training_duration']:.2f}s")
            print("\nValidation metrics:")
            for k, v in training_result["metrics"].items():
                if k.startswith("val_"):
                    print(f"  {k}: {v:.4f}")
            print("\nTest metrics:")
            for k, v in test_metrics.items():
                print(f"  {k}: {v:.4f}")

        else:
            # Train all targets for position
            print(f"\nTraining all models for {args.position}...")
            position_results = trainer.train_position_models(
                position=args.position,
                hyperparameter_tune=args.tune,
                n_trials=args.n_trials,
                season_start=args.season_start,
                season_end=args.season_end,
            )

            # Convert to DataFrame for reporting
            import pandas as pd

            rows = []
            for target, result in position_results.items():
                row = {
                    "position": args.position,
                    "target": target,
                    "status": result.get("status", "unknown"),
                }
                if result.get("status") == "success":
                    metrics = result.get("training_metrics", {})
                    test_metrics = result.get("test_metrics", {})
                    row.update(
                        {
                            "val_rmse": metrics.get("val_rmse"),
                            "test_rmse": test_metrics.get("test_rmse"),
                            "test_r2": test_metrics.get("test_r2"),
                            "n_train": result.get("n_train"),
                        }
                    )
                rows.append(row)

            results = pd.DataFrame(rows)

            print("\n" + "=" * 70)
            print(f"TRAINING COMPLETE - {args.position}")
            print("=" * 70)
            print(results.to_string(index=False))

    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    # Generate report if requested
    if args.report and results is not None:
        print(f"\nGenerating training report: {args.report}")
        trainer.generate_training_report(results, output_file=args.report)
        print(f"Report saved: {args.report}")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
