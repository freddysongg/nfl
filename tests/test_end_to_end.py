"""
End-to-End Integration Test for NFL Prediction System.

This test simulates the complete workflow from raw data to predictions:
1. Create sample raw data (simulating Stage 1 output)
2. Run data pipeline (Stages 2-4)
3. Train ML models
4. Make predictions
5. Evaluate performance

This is the ultimate integration test that verifies the entire system works together.
"""

import pytest
import duckdb
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import json

from src.data_pipeline import NFLDataPipeline
from src.training.trainer import NFLTrainer
from src.models.xgboost_predictor import XGBoostPredictor
from src.table_schemas import (
    create_raw_player_stats_table,
    create_raw_team_stats_table,
    create_raw_schedules_table,
    create_raw_rosters_weekly_table,
    create_player_lifecycle_table,
    create_team_roster_snapshots_table,
    create_player_experience_classification_table,
    create_player_rolling_features_table,
    create_team_rolling_features_table,
    create_ml_training_features_table,
)


@pytest.fixture
def end_to_end_db(tmp_path):
    """
    Create comprehensive test database for end-to-end testing.

    This fixture creates a realistic mini-dataset with:
    - 5 players (2 QBs, 2 WRs, 1 RB)
    - 2 complete seasons (2023-2024)
    - 15 weeks per season
    - All raw tables populated
    """
    db_path = tmp_path / "e2e_test.duckdb"
    conn = duckdb.connect(str(db_path))

    # Create all tables
    _create_all_tables(conn)

    # Insert comprehensive test data
    _insert_e2e_test_data(conn)

    conn.close()

    yield str(db_path)

    # Cleanup
    db_path.unlink(missing_ok=True)


def _create_all_tables(conn):
    """Create all necessary tables for end-to-end test."""
    # Raw tables
    create_raw_player_stats_table(conn)
    create_raw_team_stats_table(conn)
    create_raw_schedules_table(conn)
    create_raw_rosters_weekly_table(conn)

    # Stage 2 tables
    create_player_lifecycle_table(conn)
    create_team_roster_snapshots_table(conn)
    create_player_experience_classification_table(conn)

    # Stage 3 tables
    create_player_rolling_features_table(conn)
    create_team_rolling_features_table(conn)

    # Stage 4 table
    create_ml_training_features_table(conn)


def _insert_e2e_test_data(conn):
    """
    Insert comprehensive test data for end-to-end testing.

    Creates realistic NFL data patterns:
    - Player performance variability
    - Home/away differences
    - Opponent strength effects
    - Seasonal trends
    """
    np.random.seed(12345)

    # Define players
    players = [
        {
            "id": "QB_MAHOMES", "name": "Patrick Mahomes", "position": "QB",
            "team": "KC", "skill_level": 0.95
        },
        {
            "id": "QB_ALLEN", "name": "Josh Allen", "position": "QB",
            "team": "BUF", "skill_level": 0.93
        },
        {
            "id": "WR_HILL", "name": "Tyreek Hill", "position": "WR",
            "team": "MIA", "skill_level": 0.92
        },
        {
            "id": "WR_ADAMS", "name": "Davante Adams", "position": "WR",
            "team": "LV", "skill_level": 0.90
        },
        {
            "id": "RB_CHUBB", "name": "Nick Chubb", "position": "RB",
            "team": "CLE", "skill_level": 0.88
        },
    ]

    teams = ["KC", "BUF", "MIA", "LV", "CLE", "LAC", "NE", "NYJ"]

    # Generate 2 seasons of data
    for season in [2023, 2024]:
        for week in range(1, 16):  # 15 weeks
            game_date = datetime(season, 9, 1) + timedelta(weeks=week-1)

            # Insert player stats
            for player in players:
                _insert_player_week_stats(
                    conn, player, season, week, game_date
                )

            # Insert team stats
            for team in teams:
                _insert_team_week_stats(conn, team, season, week)

            # Insert schedules
            _insert_week_schedules(conn, teams, season, week, game_date)

            # Insert rosters
            for player in players:
                _insert_roster_entry(conn, player, season, week)


def _insert_player_week_stats(conn, player, season, week, game_date):
    """Insert realistic player stats for a week."""
    np.random.seed(hash(f"{player['id']}_{season}_{week}") % 2**32)

    position = player["position"]
    skill = player["skill_level"]

    # Home/away effect
    is_home = week % 2 == 0
    home_boost = 1.1 if is_home else 1.0

    # Seasonal improvement (2024 > 2023)
    season_boost = 1.05 if season == 2024 else 1.0

    # Weekly variance
    variance = np.random.uniform(0.85, 1.15)

    # Calculate stats based on position
    if position == "QB":
        base_yards = 270
        base_tds = 2.3

        passing_yards = int(base_yards * skill * home_boost * season_boost * variance)
        passing_tds = max(0, int(base_tds * skill * variance + np.random.normal(0, 0.5)))
        interceptions = max(0, int(np.random.binomial(2, 0.3)))
        completions = int(passing_yards / 12 + np.random.normal(0, 2))
        attempts = int(completions / 0.65 + np.random.normal(0, 3))
        rushing_yards = int(20 * skill * variance)
        rushing_tds = int(np.random.binomial(1, 0.15))

        conn.execute("""
            INSERT INTO raw_player_stats (
                player_id, player_name, player_display_name, position, position_group,
                season, week, season_type, team, opponent_team,
                completions, attempts, passing_yards, passing_tds, passing_interceptions,
                carries, rushing_yards, rushing_tds,
                fantasy_points_ppr
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            player["id"], player["name"], player["name"], position, position,
            season, week, "REG", player["team"], _get_opponent(player["team"], week),
            completions, attempts, passing_yards, passing_tds, interceptions,
            3, rushing_yards, rushing_tds,
            _calculate_fantasy_points(position, {
                "passing_yards": passing_yards,
                "passing_tds": passing_tds,
                "rushing_yards": rushing_yards,
                "rushing_tds": rushing_tds
            })
        ])

    elif position == "WR":
        base_yards = 85
        base_tds = 0.7

        receiving_yards = int(base_yards * skill * home_boost * season_boost * variance)
        receiving_tds = max(0, int(base_tds * skill * variance + np.random.normal(0, 0.3)))
        targets = int(8 * skill * variance + np.random.normal(0, 1.5))
        receptions = int(targets * 0.70)

        conn.execute("""
            INSERT INTO raw_player_stats (
                player_id, player_name, player_display_name, position, position_group,
                season, week, season_type, team, opponent_team,
                targets, receptions, receiving_yards, receiving_tds,
                carries, rushing_yards, rushing_tds,
                fantasy_points_ppr
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            player["id"], player["name"], player["name"], position, position,
            season, week, "REG", player["team"], _get_opponent(player["team"], week),
            targets, receptions, receiving_yards, receiving_tds,
            1, 5, 0,
            _calculate_fantasy_points(position, {
                "receiving_yards": receiving_yards,
                "receiving_tds": receiving_tds,
                "receptions": receptions
            })
        ])

    elif position == "RB":
        base_rushing = 75
        base_rushing_tds = 0.6
        base_receiving = 25

        rushing_yards = int(base_rushing * skill * home_boost * season_boost * variance)
        rushing_tds = max(0, int(base_rushing_tds * skill * variance + np.random.normal(0, 0.3)))
        carries = int(rushing_yards / 4.5)
        receiving_yards = int(base_receiving * skill * variance)
        receptions = int(receiving_yards / 8)
        targets = int(receptions / 0.75)

        conn.execute("""
            INSERT INTO raw_player_stats (
                player_id, player_name, player_display_name, position, position_group,
                season, week, season_type, team, opponent_team,
                carries, rushing_yards, rushing_tds,
                targets, receptions, receiving_yards, receiving_tds,
                fantasy_points_ppr
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            player["id"], player["name"], player["name"], position, position,
            season, week, "REG", player["team"], _get_opponent(player["team"], week),
            carries, rushing_yards, rushing_tds,
            targets, receptions, receiving_yards, 0,
            _calculate_fantasy_points(position, {
                "rushing_yards": rushing_yards,
                "rushing_tds": rushing_tds,
                "receiving_yards": receiving_yards,
                "receptions": receptions
            })
        ])


def _insert_team_week_stats(conn, team, season, week):
    """Insert team stats for a week."""
    np.random.seed(hash(f"{team}_{season}_{week}") % 2**32)

    conn.execute("""
        INSERT INTO raw_team_stats (
            team, season, week, season_type,
            points_scored, points_allowed, total_yards,
            passing_yards, rushing_yards, turnovers
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, [
        team, season, week, "REG",
        int(np.random.normal(26, 8)),
        int(np.random.normal(22, 7)),
        int(np.random.normal(360, 60)),
        int(np.random.normal(240, 50)),
        int(np.random.normal(120, 30)),
        int(np.random.binomial(3, 0.4))
    ])


def _insert_week_schedules(conn, teams, season, week, game_date):
    """Insert game schedules for a week."""
    # Pair teams for games
    for i in range(0, min(len(teams), 8), 2):
        if i + 1 < len(teams):
            home = teams[i]
            away = teams[i + 1]

            home_score = int(np.random.normal(26, 8))
            away_score = int(np.random.normal(23, 8))

            conn.execute("""
                INSERT INTO raw_schedules (
                    game_id, season, week, game_type, gameday,
                    home_team, away_team, home_score, away_score
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                f"{season}_W{week:02d}_{home}_{away}",
                season, week, "REG", game_date.strftime("%Y-%m-%d"),
                home, away, home_score, away_score
            ])


def _insert_roster_entry(conn, player, season, week):
    """Insert roster entry for a player."""
    conn.execute("""
        INSERT INTO raw_rosters_weekly (
            player_id, player_name, position, team, season, week
        ) VALUES (?, ?, ?, ?, ?, ?)
    """, [
        player["id"], player["name"], player["position"],
        player["team"], season, week
    ])


def _get_opponent(team, week):
    """Get opponent for a team based on week."""
    opponents = {
        "KC": "BUF", "BUF": "KC", "MIA": "NYJ", "LV": "LAC",
        "CLE": "NE", "LAC": "LV", "NE": "CLE", "NYJ": "MIA"
    }
    return opponents.get(team, "OPP")


def _calculate_fantasy_points(position, stats):
    """Calculate fantasy points based on stats."""
    points = 0.0

    if position == "QB":
        points += stats.get("passing_yards", 0) * 0.04
        points += stats.get("passing_tds", 0) * 4
        points += stats.get("rushing_yards", 0) * 0.1
        points += stats.get("rushing_tds", 0) * 6
        points -= stats.get("interceptions", 0) * 2
    elif position in ["WR", "TE"]:
        points += stats.get("receiving_yards", 0) * 0.1
        points += stats.get("receiving_tds", 0) * 6
        points += stats.get("receptions", 0) * 1  # PPR
        points += stats.get("rushing_yards", 0) * 0.1
        points += stats.get("rushing_tds", 0) * 6
    elif position == "RB":
        points += stats.get("rushing_yards", 0) * 0.1
        points += stats.get("rushing_tds", 0) * 6
        points += stats.get("receiving_yards", 0) * 0.1
        points += stats.get("receiving_tds", 0) * 6
        points += stats.get("receptions", 0) * 1  # PPR

    return max(0, points)


# =============================================================================
# End-to-End Test
# =============================================================================


@pytest.mark.slow
def test_full_system_end_to_end(end_to_end_db, tmp_path):
    """
    Comprehensive end-to-end test: Raw data → Features → Training → Predictions

    This test verifies the complete system workflow:
    1. Stage 2: Player lifecycle and roster management
    2. Stage 3a: Rolling statistics
    3. Stage 3b: Matchup features
    4. Stage 3c: Team aggregates
    5. Stage 4: ML dataset assembly
    6. Training: Train XGBoost models
    7. Prediction: Make predictions on test data
    8. Evaluation: Evaluate model performance

    Expected runtime: ~30-60 seconds
    """
    print("\n" + "="*70)
    print("END-TO-END INTEGRATION TEST")
    print("="*70)

    # =========================================================================
    # Step 1: Run Data Pipeline (Stages 2-4)
    # =========================================================================
    print("\n[Step 1] Running Data Pipeline (Stages 2-4)...")

    pipeline = NFLDataPipeline(db_file=end_to_end_db)

    # Stage 2: Player Lifecycle
    print("  → Stage 2: Player Lifecycle and Roster Management")
    lifecycle_count = pipeline.build_player_lifecycle_table()
    snapshot_count = pipeline.create_weekly_roster_snapshots()
    exp_count = pipeline.classify_player_experience_levels()

    assert lifecycle_count > 0, "Should create player lifecycle records"
    assert snapshot_count > 0, "Should create roster snapshots"
    assert exp_count > 0, "Should classify player experience"
    print(f"    ✓ Created {lifecycle_count} lifecycle records")
    print(f"    ✓ Created {snapshot_count} roster snapshots")
    print(f"    ✓ Classified {exp_count} experience levels")

    # Stage 3a: Rolling Statistics
    print("  → Stage 3a: Rolling Statistics")
    rolling_count = pipeline.calculate_rolling_statistics()
    assert rolling_count > 0, "Should calculate rolling features"
    print(f"    ✓ Calculated {rolling_count} rolling feature records")

    # Stage 3b: Matchup Features
    print("  → Stage 3b: Matchup Features")
    matchup_count = pipeline.build_matchup_features()
    assert matchup_count > 0, "Should build matchup features"
    print(f"    ✓ Built {matchup_count} matchup feature records")

    # Stage 3c: Team Aggregates
    print("  → Stage 3c: Team Aggregates")
    team_agg_count = pipeline.create_team_aggregates()
    assert team_agg_count > 0, "Should create team aggregates"
    print(f"    ✓ Created {team_agg_count} team aggregate records")

    # Stage 4: ML Dataset Assembly
    print("  → Stage 4: ML Dataset Assembly")
    features_count = pipeline.combine_all_features()
    scored, passed, filtered = pipeline.apply_data_quality_scoring()
    with_targets, without_targets = pipeline.create_prediction_targets()
    is_valid = pipeline.validate_temporal_consistency()

    assert features_count > 0, "Should combine features"
    assert passed > 0, "Should have high-quality data"
    assert with_targets > 0, "Should have prediction targets"
    assert is_valid, "Should pass temporal consistency"
    print(f"    ✓ Combined {features_count} feature records")
    print(f"    ✓ Quality scoring: {passed} passed, {filtered} filtered")
    print(f"    ✓ Created targets: {with_targets} with, {without_targets} without")
    print(f"    ✓ Temporal consistency: VALID")

    # =========================================================================
    # Step 2: Train ML Models
    # =========================================================================
    print("\n[Step 2] Training ML Models...")

    trainer = NFLTrainer(
        db_path=end_to_end_db,
        model_dir=str(tmp_path / "models"),
        use_mlflow=False,
        random_seed=42
    )

    # Train QB model
    print("  → Training QB passing_yards model")
    df_qb = trainer.load_training_data(position="QB")
    assert len(df_qb) > 0, "Should have QB training data"
    print(f"    ✓ Loaded {len(df_qb)} QB samples")

    train_df, val_df, test_df = trainer.split_data_temporal(df_qb)
    assert len(train_df) > 0, "Should have training data"
    assert len(test_df) > 0, "Should have test data"
    print(f"    ✓ Split: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test")

    X_train, y_train, feature_names = trainer.prepare_features_and_targets(
        train_df, "QB", "passing_yards"
    )
    X_test, y_test, _ = trainer.prepare_features_and_targets(
        test_df, "QB", "passing_yards"
    )

    assert X_train.shape[1] == 47, "Should have 47 features"
    print(f"    ✓ Prepared features: {X_train.shape}")

    predictor = XGBoostPredictor(
        model_dir=str(tmp_path / "models"),
        use_mlflow=False,
        random_seed=42
    )

    predictor.train(
        position="QB",
        target="passing_yards",
        X_train=X_train,
        y_train=y_train,
        feature_names=feature_names,
        n_estimators=20,  # Fast for testing
    )
    print(f"    ✓ Trained XGBoost model")

    # =========================================================================
    # Step 3: Make Predictions
    # =========================================================================
    print("\n[Step 3] Making Predictions...")

    predictions = predictor.predict("QB", "passing_yards", X_test)
    assert len(predictions) == len(y_test), "Should predict for all test samples"
    print(f"    ✓ Made {len(predictions)} predictions")
    print(f"    ✓ Prediction range: [{predictions.min():.1f}, {predictions.max():.1f}] yards")
    print(f"    ✓ Actual range: [{y_test.min():.1f}, {y_test.max():.1f}] yards")

    # =========================================================================
    # Step 4: Evaluate Performance
    # =========================================================================
    print("\n[Step 4] Evaluating Model Performance...")

    metrics = trainer.evaluate_model("QB", "passing_yards", predictor, X_test, y_test)

    assert "rmse" in metrics, "Should have RMSE"
    assert "mae" in metrics, "Should have MAE"
    assert "r2" in metrics, "Should have R2"

    print(f"    ✓ RMSE: {metrics['rmse']:.2f} yards")
    print(f"    ✓ MAE: {metrics['mae']:.2f} yards")
    print(f"    ✓ R²: {metrics['r2']:.4f}")
    print(f"    ✓ MAPE: {metrics['mape']:.2f}%")

    # Sanity checks on metrics
    assert metrics["rmse"] > 0, "RMSE should be positive"
    assert metrics["mae"] > 0, "MAE should be positive"
    assert metrics["rmse"] < 200, "RMSE should be reasonable (< 200 yards)"
    # R2 might be negative for small test sets, so we don't assert

    # =========================================================================
    # Step 5: Verify Model Persistence
    # =========================================================================
    print("\n[Step 5] Verifying Model Persistence...")

    # Save model
    predictor.save("QB", "passing_yards")
    model_path = tmp_path / "models" / "xgboost" / "QB_passing_yards_model.json"
    assert model_path.exists(), "Model file should exist"
    print(f"    ✓ Saved model to {model_path.name}")

    # Load model and verify it works
    new_predictor = XGBoostPredictor(
        model_dir=str(tmp_path / "models"),
        use_mlflow=False
    )
    new_predictor.load("QB", "passing_yards")

    new_predictions = new_predictor.predict("QB", "passing_yards", X_test)
    assert len(new_predictions) == len(predictions), "Should make same number of predictions"
    np.testing.assert_allclose(
        predictions, new_predictions, rtol=1e-5,
        err_msg="Loaded model should produce same predictions"
    )
    print(f"    ✓ Loaded model produces identical predictions")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "="*70)
    print("END-TO-END TEST SUMMARY")
    print("="*70)
    print(f"✓ Data Pipeline: {features_count} ML-ready records created")
    print(f"✓ Model Training: QB passing_yards model trained")
    print(f"✓ Predictions: {len(predictions)} predictions made")
    print(f"✓ Performance: RMSE={metrics['rmse']:.2f}, MAE={metrics['mae']:.2f}, R²={metrics['r2']:.4f}")
    print(f"✓ Persistence: Model saved and loaded successfully")
    print("="*70)
    print("✅ END-TO-END TEST PASSED")
    print("="*70)


@pytest.mark.slow
def test_multi_position_end_to_end(end_to_end_db, tmp_path):
    """
    Test end-to-end workflow for multiple positions (QB, WR, RB).

    This verifies that the system works correctly for different player positions
    with position-specific features and targets.
    """
    print("\n" + "="*70)
    print("MULTI-POSITION END-TO-END TEST")
    print("="*70)

    # Run data pipeline
    pipeline = NFLDataPipeline(db_file=end_to_end_db)
    pipeline.build_player_lifecycle_table()
    pipeline.classify_player_experience_levels()
    pipeline.calculate_rolling_statistics()
    pipeline.build_matchup_features()
    pipeline.create_team_aggregates()
    pipeline.combine_all_features()
    pipeline.apply_data_quality_scoring()
    pipeline.create_prediction_targets()

    trainer = NFLTrainer(
        db_path=end_to_end_db,
        model_dir=str(tmp_path / "models"),
        use_mlflow=False
    )

    positions_tested = []

    # Test each position
    for position in ["QB", "WR", "RB"]:
        print(f"\n[{position}] Testing position-specific workflow...")

        df = trainer.load_training_data(position=position)

        if len(df) < 10:
            print(f"  ⚠ Skipping {position}: insufficient data ({len(df)} samples)")
            continue

        train_df, _, test_df = trainer.split_data_temporal(df)

        if len(train_df) < 5 or len(test_df) < 2:
            print(f"  ⚠ Skipping {position}: insufficient split data")
            continue

        # Get position-specific target
        target_map = {
            "QB": "passing_yards",
            "WR": "receiving_yards",
            "RB": "rushing_yards"
        }
        target = target_map[position]

        X_train, y_train, feature_names = trainer.prepare_features_and_targets(
            train_df, position, target
        )
        X_test, y_test, _ = trainer.prepare_features_and_targets(
            test_df, position, target
        )

        # Train
        predictor = XGBoostPredictor(
            model_dir=str(tmp_path / "models"),
            use_mlflow=False
        )

        predictor.train(
            position=position,
            target=target,
            X_train=X_train,
            y_train=y_train,
            feature_names=feature_names,
            n_estimators=10,
        )

        # Predict
        predictions = predictor.predict(position, target, X_test)

        # Evaluate
        metrics = trainer.evaluate_model(position, target, predictor, X_test, y_test)

        print(f"  ✓ Trained {position} {target} model")
        print(f"  ✓ RMSE: {metrics['rmse']:.2f}, MAE: {metrics['mae']:.2f}, R²: {metrics['r2']:.4f}")

        positions_tested.append(position)

    assert len(positions_tested) >= 1, "Should successfully test at least one position"
    print(f"\n✅ Successfully tested positions: {', '.join(positions_tested)}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
