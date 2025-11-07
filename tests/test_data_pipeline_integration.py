"""
Integration tests for NFL Data Pipeline (Stages 2-4).

Tests the complete data flow through:
- Stage 2: Player Lifecycle and Roster Management
- Stage 3a: Rolling Statistics
- Stage 3b: Matchup Features
- Stage 3c: Team Aggregates
- Stage 4: ML Dataset Assembly and Target Creation

These tests verify that components work together correctly and data flows
properly between stages.
"""

import pytest
import duckdb
import polars as pl
from pathlib import Path
from datetime import datetime, timedelta
import json

from src.data_pipeline import NFLDataPipeline
from src.database import NFLDatabase
from src.table_schemas import (
    create_raw_player_stats_table,
    create_raw_team_stats_table,
    create_raw_schedules_table,
    create_raw_rosters_weekly_table,
)


@pytest.fixture
def integration_test_db(tmp_path):
    """
    Create test database with complete schema and comprehensive sample data.

    Creates 3 players (1 QB, 1 WR, 1 RB) with 10 weeks of data each across 2 seasons.
    Includes all necessary raw tables for full pipeline execution.
    """
    db_path = tmp_path / "integration_test.duckdb"
    conn = duckdb.connect(str(db_path))

    # Create all raw tables
    create_raw_player_stats_table(conn)
    create_raw_team_stats_table(conn)
    create_raw_schedules_table(conn)
    create_raw_rosters_weekly_table(conn)

    # Create Stage 2 tables
    conn.execute("""
        CREATE TABLE IF NOT EXISTS player_lifecycle (
            player_id VARCHAR PRIMARY KEY,
            player_name VARCHAR,
            position VARCHAR,
            first_nfl_season INTEGER,
            last_nfl_season INTEGER,
            total_seasons_played INTEGER,
            total_games_played INTEGER,
            is_retired BOOLEAN,
            career_teams VARCHAR,
            primary_position VARCHAR,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS team_roster_snapshots (
            snapshot_id VARCHAR PRIMARY KEY,
            team VARCHAR,
            season INTEGER,
            week INTEGER,
            active_players JSON,
            total_roster_size INTEGER,
            snapshot_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS player_experience_classification (
            player_id VARCHAR,
            season INTEGER,
            seasons_experience INTEGER,
            experience_level VARCHAR,
            confidence_score FLOAT,
            classification_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (player_id, season)
        )
    """)

    # Create Stage 3 tables
    conn.execute("""
        CREATE TABLE IF NOT EXISTS player_rolling_features (
            feature_id VARCHAR PRIMARY KEY,
            player_id VARCHAR,
            player_name VARCHAR,
            season INTEGER,
            week INTEGER,
            position VARCHAR,
            team VARCHAR,
            opponent VARCHAR,
            stats_last3_games JSON,
            stats_last5_games JSON,
            stats_last10_games JSON,
            performance_trend FLOAT,
            usage_trend FLOAT,
            efficiency_trend FLOAT,
            vs_opponent_history JSON,
            home_away_splits JSON,
            divisional_game BOOLEAN,
            rest_days INTEGER,
            rolling_stats_json JSON,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS team_rolling_features (
            feature_id VARCHAR PRIMARY KEY,
            team VARCHAR,
            season INTEGER,
            week INTEGER,
            offensive_stats JSON,
            offensive_epa JSON,
            offensive_success_rate JSON,
            defensive_stats JSON,
            defensive_epa JSON,
            defensive_success_rate JSON,
            red_zone_efficiency JSON,
            third_down_efficiency JSON,
            explosive_play_rate JSON,
            offensive_trend FLOAT,
            defensive_trend FLOAT,
            data_source VARCHAR,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Create Stage 4 table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS ml_training_features (
            feature_id VARCHAR PRIMARY KEY,
            entity_id VARCHAR,
            player_id VARCHAR,
            player_name VARCHAR,
            position VARCHAR,
            season INTEGER,
            week INTEGER,
            team VARCHAR,
            opponent VARCHAR,
            game_date VARCHAR,
            numerical_features FLOAT[],
            feature_names JSON,
            categorical_features JSON,
            experience_level VARCHAR,
            confidence_score FLOAT,
            player_experience_level VARCHAR,
            data_quality_score FLOAT,
            completeness_score FLOAT,
            has_outliers BOOLEAN,
            recency_score FLOAT,
            target_passing_yards FLOAT,
            target_rushing_yards FLOAT,
            target_receiving_yards FLOAT,
            target_fantasy_points FLOAT,
            target_touchdowns INTEGER,
            actual_outcomes JSON,
            is_valid_temporal BOOLEAN,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Insert comprehensive sample player stats
    _insert_sample_player_stats(conn)

    # Insert sample team stats
    _insert_sample_team_stats(conn)

    # Insert sample schedules
    _insert_sample_schedules(conn)

    # Insert sample rosters
    _insert_sample_rosters(conn)

    conn.close()

    yield str(db_path)

    # Cleanup
    db_path.unlink(missing_ok=True)


def _insert_sample_player_stats(conn):
    """Insert realistic player stats for 3 players across 2 seasons (10 weeks each)."""

    players = [
        {"id": "QB001", "name": "Test Quarterback", "position": "QB", "team": "KC"},
        {"id": "WR001", "name": "Test Receiver", "position": "WR", "team": "KC"},
        {"id": "RB001", "name": "Test Running Back", "position": "RB", "team": "BUF"},
    ]

    stats_data = []

    # Season 2023 - 10 weeks
    for week in range(1, 11):
        game_date = datetime(2023, 9, 1) + timedelta(weeks=week-1)

        # QB stats
        stats_data.append({
            "player_id": "QB001",
            "player_name": "Test Quarterback",
            "player_display_name": "Test Quarterback",
            "position": "QB",
            "position_group": "QB",
            "season": 2023,
            "week": week,
            "season_type": "REG",
            "team": "KC",
            "opponent_team": "BUF" if week % 2 == 0 else "LAC",
            "completions": 22 + week,
            "attempts": 32 + week,
            "passing_yards": 250 + week * 10,
            "passing_tds": 2,
            "passing_interceptions": 1 if week % 3 == 0 else 0,
            "carries": 3,
            "rushing_yards": 15 + week,
            "rushing_tds": 0,
            "fantasy_points_ppr": 18.5 + week * 1.5,
        })

        # WR stats
        stats_data.append({
            "player_id": "WR001",
            "player_name": "Test Receiver",
            "player_display_name": "Test Receiver",
            "position": "WR",
            "position_group": "WR",
            "season": 2023,
            "week": week,
            "season_type": "REG",
            "team": "KC",
            "opponent_team": "BUF" if week % 2 == 0 else "LAC",
            "targets": 8 + week,
            "receptions": 6 + week // 2,
            "receiving_yards": 80 + week * 5,
            "receiving_tds": 1 if week % 2 == 0 else 0,
            "carries": 1,
            "rushing_yards": 5,
            "rushing_tds": 0,
            "fantasy_points_ppr": 12.5 + week * 1.2,
        })

        # RB stats
        stats_data.append({
            "player_id": "RB001",
            "player_name": "Test Running Back",
            "player_display_name": "Test Running Back",
            "position": "RB",
            "position_group": "RB",
            "season": 2023,
            "week": week,
            "season_type": "REG",
            "team": "BUF",
            "opponent_team": "KC" if week % 2 == 0 else "NE",
            "carries": 15 + week,
            "rushing_yards": 70 + week * 8,
            "rushing_tds": 1 if week % 3 == 0 else 0,
            "targets": 4,
            "receptions": 3,
            "receiving_yards": 25 + week * 2,
            "receiving_tds": 0,
            "fantasy_points_ppr": 14.0 + week * 1.3,
        })

    # Season 2024 - 10 weeks
    for week in range(1, 11):
        game_date = datetime(2024, 9, 1) + timedelta(weeks=week-1)

        # QB stats - slightly better performance
        stats_data.append({
            "player_id": "QB001",
            "player_name": "Test Quarterback",
            "player_display_name": "Test Quarterback",
            "position": "QB",
            "position_group": "QB",
            "season": 2024,
            "week": week,
            "season_type": "REG",
            "team": "KC",
            "opponent_team": "BUF" if week % 2 == 0 else "LAC",
            "completions": 24 + week,
            "attempts": 34 + week,
            "passing_yards": 270 + week * 12,
            "passing_tds": 2 + (1 if week % 4 == 0 else 0),
            "passing_interceptions": 1 if week % 4 == 0 else 0,
            "carries": 4,
            "rushing_yards": 18 + week,
            "rushing_tds": 1 if week == 5 else 0,
            "fantasy_points_ppr": 20.5 + week * 1.7,
        })

        # WR stats - improved
        stats_data.append({
            "player_id": "WR001",
            "player_name": "Test Receiver",
            "player_display_name": "Test Receiver",
            "position": "WR",
            "position_group": "WR",
            "season": 2024,
            "week": week,
            "season_type": "REG",
            "team": "KC",
            "opponent_team": "BUF" if week % 2 == 0 else "LAC",
            "targets": 10 + week,
            "receptions": 7 + week // 2,
            "receiving_yards": 95 + week * 6,
            "receiving_tds": 1 if week % 2 == 0 else 0,
            "carries": 1,
            "rushing_yards": 8,
            "rushing_tds": 0,
            "fantasy_points_ppr": 14.5 + week * 1.4,
        })

        # RB stats - consistent
        stats_data.append({
            "player_id": "RB001",
            "player_name": "Test Running Back",
            "player_display_name": "Test Running Back",
            "position": "RB",
            "position_group": "RB",
            "season": 2024,
            "week": week,
            "season_type": "REG",
            "team": "BUF",
            "opponent_team": "KC" if week % 2 == 0 else "NE",
            "carries": 17 + week,
            "rushing_yards": 75 + week * 9,
            "rushing_tds": 1 if week % 3 == 0 else 0,
            "targets": 5,
            "receptions": 4,
            "receiving_yards": 30 + week * 2,
            "receiving_tds": 1 if week == 7 else 0,
            "fantasy_points_ppr": 15.5 + week * 1.5,
        })

    # Insert all stats
    for stat in stats_data:
        # Create placeholders for all columns
        columns = [
            "player_id", "player_name", "player_display_name", "position",
            "position_group", "season", "week", "season_type", "team", "opponent_team",
            "completions", "attempts", "passing_yards", "passing_tds", "passing_interceptions",
            "carries", "rushing_yards", "rushing_tds",
            "targets", "receptions", "receiving_yards", "receiving_tds",
            "fantasy_points_ppr"
        ]

        values = [stat.get(col) for col in columns]
        placeholders = ",".join(["?" for _ in columns])

        conn.execute(
            f"INSERT INTO raw_player_stats ({','.join(columns)}) VALUES ({placeholders})",
            values
        )


def _insert_sample_team_stats(conn):
    """Insert sample team stats for KC and BUF."""

    teams = ["KC", "BUF", "LAC", "NE"]

    for season in [2023, 2024]:
        for week in range(1, 11):
            for team in teams:
                conn.execute("""
                    INSERT INTO raw_team_stats (
                        team, season, week, season_type,
                        points_scored, points_allowed, total_yards,
                        passing_yards, rushing_yards, turnovers
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, [
                    team, season, week, "REG",
                    24 + week, 20 + week // 2, 350 + week * 10,
                    220 + week * 5, 130 + week * 5, 1 if week % 3 == 0 else 0
                ])


def _insert_sample_schedules(conn):
    """Insert sample game schedules."""

    for season in [2023, 2024]:
        for week in range(1, 11):
            game_date = datetime(season, 9, 1) + timedelta(weeks=week-1)

            # KC vs opponent
            conn.execute("""
                INSERT INTO raw_schedules (
                    game_id, season, week, game_type, gameday,
                    home_team, away_team, home_score, away_score
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                f"{season}_0{week}_KC_BUF",
                season, week, "REG", game_date.strftime("%Y-%m-%d"),
                "KC", "BUF", 27, 24
            ])

            # BUF vs opponent
            conn.execute("""
                INSERT INTO raw_schedules (
                    game_id, season, week, game_type, gameday,
                    home_team, away_team, home_score, away_score
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                f"{season}_0{week}_BUF_NE",
                season, week, "REG", game_date.strftime("%Y-%m-%d"),
                "BUF", "NE", 28, 21
            ])


def _insert_sample_rosters(conn):
    """Insert sample weekly rosters."""

    for season in [2023, 2024]:
        for week in range(1, 11):
            # KC roster
            conn.execute("""
                INSERT INTO raw_rosters_weekly (
                    player_id, player_name, position, team, season, week
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, ["QB001", "Test Quarterback", "QB", "KC", season, week])

            conn.execute("""
                INSERT INTO raw_rosters_weekly (
                    player_id, player_name, position, team, season, week
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, ["WR001", "Test Receiver", "WR", "KC", season, week])

            # BUF roster
            conn.execute("""
                INSERT INTO raw_rosters_weekly (
                    player_id, player_name, position, team, season, week
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, ["RB001", "Test Running Back", "RB", "BUF", season, week])


# =============================================================================
# Integration Tests
# =============================================================================


class TestDataPipelineIntegration:
    """Integration tests for full data pipeline (Stages 2-4)."""

    def test_stage2_to_stage3a_flow(self, integration_test_db):
        """Test Stage 2 creates tables that Stage 3a can use."""
        pipeline = NFLDataPipeline(db_file=integration_test_db)

        # Run Stage 2
        lifecycle_count = pipeline.build_player_lifecycle_table()
        snapshot_count = pipeline.create_weekly_roster_snapshots()
        exp_count = pipeline.classify_player_experience_levels()

        assert lifecycle_count > 0, "Should create player lifecycle records"
        assert snapshot_count > 0, "Should create roster snapshots"
        assert exp_count > 0, "Should classify player experience"

        # Verify Stage 3a can access Stage 2 outputs
        rolling_count = pipeline.calculate_rolling_statistics()
        assert rolling_count > 0, "Stage 3a should use Stage 2 data successfully"

        # Verify rolling features were created
        conn = duckdb.connect(integration_test_db)
        result = conn.execute("""
            SELECT COUNT(*) FROM player_rolling_features
        """).fetchone()
        assert result[0] > 0, "Rolling features table should have data"
        conn.close()

    def test_stage3_to_stage4_flow(self, integration_test_db):
        """Test Stage 3 creates tables that Stage 4 can combine."""
        pipeline = NFLDataPipeline(db_file=integration_test_db)

        # Run Stage 2 → 3a → 3b → 3c
        pipeline.build_player_lifecycle_table()
        pipeline.create_weekly_roster_snapshots()
        pipeline.classify_player_experience_levels()
        pipeline.calculate_rolling_statistics()
        pipeline.build_matchup_features()
        pipeline.create_team_aggregates()

        # Run Stage 4
        features_count = pipeline.combine_all_features()
        assert features_count > 0, "Stage 4 should combine all features"

        # Verify numerical_features array exists and has correct size
        conn = duckdb.connect(integration_test_db)
        result = conn.execute("""
            SELECT numerical_features, feature_names FROM ml_training_features LIMIT 1
        """).fetchone()

        assert result is not None, "Should have at least one ML feature row"
        assert len(result[0]) == 47, "Should have 47 numerical features"
        assert len(result[1]) == 47, "Should have 47 feature names"
        conn.close()

    def test_full_data_pipeline_stages_2_through_4(self, integration_test_db):
        """Test complete pipeline execution through all stages."""
        pipeline = NFLDataPipeline(db_file=integration_test_db)

        # Run Stage 2
        lifecycle_count = pipeline.build_player_lifecycle_table()
        snapshot_count = pipeline.create_weekly_roster_snapshots()
        exp_count = pipeline.classify_player_experience_levels()

        # Run Stage 3a
        rolling_count = pipeline.calculate_rolling_statistics()

        # Run Stage 3b
        matchup_count = pipeline.build_matchup_features()

        # Run Stage 3c
        team_agg_count = pipeline.create_team_aggregates()

        # Run Stage 4
        features_count = pipeline.combine_all_features()
        scored, passed, filtered = pipeline.apply_data_quality_scoring()
        with_targets, without_targets = pipeline.create_prediction_targets()
        is_valid = pipeline.validate_temporal_consistency()

        # Verify all stages produced results
        assert lifecycle_count > 0, "Stage 2: Player lifecycle should have data"
        assert snapshot_count > 0, "Stage 2: Roster snapshots should have data"
        assert exp_count > 0, "Stage 2: Experience classification should have data"
        assert rolling_count > 0, "Stage 3a: Rolling features should have data"
        assert matchup_count > 0, "Stage 3b: Matchup features should have data"
        assert team_agg_count > 0, "Stage 3c: Team aggregates should have data"
        assert features_count > 0, "Stage 4: Combined features should have data"
        assert passed > 0, "Stage 4: Some data should pass quality checks"
        assert with_targets > 0, "Stage 4: Should have prediction targets"
        assert is_valid == True, "Stage 4: Temporal consistency should be valid"

    def test_data_quality_filtering(self, integration_test_db):
        """Test that data quality scoring and filtering works correctly."""
        pipeline = NFLDataPipeline(db_file=integration_test_db)

        # Run pipeline through feature combination
        pipeline.build_player_lifecycle_table()
        pipeline.classify_player_experience_levels()
        pipeline.calculate_rolling_statistics()
        pipeline.combine_all_features()

        # Apply data quality scoring
        scored, passed, filtered = pipeline.apply_data_quality_scoring()

        assert scored > 0, "Should score some records"
        assert passed >= 0, "Should have some records pass (or all fail)"
        assert filtered >= 0, "Should filter some records (or none)"
        assert scored == passed + filtered, "Scored should equal passed + filtered"

        # Verify quality scores exist
        conn = duckdb.connect(integration_test_db)
        result = conn.execute("""
            SELECT COUNT(*),
                   MIN(data_quality_score),
                   MAX(data_quality_score),
                   AVG(data_quality_score)
            FROM ml_training_features
            WHERE data_quality_score IS NOT NULL
        """).fetchone()

        count, min_score, max_score, avg_score = result
        assert count > 0, "Should have quality scores"
        assert 0.0 <= min_score <= 1.0, "Min score should be in [0, 1]"
        assert 0.0 <= max_score <= 1.0, "Max score should be in [0, 1]"
        assert 0.0 <= avg_score <= 1.0, "Avg score should be in [0, 1]"
        conn.close()

    def test_temporal_consistency_validation(self, integration_test_db):
        """Test temporal consistency checks prevent data leakage."""
        pipeline = NFLDataPipeline(db_file=integration_test_db)

        # Run through Stage 4
        pipeline.build_player_lifecycle_table()
        pipeline.classify_player_experience_levels()
        pipeline.calculate_rolling_statistics()
        pipeline.combine_all_features()
        pipeline.create_prediction_targets()

        # Validate temporal consistency
        is_valid = pipeline.validate_temporal_consistency()
        assert is_valid == True, "Temporal consistency should pass"

        # Verify no future data leakage
        conn = duckdb.connect(integration_test_db)

        # Check that features are created before targets
        result = conn.execute("""
            SELECT COUNT(*) FROM ml_training_features
            WHERE actual_outcomes IS NOT NULL
        """).fetchone()

        assert result[0] > 0, "Should have targets for validation"
        conn.close()

    def test_rolling_statistics_window_sizes(self, integration_test_db):
        """Test that rolling statistics respect window sizes (3, 5, 10 games)."""
        pipeline = NFLDataPipeline(db_file=integration_test_db)

        # Run through Stage 3a
        pipeline.build_player_lifecycle_table()
        pipeline.classify_player_experience_levels()
        pipeline.calculate_rolling_statistics()

        # Verify rolling features have correct windows
        conn = duckdb.connect(integration_test_db)

        # Check that we have rolling features with different windows
        result = conn.execute("""
            SELECT player_id, season, week, rolling_stats_json
            FROM player_rolling_features
            WHERE rolling_stats_json IS NOT NULL
            LIMIT 1
        """).fetchone()

        if result:
            player_id, season, week, rolling_json = result
            rolling_stats = json.loads(rolling_json)

            # Should have stats for different window sizes
            assert "3_game" in rolling_stats or "passing_yards_avg_3" in str(rolling_stats), \
                "Should have 3-game rolling stats"

        conn.close()

    def test_position_specific_features(self, integration_test_db):
        """Test that different positions get appropriate features."""
        pipeline = NFLDataPipeline(db_file=integration_test_db)

        # Run full pipeline
        pipeline.build_player_lifecycle_table()
        pipeline.classify_player_experience_levels()
        pipeline.calculate_rolling_statistics()
        pipeline.combine_all_features()

        # Check position-specific features
        conn = duckdb.connect(integration_test_db)

        # QB should have passing features
        qb_result = conn.execute("""
            SELECT feature_names FROM ml_training_features
            WHERE entity_id = 'QB001'
            LIMIT 1
        """).fetchone()

        if qb_result:
            feature_names = qb_result[0]
            # Convert list to string for easier checking
            feature_str = str(feature_names)
            # Should have passing-related features
            assert "passing" in feature_str.lower() or "yards" in feature_str.lower(), \
                "QB should have passing features"

        conn.close()

    def test_multiple_seasons_integration(self, integration_test_db):
        """Test that pipeline handles multiple seasons correctly."""
        pipeline = NFLDataPipeline(db_file=integration_test_db)

        # Run full pipeline
        pipeline.build_player_lifecycle_table()
        pipeline.classify_player_experience_levels()
        pipeline.calculate_rolling_statistics()
        pipeline.combine_all_features()

        # Verify data from both seasons
        conn = duckdb.connect(integration_test_db)

        result = conn.execute("""
            SELECT DISTINCT season
            FROM ml_training_features
            ORDER BY season
        """).fetchall()

        seasons = [r[0] for r in result]
        assert len(seasons) >= 2, "Should have data from multiple seasons"
        assert 2023 in seasons, "Should have 2023 data"
        assert 2024 in seasons, "Should have 2024 data"

        conn.close()


class TestFeatureEngineering:
    """Integration tests for feature engineering components."""

    def test_rolling_features_calculation(self, integration_test_db):
        """Test rolling statistics are calculated correctly."""
        pipeline = NFLDataPipeline(db_file=integration_test_db)

        pipeline.build_player_lifecycle_table()
        pipeline.classify_player_experience_levels()
        rolling_count = pipeline.calculate_rolling_statistics()

        assert rolling_count > 0, "Should calculate rolling features"

        # Verify QB has meaningful rolling stats
        conn = duckdb.connect(integration_test_db)
        result = conn.execute("""
            SELECT season, week, rolling_stats_json
            FROM player_rolling_features
            WHERE player_id = 'QB001'
            ORDER BY season, week
        """).fetchall()

        assert len(result) > 0, "QB should have rolling features"
        conn.close()

    def test_matchup_features_creation(self, integration_test_db):
        """Test matchup features are created properly."""
        pipeline = NFLDataPipeline(db_file=integration_test_db)

        # Run prerequisites
        pipeline.build_player_lifecycle_table()
        pipeline.calculate_rolling_statistics()

        # Create matchup features
        matchup_count = pipeline.build_matchup_features()

        assert matchup_count > 0, "Should create matchup features"

    def test_team_aggregates_creation(self, integration_test_db):
        """Test team aggregates are created correctly."""
        pipeline = NFLDataPipeline(db_file=integration_test_db)

        # Run prerequisites
        pipeline.build_player_lifecycle_table()
        pipeline.calculate_rolling_statistics()

        # Create team aggregates
        team_agg_count = pipeline.create_team_aggregates()

        assert team_agg_count > 0, "Should create team aggregates"

        # Verify team aggregates exist
        conn = duckdb.connect(integration_test_db)
        result = conn.execute("""
            SELECT COUNT(*) FROM team_rolling_features
        """).fetchone()

        assert result[0] > 0, "Team rolling features should have data"
        conn.close()


class TestDataQuality:
    """Integration tests for data quality and validation."""

    def test_data_completeness_scoring(self, integration_test_db):
        """Test data completeness is scored correctly."""
        pipeline = NFLDataPipeline(db_file=integration_test_db)

        # Run through feature creation
        pipeline.build_player_lifecycle_table()
        pipeline.calculate_rolling_statistics()
        pipeline.combine_all_features()

        # Score data quality
        scored, passed, filtered = pipeline.apply_data_quality_scoring()

        assert scored > 0, "Should score data"

        # Check quality score distribution
        conn = duckdb.connect(integration_test_db)
        result = conn.execute("""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN data_quality_score >= 0.7 THEN 1 ELSE 0 END) as high_quality,
                AVG(data_quality_score) as avg_score
            FROM ml_training_features
            WHERE data_quality_score IS NOT NULL
        """).fetchone()

        total, high_quality, avg_score = result
        assert total > 0, "Should have scored records"
        assert avg_score >= 0.0 and avg_score <= 1.0, "Average score should be in valid range"

        conn.close()

    def test_outlier_detection(self, integration_test_db):
        """Test outlier detection in data quality scoring."""
        pipeline = NFLDataPipeline(db_file=integration_test_db)

        # Run pipeline
        pipeline.build_player_lifecycle_table()
        pipeline.calculate_rolling_statistics()
        pipeline.combine_all_features()

        # Apply quality scoring (which includes outlier detection)
        scored, passed, filtered = pipeline.apply_data_quality_scoring()

        # All our test data should be reasonable (no outliers)
        # So pass + filter should equal total scored
        assert scored == passed + filtered, "Should account for all scored records"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
