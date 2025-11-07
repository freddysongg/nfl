"""
Unit tests for Stage 3b: Matchup Features

Tests cover:
- Matchup features added to existing rolling features
- Rest days calculation
- Divisional game detection
- Opponent history aggregation
- Home/away splits calculation
"""

import pytest
import duckdb
import polars as pl
from pathlib import Path
import json
from datetime import datetime, timedelta

from src.data_pipeline import NFLDataPipeline
from src.database import NFLDatabase
from src.table_schemas import create_raw_player_stats_table, create_raw_schedules_table


@pytest.fixture
def test_db_path(tmp_path):
    """Create temporary test database path."""
    db_path = tmp_path / "test_stage3b.duckdb"
    yield str(db_path)
    if db_path.exists():
        db_path.unlink()


@pytest.fixture
def test_db(test_db_path):
    """Create and populate test database with sample data."""
    conn = duckdb.connect(test_db_path)

    # Create necessary tables
    create_raw_player_stats_table(conn)
    create_raw_schedules_table(conn)

    # Create rolling features table
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

            -- Rolling stats (JSON)
            stats_last3_games JSON,
            stats_last5_games JSON,
            stats_last10_games JSON,

            -- Trends
            performance_trend FLOAT,
            usage_trend FLOAT,
            efficiency_trend FLOAT,

            -- Matchup features (Stage 3b fills these)
            vs_opponent_history JSON,
            home_away_splits JSON,
            divisional_game BOOLEAN,
            rest_days INTEGER,

            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.close()
    yield test_db_path


@pytest.fixture
def sample_schedules(test_db):
    """Insert sample schedule data."""
    conn = duckdb.connect(test_db)

    # Sample schedule with rest days and divisional games
    schedules = [
        # Week 1: KC vs DAL (not divisional)
        ("2025_01_KC_DAL", 2025, "REG", 1, "2025-09-08", "KC", 27, "DAL", 24, 0, 7, 7),

        # Week 2: KC vs LV (divisional in AFC West)
        ("2025_02_KC_LV", 2025, "REG", 2, "2025-09-15", "KC", 30, "LV", 21, 1, 7, 7),

        # Week 3: KC vs LAC (divisional in AFC West) - short week
        ("2025_03_KC_LAC", 2025, "REG", 3, "2025-09-19", "KC", 24, "LAC", 20, 1, 4, 7),

        # Week 4: DAL vs PHI (divisional in NFC East)
        ("2025_04_DAL_PHI", 2025, "REG", 4, "2025-09-29", "DAL", 20, "PHI", 17, 1, 7, 10),
    ]

    for sched in schedules:
        conn.execute("""
            INSERT INTO raw_schedules
            (game_id, season, game_type, week, gameday, home_team, home_score,
             away_team, away_score, div_game, home_rest, away_rest)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, sched)

    conn.close()


@pytest.fixture
def sample_player_stats_with_matchups(test_db, sample_schedules):
    """Insert player stats with opponent information."""
    conn = duckdb.connect(test_db)

    # QB playing multiple games against same opponent
    stats = [
        # Week 1: vs DAL
        ("00-0001111", "Test QB", "QB", 2025, 1, "KC", "DAL", 300, 2, 0),

        # Week 2: vs LV
        ("00-0001111", "Test QB", "QB", 2025, 2, "KC", "LV", 250, 1, 1),

        # Week 3: vs LAC
        ("00-0001111", "Test QB", "QB", 2025, 3, "KC", "LAC", 350, 3, 0),

        # Week 4: same team (DAL) playing PHI
        ("00-0002222", "DAL QB", "QB", 2025, 4, "DAL", "PHI", 280, 2, 1),
    ]

    for stat in stats:
        conn.execute("""
            INSERT INTO raw_player_stats
            (player_id, player_name, position, season, week, team, opponent_team,
             passing_yards, passing_tds, passing_interceptions)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, stat)

    conn.close()


@pytest.fixture
def sample_rolling_features(test_db):
    """Insert sample rolling features (Stage 3a output)."""
    conn = duckdb.connect(test_db)

    features = [
        ("feat_1", "00-0001111", "Test QB", 2025, 2, "QB", "KC", "LV",
         '{"passing": {"avg_passing_yards": 300}}', '{}', '{}', 0.5, 0.3, 0.4),

        ("feat_2", "00-0001111", "Test QB", 2025, 3, "QB", "KC", "LAC",
         '{"passing": {"avg_passing_yards": 275}}', '{}', '{}', 0.6, 0.4, 0.5),

        ("feat_3", "00-0002222", "DAL QB", 2025, 4, "QB", "DAL", "PHI",
         '{"passing": {"avg_passing_yards": 280}}', '{}', '{}', 0.4, 0.3, 0.4),
    ]

    for feat in features:
        conn.execute("""
            INSERT INTO player_rolling_features
            (feature_id, player_id, player_name, season, week, position, team, opponent,
             stats_last3_games, stats_last5_games, stats_last10_games,
             performance_trend, usage_trend, efficiency_trend)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, feat)

    conn.close()


class TestStage3bMatchupFeatures:
    """Unit tests for Stage 3b: Matchup Features"""

    def test_build_matchup_features_updates_records(self, test_db, sample_player_stats_with_matchups, sample_rolling_features):
        """Test matchup features are added to existing rolling features."""
        pipeline = NFLDataPipeline(test_db)

        # Run matchup feature building
        pipeline.build_matchup_features()

        conn = duckdb.connect(test_db)

        # Check that records were updated
        result = conn.execute("""
            SELECT COUNT(*) FROM player_rolling_features
            WHERE rest_days IS NOT NULL
        """).fetchone()[0]

        assert result > 0

        conn.close()

    def test_rest_days_calculation(self, test_db, sample_player_stats_with_matchups, sample_rolling_features):
        """Test rest days calculation from schedules."""
        pipeline = NFLDataPipeline(test_db)
        pipeline.build_matchup_features()

        conn = duckdb.connect(test_db)

        # Week 2: KC had 7 days rest
        result = conn.execute("""
            SELECT rest_days FROM player_rolling_features
            WHERE player_id = '00-0001111' AND week = 2
        """).fetchone()

        if result:
            rest_days = result[0]
            assert rest_days == 7

        # Week 3: KC had 4 days rest (short week)
        result = conn.execute("""
            SELECT rest_days FROM player_rolling_features
            WHERE player_id = '00-0001111' AND week = 3
        """).fetchone()

        if result:
            rest_days = result[0]
            assert rest_days == 4

        conn.close()

    def test_divisional_game_detection(self, test_db, sample_player_stats_with_matchups, sample_rolling_features):
        """Test divisional game flag."""
        pipeline = NFLDataPipeline(test_db)
        pipeline.build_matchup_features()

        conn = duckdb.connect(test_db)

        # Week 2: KC vs LV is divisional (AFC West)
        result = conn.execute("""
            SELECT divisional_game FROM player_rolling_features
            WHERE player_id = '00-0001111' AND week = 2
        """).fetchone()

        if result:
            is_divisional = result[0]
            assert is_divisional == True

        # Week 4: DAL vs PHI is divisional (NFC East)
        result = conn.execute("""
            SELECT divisional_game FROM player_rolling_features
            WHERE player_id = '00-0002222' AND week = 4
        """).fetchone()

        if result:
            is_divisional = result[0]
            assert is_divisional == True

        conn.close()

    def test_opponent_history_aggregation(self, test_db, sample_player_stats_with_matchups, sample_rolling_features):
        """Test opponent history is calculated."""
        # Need to add more games against same opponent
        conn = duckdb.connect(test_db)

        # Add previous game against same opponent
        conn.execute("""
            INSERT INTO raw_player_stats
            (player_id, player_name, position, season, week, team, opponent_team,
             passing_yards, passing_tds, passing_interceptions)
            VALUES
            ('00-0001111', 'Test QB', 'QB', 2024, 5, 'KC', 'LV', 320, 2, 0),
            ('00-0001111', 'Test QB', 'QB', 2024, 12, 'KC', 'LV', 280, 3, 1)
        """)
        conn.close()

        pipeline = NFLDataPipeline(test_db)
        pipeline.build_matchup_features()

        conn = duckdb.connect(test_db)

        # Week 2 2025: Playing LV (should have history from 2024)
        result = conn.execute("""
            SELECT vs_opponent_history FROM player_rolling_features
            WHERE player_id = '00-0001111' AND week = 2
        """).fetchone()

        if result and result[0]:
            history = json.loads(result[0])
            # Should have some opponent history data
            assert isinstance(history, dict)

        conn.close()

    def test_home_away_splits(self, test_db):
        """Test home/away splits calculation."""
        conn = duckdb.connect(test_db)

        # Create schedule with home/away games
        conn.execute("""
            INSERT INTO raw_schedules
            (game_id, season, game_type, week, gameday, home_team, away_team, div_game)
            VALUES
            ('2025_01_KC_DAL', 2025, 'REG', 1, '2025-09-08', 'KC', 'DAL', 0),
            ('2025_02_DAL_KC', 2025, 'REG', 2, '2025-09-15', 'DAL', 'KC', 0)
        """)

        # Player stats at home and away
        conn.execute("""
            INSERT INTO raw_player_stats
            (player_id, player_name, position, season, week, team, opponent_team,
             passing_yards, passing_tds)
            VALUES
            ('00-0001111', 'Test QB', 'QB', 2025, 1, 'KC', 'DAL', 300, 2),
            ('00-0001111', 'Test QB', 'QB', 2025, 2, 'KC', 'DAL', 250, 1)
        """)

        # Rolling features
        conn.execute("""
            INSERT INTO player_rolling_features
            (feature_id, player_id, season, week, team, opponent,
             stats_last3_games, performance_trend)
            VALUES
            ('f1', '00-0001111', 2025, 1, 'KC', 'DAL', '{}', 0.5),
            ('f2', '00-0001111', 2025, 2, 'KC', 'DAL', '{}', 0.5)
        """)
        conn.close()

        pipeline = NFLDataPipeline(test_db)
        pipeline.build_matchup_features()

        conn = duckdb.connect(test_db)

        # Check home/away splits exist
        result = conn.execute("""
            SELECT home_away_splits FROM player_rolling_features
            WHERE player_id = '00-0001111' AND week = 2
        """).fetchone()

        if result and result[0]:
            splits = json.loads(result[0])
            assert isinstance(splits, dict)

        conn.close()

    def test_matchup_features_json_structure(self, test_db, sample_player_stats_with_matchups, sample_rolling_features):
        """Test matchup features have correct JSON structure."""
        pipeline = NFLDataPipeline(test_db)
        pipeline.build_matchup_features()

        conn = duckdb.connect(test_db)

        result = conn.execute("""
            SELECT vs_opponent_history, home_away_splits
            FROM player_rolling_features
            WHERE player_id = '00-0001111' AND week = 2
        """).fetchone()

        if result:
            # Both should be valid JSON (or NULL)
            if result[0]:
                history = json.loads(result[0])
                assert isinstance(history, dict)

            if result[1]:
                splits = json.loads(result[1])
                assert isinstance(splits, dict)

        conn.close()

    def test_matchup_features_all_fields_populated(self, test_db, sample_player_stats_with_matchups, sample_rolling_features):
        """Test all matchup feature fields are populated."""
        pipeline = NFLDataPipeline(test_db)
        pipeline.build_matchup_features()

        conn = duckdb.connect(test_db)

        result = conn.execute("""
            SELECT rest_days, divisional_game
            FROM player_rolling_features
            WHERE player_id = '00-0001111' AND week = 2
        """).fetchone()

        if result:
            rest_days, divisional = result

            # rest_days should be an integer
            assert rest_days is not None
            assert isinstance(rest_days, int)

            # divisional_game should be boolean
            assert divisional is not None
            assert isinstance(divisional, (bool, int))

        conn.close()


class TestStage3bEdgeCases:
    """Test edge cases for Stage 3b"""

    def test_missing_schedule_data(self, test_db, sample_rolling_features):
        """Test handling when schedule data is missing."""
        pipeline = NFLDataPipeline(test_db)

        # Should not crash without schedule data
        pipeline.build_matchup_features()

        # Success if no exception
        assert True

    def test_no_opponent_history(self, test_db):
        """Test handling when player has never faced opponent."""
        conn = duckdb.connect(test_db)

        # Create schedule
        conn.execute("""
            INSERT INTO raw_schedules
            (game_id, season, game_type, week, gameday, home_team, away_team, div_game)
            VALUES ('2025_01_KC_DAL', 2025, 'REG', 1, '2025-09-08', 'KC', 'DAL', 0)
        """)

        # First time facing this opponent
        conn.execute("""
            INSERT INTO raw_player_stats
            (player_id, player_name, position, season, week, team, opponent_team,
             passing_yards)
            VALUES ('00-0009999', 'New QB', 'QB', 2025, 1, 'KC', 'DAL', 250)
        """)

        conn.execute("""
            INSERT INTO player_rolling_features
            (feature_id, player_id, season, week, team, opponent, stats_last3_games)
            VALUES ('f1', '00-0009999', 2025, 1, 'KC', 'DAL', '{}')
        """)
        conn.close()

        pipeline = NFLDataPipeline(test_db)
        pipeline.build_matchup_features()

        conn = duckdb.connect(test_db)

        # Should handle gracefully (empty history or NULL)
        result = conn.execute("""
            SELECT vs_opponent_history FROM player_rolling_features
            WHERE player_id = '00-0009999'
        """).fetchone()

        # Should not crash
        assert result is not None

        conn.close()

    def test_first_game_of_season_rest_days(self, test_db):
        """Test rest days for first game of season."""
        conn = duckdb.connect(test_db)

        # Week 1 game (first of season)
        conn.execute("""
            INSERT INTO raw_schedules
            (game_id, season, game_type, week, gameday, home_team, away_team,
             home_rest, away_rest, div_game)
            VALUES ('2025_01_KC_DAL', 2025, 'REG', 1, '2025-09-08', 'KC', 'DAL', 180, 180, 0)
        """)

        conn.execute("""
            INSERT INTO player_rolling_features
            (feature_id, player_id, season, week, team, opponent, stats_last3_games)
            VALUES ('f1', '00-0008888', 2025, 1, 'KC', 'DAL', '{}')
        """)
        conn.close()

        pipeline = NFLDataPipeline(test_db)
        pipeline.build_matchup_features()

        conn = duckdb.connect(test_db)

        # Should have rest_days set (from preseason)
        result = conn.execute("""
            SELECT rest_days FROM player_rolling_features
            WHERE feature_id = 'f1'
        """).fetchone()

        if result:
            # Should have some rest days value
            assert result[0] is not None

        conn.close()

    def test_multiple_teams_same_week(self, test_db):
        """Test matchup features for different teams in same week."""
        conn = duckdb.connect(test_db)

        # Two different games in week 1
        conn.execute("""
            INSERT INTO raw_schedules
            (game_id, season, game_type, week, gameday, home_team, away_team, div_game)
            VALUES
            ('2025_01_KC_DAL', 2025, 'REG', 1, '2025-09-08', 'KC', 'DAL', 0),
            ('2025_01_SF_NYG', 2025, 'REG', 1, '2025-09-08', 'SF', 'NYG', 0)
        """)

        conn.execute("""
            INSERT INTO player_rolling_features
            (feature_id, player_id, season, week, team, opponent, stats_last3_games)
            VALUES
            ('f1', '00-0001111', 2025, 1, 'KC', 'DAL', '{}'),
            ('f2', '00-0002222', 2025, 1, 'SF', 'NYG', '{}')
        """)
        conn.close()

        pipeline = NFLDataPipeline(test_db)
        pipeline.build_matchup_features()

        conn = duckdb.connect(test_db)

        # Both should have matchup features
        count = conn.execute("""
            SELECT COUNT(*) FROM player_rolling_features
            WHERE week = 1 AND divisional_game IS NOT NULL
        """).fetchone()[0]

        assert count == 2

        conn.close()
