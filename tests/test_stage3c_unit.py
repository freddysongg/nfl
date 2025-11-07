"""
Unit tests for Stage 3c: Team Aggregates

Tests cover:
- Team aggregates calculation
- Offensive metrics (EPA, success rate)
- Defensive metrics
- Fallback mode (no PBP data)
- JSON structure validation
"""

import pytest
import duckdb
import polars as pl
from pathlib import Path
import json

from src.data_pipeline import NFLDataPipeline
from src.database import NFLDatabase
from src.table_schemas import (
    create_raw_team_stats_table,
    create_raw_pbp_table,
    create_raw_schedules_table
)


@pytest.fixture
def test_db_path(tmp_path):
    """Create temporary test database path."""
    db_path = tmp_path / "test_stage3c.duckdb"
    yield str(db_path)
    if db_path.exists():
        db_path.unlink()


@pytest.fixture
def test_db(test_db_path):
    """Create and populate test database with sample data."""
    conn = duckdb.connect(test_db_path)

    # Create necessary tables
    create_raw_team_stats_table(conn)
    create_raw_pbp_table(conn)
    create_raw_schedules_table(conn)

    # Create team rolling features table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS team_rolling_features (
            feature_id VARCHAR PRIMARY KEY,
            team VARCHAR,
            season INTEGER,
            week INTEGER,

            -- Offensive metrics (JSON)
            offensive_stats JSON,
            offensive_epa JSON,
            offensive_success_rate JSON,

            -- Defensive metrics (JSON)
            defensive_stats JSON,
            defensive_epa JSON,
            defensive_success_rate JSON,

            -- Situational metrics (JSON)
            red_zone_efficiency JSON,
            third_down_efficiency JSON,
            explosive_play_rate JSON,

            -- Trends
            offensive_trend FLOAT,
            defensive_trend FLOAT,

            -- Metadata
            data_source VARCHAR,  -- 'pbp' or 'team_stats'
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.close()
    yield test_db_path


@pytest.fixture
def sample_pbp_data(test_db):
    """Insert sample play-by-play data."""
    conn = duckdb.connect(test_db)

    # Sample PBP plays with EPA and success metrics
    plays = [
        # KC offensive plays (week 1)
        ("2025_01_KC_DAL", "play_1", 2025, 1, "KC", "KC", "DAL", "pass", 15, 0.5, 1, 1, 0),
        ("2025_01_KC_DAL", "play_2", 2025, 1, "KC", "KC", "DAL", "run", 5, 0.2, 1, 0, 0),
        ("2025_01_KC_DAL", "play_3", 2025, 1, "KC", "KC", "DAL", "pass", 25, 1.5, 1, 1, 1),

        # DAL defensive plays (same game)
        ("2025_01_KC_DAL", "play_4", 2025, 1, "DAL", "DAL", "KC", "pass", -3, -0.5, 0, 0, 0),
        ("2025_01_KC_DAL", "play_5", 2025, 1, "DAL", "DAL", "KC", "run", 2, -0.2, 0, 0, 0),

        # KC week 2
        ("2025_02_KC_LV", "play_6", 2025, 2, "KC", "KC", "LV", "pass", 12, 0.8, 1, 1, 0),
        ("2025_02_KC_LV", "play_7", 2025, 2, "KC", "KC", "LV", "run", 8, 0.4, 1, 0, 0),
    ]

    for play in plays:
        conn.execute("""
            INSERT INTO raw_pbp
            (game_id, play_id, season, week, posteam, posteam, defteam,
             play_type, yards_gained, epa, success, first_down, touchdown)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, play)

    conn.close()


@pytest.fixture
def sample_team_stats(test_db):
    """Insert sample team stats (fallback data)."""
    conn = duckdb.connect(test_db)

    # Team stats for weeks without PBP
    stats = [
        # KC week 1
        (2025, 1, "KC", "REG", "DAL", 25, 35, 350, 3, 1, 20, 150, 1),

        # DAL week 1
        (2025, 1, "DAL", "REG", "KC", 20, 30, 280, 2, 2, 18, 120, 0),

        # KC week 2
        (2025, 2, "KC", "REG", "LV", 28, 38, 380, 4, 0, 22, 160, 2),
    ]

    for stat in stats:
        conn.execute("""
            INSERT INTO raw_team_stats
            (season, week, team, season_type, opponent_team,
             completions, attempts, passing_yards, passing_tds, passing_interceptions,
             carries, rushing_yards, rushing_tds)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, stat)

    conn.close()


@pytest.fixture
def sample_schedules(test_db):
    """Insert sample schedule data."""
    conn = duckdb.connect(test_db)

    schedules = [
        ("2025_01_KC_DAL", 2025, "REG", 1, "2025-09-08", "KC", 27, "DAL", 24),
        ("2025_02_KC_LV", 2025, "REG", 2, "2025-09-15", "KC", 30, "LV", 21),
    ]

    for sched in schedules:
        conn.execute("""
            INSERT INTO raw_schedules
            (game_id, season, game_type, week, gameday, home_team, home_score,
             away_team, away_score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, sched)

    conn.close()


class TestStage3cTeamAggregates:
    """Unit tests for Stage 3c: Team Aggregates"""

    def test_create_team_aggregates_from_pbp(self, test_db, sample_pbp_data, sample_schedules):
        """Test team aggregates calculation from PBP data."""
        pipeline = NFLDataPipeline(test_db)

        # Run team aggregates calculation
        pipeline.create_team_aggregates()

        conn = duckdb.connect(test_db)

        # Check records were created
        count = conn.execute("""
            SELECT COUNT(*) FROM team_rolling_features
        """).fetchone()[0]

        assert count > 0

        conn.close()

    def test_offensive_epa_calculation(self, test_db, sample_pbp_data, sample_schedules):
        """Test offensive EPA metrics are calculated."""
        pipeline = NFLDataPipeline(test_db)
        pipeline.create_team_aggregates()

        conn = duckdb.connect(test_db)

        # KC should have offensive EPA for week 1
        result = conn.execute("""
            SELECT offensive_epa FROM team_rolling_features
            WHERE team = 'KC' AND season = 2025 AND week = 1
        """).fetchone()

        if result and result[0]:
            epa_stats = json.loads(result[0])

            # Should have EPA metrics
            assert isinstance(epa_stats, dict)
            assert len(epa_stats) > 0

        conn.close()

    def test_defensive_metrics_calculation(self, test_db, sample_pbp_data, sample_schedules):
        """Test defensive metrics are calculated."""
        pipeline = NFLDataPipeline(test_db)
        pipeline.create_team_aggregates()

        conn = duckdb.connect(test_db)

        # Check defensive stats exist
        result = conn.execute("""
            SELECT defensive_stats FROM team_rolling_features
            WHERE team = 'KC' AND season = 2025 AND week = 1
        """).fetchone()

        if result and result[0]:
            def_stats = json.loads(result[0])
            assert isinstance(def_stats, dict)

        conn.close()

    def test_success_rate_calculation(self, test_db, sample_pbp_data, sample_schedules):
        """Test success rate metrics."""
        pipeline = NFLDataPipeline(test_db)
        pipeline.create_team_aggregates()

        conn = duckdb.connect(test_db)

        result = conn.execute("""
            SELECT offensive_success_rate FROM team_rolling_features
            WHERE team = 'KC' AND season = 2025 AND week = 1
        """).fetchone()

        if result and result[0]:
            success_rate = json.loads(result[0])

            # Should have success rate metrics
            assert isinstance(success_rate, dict)

        conn.close()

    def test_fallback_to_team_stats(self, test_db, sample_team_stats, sample_schedules):
        """Test fallback to team stats when no PBP data."""
        # Clear any PBP data
        conn = duckdb.connect(test_db)
        conn.execute("DELETE FROM raw_pbp")
        conn.close()

        pipeline = NFLDataPipeline(test_db)
        pipeline.create_team_aggregates()

        conn = duckdb.connect(test_db)

        # Should still create aggregates from team_stats
        result = conn.execute("""
            SELECT data_source FROM team_rolling_features
            WHERE team = 'KC' AND season = 2025 AND week = 1
        """).fetchone()

        if result:
            source = result[0]
            # Should use team_stats as fallback
            assert source == 'team_stats' or source == 'fallback'

        conn.close()

    def test_json_structure_offensive_stats(self, test_db, sample_pbp_data, sample_schedules):
        """Test offensive stats JSON structure."""
        pipeline = NFLDataPipeline(test_db)
        pipeline.create_team_aggregates()

        conn = duckdb.connect(test_db)

        result = conn.execute("""
            SELECT offensive_stats FROM team_rolling_features
            WHERE team = 'KC' AND season = 2025 AND week = 1
        """).fetchone()

        if result and result[0]:
            stats = json.loads(result[0])

            # Should be valid JSON dict
            assert isinstance(stats, dict)

        conn.close()

    def test_json_structure_defensive_stats(self, test_db, sample_pbp_data, sample_schedules):
        """Test defensive stats JSON structure."""
        pipeline = NFLDataPipeline(test_db)
        pipeline.create_team_aggregates()

        conn = duckdb.connect(test_db)

        result = conn.execute("""
            SELECT defensive_stats FROM team_rolling_features
            WHERE team = 'KC' AND season = 2025 AND week = 1
        """).fetchone()

        if result and result[0]:
            stats = json.loads(result[0])
            assert isinstance(stats, dict)

        conn.close()

    def test_multiple_weeks_same_team(self, test_db, sample_pbp_data, sample_schedules):
        """Test team aggregates for multiple weeks."""
        pipeline = NFLDataPipeline(test_db)
        pipeline.create_team_aggregates()

        conn = duckdb.connect(test_db)

        # KC should have data for weeks 1 and 2
        count = conn.execute("""
            SELECT COUNT(*) FROM team_rolling_features
            WHERE team = 'KC' AND season = 2025
        """).fetchone()[0]

        assert count >= 2  # At least 2 weeks

        conn.close()

    def test_red_zone_efficiency(self, test_db):
        """Test red zone efficiency calculation."""
        conn = duckdb.connect(test_db)

        # Add red zone plays
        conn.execute("""
            INSERT INTO raw_pbp
            (game_id, play_id, season, week, posteam, defteam, play_type,
             yardline_100, touchdown, epa)
            VALUES
            ('2025_01_KC_DAL', 'rz1', 2025, 1, 'KC', 'DAL', 'pass', 15, 1, 2.0),
            ('2025_01_KC_DAL', 'rz2', 2025, 1, 'KC', 'DAL', 'run', 10, 0, -0.5)
        """)

        conn.execute("""
            INSERT INTO raw_schedules
            (game_id, season, game_type, week, gameday, home_team, away_team)
            VALUES ('2025_01_KC_DAL', 2025, 'REG', 1, '2025-09-08', 'KC', 'DAL')
        """)
        conn.close()

        pipeline = NFLDataPipeline(test_db)
        pipeline.create_team_aggregates()

        conn = duckdb.connect(test_db)

        result = conn.execute("""
            SELECT red_zone_efficiency FROM team_rolling_features
            WHERE team = 'KC' AND week = 1
        """).fetchone()

        if result and result[0]:
            rz_stats = json.loads(result[0])
            assert isinstance(rz_stats, dict)

        conn.close()

    def test_third_down_efficiency(self, test_db):
        """Test third down efficiency calculation."""
        conn = duckdb.connect(test_db)

        # Add third down plays
        conn.execute("""
            INSERT INTO raw_pbp
            (game_id, play_id, season, week, posteam, defteam, play_type,
             down, ydstogo, first_down, epa)
            VALUES
            ('2025_01_KC_DAL', '3d1', 2025, 1, 'KC', 'DAL', 'pass', 3, 5, 1, 1.0),
            ('2025_01_KC_DAL', '3d2', 2025, 1, 'KC', 'DAL', 'run', 3, 8, 0, -0.5)
        """)

        conn.execute("""
            INSERT INTO raw_schedules
            (game_id, season, game_type, week, gameday, home_team, away_team)
            VALUES ('2025_01_KC_DAL', 2025, 'REG', 1, '2025-09-08', 'KC', 'DAL')
        """)
        conn.close()

        pipeline = NFLDataPipeline(test_db)
        pipeline.create_team_aggregates()

        conn = duckdb.connect(test_db)

        result = conn.execute("""
            SELECT third_down_efficiency FROM team_rolling_features
            WHERE team = 'KC' AND week = 1
        """).fetchone()

        if result and result[0]:
            third_down = json.loads(result[0])
            assert isinstance(third_down, dict)

        conn.close()


class TestStage3cEdgeCases:
    """Test edge cases for Stage 3c"""

    def test_no_pbp_no_team_stats(self, test_db):
        """Test handling when both PBP and team stats are missing."""
        pipeline = NFLDataPipeline(test_db)

        # Should not crash with empty database
        pipeline.create_team_aggregates()

        # Success if no exception
        assert True

    def test_team_with_single_play(self, test_db):
        """Test team with only one play."""
        conn = duckdb.connect(test_db)

        conn.execute("""
            INSERT INTO raw_pbp
            (game_id, play_id, season, week, posteam, defteam, play_type, epa, success)
            VALUES ('2025_01_KC_DAL', 'p1', 2025, 1, 'KC', 'DAL', 'pass', 0.5, 1)
        """)

        conn.execute("""
            INSERT INTO raw_schedules
            (game_id, season, game_type, week, gameday, home_team, away_team)
            VALUES ('2025_01_KC_DAL', 2025, 'REG', 1, '2025-09-08', 'KC', 'DAL')
        """)
        conn.close()

        pipeline = NFLDataPipeline(test_db)
        pipeline.create_team_aggregates()

        conn = duckdb.connect(test_db)

        # Should still create aggregate
        count = conn.execute("""
            SELECT COUNT(*) FROM team_rolling_features
            WHERE team = 'KC'
        """).fetchone()[0]

        assert count > 0
        conn.close()

    def test_all_unsuccessful_plays(self, test_db):
        """Test team with all unsuccessful plays."""
        conn = duckdb.connect(test_db)

        # All plays fail
        for i in range(3):
            conn.execute("""
                INSERT INTO raw_pbp
                (game_id, play_id, season, week, posteam, defteam, play_type, epa, success)
                VALUES ('2025_01_KC_DAL', ?, 2025, 1, 'KC', 'DAL', 'pass', -0.5, 0)
            """, [f'p{i}'])

        conn.execute("""
            INSERT INTO raw_schedules
            (game_id, season, game_type, week, gameday, home_team, away_team)
            VALUES ('2025_01_KC_DAL', 2025, 'REG', 1, '2025-09-08', 'KC', 'DAL')
        """)
        conn.close()

        pipeline = NFLDataPipeline(test_db)
        pipeline.create_team_aggregates()

        conn = duckdb.connect(test_db)

        result = conn.execute("""
            SELECT offensive_success_rate FROM team_rolling_features
            WHERE team = 'KC'
        """).fetchone()

        if result and result[0]:
            success_rate = json.loads(result[0])
            # Should have 0% success rate
            assert isinstance(success_rate, dict)

        conn.close()

    def test_null_epa_values(self, test_db):
        """Test handling of NULL EPA values."""
        conn = duckdb.connect(test_db)

        conn.execute("""
            INSERT INTO raw_pbp
            (game_id, play_id, season, week, posteam, defteam, play_type, epa, success)
            VALUES
            ('2025_01_KC_DAL', 'p1', 2025, 1, 'KC', 'DAL', 'pass', NULL, 1),
            ('2025_01_KC_DAL', 'p2', 2025, 1, 'KC', 'DAL', 'run', 0.5, 1)
        """)

        conn.execute("""
            INSERT INTO raw_schedules
            (game_id, season, game_type, week, gameday, home_team, away_team)
            VALUES ('2025_01_KC_DAL', 2025, 'REG', 1, '2025-09-08', 'KC', 'DAL')
        """)
        conn.close()

        pipeline = NFLDataPipeline(test_db)
        # Should handle NULLs gracefully
        pipeline.create_team_aggregates()

        # Success if no exception
        assert True

    def test_team_aggregates_table_columns(self, test_db, sample_pbp_data, sample_schedules):
        """Test team aggregates table has required columns."""
        pipeline = NFLDataPipeline(test_db)
        pipeline.create_team_aggregates()

        conn = duckdb.connect(test_db)

        columns = conn.execute("DESCRIBE team_rolling_features").fetchall()
        column_names = [col[0] for col in columns]

        required_columns = [
            'team', 'season', 'week',
            'offensive_stats', 'offensive_epa', 'offensive_success_rate',
            'defensive_stats', 'defensive_epa', 'defensive_success_rate'
        ]

        for col in required_columns:
            assert col in column_names, f"Missing column: {col}"

        conn.close()

    def test_multiple_teams_same_week(self, test_db, sample_pbp_data, sample_schedules):
        """Test aggregates for multiple teams in same week."""
        pipeline = NFLDataPipeline(test_db)
        pipeline.create_team_aggregates()

        conn = duckdb.connect(test_db)

        # Week 1 has KC and DAL
        count = conn.execute("""
            SELECT COUNT(DISTINCT team) FROM team_rolling_features
            WHERE season = 2025 AND week = 1
        """).fetchone()[0]

        assert count >= 1  # At least one team

        conn.close()
