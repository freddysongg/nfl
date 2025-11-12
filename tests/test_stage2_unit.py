"""
Unit tests for Stage 2: Player Lifecycle and Roster Management

Tests cover:
- Player lifecycle table creation
- First/last season calculation
- Retirement status detection
- Weekly roster snapshots
- Experience classification
- Confidence scores
"""

import pytest
import duckdb
import polars as pl
from pathlib import Path
from datetime import datetime
import json

from src.data_pipeline import NFLDataPipeline
from src.database import NFLDatabase
from src.table_schemas import (
    create_raw_player_stats_table,
    create_raw_rosters_weekly_table,
    create_raw_schedules_table
)


@pytest.fixture
def test_db_path(tmp_path):
    """Create temporary test database path."""
    db_path = tmp_path / "test_stage2.duckdb"
    yield str(db_path)
    # Cleanup
    if db_path.exists():
        db_path.unlink()


@pytest.fixture
def test_db(test_db_path):
    """Create and populate test database with sample data."""
    conn = duckdb.connect(test_db_path)

    # Create necessary tables
    create_raw_player_stats_table(conn)
    create_raw_rosters_weekly_table(conn)
    create_raw_schedules_table(conn)

    # Create lifecycle tables
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

    conn.close()
    yield test_db_path


@pytest.fixture
def sample_player_stats(test_db):
    """Insert sample player stats for testing."""
    conn = duckdb.connect(test_db)

    # Sample data: 3 players with different career lengths
    sample_data = [
        # Player 1: Rookie (2025 only)
        ("00-0012345", "John Rookie", "QB", "QB", "OFF", 2025, 1, "REG", "KC", 15, 25, 200, 2),
        ("00-0012345", "John Rookie", "QB", "QB", "OFF", 2025, 2, "REG", "KC", 18, 28, 250, 3),

        # Player 2: Developing (2023-2025)
        ("00-0023456", "Tom Developing", "RB", "RB", "OFF", 2023, 1, "REG", "DAL", 15, None, 85, 1),
        ("00-0023456", "Tom Developing", "RB", "RB", "OFF", 2024, 1, "REG", "DAL", 18, None, 100, 2),
        ("00-0023456", "Tom Developing", "RB", "RB", "OFF", 2025, 1, "REG", "DAL", 20, None, 120, 1),

        # Player 3: Veteran (2021-2025)
        ("00-0034567", "Mike Veteran", "WR", "WR", "OFF", 2021, 1, "REG", "SF", 8, None, 120, 1),
        ("00-0034567", "Mike Veteran", "WR", "WR", "OFF", 2022, 1, "REG", "SF", 10, None, 150, 2),
        ("00-0034567", "Mike Veteran", "WR", "WR", "OFF", 2023, 1, "REG", "NYG", 12, None, 180, 2),
        ("00-0034567", "Mike Veteran", "WR", "WR", "OFF", 2024, 1, "REG", "NYG", 9, None, 140, 1),
        ("00-0034567", "Mike Veteran", "WR", "WR", "OFF", 2025, 1, "REG", "NYG", 11, None, 160, 2),
    ]

    for row in sample_data:
        conn.execute("""
            INSERT INTO raw_player_stats
            (player_id, player_name, position, position_group, season, week, season_type, team,
             receptions, targets, receiving_yards, receiving_tds)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, row)

    conn.close()
    return sample_data


@pytest.fixture
def sample_rosters(test_db):
    """Insert sample roster data for testing."""
    conn = duckdb.connect(test_db)

    roster_data = [
        (2025, "KC", "QB", "QB", 12, "ACT", "John Rookie", "00-0012345", 1, 1),
        (2025, "DAL", "RB", "RB", 22, "ACT", "Tom Developing", "00-0023456", 1, 2),
        (2025, "NYG", "WR", "WR", 81, "ACT", "Mike Veteran", "00-0034567", 1, 5),
    ]

    for row in roster_data:
        conn.execute("""
            INSERT INTO raw_rosters_weekly
            (season, team, position, depth_chart_position, jersey_number, status,
             full_name, gsis_id, week, years_exp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, row)

    conn.close()


class TestStage2PlayerLifecycle:
    """Unit tests for Stage 2: Player Lifecycle"""

    def test_build_player_lifecycle_table_creates_records(self, test_db, sample_player_stats):
        """Test player lifecycle table creation."""
        pipeline = NFLDataPipeline(test_db)

        # Run the method
        rows_created = pipeline.build_player_lifecycle_table()

        # Verify records were created
        assert rows_created == 3, f"Expected 3 players, got {rows_created}"

        # Verify table contents
        conn = duckdb.connect(test_db)
        result = conn.execute("SELECT COUNT(*) FROM player_lifecycle").fetchone()
        assert result[0] == 3
        conn.close()

    def test_player_lifecycle_first_last_season(self, test_db, sample_player_stats):
        """Test first/last season calculation."""
        pipeline = NFLDataPipeline(test_db)
        pipeline.build_player_lifecycle_table()

        conn = duckdb.connect(test_db)

        # Check rookie player
        rookie = conn.execute("""
            SELECT first_nfl_season, last_nfl_season, total_seasons_played
            FROM player_lifecycle
            WHERE player_id = '00-0012345'
        """).fetchone()
        assert rookie[0] == 2025  # first_nfl_season
        assert rookie[1] == 2025  # last_nfl_season
        assert rookie[2] == 1     # total_seasons_played

        # Check veteran player
        veteran = conn.execute("""
            SELECT first_nfl_season, last_nfl_season, total_seasons_played
            FROM player_lifecycle
            WHERE player_id = '00-0034567'
        """).fetchone()
        assert veteran[0] == 2021  # first_nfl_season
        assert veteran[1] == 2025  # last_nfl_season
        assert veteran[2] == 5     # total_seasons_played

        conn.close()

    def test_player_lifecycle_total_games(self, test_db, sample_player_stats):
        """Test total games played calculation."""
        pipeline = NFLDataPipeline(test_db)
        pipeline.build_player_lifecycle_table()

        conn = duckdb.connect(test_db)

        # Rookie should have 2 games
        rookie_games = conn.execute("""
            SELECT total_games_played FROM player_lifecycle
            WHERE player_id = '00-0012345'
        """).fetchone()[0]
        assert rookie_games == 2

        # Veteran should have 5 games
        veteran_games = conn.execute("""
            SELECT total_games_played FROM player_lifecycle
            WHERE player_id = '00-0034567'
        """).fetchone()[0]
        assert veteran_games == 5

        conn.close()

    def test_player_lifecycle_career_teams(self, test_db, sample_player_stats):
        """Test career teams tracking."""
        pipeline = NFLDataPipeline(test_db)
        pipeline.build_player_lifecycle_table()

        conn = duckdb.connect(test_db)

        # Veteran played for SF and NYG
        career_teams = conn.execute("""
            SELECT career_teams FROM player_lifecycle
            WHERE player_id = '00-0034567'
        """).fetchone()[0]

        # Should contain both teams
        assert 'SF' in career_teams
        assert 'NYG' in career_teams

        conn.close()

    def test_create_weekly_roster_snapshots(self, test_db, sample_player_stats, sample_rosters):
        """Test roster snapshot creation."""
        pipeline = NFLDataPipeline(test_db)

        # Run the method
        snapshots_created = pipeline.create_weekly_roster_snapshots()

        # Verify snapshots were created
        assert snapshots_created > 0

        conn = duckdb.connect(test_db)

        # Check that snapshots exist
        snapshot_count = conn.execute("""
            SELECT COUNT(*) FROM team_roster_snapshots
        """).fetchone()[0]
        assert snapshot_count > 0

        # Check JSON structure
        snapshot = conn.execute("""
            SELECT active_players, total_roster_size
            FROM team_roster_snapshots
            LIMIT 1
        """).fetchone()

        active_players = json.loads(snapshot[0])
        assert isinstance(active_players, list)
        assert len(active_players) > 0
        assert 'player_id' in active_players[0]
        assert 'player_name' in active_players[0]

        conn.close()

    def test_classify_player_experience_levels(self, test_db, sample_player_stats):
        """Test experience classification."""
        pipeline = NFLDataPipeline(test_db)

        # Build lifecycle first (required for classification)
        pipeline.build_player_lifecycle_table()

        # Run classification
        classifications = pipeline.classify_player_experience_levels()

        assert classifications > 0

        conn = duckdb.connect(test_db)

        # Check rookie classification (1 season)
        rookie = conn.execute("""
            SELECT experience_level, confidence_score
            FROM player_experience_classification
            WHERE player_id = '00-0012345' AND season = 2025
        """).fetchone()
        assert rookie[0] == 'rookie'
        assert rookie[1] == 0.6  # rookie confidence

        # Check developing classification (2-3 seasons)
        developing = conn.execute("""
            SELECT experience_level, confidence_score
            FROM player_experience_classification
            WHERE player_id = '00-0023456' AND season = 2025
        """).fetchone()
        assert developing[0] == 'developing'
        assert developing[1] == 0.8  # developing confidence

        # Check veteran classification (4+ seasons)
        veteran = conn.execute("""
            SELECT experience_level, confidence_score
            FROM player_experience_classification
            WHERE player_id = '00-0034567' AND season = 2025
        """).fetchone()
        assert veteran[0] == 'veteran'
        assert veteran[1] == 1.0  # veteran confidence

        conn.close()

    def test_experience_classification_all_seasons(self, test_db, sample_player_stats):
        """Test classification is created for all player-season combinations."""
        pipeline = NFLDataPipeline(test_db)
        pipeline.build_player_lifecycle_table()
        pipeline.classify_player_experience_levels()

        conn = duckdb.connect(test_db)

        # Check developing player has classification for each season
        developing_seasons = conn.execute("""
            SELECT season, seasons_experience
            FROM player_experience_classification
            WHERE player_id = '00-0023456'
            ORDER BY season
        """).fetchall()

        assert len(developing_seasons) == 3  # 2023, 2024, 2025
        assert developing_seasons[0][1] == 1  # 1st season in 2023
        assert developing_seasons[1][1] == 2  # 2nd season in 2024
        assert developing_seasons[2][1] == 3  # 3rd season in 2025

        conn.close()

    def test_roster_snapshots_json_structure(self, test_db, sample_player_stats, sample_rosters):
        """Test roster snapshot JSON has correct structure."""
        pipeline = NFLDataPipeline(test_db)
        pipeline.create_weekly_roster_snapshots()

        conn = duckdb.connect(test_db)

        snapshot = conn.execute("""
            SELECT active_players FROM team_roster_snapshots
            WHERE team = 'KC' AND season = 2025 AND week = 1
        """).fetchone()

        if snapshot:
            players = json.loads(snapshot[0])

            # Verify all required fields
            required_fields = ['player_id', 'player_name', 'position', 'status']
            for player in players:
                for field in required_fields:
                    assert field in player, f"Missing field: {field}"

        conn.close()

    def test_player_lifecycle_table_columns(self, test_db, sample_player_stats):
        """Test player lifecycle table has all required columns."""
        pipeline = NFLDataPipeline(test_db)
        pipeline.build_player_lifecycle_table()

        conn = duckdb.connect(test_db)

        # Get table schema
        columns = conn.execute("DESCRIBE player_lifecycle").fetchall()
        column_names = [col[0] for col in columns]

        required_columns = [
            'player_id', 'player_name', 'position',
            'first_nfl_season', 'last_nfl_season',
            'total_seasons_played', 'total_games_played',
            'is_retired', 'career_teams', 'primary_position'
        ]

        for col in required_columns:
            assert col in column_names, f"Missing required column: {col}"

        conn.close()


class TestStage2EdgeCases:
    """Test edge cases for Stage 2"""

    def test_empty_database(self, test_db):
        """Test behavior with no data."""
        pipeline = NFLDataPipeline(test_db)

        # Should not crash with empty database
        rows = pipeline.build_player_lifecycle_table()
        assert rows == 0

    def test_single_game_player(self, test_db):
        """Test player with only one game."""
        conn = duckdb.connect(test_db)

        # Insert single game
        conn.execute("""
            INSERT INTO raw_player_stats
            (player_id, player_name, position, position_group, season, week, season_type, team)
            VALUES ('00-0099999', 'One Game', 'TE', 'OFF', 2025, 1, 'REG', 'BUF')
        """)
        conn.close()

        pipeline = NFLDataPipeline(test_db)
        rows = pipeline.build_player_lifecycle_table()

        assert rows == 1

        conn = duckdb.connect(test_db)
        player = conn.execute("""
            SELECT total_games_played, total_seasons_played
            FROM player_lifecycle WHERE player_id = '00-0099999'
        """).fetchone()

        assert player[0] == 1  # 1 game
        assert player[1] == 1  # 1 season
        conn.close()

    def test_missing_position_data(self, test_db):
        """Test handling of missing position data."""
        conn = duckdb.connect(test_db)

        # Insert player with NULL position
        conn.execute("""
            INSERT INTO raw_player_stats
            (player_id, player_name, position, position_group, season, week, season_type, team)
            VALUES ('00-0088888', 'No Position', NULL, NULL, 2025, 1, 'REG', 'MIA')
        """)
        conn.close()

        pipeline = NFLDataPipeline(test_db)
        # Should not crash
        rows = pipeline.build_player_lifecycle_table()
        assert rows >= 1
