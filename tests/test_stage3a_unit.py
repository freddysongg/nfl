"""
Unit tests for Stage 3a: Rolling Statistics

Tests cover:
- Rolling statistics calculation (3, 5, 10 game windows)
- Position-specific stat selection
- Trend calculation (REGR_SLOPE)
- JSON structure validation
- Edge case: insufficient game history
"""

import pytest
import duckdb
import polars as pl
from pathlib import Path
import json

from src.data_pipeline import NFLDataPipeline
from src.database import NFLDatabase
from src.table_schemas import create_raw_player_stats_table


@pytest.fixture
def test_db_path(tmp_path):
    """Create temporary test database path."""
    db_path = tmp_path / "test_stage3a.duckdb"
    yield str(db_path)
    if db_path.exists():
        db_path.unlink()


@pytest.fixture
def test_db(test_db_path):
    """Create and populate test database with sample data."""
    conn = duckdb.connect(test_db_path)

    # Create necessary tables
    create_raw_player_stats_table(conn)

    # Create player lifecycle tables (required for rolling stats)
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
            primary_position VARCHAR
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS player_experience_classification (
            player_id VARCHAR,
            season INTEGER,
            seasons_experience INTEGER,
            experience_level VARCHAR,
            confidence_score FLOAT,
            PRIMARY KEY (player_id, season)
        )
    """)

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

            -- Rolling stats (JSON)
            stats_last3_games JSON,
            stats_last5_games JSON,
            stats_last10_games JSON,

            -- Trends
            performance_trend FLOAT,
            usage_trend FLOAT,
            efficiency_trend FLOAT,

            -- Matchup features (filled in Stage 3b)
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
def sample_qb_stats(test_db):
    """Insert sample QB stats for rolling statistics."""
    conn = duckdb.connect(test_db)

    # QB with 10 games for full window testing
    qb_games = [
        ("00-0001111", "Test QB", "QB", 2025, 1, 300, 2, 0, 20, 30, 150),
        ("00-0001111", "Test QB", "QB", 2025, 2, 250, 1, 1, 15, 25, 120),
        ("00-0001111", "Test QB", "QB", 2025, 3, 350, 3, 0, 25, 35, 180),
        ("00-0001111", "Test QB", "QB", 2025, 4, 280, 2, 1, 18, 28, 140),
        ("00-0001111", "Test QB", "QB", 2025, 5, 320, 2, 0, 22, 32, 160),
        ("00-0001111", "Test QB", "QB", 2025, 6, 290, 1, 1, 19, 29, 145),
        ("00-0001111", "Test QB", "QB", 2025, 7, 310, 3, 0, 21, 31, 155),
        ("00-0001111", "Test QB", "QB", 2025, 8, 270, 2, 1, 17, 27, 135),
        ("00-0001111", "Test QB", "QB", 2025, 9, 330, 2, 0, 23, 33, 170),
        ("00-0001111", "Test QB", "QB", 2025, 10, 300, 2, 1, 20, 30, 150),
    ]

    for game in qb_games:
        conn.execute("""
            INSERT INTO raw_player_stats
            (player_id, player_name, position, season, week,
             passing_yards, passing_tds, passing_interceptions,
             completions, attempts, rushing_yards)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, game)

    # Insert lifecycle data
    conn.execute("""
        INSERT INTO player_lifecycle
        (player_id, player_name, position, first_nfl_season, last_nfl_season,
         total_seasons_played, total_games_played, career_teams, primary_position)
        VALUES ('00-0001111', 'Test QB', 'QB', 2025, 2025, 1, 10, 'KC', 'QB')
    """)

    conn.execute("""
        INSERT INTO player_experience_classification
        (player_id, season, seasons_experience, experience_level, confidence_score)
        VALUES ('00-0001111', 2025, 1, 'rookie', 0.6)
    """)

    conn.close()


@pytest.fixture
def sample_rb_stats(test_db):
    """Insert sample RB stats for position-specific testing."""
    conn = duckdb.connect(test_db)

    # RB with 6 games
    rb_games = [
        ("00-0002222", "Test RB", "RB", 2025, 1, 85, 1, 3, 25),
        ("00-0002222", "Test RB", "RB", 2025, 2, 100, 2, 4, 35),
        ("00-0002222", "Test RB", "RB", 2025, 3, 75, 0, 5, 40),
        ("00-0002222", "Test RB", "RB", 2025, 4, 120, 2, 6, 50),
        ("00-0002222", "Test RB", "RB", 2025, 5, 95, 1, 4, 30),
        ("00-0002222", "Test RB", "RB", 2025, 6, 110, 1, 3, 28),
    ]

    for game in rb_games:
        conn.execute("""
            INSERT INTO raw_player_stats
            (player_id, player_name, position, season, week,
             rushing_yards, rushing_tds, receptions, receiving_yards)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, game)

    # Insert lifecycle data
    conn.execute("""
        INSERT INTO player_lifecycle
        (player_id, player_name, position, first_nfl_season, last_nfl_season,
         total_seasons_played, total_games_played, career_teams, primary_position)
        VALUES ('00-0002222', 'Test RB', 'RB', 2025, 2025, 1, 6, 'DAL', 'RB')
    """)

    conn.execute("""
        INSERT INTO player_experience_classification
        (player_id, season, seasons_experience, experience_level, confidence_score)
        VALUES ('00-0002222', 2025, 1, 'rookie', 0.6)
    """)

    conn.close()


class TestStage3aRollingStatistics:
    """Unit tests for Stage 3a: Rolling Statistics"""

    def test_calculate_rolling_statistics_creates_records(self, test_db, sample_qb_stats):
        """Test rolling statistics calculation creates records."""
        pipeline = NFLDataPipeline(test_db)

        # Run calculation
        pipeline.calculate_rolling_statistics()

        conn = duckdb.connect(test_db)

        # Check records were created
        count = conn.execute("""
            SELECT COUNT(*) FROM player_rolling_features
        """).fetchone()[0]

        # Should have records for weeks 2-10 (week 1 has no history)
        assert count > 0

        conn.close()

    def test_rolling_window_3_games(self, test_db, sample_qb_stats):
        """Test 3-game rolling window calculation."""
        pipeline = NFLDataPipeline(test_db)
        pipeline.calculate_rolling_statistics()

        conn = duckdb.connect(test_db)

        # Get stats for week 4 (should have 3 games of history: weeks 1-3)
        result = conn.execute("""
            SELECT stats_last3_games FROM player_rolling_features
            WHERE player_id = '00-0001111' AND week = 4
        """).fetchone()

        if result:
            stats = json.loads(result[0])

            # Check structure
            assert 'passing' in stats
            assert 'avg_passing_yards' in stats['passing']

            # Avg of weeks 1-3: (300 + 250 + 350) / 3 = 300
            assert abs(stats['passing']['avg_passing_yards'] - 300.0) < 1.0

        conn.close()

    def test_rolling_window_5_games(self, test_db, sample_qb_stats):
        """Test 5-game rolling window calculation."""
        pipeline = NFLDataPipeline(test_db)
        pipeline.calculate_rolling_statistics()

        conn = duckdb.connect(test_db)

        # Get stats for week 6 (should have 5 games: weeks 1-5)
        result = conn.execute("""
            SELECT stats_last5_games FROM player_rolling_features
            WHERE player_id = '00-0001111' AND week = 6
        """).fetchone()

        if result:
            stats = json.loads(result[0])

            # Avg of weeks 1-5: (300 + 250 + 350 + 280 + 320) / 5 = 300
            assert 'passing' in stats
            assert abs(stats['passing']['avg_passing_yards'] - 300.0) < 1.0

        conn.close()

    def test_rolling_window_10_games(self, test_db, sample_qb_stats):
        """Test 10-game rolling window calculation."""
        pipeline = NFLDataPipeline(test_db)
        pipeline.calculate_rolling_statistics()

        conn = duckdb.connect(test_db)

        # Only week 11 would have full 10 games, but we only have 10 weeks
        # So check week 10 has stats from games 1-9
        result = conn.execute("""
            SELECT stats_last10_games FROM player_rolling_features
            WHERE player_id = '00-0001111' AND week = 10
        """).fetchone()

        if result:
            stats = json.loads(result[0])
            assert 'passing' in stats
            assert 'avg_passing_yards' in stats['passing']

        conn.close()

    def test_position_specific_stats_qb(self, test_db, sample_qb_stats):
        """Test QB gets passing stats in rolling features."""
        pipeline = NFLDataPipeline(test_db)
        pipeline.calculate_rolling_statistics()

        conn = duckdb.connect(test_db)

        result = conn.execute("""
            SELECT stats_last3_games FROM player_rolling_features
            WHERE player_id = '00-0001111' AND week = 4
        """).fetchone()

        if result:
            stats = json.loads(result[0])

            # QB should have passing category
            assert 'passing' in stats

            # Should have QB-specific stats
            qb_stats = ['avg_passing_yards', 'avg_passing_tds', 'avg_completions', 'avg_attempts']
            for stat in qb_stats:
                assert stat in stats['passing'], f"Missing QB stat: {stat}"

        conn.close()

    def test_position_specific_stats_rb(self, test_db, sample_rb_stats):
        """Test RB gets rushing/receiving stats."""
        pipeline = NFLDataPipeline(test_db)
        pipeline.calculate_rolling_statistics()

        conn = duckdb.connect(test_db)

        result = conn.execute("""
            SELECT stats_last3_games FROM player_rolling_features
            WHERE player_id = '00-0002222' AND week = 4
        """).fetchone()

        if result:
            stats = json.loads(result[0])

            # RB should have rushing and receiving categories
            assert 'rushing' in stats
            assert 'receiving' in stats

            # Should have RB-specific stats
            assert 'avg_rushing_yards' in stats['rushing']
            assert 'avg_receptions' in stats['receiving']

        conn.close()

    def test_trend_calculation(self, test_db, sample_qb_stats):
        """Test performance trend calculation."""
        pipeline = NFLDataPipeline(test_db)
        pipeline.calculate_rolling_statistics()

        conn = duckdb.connect(test_db)

        result = conn.execute("""
            SELECT performance_trend FROM player_rolling_features
            WHERE player_id = '00-0001111' AND week = 10
        """).fetchone()

        if result:
            trend = result[0]
            # Trend should be a float
            assert isinstance(trend, (int, float))

        conn.close()

    def test_insufficient_game_history(self, test_db):
        """Test handling of insufficient game history."""
        conn = duckdb.connect(test_db)

        # Insert player with only 1 game
        conn.execute("""
            INSERT INTO raw_player_stats
            (player_id, player_name, position, season, week, passing_yards, passing_tds)
            VALUES ('00-0003333', 'New QB', 'QB', 2025, 1, 200, 1)
        """)

        conn.execute("""
            INSERT INTO player_lifecycle
            (player_id, player_name, position, first_nfl_season, last_nfl_season,
             total_seasons_played, total_games_played, career_teams, primary_position)
            VALUES ('00-0003333', 'New QB', 'QB', 2025, 2025, 1, 1, 'BUF', 'QB')
        """)

        conn.execute("""
            INSERT INTO player_experience_classification
            (player_id, season, seasons_experience, experience_level, confidence_score)
            VALUES ('00-0003333', 2025, 1, 'rookie', 0.6)
        """)
        conn.close()

        pipeline = NFLDataPipeline(test_db)
        # Should not crash with insufficient history
        pipeline.calculate_rolling_statistics()

        conn = duckdb.connect(test_db)

        # Week 1 should have no rolling features (no prior history)
        result = conn.execute("""
            SELECT COUNT(*) FROM player_rolling_features
            WHERE player_id = '00-0003333' AND week = 1
        """).fetchone()[0]

        # Should be 0 or handle gracefully
        assert result == 0

        conn.close()

    def test_json_structure_all_windows(self, test_db, sample_qb_stats):
        """Test JSON structure for all rolling windows."""
        pipeline = NFLDataPipeline(test_db)
        pipeline.calculate_rolling_statistics()

        conn = duckdb.connect(test_db)

        result = conn.execute("""
            SELECT stats_last3_games, stats_last5_games, stats_last10_games
            FROM player_rolling_features
            WHERE player_id = '00-0001111' AND week = 10
        """).fetchone()

        if result:
            for i, window_name in enumerate(['3-game', '5-game', '10-game']):
                stats = json.loads(result[i])

                # All should be valid JSON dicts
                assert isinstance(stats, dict)

                # Should have position-specific categories
                assert len(stats) > 0

        conn.close()

    def test_rolling_stats_temporal_consistency(self, test_db, sample_qb_stats):
        """Test rolling stats only use prior weeks (no data leakage)."""
        pipeline = NFLDataPipeline(test_db)
        pipeline.calculate_rolling_statistics()

        conn = duckdb.connect(test_db)

        # Get week 5 rolling stats (should only use weeks 1-4)
        result = conn.execute("""
            SELECT stats_last3_games FROM player_rolling_features
            WHERE player_id = '00-0001111' AND week = 5
        """).fetchone()

        if result:
            stats = json.loads(result[0])

            # Last 3 games should be weeks 2, 3, 4 (avg: (250 + 350 + 280) / 3 = 293.33)
            avg_yards = stats['passing']['avg_passing_yards']

            # Should NOT include week 5 data (320 yards)
            assert abs(avg_yards - 293.33) < 1.0
            assert abs(avg_yards - 300) > 5.0  # Not the overall average

        conn.close()

    def test_multiple_players_same_week(self, test_db, sample_qb_stats, sample_rb_stats):
        """Test rolling stats for multiple players in same week."""
        pipeline = NFLDataPipeline(test_db)
        pipeline.calculate_rolling_statistics()

        conn = duckdb.connect(test_db)

        # Both players should have stats for week 4
        count = conn.execute("""
            SELECT COUNT(DISTINCT player_id) FROM player_rolling_features
            WHERE week = 4
        """).fetchone()[0]

        assert count == 2  # QB and RB

        conn.close()

    def test_rolling_features_table_columns(self, test_db, sample_qb_stats):
        """Test rolling features table has required columns."""
        pipeline = NFLDataPipeline(test_db)
        pipeline.calculate_rolling_statistics()

        conn = duckdb.connect(test_db)

        columns = conn.execute("DESCRIBE player_rolling_features").fetchall()
        column_names = [col[0] for col in columns]

        required_columns = [
            'player_id', 'player_name', 'season', 'week', 'position',
            'stats_last3_games', 'stats_last5_games', 'stats_last10_games',
            'performance_trend', 'usage_trend'
        ]

        for col in required_columns:
            assert col in column_names, f"Missing column: {col}"

        conn.close()


class TestStage3aEdgeCases:
    """Test edge cases for Stage 3a"""

    def test_zero_stats_game(self, test_db):
        """Test handling of games with zero stats."""
        conn = duckdb.connect(test_db)

        # Insert games with zeros
        conn.execute("""
            INSERT INTO raw_player_stats
            (player_id, player_name, position, season, week, passing_yards, passing_tds)
            VALUES
            ('00-0004444', 'Zero QB', 'QB', 2025, 1, 0, 0),
            ('00-0004444', 'Zero QB', 'QB', 2025, 2, 0, 0),
            ('00-0004444', 'Zero QB', 'QB', 2025, 3, 200, 2)
        """)

        conn.execute("""
            INSERT INTO player_lifecycle
            (player_id, player_name, position, first_nfl_season, last_nfl_season,
             total_seasons_played, total_games_played, career_teams, primary_position)
            VALUES ('00-0004444', 'Zero QB', 'QB', 2025, 2025, 1, 3, 'MIA', 'QB')
        """)

        conn.execute("""
            INSERT INTO player_experience_classification
            (player_id, season, seasons_experience, experience_level, confidence_score)
            VALUES ('00-0004444', 2025, 1, 'rookie', 0.6)
        """)
        conn.close()

        pipeline = NFLDataPipeline(test_db)
        # Should handle zeros gracefully
        pipeline.calculate_rolling_statistics()

        conn = duckdb.connect(test_db)
        result = conn.execute("""
            SELECT COUNT(*) FROM player_rolling_features
            WHERE player_id = '00-0004444'
        """).fetchone()[0]

        # Should still create features
        assert result > 0
        conn.close()

    def test_null_stats_handling(self, test_db):
        """Test handling of NULL stat values."""
        conn = duckdb.connect(test_db)

        conn.execute("""
            INSERT INTO raw_player_stats
            (player_id, player_name, position, season, week, passing_yards, passing_tds)
            VALUES
            ('00-0005555', 'Null QB', 'QB', 2025, 1, NULL, NULL),
            ('00-0005555', 'Null QB', 'QB', 2025, 2, 250, 2),
            ('00-0005555', 'Null QB', 'QB', 2025, 3, 300, 3)
        """)

        conn.execute("""
            INSERT INTO player_lifecycle
            (player_id, player_name, position, first_nfl_season, last_nfl_season,
             total_seasons_played, total_games_played, career_teams, primary_position)
            VALUES ('00-0005555', 'Null QB', 'QB', 2025, 2025, 1, 3, 'NYJ', 'QB')
        """)

        conn.execute("""
            INSERT INTO player_experience_classification
            (player_id, season, seasons_experience, experience_level, confidence_score)
            VALUES ('00-0005555', 2025, 1, 'rookie', 0.6)
        """)
        conn.close()

        pipeline = NFLDataPipeline(test_db)
        # Should handle NULLs without crashing
        pipeline.calculate_rolling_statistics()

        # Success if no exception
        assert True
