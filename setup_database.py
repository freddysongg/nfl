#!/usr/bin/env python3
"""
Database setup script for NFL Prediction System.
Creates DuckDB database with comprehensive schema from DATA_SETUP.md
"""

import duckdb
import os
from pathlib import Path
from src.table_schemas import (
    create_all_raw_tables,
    create_indexes_for_raw_tables
)


class NFLDatabaseSetup:
    """Setup and initialize DuckDB database with complete schema"""

    def __init__(self, db_file="nfl_predictions.duckdb"):
        self.db_file = db_file
        self.conn = None

    def connect(self):
        """Connect to DuckDB and enable Polars integration"""
        print(f"Connecting to database: {self.db_file}")
        self.conn = duckdb.connect(self.db_file)

        try:
            self.conn.execute("INSTALL polars")
            self.conn.execute("LOAD polars")
            print("✅ Polars integration enabled")
        except Exception as e:
            print(f"⚠️  Polars integration not available: {e}")

    def create_raw_data_tables(self):
        """Create all raw data storage tables using updated 2025 schemas"""
        print("\n📊 Creating raw data tables...")

        create_all_raw_tables(self.conn)

    def create_player_lifecycle_tables(self):
        """Create player lifecycle management tables"""
        print("\n👥 Creating player lifecycle tables...")

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS player_lifecycle (
                player_id VARCHAR PRIMARY KEY,
                gsis_id VARCHAR,
                espn_id VARCHAR,
                first_nfl_season INTEGER,
                last_nfl_season INTEGER,
                career_teams VARCHAR[],
                primary_position VARCHAR(5),
                draft_year INTEGER,
                draft_round INTEGER,
                draft_pick INTEGER,
                college VARCHAR,
                retirement_status VARCHAR(20) DEFAULT 'active',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        print("✅ player_lifecycle table created")

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS player_experience_classification (
                player_id VARCHAR,
                season INTEGER,
                experience_category VARCHAR(20),
                seasons_played INTEGER,
                prediction_strategy VARCHAR(50),
                confidence_multiplier FLOAT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (player_id, season)
            )
        """)
        print("✅ player_experience_classification table created")

    def create_roster_management_tables(self):
        """Create roster snapshot and management tables"""
        print("\n📋 Creating roster management tables...")

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS team_roster_snapshots (
                snapshot_id VARCHAR PRIMARY KEY,
                team VARCHAR(3),
                season INTEGER,
                week INTEGER,
                snapshot_date DATE,
                active_players JSON,
                depth_chart JSON,
                key_changes JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        print("✅ team_roster_snapshots table created")

    def create_feature_engineering_tables(self):
        """Create feature engineering and ML training tables"""
        print("\n⚙️ Creating feature engineering tables...")

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS player_rolling_features (
                player_id VARCHAR,
                season INTEGER,
                week INTEGER,
                position VARCHAR(5),

                -- Rolling statistics
                stats_last3_games JSON,
                stats_last5_games JSON,
                stats_season_avg JSON,

                -- Trend analysis
                performance_trend FLOAT,
                usage_trend FLOAT,
                target_share_trend FLOAT,

                -- Matchup context
                vs_opponent_history JSON,
                opp_rank_vs_position INTEGER,
                opp_avg_allowed_to_position JSON,

                -- Situational context
                home_away_splits JSON,
                divisional_game BOOLEAN,
                rest_days INTEGER,

                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (player_id, season, week)
            )
        """)
        print("✅ player_rolling_features table created")

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS team_rolling_features (
                team VARCHAR(3),
                season INTEGER,
                week INTEGER,

                -- Offensive metrics
                off_epa_per_play_last3 FLOAT,
                off_success_rate_last3 FLOAT,
                off_explosive_play_rate FLOAT,
                off_red_zone_efficiency FLOAT,
                off_third_down_conv FLOAT,

                -- Defensive metrics
                def_epa_per_play_last3 FLOAT,
                def_success_rate_last3 FLOAT,
                def_pressure_rate FLOAT,
                def_turnover_rate FLOAT,

                -- Situational tendencies
                pass_rate_neutral FLOAT,
                pace_of_play FLOAT,
                time_of_possession_avg FLOAT,

                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (team, season, week)
            )
        """)
        print("✅ team_rolling_features table created")

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS ml_training_features (
                feature_id VARCHAR PRIMARY KEY,
                entity_type VARCHAR(10),
                entity_id VARCHAR,
                prediction_target VARCHAR(20),

                -- Time context
                season INTEGER,
                week INTEGER,
                game_date DATE,

                -- Roster context
                roster_snapshot_id VARCHAR,
                player_experience_level VARCHAR(20),

                -- Features
                numerical_features FLOAT[],
                feature_names VARCHAR[],
                categorical_features JSON,

                -- Targets
                actual_outcomes JSON,

                -- Data quality
                data_quality_score FLOAT,
                missing_data_flags VARCHAR[],

                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        print("✅ ml_training_features table created")

    def create_indexes(self):
        """Create performance indexes"""
        print("\n🚀 Creating performance indexes...")

        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_player_stats_player_season_week ON raw_player_stats(player_id, season, week)",
            "CREATE INDEX IF NOT EXISTS idx_player_stats_team_season_week ON raw_player_stats(team, season, week)",
            "CREATE INDEX IF NOT EXISTS idx_player_stats_position ON raw_player_stats(position)",
            "CREATE INDEX IF NOT EXISTS idx_team_stats_team_season_week ON raw_team_stats(team, season, week)",
            "CREATE INDEX IF NOT EXISTS idx_depth_charts_team_season_week ON raw_depth_charts(club_code, season, week)",
            "CREATE INDEX IF NOT EXISTS idx_rosters_weekly_team_season_week ON raw_rosters_weekly(team, season, week)",
            "CREATE INDEX IF NOT EXISTS idx_schedules_season_week ON raw_schedules(season, week)",
            "CREATE INDEX IF NOT EXISTS idx_nextgen_passing_player_season_week ON raw_nextgen_passing(player_gsis_id, season, week)",

            "CREATE INDEX IF NOT EXISTS idx_ml_season_week ON ml_training_features(season, week)",
            "CREATE INDEX IF NOT EXISTS idx_ml_entity ON ml_training_features(entity_type, entity_id)",
            "CREATE INDEX IF NOT EXISTS idx_ml_target ON ml_training_features(prediction_target)",
            "CREATE INDEX IF NOT EXISTS idx_ml_quality ON ml_training_features(data_quality_score)",

            "CREATE INDEX IF NOT EXISTS idx_player_features_player_season ON player_rolling_features(player_id, season)",
            "CREATE INDEX IF NOT EXISTS idx_team_features_team_season ON team_rolling_features(team, season)",
        ]

        for idx_sql in indexes:
            self.conn.execute(idx_sql)

        print(f"✅ Created {len(indexes)} performance indexes")

    def create_views(self):
        """Create useful views for common queries"""
        print("\n👁️  Creating database views...")

        self.conn.execute("""
            CREATE VIEW IF NOT EXISTS current_season_players AS
            SELECT DISTINCT
                player_id,
                player_name,
                team,
                position,
                MAX(week) as latest_week
            FROM raw_player_stats
            WHERE season = (SELECT MAX(season) FROM raw_player_stats)
            GROUP BY player_id, player_name, team, position
        """)
        print("✅ current_season_players view created")

        self.conn.execute("""
            CREATE VIEW IF NOT EXISTS latest_team_rosters AS
            SELECT
                team,
                position,
                COUNT(*) as player_count,
                STRING_AGG(player_name, ', ') as players
            FROM current_season_players
            GROUP BY team, position
            ORDER BY team, position
        """)
        print("✅ latest_team_rosters view created")

        self.conn.execute("""
            CREATE VIEW IF NOT EXISTS season_summary AS
            SELECT
                season,
                COUNT(DISTINCT player_id) as total_players,
                COUNT(DISTINCT team) as total_teams,
                MAX(week) as max_week,
                COUNT(*) as total_records
            FROM raw_player_stats
            GROUP BY season
            ORDER BY season DESC
        """)
        print("✅ season_summary view created")

    def setup_database(self):
        """Complete database setup process"""
        print("🏗️  Setting up NFL Predictions Database")
        print("=" * 50)

        self.connect()

        self.create_raw_data_tables()
        self.create_player_lifecycle_tables()
        self.create_roster_management_tables()
        self.create_feature_engineering_tables()

        create_indexes_for_raw_tables(self.conn)
        self.create_indexes()
        self.create_views()

        print("\n🎉 Database setup complete!")
        print(f"📁 Database file: {self.db_file}")
        print(f"📊 File size: {os.path.getsize(self.db_file) / 1024:.1f} KB")

        return self.conn

    def show_database_info(self):
        """Display database information"""
        print("\n📋 Database Information")
        print("=" * 30)

        tables = self.conn.execute("SHOW TABLES").fetchall()
        print(f"📊 Tables: {len(tables)}")
        for table in tables:
            print(f"  - {table[0]}")

        views = self.conn.execute("SELECT table_name FROM information_schema.tables WHERE table_type = 'VIEW'").fetchall()
        print(f"\n👁️  Views: {len(views)}")
        for view in views:
            print(f"  - {view[0]}")

        if os.path.exists(self.db_file):
            size_mb = os.path.getsize(self.db_file) / (1024 * 1024)
            print(f"\n💾 Database file: {self.db_file}")
            print(f"📏 Size: {size_mb:.2f} MB")

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            print("🔒 Database connection closed")


def main():
    """Main setup function"""
    db_setup = NFLDatabaseSetup()

    try:
        conn = db_setup.setup_database()

        db_setup.show_database_info()
        
        print("\n🧪 Testing database functionality...")
        result = conn.execute("SELECT 1 as test").fetchone()
        print(f"✅ Database test successful: {result[0]}")

    except Exception as e:
        print(f"❌ Database setup failed: {e}")
        raise
    finally:
        db_setup.close()


if __name__ == "__main__":
    main()