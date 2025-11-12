"""
Database utilities for NFL Prediction System.
Provides connection management and common operations for DuckDB.
"""

import duckdb
import polars as pl
from pathlib import Path
from typing import Optional, List, Dict, Any
import json


class NFLDatabase:
    """Main database connection and utilities class"""

    def __init__(self, db_file: str = "nfl_predictions.duckdb"):
        self.db_file = db_file
        self.conn = None

    def connect(self) -> duckdb.DuckDBPyConnection:
        """Connect to the database"""
        if self.conn is None:
            self.conn = duckdb.connect(self.db_file)
        return self.conn

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            self.conn = None

    def __enter__(self):
        """Context manager entry"""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()

    def execute(self, query: str) -> List[tuple]:
        """Execute query and return results"""
        return self.connect().execute(query).fetchall()

    def execute_one(self, query: str) -> Optional[tuple]:
        """Execute query and return first result"""
        result = self.connect().execute(query).fetchone()
        return result

    def store_dataframe(self, df: pl.DataFrame, table_name: str, if_exists: str = "append") -> int:
        """
        Store a Polars DataFrame in the database

        Args:
            df: Polars DataFrame to store
            table_name: Target table name
            if_exists: 'append', 'replace', or 'fail'

        Returns:
            Number of rows inserted
        """
        conn = self.connect()

        if if_exists == "replace":
            conn.execute(f"DELETE FROM {table_name}")
        elif if_exists == "fail":
            count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
            if count > 0:
                raise ValueError(f"Table {table_name} already contains data")

        pandas_df = df.to_pandas()

        conn.register("temp_df", pandas_df)

 

        # Get table schema to know which columns exist

        try:

            table_columns_result = conn.execute(f"DESCRIBE {table_name}").fetchall()

            table_columns = [col[0] for col in table_columns_result]

        except Exception:

            # If table doesn't exist or can't be described, fallback to SELECT *

            conn.execute(f"INSERT INTO {table_name} SELECT * FROM temp_df")

            return len(df)

 

        # Find common columns between dataframe and table

        df_columns = df.columns

        common_columns = [col for col in df_columns if col in table_columns]

 

        if not common_columns:

            raise ValueError(f"No common columns found between dataframe and table {table_name}")

 

        # Build INSERT statement with explicit column names (order-independent)

        cols_str = ", ".join(common_columns)

        conn.execute(f"INSERT INTO {table_name} ({cols_str}) SELECT {cols_str} FROM temp_df")

        return len(df)

    def get_table_info(self, table_name: str) -> Dict[str, Any]:
        """Get information about a table"""
        conn = self.connect()

        count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]

        columns = conn.execute(f"DESCRIBE {table_name}").fetchall()

        return {
            "table_name": table_name,
            "row_count": count,
            "columns": [{"name": col[0], "type": col[1]} for col in columns],
            "column_count": len(columns)
        }

    def list_tables(self) -> List[str]:
        """List all tables in the database"""
        tables = self.execute("SHOW TABLES")
        return [table[0] for table in tables]

    def get_database_summary(self) -> Dict[str, Any]:
        """Get comprehensive database summary"""
        tables = self.list_tables()
        table_info = {}
        total_rows = 0

        for table in tables:
            info = self.get_table_info(table)
            table_info[table] = info
            total_rows += info["row_count"]

        return {
            "total_tables": len(tables),
            "total_rows": total_rows,
            "tables": table_info
        }


class PlayerStatsManager:
    """Specialized manager for player statistics operations"""

    def __init__(self, db: NFLDatabase):
        self.db = db

    def get_player_stats(self, player_id: str, season: Optional[int] = None,
                        week: Optional[int] = None) -> pl.DataFrame:
        """Get player statistics with optional filters"""
        query = "SELECT * FROM raw_player_stats WHERE player_id = ?"
        params = [player_id]

        if season:
            query += " AND season = ?"
            params.append(season)

        if week:
            query += " AND week = ?"
            params.append(week)

        query += " ORDER BY season DESC, week DESC"

        result = self.db.connect().execute(query, params).fetchdf()
        return pl.from_pandas(result)

    def get_team_roster(self, team: str, season: int, week: Optional[int] = None) -> pl.DataFrame:
        """Get team roster for specific season/week"""
        query = """
            SELECT DISTINCT player_id, player_name, position, team
            FROM raw_player_stats
            WHERE team = ? AND season = ?
        """
        params = [team, season]

        if week:
            query += " AND week = ?"
            params.append(week)

        result = self.db.connect().execute(query, params).fetchdf()
        return pl.from_pandas(result)

    def get_position_players(self, position: str, season: int) -> pl.DataFrame:
        """Get all players at a specific position for a season"""
        query = """
            SELECT DISTINCT player_id, player_name, team, position,
                   COUNT(*) as games_played,
                   AVG(fantasy_points) as avg_fantasy_points
            FROM raw_player_stats
            WHERE position = ? AND season = ?
            GROUP BY player_id, player_name, team, position
            ORDER BY avg_fantasy_points DESC
        """

        result = self.db.connect().execute(query, [position, season]).fetchdf()
        return pl.from_pandas(result)


class FeatureManager:
    """Manager for feature engineering operations"""

    def __init__(self, db: NFLDatabase):
        self.db = db

    def calculate_rolling_stats(self, player_id: str, target_season: int,
                               target_week: int, window: int = 3) -> Dict[str, float]:
        """Calculate rolling statistics for a player"""
        query = """
            SELECT
                AVG(passing_yards) as avg_passing_yards,
                AVG(rushing_yards) as avg_rushing_yards,
                AVG(receiving_yards) as avg_receiving_yards,
                AVG(fantasy_points) as avg_fantasy_points,
                COUNT(*) as games_in_window
            FROM raw_player_stats
            WHERE player_id = ?
                AND ((season = ? AND week < ?) OR season < ?)
            ORDER BY season DESC, week DESC
            LIMIT ?
        """

        result = self.db.execute_one(query, [
            player_id, target_season, target_week, target_season, window
        ])

        if result:
            return {
                "avg_passing_yards": result[0] or 0,
                "avg_rushing_yards": result[1] or 0,
                "avg_receiving_yards": result[2] or 0,
                "avg_fantasy_points": result[3] or 0,
                "games_in_window": result[4] or 0
            }
        return {}

    def store_player_features(self, player_id: str, season: int, week: int,
                             features: Dict[str, Any]):
        """Store calculated features for a player"""
        stats_json = json.dumps(features.get('rolling_stats', {}))
        trends_json = json.dumps(features.get('trends', {}))
        matchup_json = json.dumps(features.get('matchup', {}))
        context_json = json.dumps(features.get('context', {}))

        query = """
            INSERT INTO player_rolling_features
            (player_id, season, week, position, stats_last3_games,
             performance_trend, usage_trend, vs_opponent_history,
             home_away_splits, divisional_game, rest_days)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        self.db.connect().execute(query, [
            player_id, season, week,
            features.get('position', 'UNK'),
            stats_json,
            features.get('performance_trend', 0.0),
            features.get('usage_trend', 0.0),
            json.dumps(features.get('vs_opponent_history', {})),
            json.dumps(features.get('home_away_splits', {})),
            features.get('divisional_game', False),
            features.get('rest_days', 7)
        ])


class DataQualityManager:
    """Manager for data quality operations"""

    def __init__(self, db: NFLDatabase):
        self.db = db

    def check_data_completeness(self) -> Dict[str, Any]:
        """Check data completeness across tables"""
        results = {}

        player_stats_check = self.db.execute_one("""
            SELECT
                COUNT(*) as total_records,
                COUNT(DISTINCT player_id) as unique_players,
                COUNT(DISTINCT season) as seasons_covered,
                MIN(season) as earliest_season,
                MAX(season) as latest_season
            FROM raw_player_stats
        """)

        results['player_stats'] = {
            "total_records": player_stats_check[0],
            "unique_players": player_stats_check[1],
            "seasons_covered": player_stats_check[2],
            "earliest_season": player_stats_check[3],
            "latest_season": player_stats_check[4]
        }

        return results

    def detect_outliers(self, table: str, column: str, threshold: float = 3.0) -> List[Dict]:
        """Detect statistical outliers using z-score"""
        query = f"""
            WITH stats AS (
                SELECT
                    AVG({column}) as mean_val,
                    STDDEV({column}) as std_val
                FROM {table}
                WHERE {column} IS NOT NULL
            ),
            z_scores AS (
                SELECT *,
                    ABS(({column} - stats.mean_val) / stats.std_val) as z_score
                FROM {table}, stats
                WHERE {column} IS NOT NULL
            )
            SELECT * FROM z_scores
            WHERE z_score > {threshold}
            ORDER BY z_score DESC
            LIMIT 100
        """

        results = self.db.execute(query)
        return [dict(zip([col[0] for col in self.db.connect().description], row))
                for row in results]


def get_database() -> NFLDatabase:
    """Factory function to get database instance"""
    return NFLDatabase()