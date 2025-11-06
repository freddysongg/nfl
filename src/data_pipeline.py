"""
NFL Data Collection Pipeline
Implements the 4-stage pipeline from DATA_SETUP.md
"""

import nflreadpy as nfl
import polars as pl
from typing import List, Optional, Dict, Any
import time
import logging
from pathlib import Path
import json
from datetime import datetime, date

from .database import NFLDatabase, PlayerStatsManager, FeatureManager
from .config import config
from .batch_processor import BatchProcessor, ProgressTracker
from .table_schemas import (
    create_raw_nextgen_passing_table, create_raw_nextgen_rushing_table,
    create_raw_nextgen_receiving_table, create_raw_snap_counts_table,
    create_raw_pbp_table, create_raw_players_table, create_raw_ftn_charting_table,
    create_raw_participation_table, create_raw_draft_picks_table, create_raw_combine_table
)


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class NFLDataPipeline:
    """
    Master data collection and processing pipeline
    Implements 4-stage architecture from DATA_SETUP.md
    """

    def __init__(self, db_file: str = None):
        self.db = NFLDatabase(db_file or str(config.db_file))
        self.player_manager = PlayerStatsManager(self.db)
        self.feature_manager = FeatureManager(self.db)
        self.batch_processor = BatchProcessor(self.db)
        self.progress_tracker = ProgressTracker()

        self.seasons_to_collect = config.data_collection_config["seasons_to_collect"]
        self.batch_size = config.data_collection_config["batch_size"]
        self.retry_attempts = config.data_collection_config["retry_attempts"]
        self.rate_limit_delay = config.data_collection_config["rate_limit_delay"]

        logger.info(f"Pipeline initialized for seasons: {self.seasons_to_collect}")
        logger.info(f"Batch processing enabled with size: {self.batch_size}")

    def full_historical_load(self, seasons: Optional[List[int]] = None):
        """
        Stage 1: Load all raw data from nflreadr
        From DATA_SETUP.md pipeline architecture
        """
        seasons = seasons or self.seasons_to_collect
        logger.info(f"🚀 Starting full historical load for seasons: {seasons}")

        total_seasons = len(seasons)
        for i, season in enumerate(seasons, 1):
            logger.info(f"📅 Processing season {season} ({i}/{total_seasons})")

            try:
                self.load_player_data(season)

                self.load_team_data(season)

                self.load_roster_data(season)

                self.load_advanced_data(season)

                logger.info(f"✅ Season {season} completed successfully")

                if i < total_seasons:
                    time.sleep(self.rate_limit_delay)

            except Exception as e:
                logger.error(f"❌ Failed to process season {season}: {e}")
                if self.retry_attempts > 0:
                    logger.info(f"🔄 Retrying season {season}...")
                    self._retry_season_load(season)
                else:
                    raise

        self.progress_tracker.log_summary("NFL Data Pipeline")
        logger.info("🎉 Full historical load completed!")

    def load_player_data(self, season: int):
        """Load all player-related data for a season"""
        logger.info(f"📊 Loading player data for {season}")

        try:
            logger.info("  Loading player stats...")
            player_stats = nfl.load_player_stats(seasons=season)

            if not player_stats.is_empty():
                result = self.batch_processor.process_dataframe_to_table(
                    player_stats, 'raw_player_stats', f'Player Stats {season}'
                )
                self.progress_tracker.add_result(result)
            else:
                logger.warning(f"  ⚠️  No player stats found for {season}")

            if season >= 2016:
                for stat_type in ['passing', 'rushing', 'receiving']:
                    try:
                        logger.info(f"  Loading Next Gen {stat_type} stats...")
                        ngs = nfl.load_nextgen_stats(stat_type=stat_type, seasons=season)

                        if not ngs.is_empty():
                            table_name = f'raw_nextgen_{stat_type}'
                            self._ensure_nextgen_table_exists(table_name, stat_type)
                            result = self.batch_processor.process_dataframe_to_table(
                                ngs, table_name, f'Next Gen {stat_type.title()} {season}'
                            )
                            self.progress_tracker.add_result(result)

                    except Exception as e:
                        logger.warning(f"  ⚠️  Next Gen {stat_type} failed for {season}: {e}")

            if season >= 2012:
                try:
                    logger.info("  Loading snap counts...")
                    snaps = nfl.load_snap_counts(seasons=season)

                    if not snaps.is_empty():
                        self._ensure_snap_counts_table_exists()
                        result = self.batch_processor.process_dataframe_to_table(
                            snaps, 'raw_snap_counts', f'Snap Counts {season}'
                        )
                        self.progress_tracker.add_result(result)

                except Exception as e:
                    logger.warning(f"  ⚠️  Snap counts failed for {season}: {e}")

        except Exception as e:
            logger.error(f"❌ Player data load failed for {season}: {e}")
            raise

    def load_team_data(self, season: int):
        """Load team-related data for a season"""
        logger.info(f"🏈 Loading team data for {season}")

        try:
            logger.info("  Loading team stats...")
            team_stats = nfl.load_team_stats(seasons=season)

            if not team_stats.is_empty():
                result = self.batch_processor.process_dataframe_to_table(
                    team_stats, 'raw_team_stats', f'Team Stats {season}'
                )
                self.progress_tracker.add_result(result)

            logger.info("  Loading schedules...")
            schedules = nfl.load_schedules(seasons=season)

            if not schedules.is_empty():
                season_schedules = schedules.filter(pl.col("season") == season)
                if not season_schedules.is_empty():
                    result = self.batch_processor.process_dataframe_to_table(
                        season_schedules, 'raw_schedules', f'Schedules {season}'
                    )
                    self.progress_tracker.add_result(result)

            if season >= 2022:
                try:
                    logger.info("  Loading play-by-play data...")
                    pbp = nfl.load_pbp(seasons=season)

                    if not pbp.is_empty():
                        self._ensure_pbp_table_exists()
                        result = self.batch_processor.process_dataframe_to_table(
                            pbp, 'raw_pbp', f'Play-by-Play {season}'
                        )
                        self.progress_tracker.add_result(result)

                except Exception as e:
                    logger.warning(f"  ⚠️  PBP data failed for {season}: {e}")

        except Exception as e:
            logger.error(f"❌ Team data load failed for {season}: {e}")
            raise

    def load_roster_data(self, season: int):
        """Load roster and depth chart data with proper 2025+ handling"""
        logger.info(f"👥 Loading roster data for {season}")

        try:
            logger.info("  Loading weekly rosters...")
            rosters_weekly = nfl.load_rosters_weekly(seasons=season)

            if not rosters_weekly.is_empty():
                result = self.batch_processor.process_dataframe_to_table(
                    rosters_weekly, 'raw_rosters_weekly', f'Weekly Rosters {season}'
                )
                self.progress_tracker.add_result(result)

            logger.info("  Loading depth charts...")
            depth_charts = nfl.load_depth_charts(seasons=season)

            if not depth_charts.is_empty():
                if season >= 2025:
                    depth_charts = self.process_2025_depth_charts(depth_charts, season)
                    logger.info("  📅 Applied 2025+ timestamp processing")

                result = self.batch_processor.process_dataframe_to_table(
                    depth_charts, 'raw_depth_charts', f'Depth Charts {season}'
                )
                self.progress_tracker.add_result(result)

            if season == min(self.seasons_to_collect):
                logger.info("  Loading player metadata...")
                players = nfl.load_players()

                if not players.is_empty():
                    self._ensure_players_table_exists()
                    result = self.batch_processor.process_dataframe_to_table(
                        players, 'raw_players', f'Player Metadata'
                    )
                    self.progress_tracker.add_result(result)

        except Exception as e:
            logger.error(f"❌ Roster data load failed for {season}: {e}")
            raise

    def load_advanced_data(self, season: int):
        """Load advanced metrics when available"""
        logger.info(f"🔬 Loading advanced data for {season}")

        if season >= 2022:
            try:
                logger.info("  Loading FTN charting data...")
                ftn = nfl.load_ftn_charting(seasons=season)

                if not ftn.is_empty():
                    self._ensure_ftn_table_exists()
                    rows_inserted = self.db.store_dataframe(ftn, 'raw_ftn_charting')
                    logger.info(f"  ✅ Stored {rows_inserted:,} FTN charting records")

            except Exception as e:
                logger.warning(f"  ⚠️  FTN charting failed for {season}: {e}")

        if season < 2025:
            try:
                logger.info("  Loading participation data...")
                participation = nfl.load_participation(seasons=season)

                if not participation.is_empty():
                    self._ensure_participation_table_exists()
                    rows_inserted = self.db.store_dataframe(participation, 'raw_participation')
                    logger.info(f"  ✅ Stored {rows_inserted:,} participation records")

            except Exception as e:
                logger.warning(f"  ⚠️  Participation data failed for {season}: {e}")

        if season == min(self.seasons_to_collect):
            try:
                logger.info("  Loading draft picks...")
                draft_picks = nfl.load_draft_picks()

                if not draft_picks.is_empty():
                    self._ensure_draft_table_exists()
                    rows_inserted = self.db.store_dataframe(draft_picks, 'raw_draft_picks')
                    logger.info(f"  ✅ Stored {rows_inserted:,} draft pick records")

                logger.info("  Loading combine data...")
                combine = nfl.load_combine()

                if not combine.is_empty():
                    self._ensure_combine_table_exists()
                    rows_inserted = self.db.store_dataframe(combine, 'raw_combine')
                    logger.info(f"  ✅ Stored {rows_inserted:,} combine records")

            except Exception as e:
                logger.warning(f"  ⚠️  Draft/combine data failed: {e}")

    def process_2025_depth_charts(self, df: pl.DataFrame, season: int) -> pl.DataFrame:
        """
        Handle 2025+ ISO8601 timestamp format
        From DATA_SETUP.md depth chart strategy
        """
        logger.info("📅 Processing 2025+ depth chart timestamp format")

        try:
            processed_df = df.with_columns([
                pl.col("dt").str.strptime(pl.Datetime, "%Y-%m-%dT%H:%M:%SZ").alias("dt_timestamp"),

                pl.when(pl.col("dt").str.strptime(pl.Datetime, "%Y-%m-%dT%H:%M:%SZ").dt.month() <= 8)
                .then(1)
                .otherwise(
                    ((pl.col("dt").str.strptime(pl.Datetime, "%Y-%m-%dT%H:%M:%SZ").dt.ordinal_day() - 244) / 7 + 1).cast(pl.Int32)
                ).alias("week"),

                pl.lit(season).alias("season")
            ])

            logger.info("✅ 2025+ timestamp processing completed")
            return processed_df

        except Exception as e:
            logger.error(f"❌ 2025+ timestamp processing failed: {e}")
            return df.with_columns([
                pl.col("dt").alias("dt_timestamp"),
                pl.lit(1).alias("week")
            ])

    def process_roster_snapshots(self):
        """
        Stage 2: Create time-aware roster snapshots
        From DATA_SETUP.md pipeline architecture
        """
        logger.info("🗂️  Stage 2: Processing roster snapshots")

        self.build_player_lifecycle_table()
        self.create_weekly_roster_snapshots()
        self.classify_player_experience_levels()

        logger.info("✅ Roster snapshot processing completed")

    def engineer_features(self):
        """
        Stage 3: Feature engineering
        From DATA_SETUP.md pipeline architecture
        """
        logger.info("⚙️ Stage 3: Engineering features")

        self.calculate_rolling_statistics()
        self.build_matchup_features()
        self.create_team_aggregates()
        self.handle_rookie_veteran_features()

        logger.info("✅ Feature engineering completed")

    def build_ml_dataset(self):
        """
        Stage 4: Create ML-ready training set
        From DATA_SETUP.md pipeline architecture
        """
        logger.info("🤖 Stage 4: Building ML dataset")

        self.combine_all_features()
        self.apply_data_quality_scoring()
        self.create_prediction_targets()
        self.validate_temporal_consistency()

        logger.info("✅ ML dataset creation completed")

    def run_full_pipeline(self, seasons: Optional[List[int]] = None):
        """Execute complete 4-stage pipeline"""
        logger.info("🚀 Starting complete NFL data pipeline")

        start_time = datetime.now()

        try:
            self.full_historical_load(seasons)

            self.process_roster_snapshots()

            self.engineer_features()

            self.build_ml_dataset()

            duration = datetime.now() - start_time
            logger.info(f"🎉 Complete pipeline finished in {duration}")

        except Exception as e:
            logger.error(f"❌ Pipeline failed: {e}")
            raise

    def _ensure_nextgen_table_exists(self, table_name: str, stat_type: str):
        """Create Next Gen Stats table if it doesn't exist"""
        conn = self.db.connect()
        if stat_type == 'passing':
            create_raw_nextgen_passing_table(conn)
        elif stat_type == 'rushing':
            create_raw_nextgen_rushing_table(conn)
        elif stat_type == 'receiving':
            create_raw_nextgen_receiving_table(conn)

    def _ensure_snap_counts_table_exists(self):
        """Create snap counts table if it doesn't exist"""
        conn = self.db.connect()
        create_raw_snap_counts_table(conn)

    def _ensure_pbp_table_exists(self):
        """Create play-by-play table if it doesn't exist"""
        conn = self.db.connect()
        create_raw_pbp_table(conn)

    def _ensure_players_table_exists(self):
        """Create players metadata table if it doesn't exist"""
        conn = self.db.connect()
        create_raw_players_table(conn)

    def _ensure_ftn_table_exists(self):
        """Create FTN charting table if it doesn't exist"""
        conn = self.db.connect()
        create_raw_ftn_charting_table(conn)

    def _ensure_participation_table_exists(self):
        """Create participation table if it doesn't exist"""
        conn = self.db.connect()
        create_raw_participation_table(conn)

    def _ensure_draft_table_exists(self):
        """Create draft picks table if it doesn't exist"""
        conn = self.db.connect()
        create_raw_draft_picks_table(conn)

    def _ensure_combine_table_exists(self):
        """Create combine table if it doesn't exist"""
        conn = self.db.connect()
        create_raw_combine_table(conn)

    def _retry_season_load(self, season: int):
        """Retry failed season load with exponential backoff"""
        for attempt in range(self.retry_attempts):
            try:
                wait_time = (2 ** attempt) * self.rate_limit_delay
                logger.info(f"⏰ Retry attempt {attempt + 1}, waiting {wait_time}s")
                time.sleep(wait_time)

                self.load_player_data(season)
                self.load_team_data(season)
                self.load_roster_data(season)
                self.load_advanced_data(season)

                logger.info(f"✅ Retry successful for season {season}")
                return

            except Exception as e:
                logger.warning(f"⚠️  Retry {attempt + 1} failed: {e}")

        logger.error(f"❌ All retries exhausted for season {season}")
        raise Exception(f"Failed to load season {season} after {self.retry_attempts} attempts")

    def build_player_lifecycle_table(self) -> int:
        """
        Build player career tracking table.

        Combines data from raw_players, raw_rosters_weekly, and raw_player_stats
        to create comprehensive player lifecycle records.

        Returns:
            int: Number of player records created

        Raises:
            ValueError: If required raw tables don't exist
            Exception: If database operation fails
        """
        logger.info("👥 Building player lifecycle table...")

        try:
            conn = self.db.connect()

            # Check if required tables exist
            tables = self.db.list_tables()
            required = ['raw_players', 'raw_rosters_weekly', 'raw_player_stats']
            missing = [t for t in required if t not in tables]
            if missing:
                raise ValueError(f"Missing required tables: {missing}. Please run Stage 1 first.")

            # Check for data in source tables
            player_count = conn.execute("SELECT COUNT(*) FROM raw_players").fetchone()[0]
            if player_count == 0:
                logger.warning("⚠️  raw_players is empty, skipping lifecycle build")
                return 0

            logger.info("  📊 Loading player metadata...")

            # Clear existing data
            conn.execute("DELETE FROM player_lifecycle")
            logger.info("  📊 Cleared existing lifecycle data")

            # Build lifecycle table using SQL
            logger.info("  📊 Calculating career spans...")
            conn.execute("""
                INSERT INTO player_lifecycle
                WITH player_metadata AS (
                    SELECT
                        player_id,
                        gsis_id,
                        espn_id,
                        position,
                        position_group,
                        draft_year,
                        draft_round,
                        draft_pick,
                        draft_ovr,
                        college,
                        status
                    FROM raw_players
                    WHERE player_id IS NOT NULL
                ),

                career_span_rosters AS (
                    SELECT
                        gsis_id,
                        MIN(season) as first_nfl_season,
                        MAX(season) as last_nfl_season,
                        LIST(DISTINCT team ORDER BY team) as career_teams
                    FROM raw_rosters_weekly
                    WHERE gsis_id IS NOT NULL
                        AND season IS NOT NULL
                        AND team IS NOT NULL
                    GROUP BY gsis_id
                ),

                career_span_stats AS (
                    SELECT
                        player_id,
                        MIN(season) as stats_first_season,
                        MAX(season) as stats_last_season,
                        LIST(DISTINCT team ORDER BY team) as stats_teams
                    FROM raw_player_stats
                    WHERE player_id IS NOT NULL
                        AND season IS NOT NULL
                        AND team IS NOT NULL
                    GROUP BY player_id
                ),

                max_season AS (
                    SELECT MAX(season) as current_season
                    FROM raw_player_stats
                )

                SELECT
                    pm.player_id,
                    pm.gsis_id,
                    pm.espn_id,

                    -- Career span: prefer rosters_weekly, fallback to stats
                    COALESCE(csr.first_nfl_season, css.stats_first_season) as first_nfl_season,
                    COALESCE(csr.last_nfl_season, css.stats_last_season) as last_nfl_season,

                    -- Career teams: merge both sources
                    CASE
                        WHEN csr.career_teams IS NOT NULL AND css.stats_teams IS NOT NULL
                        THEN LIST_DISTINCT(LIST_CONCAT(csr.career_teams, css.stats_teams))
                        ELSE COALESCE(csr.career_teams, css.stats_teams)
                    END as career_teams,

                    -- Position: use position_group as primary, position as fallback
                    COALESCE(pm.position_group, pm.position) as primary_position,

                    -- Draft info
                    pm.draft_year,
                    pm.draft_round,
                    pm.draft_pick,

                    -- College
                    pm.college,

                    -- Retirement status logic
                    CASE
                        WHEN pm.status = 'RET' THEN 'retired'
                        WHEN COALESCE(csr.last_nfl_season, css.stats_last_season) < (SELECT current_season FROM max_season) - 1
                        THEN 'inactive'
                        ELSE 'active'
                    END as retirement_status,

                    CURRENT_TIMESTAMP as created_at,
                    CURRENT_TIMESTAMP as updated_at

                FROM player_metadata pm
                LEFT JOIN career_span_rosters csr ON pm.gsis_id = csr.gsis_id
                LEFT JOIN career_span_stats css ON pm.player_id = css.player_id
            """)

            # Get count of inserted records
            result = conn.execute("SELECT COUNT(*) FROM player_lifecycle").fetchone()
            rows_inserted = result[0] if result else 0

            logger.info(f"✅ Built player lifecycle for {rows_inserted:,} players")

            # Log summary statistics
            summary = conn.execute("""
                SELECT
                    COUNT(*) as total_players,
                    COUNT(*) FILTER (WHERE retirement_status = 'active') as active,
                    COUNT(*) FILTER (WHERE retirement_status = 'retired') as retired,
                    COUNT(*) FILTER (WHERE retirement_status = 'inactive') as inactive,
                    MIN(first_nfl_season) as earliest_season,
                    MAX(last_nfl_season) as latest_season
                FROM player_lifecycle
            """).fetchone()

            if summary:
                logger.info(f"  📊 Career span: {summary[4]} - {summary[5]}")
                logger.info(f"  📊 Active: {summary[1]:,} | Retired: {summary[2]:,} | Inactive: {summary[3]:,}")

            return rows_inserted

        except Exception as e:
            logger.error(f"❌ Failed to build lifecycle table: {e}")
            raise

    def create_weekly_roster_snapshots(self) -> int:
        """
        Create time-aware roster snapshots for each team/week.

        Generates snapshot records with active players, depth chart info,
        and roster changes compared to previous week.

        Returns:
            int: Number of snapshot records created

        Raises:
            ValueError: If required raw tables don't exist
            Exception: If database operation fails
        """
        logger.info("📸 Creating weekly roster snapshots...")

        try:
            conn = self.db.connect()

            # Check if required tables exist
            tables = self.db.list_tables()
            required = ['raw_rosters_weekly']
            missing = [t for t in required if t not in tables]
            if missing:
                raise ValueError(f"Missing required tables: {missing}. Please run Stage 1 first.")

            # Clear existing data
            conn.execute("DELETE FROM team_roster_snapshots")
            logger.info("  📊 Cleared existing snapshot data")

            # Build roster snapshots using SQL
            logger.info("  📊 Building roster snapshots...")
            conn.execute("""
                INSERT INTO team_roster_snapshots
                WITH roster_base AS (
                    SELECT
                        team,
                        season,
                        week,
                        gsis_id,
                        full_name,
                        position,
                        depth_chart_position,
                        jersey_number,
                        status,
                        years_exp,
                        height,
                        weight
                    FROM raw_rosters_weekly
                    WHERE gsis_id IS NOT NULL
                        AND team IS NOT NULL
                        AND season IS NOT NULL
                        AND week IS NOT NULL
                ),

                roster_snapshots AS (
                    SELECT
                        team || '_' || season::VARCHAR || '_W' || week::VARCHAR as snapshot_id,
                        team,
                        season,
                        week,

                        -- Build active players JSON
                        TO_JSON(LIST({
                            'gsis_id': gsis_id,
                            'name': full_name,
                            'position': position,
                            'depth_position': depth_chart_position,
                            'jersey_number': jersey_number,
                            'status': status,
                            'years_exp': years_exp,
                            'height': height,
                            'weight': weight
                        })) as active_players

                    FROM roster_base
                    WHERE status IN ('ACT', 'RES')
                    GROUP BY team, season, week
                ),

                snapshot_dates AS (
                    SELECT DISTINCT
                        rs.snapshot_id,
                        rs.team,
                        rs.season,
                        rs.week,
                        (
                            SELECT MIN(gameday)
                            FROM raw_schedules s
                            WHERE s.season = rs.season
                                AND s.week = rs.week
                                AND (s.home_team = rs.team OR s.away_team = rs.team)
                        ) as snapshot_date
                    FROM roster_snapshots rs
                ),

                depth_info AS (
                    SELECT
                        club_code as team,
                        season,
                        week,
                        TO_JSON(LIST({
                            'gsis_id': gsis_id,
                            'position': position,
                            'formation': formation,
                            'depth_team': depth_team,
                            'name': first_name || ' ' || last_name
                        })) as depth_chart
                    FROM raw_depth_charts
                    WHERE gsis_id IS NOT NULL
                        AND club_code IS NOT NULL
                        AND season IS NOT NULL
                        AND week IS NOT NULL
                    GROUP BY club_code, season, week
                ),

                roster_changes AS (
                    SELECT
                        curr.snapshot_id,
                        curr.team,
                        curr.season,
                        curr.week,

                        TO_JSON({
                            'new_players': COALESCE(
                                (
                                    SELECT LIST(gsis_id)
                                    FROM roster_base curr_r
                                    WHERE curr_r.team = curr.team
                                        AND curr_r.season = curr.season
                                        AND curr_r.week = curr.week
                                        AND curr_r.gsis_id NOT IN (
                                            SELECT prev_r.gsis_id
                                            FROM roster_base prev_r
                                            WHERE prev_r.team = curr.team
                                                AND prev_r.season = curr.season
                                                AND prev_r.week = curr.week - 1
                                        )
                                ), []
                            ),
                            'removed_players': COALESCE(
                                (
                                    SELECT LIST(gsis_id)
                                    FROM roster_base prev_r
                                    WHERE prev_r.team = curr.team
                                        AND prev_r.season = curr.season
                                        AND prev_r.week = curr.week - 1
                                        AND prev_r.gsis_id NOT IN (
                                            SELECT curr_r.gsis_id
                                            FROM roster_base curr_r
                                            WHERE curr_r.team = curr.team
                                                AND curr_r.season = curr.season
                                                AND curr_r.week = curr.week
                                        )
                                ), []
                            ),
                            'position_changes': []
                        }) as key_changes

                    FROM roster_snapshots curr
                )

                SELECT
                    rs.snapshot_id,
                    rs.team,
                    rs.season,
                    rs.week,
                    sd.snapshot_date,
                    rs.active_players,
                    COALESCE(di.depth_chart, TO_JSON([])) as depth_chart,
                    COALESCE(rc.key_changes, TO_JSON({})) as key_changes,
                    CURRENT_TIMESTAMP as created_at
                FROM roster_snapshots rs
                LEFT JOIN snapshot_dates sd
                    ON rs.snapshot_id = sd.snapshot_id
                LEFT JOIN depth_info di
                    ON rs.team = di.team
                    AND rs.season = di.season
                    AND rs.week = di.week
                LEFT JOIN roster_changes rc
                    ON rs.snapshot_id = rc.snapshot_id
            """)

            # Get count of inserted records
            result = conn.execute("SELECT COUNT(*) FROM team_roster_snapshots").fetchone()
            rows_inserted = result[0] if result else 0

            logger.info(f"✅ Created {rows_inserted:,} roster snapshots")

            # Log summary statistics
            summary = conn.execute("""
                SELECT
                    COUNT(DISTINCT team) as teams,
                    COUNT(DISTINCT season) as seasons,
                    COUNT(DISTINCT week) as weeks
                FROM team_roster_snapshots
            """).fetchone()

            if summary:
                logger.info(f"  📊 Teams: {summary[0]} | Seasons: {summary[1]} | Weeks: {summary[2]}")

            return rows_inserted

        except Exception as e:
            logger.error(f"❌ Failed to create roster snapshots: {e}")
            raise

    def classify_player_experience_levels(self) -> int:
        """
        Classify players by experience level for each season.

        Uses player_lifecycle.first_nfl_season to calculate seasons_played
        and classify into rookie/developing/veteran categories.

        Returns:
            int: Number of player-season classifications created

        Raises:
            ValueError: If player_lifecycle table doesn't exist or is empty
            Exception: If database operation fails

        Pre-conditions:
            - build_player_lifecycle_table() must be run first
        """
        logger.info("🎓 Classifying player experience levels...")

        try:
            conn = self.db.connect()

            # Check if lifecycle table exists and has data
            tables = self.db.list_tables()
            if 'player_lifecycle' not in tables:
                raise ValueError("player_lifecycle table doesn't exist. Run build_player_lifecycle_table() first.")

            lifecycle_count = conn.execute("SELECT COUNT(*) FROM player_lifecycle").fetchone()[0]
            if lifecycle_count == 0:
                logger.warning("⚠️  player_lifecycle is empty, skipping classification")
                return 0

            # Get thresholds from config
            thresholds = config.feature_engineering_config["experience_thresholds"]
            confidence = config.ml_config["confidence_thresholds"]

            logger.info("  📊 Calculating experience classifications...")

            # Clear existing data
            conn.execute("DELETE FROM player_experience_classification")

            # Build classification table using SQL
            conn.execute(f"""
                INSERT INTO player_experience_classification
                WITH player_seasons AS (
                    SELECT DISTINCT
                        ps.player_id,
                        ps.season,
                        pl.first_nfl_season,
                        ps.season - pl.first_nfl_season + 1 as seasons_played

                    FROM raw_player_stats ps
                    INNER JOIN player_lifecycle pl ON ps.player_id = pl.player_id
                    WHERE ps.player_id IS NOT NULL
                        AND ps.season IS NOT NULL
                        AND pl.first_nfl_season IS NOT NULL
                )

                SELECT
                    player_id,
                    season,

                    -- Experience category
                    CASE
                        WHEN seasons_played <= {thresholds['rookie']} THEN 'rookie'
                        WHEN seasons_played IN ({', '.join(map(str, thresholds['developing']))}) THEN 'developing'
                        ELSE 'veteran'
                    END as experience_category,

                    seasons_played,

                    -- Prediction strategy based on experience
                    CASE
                        WHEN seasons_played <= {thresholds['rookie']} THEN 'high_variance_model'
                        WHEN seasons_played IN ({', '.join(map(str, thresholds['developing']))}) THEN 'mixed_model'
                        ELSE 'historical_trend_model'
                    END as prediction_strategy,

                    -- Confidence multiplier from config
                    CASE
                        WHEN seasons_played <= {thresholds['rookie']} THEN {confidence['rookie']}
                        WHEN seasons_played IN ({', '.join(map(str, thresholds['developing']))}) THEN {confidence['developing']}
                        ELSE {confidence['veteran']}
                    END as confidence_multiplier,

                    CURRENT_TIMESTAMP as created_at

                FROM player_seasons
            """)

            # Get count of inserted records
            result = conn.execute("SELECT COUNT(*) FROM player_experience_classification").fetchone()
            rows_inserted = result[0] if result else 0

            logger.info(f"✅ Classified {rows_inserted:,} player-season combinations")

            # Log distribution
            distribution = conn.execute("""
                SELECT
                    experience_category,
                    COUNT(*) as count,
                    AVG(confidence_multiplier) as avg_confidence
                FROM player_experience_classification
                GROUP BY experience_category
                ORDER BY experience_category
            """).fetchall()

            for row in distribution:
                logger.info(f"  📊 {row[0]}: {row[1]:,} records (confidence: {row[2]:.1f})")

            return rows_inserted

        except Exception as e:
            logger.error(f"❌ Failed to classify experience levels: {e}")
            raise

    def calculate_rolling_statistics(self):
        """Calculate rolling averages and trends"""
        logger.info("📈 Calculating rolling statistics...")
        # TODO: Implement from DATA_SETUP.md

    def build_matchup_features(self):
        """Build matchup-specific features"""
        logger.info("⚔️ Building matchup features...")
        # TODO: Implement from DATA_SETUP.md

    def create_team_aggregates(self):
        """Create team aggregate features"""
        logger.info("🏈 Creating team aggregates...")
        # TODO: Implement from DATA_SETUP.md

    def handle_rookie_veteran_features(self):
        """Handle rookie/veteran feature differences"""
        logger.info("🆕 Handling rookie/veteran features...")
        # TODO: Implement from DATA_SETUP.md

    def combine_all_features(self):
        """Combine all features into ML-ready format"""
        logger.info("🔗 Combining all features...")
        # TODO: Implement from DATA_SETUP.md

    def apply_data_quality_scoring(self):
        """Apply data quality scoring"""
        logger.info("✨ Applying data quality scoring...")
        # TODO: Implement from DATA_SETUP.md

    def create_prediction_targets(self):
        """Create prediction target variables"""
        logger.info("🎯 Creating prediction targets...")
        # TODO: Implement from DATA_SETUP.md

    def validate_temporal_consistency(self):
        """Validate no future data leakage"""
        logger.info("⏰ Validating temporal consistency...")
        # TODO: Implement from DATA_SETUP.md


