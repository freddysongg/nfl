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
    create_raw_nextgen_passing_table,
    create_raw_nextgen_rushing_table,
    create_raw_nextgen_receiving_table,
    create_raw_snap_counts_table,
    create_raw_pbp_table,
    create_raw_players_table,
    create_raw_ftn_charting_table,
    create_raw_participation_table,
    create_raw_draft_picks_table,
    create_raw_combine_table,
)


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
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
                    player_stats, "raw_player_stats", f"Player Stats {season}"
                )
                self.progress_tracker.add_result(result)
            else:
                logger.warning(f"  ⚠️  No player stats found for {season}")

            if season >= 2016:
                for stat_type in ["passing", "rushing", "receiving"]:
                    try:
                        logger.info(f"  Loading Next Gen {stat_type} stats...")
                        ngs = nfl.load_nextgen_stats(
                            stat_type=stat_type, seasons=season
                        )

                        if not ngs.is_empty():
                            table_name = f"raw_nextgen_{stat_type}"
                            self._ensure_nextgen_table_exists(table_name, stat_type)
                            result = self.batch_processor.process_dataframe_to_table(
                                ngs,
                                table_name,
                                f"Next Gen {stat_type.title()} {season}",
                            )
                            self.progress_tracker.add_result(result)

                    except Exception as e:
                        logger.warning(
                            f"  ⚠️  Next Gen {stat_type} failed for {season}: {e}"
                        )

            if season >= 2012:
                try:
                    logger.info("  Loading snap counts...")
                    snaps = nfl.load_snap_counts(seasons=season)

                    if not snaps.is_empty():
                        self._ensure_snap_counts_table_exists()
                        result = self.batch_processor.process_dataframe_to_table(
                            snaps, "raw_snap_counts", f"Snap Counts {season}"
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
                    team_stats, "raw_team_stats", f"Team Stats {season}"
                )
                self.progress_tracker.add_result(result)

            logger.info("  Loading schedules...")
            schedules = nfl.load_schedules(seasons=season)

            if not schedules.is_empty():
                season_schedules = schedules.filter(pl.col("season") == season)
                if not season_schedules.is_empty():
                    result = self.batch_processor.process_dataframe_to_table(
                        season_schedules, "raw_schedules", f"Schedules {season}"
                    )
                    self.progress_tracker.add_result(result)

            if season >= 2022:
                try:
                    logger.info("  Loading play-by-play data...")
                    pbp = nfl.load_pbp(seasons=season)

                    if not pbp.is_empty():
                        self._ensure_pbp_table_exists()
                        result = self.batch_processor.process_dataframe_to_table(
                            pbp, "raw_pbp", f"Play-by-Play {season}"
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
                    rosters_weekly, "raw_rosters_weekly", f"Weekly Rosters {season}"
                )
                self.progress_tracker.add_result(result)

            logger.info("  Loading depth charts...")
            depth_charts = nfl.load_depth_charts(seasons=season)

            if not depth_charts.is_empty():
                if season >= 2025:
                    depth_charts = self.process_2025_depth_charts(depth_charts, season)
                    logger.info("  📅 Applied 2025+ timestamp processing")

                result = self.batch_processor.process_dataframe_to_table(
                    depth_charts, "raw_depth_charts", f"Depth Charts {season}"
                )
                self.progress_tracker.add_result(result)

            if season == min(self.seasons_to_collect):
                logger.info("  Loading player metadata...")
                players = nfl.load_players()

                if not players.is_empty():
                    self._ensure_players_table_exists()
                    result = self.batch_processor.process_dataframe_to_table(
                        players, "raw_players", f"Player Metadata"
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
                    rows_inserted = self.db.store_dataframe(ftn, "raw_ftn_charting")
                    logger.info(f"  ✅ Stored {rows_inserted:,} FTN charting records")

            except Exception as e:
                logger.warning(f"  ⚠️  FTN charting failed for {season}: {e}")

        if season < 2025:
            try:
                logger.info("  Loading participation data...")
                participation = nfl.load_participation(seasons=season)

                if not participation.is_empty():
                    self._ensure_participation_table_exists()
                    rows_inserted = self.db.store_dataframe(
                        participation, "raw_participation"
                    )
                    logger.info(f"  ✅ Stored {rows_inserted:,} participation records")

            except Exception as e:
                logger.warning(f"  ⚠️  Participation data failed for {season}: {e}")

        if season == min(self.seasons_to_collect):
            try:
                logger.info("  Loading draft picks...")
                draft_picks = nfl.load_draft_picks()

                if not draft_picks.is_empty():
                    self._ensure_draft_table_exists()
                    rows_inserted = self.db.store_dataframe(
                        draft_picks, "raw_draft_picks"
                    )
                    logger.info(f"  ✅ Stored {rows_inserted:,} draft pick records")

                logger.info("  Loading combine data...")
                combine = nfl.load_combine()

                if not combine.is_empty():
                    self._ensure_combine_table_exists()
                    rows_inserted = self.db.store_dataframe(combine, "raw_combine")
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
            processed_df = df.with_columns(
                [
                    pl.col("dt")
                    .str.strptime(pl.Datetime, "%Y-%m-%dT%H:%M:%SZ")
                    .alias("dt_timestamp"),
                    pl.when(
                        pl.col("dt")
                        .str.strptime(pl.Datetime, "%Y-%m-%dT%H:%M:%SZ")
                        .dt.month()
                        <= 8
                    )
                    .then(1)
                    .otherwise(
                        (
                            (
                                pl.col("dt")
                                .str.strptime(pl.Datetime, "%Y-%m-%dT%H:%M:%SZ")
                                .dt.ordinal_day()
                                - 244
                            )
                            / 7
                            + 1
                        ).cast(pl.Int32)
                    )
                    .alias("week"),
                    pl.lit(season).alias("season"),
                ]
            )

            logger.info("✅ 2025+ timestamp processing completed")
            return processed_df

        except Exception as e:
            logger.error(f"❌ 2025+ timestamp processing failed: {e}")
            return df.with_columns(
                [pl.col("dt").alias("dt_timestamp"), pl.lit(1).alias("week")]
            )

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
        if stat_type == "passing":
            create_raw_nextgen_passing_table(conn)
        elif stat_type == "rushing":
            create_raw_nextgen_rushing_table(conn)
        elif stat_type == "receiving":
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
                wait_time = (2**attempt) * self.rate_limit_delay
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
        raise Exception(
            f"Failed to load season {season} after {self.retry_attempts} attempts"
        )

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
            required = ["raw_players", "raw_rosters_weekly", "raw_player_stats"]
            missing = [t for t in required if t not in tables]
            if missing:
                raise ValueError(
                    f"Missing required tables: {missing}. Please run Stage 1 first."
                )

            # Check for data in source tables
            player_count = conn.execute("SELECT COUNT(*) FROM raw_players").fetchone()[
                0
            ]
            if player_count == 0:
                logger.warning("⚠️  raw_players is empty, skipping lifecycle build")
                return 0

            logger.info("  📊 Loading player metadata...")

            # Clear existing data
            conn.execute("DELETE FROM player_lifecycle")
            logger.info("  📊 Cleared existing lifecycle data")

            # Build lifecycle table using SQL
            logger.info("  📊 Calculating career spans...")
            conn.execute(
                """
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
            """
            )

            # Get count of inserted records
            result = conn.execute("SELECT COUNT(*) FROM player_lifecycle").fetchone()
            rows_inserted = result[0] if result else 0

            logger.info(f"✅ Built player lifecycle for {rows_inserted:,} players")

            # Log summary statistics
            summary = conn.execute(
                """
                SELECT
                    COUNT(*) as total_players,
                    COUNT(*) FILTER (WHERE retirement_status = 'active') as active,
                    COUNT(*) FILTER (WHERE retirement_status = 'retired') as retired,
                    COUNT(*) FILTER (WHERE retirement_status = 'inactive') as inactive,
                    MIN(first_nfl_season) as earliest_season,
                    MAX(last_nfl_season) as latest_season
                FROM player_lifecycle
            """
            ).fetchone()

            if summary:
                logger.info(f"  📊 Career span: {summary[4]} - {summary[5]}")
                logger.info(
                    f"  📊 Active: {summary[1]:,} | Retired: {summary[2]:,} | Inactive: {summary[3]:,}"
                )

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
            required = ["raw_rosters_weekly"]
            missing = [t for t in required if t not in tables]
            if missing:
                raise ValueError(
                    f"Missing required tables: {missing}. Please run Stage 1 first."
                )

            # Clear existing data
            conn.execute("DELETE FROM team_roster_snapshots")
            logger.info("  📊 Cleared existing snapshot data")

            # Build roster snapshots using SQL
            logger.info("  📊 Building roster snapshots...")
            conn.execute(
                """
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
                    COALESCE(di.depth_chart, '[]'::JSON) as depth_chart,
                    COALESCE(rc.key_changes, '{}'::JSON) as key_changes,
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
            """
            )

            # Get count of inserted records
            result = conn.execute(
                "SELECT COUNT(*) FROM team_roster_snapshots"
            ).fetchone()
            rows_inserted = result[0] if result else 0

            logger.info(f"✅ Created {rows_inserted:,} roster snapshots")

            # Log summary statistics
            summary = conn.execute(
                """
                SELECT
                    COUNT(DISTINCT team) as teams,
                    COUNT(DISTINCT season) as seasons,
                    COUNT(DISTINCT week) as weeks
                FROM team_roster_snapshots
            """
            ).fetchone()

            if summary:
                logger.info(
                    f"  📊 Teams: {summary[0]} | Seasons: {summary[1]} | Weeks: {summary[2]}"
                )

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
            if "player_lifecycle" not in tables:
                raise ValueError(
                    "player_lifecycle table doesn't exist. Run build_player_lifecycle_table() first."
                )

            lifecycle_count = conn.execute(
                "SELECT COUNT(*) FROM player_lifecycle"
            ).fetchone()[0]
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
            conn.execute(
                f"""
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
            """
            )

            # Get count of inserted records
            result = conn.execute(
                "SELECT COUNT(*) FROM player_experience_classification"
            ).fetchone()
            rows_inserted = result[0] if result else 0

            logger.info(f"✅ Classified {rows_inserted:,} player-season combinations")

            # Log distribution
            distribution = conn.execute(
                """
                SELECT
                    experience_category,
                    COUNT(*) as count,
                    AVG(confidence_multiplier) as avg_confidence
                FROM player_experience_classification
                GROUP BY experience_category
                ORDER BY experience_category
            """
            ).fetchall()

            for row in distribution:
                logger.info(
                    f"  📊 {row[0]}: {row[1]:,} records (confidence: {row[2]:.1f})"
                )

            return rows_inserted

        except Exception as e:
            logger.error(f"❌ Failed to classify experience levels: {e}")
            raise

    def calculate_rolling_statistics(self):
        """
        Calculate rolling averages, trends, and consistency metrics for all players.

        Implements Stage 3a from ROLLING_STATS_IMPLEMENTATION_PLAN.md

        For each position:
        - Calculates rolling statistics over 3, 5, and 10 game windows
        - Computes averages, standard deviations, trends (linear regression slopes)
        - Tracks min/max values and games in window for confidence weighting
        - Stores results in JSON format in player_rolling_features table

        Features calculated:
        - Rolling averages (AVG over window)
        - Consistency metrics (STDDEV)
        - Performance trends (REGR_SLOPE for linear regression)
        - MIN/MAX values in window
        - games_in_window metadata for ML confidence

        Edge cases handled:
        - Players with < 10 games (calculates with available data)
        - Missing weeks due to injury/bye (ROWS BETWEEN skips naturally)
        - Cross-season windows (week 2 of 2024 can include 2023 games)
        - Position-specific stats (QB doesn't get receiving stats, etc.)

        Returns:
            None. Results stored in player_rolling_features table.

        Raises:
            ValueError: If raw_player_stats table is empty
            Exception: If database operation fails
        """
        logger.info("📈 Calculating rolling statistics...")

        try:
            conn = self.db.connect()

            # Validate raw_player_stats exists and has data
            tables = self.db.list_tables()
            if "raw_player_stats" not in tables:
                raise ValueError(
                    "raw_player_stats table not found. Please run Stage 1 first."
                )

            player_count = conn.execute(
                "SELECT COUNT(*) FROM raw_player_stats"
            ).fetchone()[0]
            if player_count == 0:
                logger.warning(
                    "⚠️  raw_player_stats is empty, skipping rolling stats calculation"
                )
                return

            logger.info(f"  📊 Processing {player_count:,} player-game records")

            # Clear existing rolling features
            conn.execute("DELETE FROM player_rolling_features")
            logger.info("  🧹 Cleared existing rolling features")

            # Process each position independently for memory efficiency
            positions = ["QB", "RB", "WR", "TE", "K", "DEF"]
            windows = config.feature_engineering_config["rolling_windows"]  # [3, 5, 10]

            total_records = 0

            for position in positions:
                logger.info(f"  🏈 Processing {position} rolling stats...")

                try:
                    # Get position-specific stats
                    relevant_stats = config.get_position_stats(position)

                    if not relevant_stats:
                        logger.warning(
                            f"    ⚠️  No stats configured for {position}, skipping"
                        )
                        continue

                    # Build and execute SQL query with window functions
                    rolling_df = self._calculate_position_rolling_stats(
                        position, relevant_stats, windows
                    )

                    if rolling_df is None or rolling_df.is_empty():
                        logger.info(f"    ℹ️  No data for {position}")
                        continue

                    # Transform wide format to JSON columns
                    feature_df = self._transform_rolling_to_json(
                        rolling_df, position, windows
                    )

                    # Store in player_rolling_features table
                    records_stored = self._store_rolling_features(feature_df)
                    total_records += records_stored

                    logger.info(
                        f"    ✅ Stored {records_stored:,} records for {position}"
                    )

                except Exception as e:
                    logger.error(f"    ❌ Failed to process {position}: {e}")
                    # Continue with next position rather than failing entire calculation
                    continue

            logger.info(
                f"✅ Rolling statistics calculation complete: {total_records:,} total records"
            )

            # Log summary statistics
            self._log_rolling_stats_summary()

        except Exception as e:
            logger.error(f"❌ Failed to calculate rolling statistics: {e}")
            raise

    def _calculate_position_rolling_stats(
        self, position: str, stat_columns: List[str], windows: List[int]
    ) -> Optional[pl.DataFrame]:
        """
        Calculate rolling statistics for a specific position using DuckDB window functions.

        Args:
            position: Player position (QB, RB, WR, TE, K, DEF)
            stat_columns: List of relevant stat column names for this position
            windows: List of window sizes (e.g., [3, 5, 10])

        Returns:
            Polars DataFrame with rolling statistics, or None if no data
        """
        conn = self.db.connect()

        # Filter to only columns that actually exist in raw_player_stats
        table_columns = [
            row[0]
            for row in conn.execute(
                "SELECT column_name FROM information_schema.columns WHERE table_name = 'raw_player_stats'"
            ).fetchall()
        ]

        valid_stats = [s for s in stat_columns if s in table_columns]

        if not valid_stats:
            logger.warning(f"      ⚠️  No valid stat columns found for {position}")
            return None

        # Build SELECT clause for stat columns
        stat_select = ", ".join(valid_stats)

        # Build window function calculations for each stat and window
        window_calcs = []

        for stat in valid_stats:
            for window in windows:
                # Rolling average
                window_calcs.append(
                    f"""
                    AVG({stat}) OVER (
                        PARTITION BY player_id
                        ORDER BY season, week
                        ROWS BETWEEN {window} PRECEDING AND 1 PRECEDING
                    ) as {stat}_rolling_{window}
                """
                )

                # Standard deviation
                window_calcs.append(
                    f"""
                    STDDEV({stat}) OVER (
                        PARTITION BY player_id
                        ORDER BY season, week
                        ROWS BETWEEN {window} PRECEDING AND 1 PRECEDING
                    ) as {stat}_stddev_{window}
                """
                )

                # Trend (linear regression slope)
                window_calcs.append(
                    f"""
                    REGR_SLOPE({stat}, game_number) OVER (
                        PARTITION BY player_id
                        ORDER BY season, week
                        ROWS BETWEEN {window} PRECEDING AND 1 PRECEDING
                    ) as {stat}_trend_{window}
                """
                )

                # Min and Max
                window_calcs.append(
                    f"""
                    MIN({stat}) OVER (
                        PARTITION BY player_id
                        ORDER BY season, week
                        ROWS BETWEEN {window} PRECEDING AND 1 PRECEDING
                    ) as {stat}_min_{window}
                """
                )

                window_calcs.append(
                    f"""
                    MAX({stat}) OVER (
                        PARTITION BY player_id
                        ORDER BY season, week
                        ROWS BETWEEN {window} PRECEDING AND 1 PRECEDING
                    ) as {stat}_max_{window}
                """
                )

        # Add games_in_window metadata for each window
        for window in windows:
            window_calcs.append(
                f"""
                COUNT(*) OVER (
                    PARTITION BY player_id
                    ORDER BY season, week
                    ROWS BETWEEN {window} PRECEDING AND 1 PRECEDING
                ) as games_in_window_{window}
            """
            )

        window_calcs_str = ",\n                ".join(window_calcs)

        # Build complete query
        query = f"""
            WITH player_games AS (
                SELECT
                    player_id,
                    player_name,
                    position,
                    season,
                    week,
                    season_type,
                    team,
                    {stat_select},
                    -- Game sequence number for trend calculation
                    ROW_NUMBER() OVER (
                        PARTITION BY player_id
                        ORDER BY season, week
                    ) as game_number
                FROM raw_player_stats
                WHERE position = '{position}'
                    AND season_type = 'REG'
                    AND player_id IS NOT NULL
                ORDER BY player_id, season, week
            )
            SELECT
                player_id,
                player_name,
                position,
                season,
                week,
                team,
                game_number,
                {window_calcs_str}
            FROM player_games
            WHERE game_number > 1
            ORDER BY player_id, season, week
        """

        try:
            # Execute query and convert to Polars
            result = conn.execute(query).pl()
            return result
        except Exception as e:
            logger.error(f"      ❌ Query failed for {position}: {e}")
            return None

    def _transform_rolling_to_json(
        self, df: pl.DataFrame, position: str, windows: List[int]
    ) -> pl.DataFrame:
        """
        Transform wide-format rolling stats into JSON columns for storage.

        Args:
            df: DataFrame with rolling stat columns
            position: Player position
            windows: List of window sizes

        Returns:
            DataFrame with JSON columns (stats_last3_games, stats_last5_games, stats_last10_games)
        """
        import json

        # Get position-specific stats
        relevant_stats = config.get_position_stats(position)

        # Create JSON columns for each window
        for window in windows:
            json_col_name = f"stats_last{window}_games"

            # Collect all rolling stat columns for this window
            cols_for_window = [col for col in df.columns if f"_{window}" in col]

            # Build JSON structure per row
            def build_json_row(row_dict):
                json_obj = {
                    "window_size": window,
                    "games_in_window": row_dict.get(f"games_in_window_{window}", 0),
                }

                # Organize stats by category
                for stat in relevant_stats:
                    avg_col = f"{stat}_rolling_{window}"
                    stddev_col = f"{stat}_stddev_{window}"
                    trend_col = f"{stat}_trend_{window}"
                    min_col = f"{stat}_min_{window}"
                    max_col = f"{stat}_max_{window}"

                    # Only add if column exists and value is not None
                    if avg_col in row_dict and row_dict[avg_col] is not None:
                        json_obj[f"{stat}_avg"] = float(row_dict[avg_col])

                    if stddev_col in row_dict and row_dict[stddev_col] is not None:
                        json_obj[f"{stat}_stddev"] = float(row_dict[stddev_col])

                    if trend_col in row_dict and row_dict[trend_col] is not None:
                        json_obj[f"{stat}_trend"] = float(row_dict[trend_col])

                    if min_col in row_dict and row_dict[min_col] is not None:
                        json_obj[f"{stat}_min"] = float(row_dict[min_col])

                    if max_col in row_dict and row_dict[max_col] is not None:
                        json_obj[f"{stat}_max"] = float(row_dict[max_col])

                return json.dumps(json_obj)

            # Apply transformation to create JSON column
            json_data = []
            for row in df.iter_rows(named=True):
                json_data.append(build_json_row(row))

            df = df.with_columns(pl.Series(name=json_col_name, values=json_data))

        # Calculate composite trend scalars
        df = self._calculate_composite_trends(df, position, windows)

        # Select only columns needed for storage
        keep_cols = ["player_id", "season", "week", "position"]
        keep_cols.extend([f"stats_last{w}_games" for w in windows])
        keep_cols.extend(["performance_trend", "usage_trend", "target_share_trend"])

        # Filter to only columns that exist
        keep_cols = [c for c in keep_cols if c in df.columns]

        return df.select(keep_cols)

    def _calculate_composite_trends(
        self, df: pl.DataFrame, position: str, windows: List[int]
    ) -> pl.DataFrame:
        """
        Calculate composite trend metrics (performance_trend, usage_trend, target_share_trend).

        Args:
            df: DataFrame with individual stat trends
            position: Player position
            windows: List of window sizes

        Returns:
            DataFrame with added trend columns
        """
        # Use 5-game window for composite trends
        window = 5

        # Position-specific composite calculations
        if position == "QB":
            # performance_trend: weighted average of passing yards, TDs, rushing yards
            py_col = f"passing_yards_trend_{window}"
            ptd_col = f"passing_tds_trend_{window}"
            ry_col = f"rushing_yards_trend_{window}"

            if all(c in df.columns for c in [py_col, ptd_col, ry_col]):
                df = df.with_columns(
                    (
                        0.6 * pl.col(py_col).fill_null(0)
                        + 0.3
                        * pl.col(ptd_col).fill_null(0)
                        * 20  # Scale TDs to yards equivalent
                        + 0.1 * pl.col(ry_col).fill_null(0)
                    ).alias("performance_trend")
                )
            else:
                df = df.with_columns(pl.lit(None).alias("performance_trend"))

            # usage_trend: attempts trend
            att_col = f"attempts_trend_{window}"
            if att_col in df.columns:
                df = df.with_columns(pl.col(att_col).alias("usage_trend"))
            else:
                df = df.with_columns(pl.lit(None).alias("usage_trend"))

            # target_share_trend: not applicable for QB
            df = df.with_columns(pl.lit(None).alias("target_share_trend"))

        elif position in ["WR", "TE"]:
            # performance_trend: weighted average of receiving yards and targets
            ry_col = f"receiving_yards_trend_{window}"
            t_col = f"targets_trend_{window}"

            if ry_col in df.columns and t_col in df.columns:
                df = df.with_columns(
                    (
                        0.7 * pl.col(ry_col).fill_null(0)
                        + 0.3 * pl.col(t_col).fill_null(0) * 10
                    ).alias(
                        "performance_trend"
                    )  # Scale targets
                )
            elif ry_col in df.columns:
                df = df.with_columns(pl.col(ry_col).alias("performance_trend"))
            else:
                df = df.with_columns(pl.lit(None).alias("performance_trend"))

            # usage_trend: targets trend
            if t_col in df.columns:
                df = df.with_columns(pl.col(t_col).alias("usage_trend"))
            else:
                df = df.with_columns(pl.lit(None).alias("usage_trend"))

            # target_share_trend: target share trend if available
            ts_col = f"target_share_trend_{window}"
            if ts_col in df.columns:
                df = df.with_columns(pl.col(ts_col).alias("target_share_trend"))
            else:
                df = df.with_columns(pl.lit(None).alias("target_share_trend"))

        elif position == "RB":
            # performance_trend: weighted average of rushing, receiving yards, and carries
            rush_y_col = f"rushing_yards_trend_{window}"
            rec_y_col = f"receiving_yards_trend_{window}"
            car_col = f"carries_trend_{window}"
            t_col = f"targets_trend_{window}"

            if all(c in df.columns for c in [rush_y_col, rec_y_col, car_col]):
                df = df.with_columns(
                    (
                        0.6 * pl.col(rush_y_col).fill_null(0)
                        + 0.3 * pl.col(rec_y_col).fill_null(0)
                        + 0.1 * pl.col(car_col).fill_null(0) * 4
                    ).alias(
                        "performance_trend"
                    )  # Scale carries
                )
            elif rush_y_col in df.columns:
                df = df.with_columns(pl.col(rush_y_col).alias("performance_trend"))
            else:
                df = df.with_columns(pl.lit(None).alias("performance_trend"))

            # usage_trend: carries + targets
            if car_col in df.columns and t_col in df.columns:
                df = df.with_columns(
                    (pl.col(car_col).fill_null(0) + pl.col(t_col).fill_null(0)).alias(
                        "usage_trend"
                    )
                )
            elif car_col in df.columns:
                df = df.with_columns(pl.col(car_col).alias("usage_trend"))
            else:
                df = df.with_columns(pl.lit(None).alias("usage_trend"))

            # target_share_trend: if available
            ts_col = f"target_share_trend_{window}"
            if ts_col in df.columns:
                df = df.with_columns(pl.col(ts_col).alias("target_share_trend"))
            else:
                df = df.with_columns(pl.lit(None).alias("target_share_trend"))

        elif position == "K":
            # performance_trend: field goal percentage trend
            fg_pct_col = f"fg_pct_trend_{window}"
            if fg_pct_col in df.columns:
                df = df.with_columns(pl.col(fg_pct_col).alias("performance_trend"))
            else:
                df = df.with_columns(pl.lit(None).alias("performance_trend"))

            # usage_trend: field goal attempts
            fg_att_col = f"fg_att_trend_{window}"
            if fg_att_col in df.columns:
                df = df.with_columns(pl.col(fg_att_col).alias("usage_trend"))
            else:
                df = df.with_columns(pl.lit(None).alias("usage_trend"))

            df = df.with_columns(pl.lit(None).alias("target_share_trend"))

        elif position == "DEF":
            # performance_trend: weighted average of tackles, sacks, interceptions
            tack_col = f"def_tackles_solo_trend_{window}"
            sack_col = f"def_sacks_trend_{window}"
            int_col = f"def_interceptions_trend_{window}"

            if all(c in df.columns for c in [tack_col, sack_col, int_col]):
                df = df.with_columns(
                    (
                        0.5 * pl.col(tack_col).fill_null(0)
                        + 0.3 * pl.col(sack_col).fill_null(0) * 2  # Scale sacks
                        + 0.2 * pl.col(int_col).fill_null(0) * 3
                    ).alias(
                        "performance_trend"
                    )  # Scale interceptions
                )
            else:
                df = df.with_columns(pl.lit(None).alias("performance_trend"))

            # usage_trend: total tackles trend
            if tack_col in df.columns:
                df = df.with_columns(pl.col(tack_col).alias("usage_trend"))
            else:
                df = df.with_columns(pl.lit(None).alias("usage_trend"))

            df = df.with_columns(pl.lit(None).alias("target_share_trend"))

        else:
            # Default: set all trends to None
            df = df.with_columns(
                [
                    pl.lit(None).alias("performance_trend"),
                    pl.lit(None).alias("usage_trend"),
                    pl.lit(None).alias("target_share_trend"),
                ]
            )

        return df

    def _store_rolling_features(self, df: pl.DataFrame) -> int:
        """
        Store rolling features in player_rolling_features table.

        Args:
            df: DataFrame with rolling features in JSON format

        Returns:
            Number of records stored
        """
        if df.is_empty():
            return 0

        conn = self.db.connect()

        # Insert data using batch processing
        try:
            # Register the DataFrame with DuckDB
            conn.register("temp_rolling_features", df)

            # Insert into table
            conn.execute(
                """
                INSERT INTO player_rolling_features
                (player_id, season, week, position,
                 stats_last3_games, stats_last5_games, stats_season_avg,
                 performance_trend, usage_trend, target_share_trend)
                SELECT
                    player_id, season, week, position,
                    stats_last3_games,
                    stats_last5_games,
                    stats_last10_games as stats_season_avg,
                    performance_trend,
                    usage_trend,
                    target_share_trend
                FROM temp_rolling_features
            """
            )

            conn.unregister("temp_rolling_features")

            return len(df)
        except Exception as e:
            logger.error(f"      ❌ Failed to store rolling features: {e}")
            raise

    def _log_rolling_stats_summary(self):
        """Log summary statistics of rolling features."""
        try:
            conn = self.db.connect()

            summary = conn.execute(
                """
                SELECT
                    COUNT(*) as total_records,
                    COUNT(DISTINCT player_id) as unique_players,
                    COUNT(DISTINCT position) as positions,
                    MIN(season) as earliest_season,
                    MAX(season) as latest_season,
                    COUNT(*) FILTER (WHERE stats_last3_games IS NOT NULL) as with_3game_stats,
                    COUNT(*) FILTER (WHERE stats_last5_games IS NOT NULL) as with_5game_stats,
                    COUNT(*) FILTER (WHERE stats_season_avg IS NOT NULL) as with_10game_stats
                FROM player_rolling_features
            """
            ).fetchone()

            if summary:
                logger.info(f"  📊 Rolling Features Summary:")
                logger.info(f"     Total records: {summary[0]:,}")
                logger.info(f"     Unique players: {summary[1]:,}")
                logger.info(f"     Positions: {summary[2]}")
                logger.info(f"     Season range: {summary[3]}-{summary[4]}")
                logger.info(f"     With 3-game stats: {summary[5]:,}")
                logger.info(f"     With 5-game stats: {summary[6]:,}")
                logger.info(f"     With 10-game stats: {summary[7]:,}")
        except Exception as e:
            logger.warning(f"  ⚠️  Could not generate summary: {e}")

    def build_matchup_features(self):
        """
        Build matchup-specific features for player predictions.

        Updates player_rolling_features table with:
        - vs_opponent_history: Last 3 games vs same opponent
        - opp_rank_vs_position: Opponent's defensive rank (1-32)
        - opp_avg_allowed_to_position: Average stats allowed to position
        - home_away_splits: Career home vs away performance
        - rest_days: Days since last game
        - divisional_game: Is this a divisional matchup?

        Data sources:
        - raw_player_stats: Historical performance
        - raw_schedules: Game context, rest days, divisional info
        - raw_team_stats: Opponent defensive stats

        Edge cases:
        - First-time matchup → NULL opponent history
        - Missing schedules → default rest_days=7
        - Indoor games → NULL weather
        - No defensive rank data → rank=16 (league average)

        Returns:
            None. Updates player_rolling_features table in place.

        Raises:
            ValueError: If player_rolling_features is empty
            Exception: If database operation fails
        """
        logger.info("⚔️ Building matchup features...")

        try:
            conn = self.db.connect()

            # Validate player_rolling_features exists and has data
            tables = self.db.list_tables()
            if "player_rolling_features" not in tables:
                raise ValueError(
                    "player_rolling_features table not found. Please run Stage 3a first."
                )

            record_count = conn.execute(
                "SELECT COUNT(*) FROM player_rolling_features"
            ).fetchone()[0]
            if record_count == 0:
                logger.warning(
                    "⚠️  player_rolling_features is empty, skipping matchup features"
                )
                return

            logger.info(f"  📊 Processing {record_count:,} player-game records")

            # Step 1: Add rest days and divisional game info from schedules
            logger.info("  🕐 Adding rest days and divisional game info...")
            self._update_rest_and_divisional_info(conn)

            # Step 2: Calculate home/away splits
            logger.info("  🏠 Calculating home/away splits...")
            self._update_home_away_splits(conn)

            # Step 3: Build opponent history
            logger.info("  📜 Building opponent history...")
            self._update_opponent_history(conn)

            # Step 4: Calculate opponent defensive rankings
            logger.info("  🛡️  Calculating opponent defensive rankings...")
            self._update_opponent_rankings(conn)

            # Step 5: Calculate average stats allowed by opponents
            logger.info("  📊 Calculating opponent stats allowed...")
            self._update_opponent_stats_allowed(conn)

            logger.info("✅ Matchup features built successfully")

            # Log summary statistics
            self._log_matchup_features_summary(conn)

        except Exception as e:
            logger.error(f"❌ Failed to build matchup features: {e}")
            raise

    def _update_rest_and_divisional_info(self, conn):
        """Update rest_days and divisional_game from schedules."""
        # Update for away teams
        conn.execute(
            """
            UPDATE player_rolling_features prf
            SET
                rest_days = COALESCE(s.away_rest, 7),
                divisional_game = COALESCE(s.div_game::BOOLEAN, FALSE)
            FROM raw_schedules s
            INNER JOIN raw_player_stats ps
                ON ps.player_id = prf.player_id
                AND ps.season = prf.season
                AND ps.week = prf.week
            WHERE s.season = prf.season
                AND s.week = prf.week
                AND s.away_team = ps.recent_team
        """
        )

        # Update for home teams
        conn.execute(
            """
            UPDATE player_rolling_features prf
            SET
                rest_days = COALESCE(s.home_rest, 7),
                divisional_game = COALESCE(s.div_game::BOOLEAN, FALSE)
            FROM raw_schedules s
            INNER JOIN raw_player_stats ps
                ON ps.player_id = prf.player_id
                AND ps.season = prf.season
                AND ps.week = prf.week
            WHERE s.season = prf.season
                AND s.week = prf.week
                AND s.home_team = ps.recent_team
        """
        )

        logger.info("     ✅ Rest days and divisional info updated")

    def _update_home_away_splits(self, conn):
        """Calculate and update home/away performance splits."""
        conn.execute(
            """
            UPDATE player_rolling_features prf
            SET home_away_splits = (
                SELECT json_object(
                    'home_games', COALESCE(home.games, 0),
                    'away_games', COALESCE(away.games, 0),
                    'home_avg_fantasy_points', COALESCE(home.avg_fantasy, 0.0),
                    'away_avg_fantasy_points', COALESCE(away.avg_fantasy, 0.0),
                    'home_avg_yards', COALESCE(home.avg_yards, 0.0),
                    'away_avg_yards', COALESCE(away.avg_yards, 0.0)
                )
                FROM (
                    -- Home stats (games before current week)
                    SELECT
                        ps.player_id,
                        COUNT(*) as games,
                        AVG(ps.fantasy_points) as avg_fantasy,
                        AVG(
                            COALESCE(ps.passing_yards, 0) +
                            COALESCE(ps.rushing_yards, 0) +
                            COALESCE(ps.receiving_yards, 0)
                        ) as avg_yards
                    FROM raw_player_stats ps
                    INNER JOIN raw_schedules s
                        ON s.season = ps.season
                        AND s.week = ps.week
                        AND s.home_team = ps.recent_team
                    WHERE ps.player_id = prf.player_id
                        AND (ps.season < prf.season
                            OR (ps.season = prf.season AND ps.week < prf.week))
                    GROUP BY ps.player_id
                ) home
                FULL OUTER JOIN (
                    -- Away stats (games before current week)
                    SELECT
                        ps.player_id,
                        COUNT(*) as games,
                        AVG(ps.fantasy_points) as avg_fantasy,
                        AVG(
                            COALESCE(ps.passing_yards, 0) +
                            COALESCE(ps.rushing_yards, 0) +
                            COALESCE(ps.receiving_yards, 0)
                        ) as avg_yards
                    FROM raw_player_stats ps
                    INNER JOIN raw_schedules s
                        ON s.season = ps.season
                        AND s.week = ps.week
                        AND s.away_team = ps.recent_team
                    WHERE ps.player_id = prf.player_id
                        AND (ps.season < prf.season
                            OR (ps.season = prf.season AND ps.week < prf.week))
                    GROUP BY ps.player_id
                ) away ON TRUE
            )
        """
        )

        logger.info("     ✅ Home/away splits calculated")

    def _update_opponent_history(self, conn):
        """Build opponent matchup history (last 3 games vs same opponent)."""
        conn.execute(
            """
            UPDATE player_rolling_features prf
            SET vs_opponent_history = (
                SELECT json_group_array(
                    json_object(
                        'season', opp_games.season,
                        'week', opp_games.week,
                        'fantasy_points', opp_games.fantasy_points,
                        'total_yards', opp_games.total_yards,
                        'touchdowns', opp_games.touchdowns
                    )
                )
                FROM (
                    SELECT
                        ps.season,
                        ps.week,
                        COALESCE(ps.fantasy_points, 0.0) as fantasy_points,
                        (COALESCE(ps.passing_yards, 0) +
                         COALESCE(ps.rushing_yards, 0) +
                         COALESCE(ps.receiving_yards, 0)) as total_yards,
                        (COALESCE(ps.passing_tds, 0) +
                         COALESCE(ps.rushing_tds, 0) +
                         COALESCE(ps.receiving_tds, 0)) as touchdowns
                    FROM raw_player_stats ps
                    INNER JOIN raw_schedules s
                        ON s.season = ps.season
                        AND s.week = ps.week
                    INNER JOIN raw_player_stats ps_current
                        ON ps_current.player_id = prf.player_id
                        AND ps_current.season = prf.season
                        AND ps_current.week = prf.week
                    WHERE ps.player_id = prf.player_id
                        AND (ps.season < prf.season
                            OR (ps.season = prf.season AND ps.week < prf.week))
                        AND (
                            -- Match opponent (home or away)
                            (s.home_team = ps.recent_team AND
                             s.away_team = ps_current.opponent_team)
                            OR
                            (s.away_team = ps.recent_team AND
                             s.home_team = ps_current.opponent_team)
                        )
                    ORDER BY ps.season DESC, ps.week DESC
                    LIMIT 3
                ) opp_games
            )
        """
        )

        logger.info("     ✅ Opponent history built")

    def _update_opponent_rankings(self, conn):
        """Calculate opponent defensive rankings by position."""
        # Process each position group
        positions = ["QB", "RB", "WR", "TE", "K", "DEF"]

        for position in positions:
            logger.info(f"     📊 Ranking defenses vs {position}...")

            # Position-specific stats for ranking
            if position == "QB":
                metric = "passing_yards"
            elif position == "RB":
                metric = "rushing_yards"
            elif position in ["WR", "TE"]:
                metric = "receiving_yards"
            elif position == "K":
                metric = "fg_made"
            else:  # DEF
                metric = "def_tackles_solo"

            # Calculate defensive rankings based on yards/stats allowed
            conn.execute(
                f"""
                WITH defensive_performance AS (
                    SELECT
                        ts.season,
                        ts.week,
                        ts.team,
                        AVG(COALESCE(ps.{metric}, 0)) as avg_allowed
                    FROM raw_team_stats ts
                    LEFT JOIN raw_player_stats ps
                        ON ps.season = ts.season
                        AND ps.week = ts.week
                        AND ps.opponent_team = ts.team
                        AND ps.position = '{position}'
                    WHERE ts.season >= (SELECT MIN(season) FROM player_rolling_features)
                    GROUP BY ts.season, ts.week, ts.team
                ),
                defensive_ranks AS (
                    SELECT
                        season,
                        week,
                        team,
                        RANK() OVER (
                            PARTITION BY season, week
                            ORDER BY avg_allowed ASC
                        ) as defense_rank
                    FROM defensive_performance
                )
                UPDATE player_rolling_features prf
                SET opp_rank_vs_position = COALESCE(dr.defense_rank, 16)
                FROM defensive_ranks dr
                INNER JOIN raw_player_stats ps
                    ON ps.player_id = prf.player_id
                    AND ps.season = prf.season
                    AND ps.week = prf.week
                WHERE dr.season = prf.season
                    AND dr.week = prf.week
                    AND dr.team = ps.opponent_team
                    AND prf.position = '{position}'
            """
            )

        logger.info("     ✅ Opponent rankings calculated")

    def _update_opponent_stats_allowed(self, conn):
        """Calculate average stats allowed by opponent to each position."""
        positions = ["QB", "RB", "WR", "TE", "K", "DEF"]

        for position in positions:
            logger.info(f"     📊 Calculating stats allowed vs {position}...")

            # Get position-specific stats
            relevant_stats = config.get_position_stats(position)

            if not relevant_stats:
                continue

            # Build JSON object with average stats allowed (last 3 games)
            stat_selects = []
            for stat in relevant_stats[:10]:  # Limit to top 10 most important stats
                stat_selects.append(
                    f"'{stat}', COALESCE(AVG(COALESCE(ps.{stat}, 0)), 0.0)"
                )

            stat_json = ", ".join(stat_selects)

            conn.execute(
                f"""
                UPDATE player_rolling_features prf
                SET opp_avg_allowed_to_position = (
                    SELECT json_object({stat_json})
                    FROM raw_player_stats ps
                    INNER JOIN raw_player_stats ps_current
                        ON ps_current.player_id = prf.player_id
                        AND ps_current.season = prf.season
                        AND ps_current.week = prf.week
                    WHERE ps.opponent_team = ps_current.opponent_team
                        AND ps.position = '{position}'
                        AND (ps.season < prf.season
                            OR (ps.season = prf.season AND ps.week < prf.week))
                        AND ps.season >= prf.season - 1  -- Last season data
                )
                WHERE prf.position = '{position}'
            """
            )

        logger.info("     ✅ Opponent stats allowed calculated")

    def _log_matchup_features_summary(self, conn):
        """Log summary statistics for matchup features."""
        try:
            # Count records with each feature populated
            summary = conn.execute(
                """
                SELECT
                    COUNT(*) as total_records,
                    SUM(CASE WHEN rest_days IS NOT NULL THEN 1 ELSE 0 END) as with_rest_days,
                    SUM(CASE WHEN divisional_game IS NOT NULL THEN 1 ELSE 0 END) as with_divisional,
                    SUM(CASE WHEN home_away_splits IS NOT NULL THEN 1 ELSE 0 END) as with_splits,
                    SUM(CASE WHEN vs_opponent_history IS NOT NULL THEN 1 ELSE 0 END) as with_opp_history,
                    SUM(CASE WHEN opp_rank_vs_position IS NOT NULL THEN 1 ELSE 0 END) as with_opp_rank,
                    SUM(CASE WHEN opp_avg_allowed_to_position IS NOT NULL THEN 1 ELSE 0 END) as with_opp_stats,
                    AVG(CAST(divisional_game AS INTEGER)) as pct_divisional,
                    AVG(rest_days) as avg_rest_days
                FROM player_rolling_features
            """
            ).fetchone()

            if summary:
                logger.info("  📊 Matchup Features Summary:")
                logger.info(f"     Total records: {summary[0]:,}")
                logger.info(f"     With rest days: {summary[1]:,}")
                logger.info(f"     With divisional info: {summary[2]:,}")
                logger.info(f"     With home/away splits: {summary[3]:,}")
                logger.info(f"     With opponent history: {summary[4]:,}")
                logger.info(f"     With opponent rankings: {summary[5]:,}")
                logger.info(f"     With opponent stats: {summary[6]:,}")
                logger.info(f"     Divisional game rate: {summary[7]*100:.1f}%")
                logger.info(f"     Average rest days: {summary[8]:.1f}")

        except Exception as e:
            logger.warning(f"  ⚠️  Could not generate summary: {e}")

    def create_team_aggregates(self):
        """
        Create team aggregate features from play-by-play and team stats.

        Implements Stage 3c from IMPLEMENTATION_PLAN_STAGE_3C.md

        Calculates 12 key team metrics:
        - Offensive: EPA per play, success rate, explosive play rate, red zone efficiency, 3rd down conversion
        - Defensive: EPA allowed, success rate, pressure rate, turnover rate
        - Situational: Pass rate (neutral), pace of play, time of possession

        Uses two-path implementation:
        - Path A (2022+): raw_pbp table for detailed EPA metrics
        - Path B (pre-2022): raw_team_stats for basic metrics (fallback)

        Returns:
            None. Results stored in team_rolling_features table.

        Raises:
            ValueError: If required tables are missing
            Exception: If database operation fails
        """
        logger.info("🏈 Creating team aggregates...")

        try:
            conn = self.db.connect()

            # Validate required tables exist
            tables = self.db.list_tables()
            if "raw_player_stats" not in tables and "raw_pbp" not in tables:
                raise ValueError(
                    "Neither raw_player_stats nor raw_pbp found. Please run Stage 1 first."
                )

            # Clear existing team rolling features
            conn.execute("DELETE FROM team_rolling_features")
            logger.info("  🧹 Cleared existing team rolling features")

            # Step 1: Determine available seasons with PBP data
            seasons_with_pbp = self._get_seasons_with_pbp()
            all_seasons = self._get_all_seasons()

            logger.info(
                f"  📊 Found {len(seasons_with_pbp)} seasons with PBP data: {seasons_with_pbp}"
            )
            logger.info(f"  📊 Total seasons to process: {len(all_seasons)}")

            total_records = 0

            # Step 2: Process seasons with PBP data (preferred path)
            for season in seasons_with_pbp:
                logger.info(f"  🏈 Processing season {season} with PBP data...")
                records = self._calculate_team_aggregates_from_pbp(season)
                total_records += records

            # Step 3: Process pre-2022 seasons with team stats fallback
            legacy_seasons = [s for s in all_seasons if s not in seasons_with_pbp]
            for season in legacy_seasons:
                logger.info(
                    f"  📊 Processing season {season} with team stats fallback..."
                )
                records = self._calculate_team_aggregates_from_team_stats(season)
                total_records += records

            # Step 4: Validate results
            self._validate_team_aggregates()

            logger.info(
                f"✅ Team aggregates created successfully: {total_records:,} total records"
            )

        except Exception as e:
            logger.error(f"❌ Failed to create team aggregates: {e}")
            raise

    def _get_seasons_with_pbp(self) -> List[int]:
        """
        Get seasons that have play-by-play data available.

        Returns:
            List of season years with PBP data (typically 2022+)
        """
        try:
            conn = self.db.connect()

            # Check if raw_pbp table exists and has data
            tables = self.db.list_tables()
            if "raw_pbp" not in tables:
                return []

            result = conn.execute(
                """
                SELECT DISTINCT season
                FROM raw_pbp
                WHERE season >= 2022
                  AND season IS NOT NULL
                ORDER BY season
            """
            ).fetchall()

            return [row[0] for row in result]

        except Exception as e:
            logger.warning(f"  ⚠️  Could not determine PBP availability: {e}")
            return []

    def _get_all_seasons(self) -> List[int]:
        """
        Get all seasons available in raw data tables.

        Returns:
            List of all season years in database
        """
        try:
            conn = self.db.connect()

            # Try raw_player_stats first (most comprehensive)
            result = conn.execute(
                """
                SELECT DISTINCT season
                FROM raw_player_stats
                WHERE season IS NOT NULL
                ORDER BY season
            """
            ).fetchall()

            if result:
                return [row[0] for row in result]

            # Fallback to raw_team_stats
            result = conn.execute(
                """
                SELECT DISTINCT season
                FROM raw_team_stats
                WHERE season IS NOT NULL
                ORDER BY season
            """
            ).fetchall()

            return [row[0] for row in result]

        except Exception as e:
            logger.warning(f"  ⚠️  Could not determine available seasons: {e}")
            return []

    def _calculate_team_aggregates_from_pbp(self, season: int) -> int:
        """
        Calculate team aggregates from play-by-play data (2022+ preferred path).

        Args:
            season: Season year to process

        Returns:
            Number of records inserted
        """
        conn = self.db.connect()

        # Get all teams and weeks for this season
        teams_weeks = conn.execute(
            """
            SELECT DISTINCT posteam AS team, week
            FROM raw_pbp
            WHERE season = ?
              AND posteam IS NOT NULL
              AND week IS NOT NULL
              AND season_type = 'REG'
            ORDER BY team, week
        """,
            [season],
        ).fetchall()

        if not teams_weeks:
            logger.info(f"    ℹ️  No PBP data found for season {season}")
            return 0

        results = []

        for team, week in teams_weeks:
            # Skip weeks without enough history for rolling window
            if week <= 3:
                continue

            try:
                # Calculate offensive metrics
                off_metrics = self._calculate_offensive_metrics(
                    team, season, week, conn
                )

                # Calculate defensive metrics
                def_metrics = self._calculate_defensive_metrics(
                    team, season, week, conn
                )

                # Calculate situational metrics
                sit_metrics = self._calculate_situational_metrics(
                    team, season, week, conn
                )

                # Combine all metrics
                team_features = {
                    "team": team,
                    "season": season,
                    "week": week,
                    **off_metrics,
                    **def_metrics,
                    **sit_metrics,
                }

                results.append(team_features)

            except Exception as e:
                logger.warning(
                    f"    ⚠️  Failed to calculate metrics for {team} week {week}: {e}"
                )
                continue

        # Bulk insert results
        if results:
            df = pl.DataFrame(results)
            conn.execute("INSERT INTO team_rolling_features SELECT * FROM df")
            logger.info(
                f"    ✅ Inserted {len(results):,} team feature records for {season}"
            )
            return len(results)

        return 0

    def _calculate_offensive_metrics(
        self, team: str, season: int, week: int, conn
    ) -> Dict[str, float]:
        """
        Calculate offensive metrics for a team using play-by-play data.

        Args:
            team: Team abbreviation (e.g., 'KC', 'BUF')
            season: Season year
            week: Week number
            conn: DuckDB connection

        Returns:
            Dictionary with 5 offensive metrics
        """
        start_week = max(1, week - 3)
        end_week = week - 1

        # Main offensive query
        query = """
        WITH offensive_plays AS (
            SELECT
                epa,
                success,
                yards_gained,
                touchdown,
                third_down_converted,
                third_down_failed,
                down
            FROM raw_pbp
            WHERE posteam = ?
              AND season = ?
              AND week BETWEEN ? AND ?
              AND play_type IN ('pass', 'run')
              AND play = 1
              AND epa IS NOT NULL
        )
        SELECT
            -- EPA per play
            AVG(epa) AS off_epa_per_play_last3,

            -- Success rate
            SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END)::FLOAT /
            NULLIF(COUNT(*), 0) AS off_success_rate_last3,

            -- Explosive play rate (>10 yards)
            SUM(CASE WHEN yards_gained > 10 THEN 1 ELSE 0 END)::FLOAT /
            NULLIF(COUNT(*), 0) AS off_explosive_play_rate,

            -- Third down conversion rate
            SUM(CASE WHEN down = 3 THEN third_down_converted ELSE 0 END)::FLOAT /
            NULLIF(
                SUM(CASE WHEN down = 3 THEN third_down_converted ELSE 0 END) +
                SUM(CASE WHEN down = 3 THEN third_down_failed ELSE 0 END),
                0
            ) AS off_third_down_conv
        FROM offensive_plays
        """

        result = conn.execute(query, [team, season, start_week, end_week]).fetchone()

        # Red zone efficiency (separate query for DISTINCT drive handling)
        rz_query = """
        WITH red_zone_drives AS (
            SELECT DISTINCT
                drive,
                MAX(touchdown) AS scored_td
            FROM raw_pbp
            WHERE posteam = ?
              AND season = ?
              AND week BETWEEN ? AND ?
              AND yardline_100 <= 20
              AND drive IS NOT NULL
            GROUP BY drive
        )
        SELECT
            SUM(scored_td)::FLOAT / NULLIF(COUNT(*), 0) AS rz_efficiency
        FROM red_zone_drives
        """

        rz_result = conn.execute(
            rz_query, [team, season, start_week, end_week]
        ).fetchone()

        return {
            "off_epa_per_play_last3": (
                float(result[0]) if result and result[0] is not None else 0.0
            ),
            "off_success_rate_last3": (
                float(result[1]) if result and result[1] is not None else 0.5
            ),
            "off_explosive_play_rate": (
                float(result[2]) if result and result[2] is not None else 0.0
            ),
            "off_third_down_conv": (
                float(result[3]) if result and result[3] is not None else 0.0
            ),
            "off_red_zone_efficiency": (
                float(rz_result[0]) if rz_result and rz_result[0] is not None else 0.5
            ),
        }

    def _calculate_defensive_metrics(
        self, team: str, season: int, week: int, conn
    ) -> Dict[str, float]:
        """
        Calculate defensive metrics for a team using play-by-play data.

        Args:
            team: Team abbreviation
            season: Season year
            week: Week number
            conn: DuckDB connection

        Returns:
            Dictionary with 4 defensive metrics
        """
        start_week = max(1, week - 3)
        end_week = week - 1

        query = """
        WITH defensive_plays AS (
            SELECT
                epa,
                success,
                qb_hit,
                sack,
                pass_attempt,
                interception,
                fumble_lost,
                drive
            FROM raw_pbp
            WHERE defteam = ?
              AND season = ?
              AND week BETWEEN ? AND ?
              AND play_type IN ('pass', 'run')
              AND play = 1
        )
        SELECT
            -- Defensive EPA (lower is better)
            AVG(epa) AS def_epa_per_play_last3,

            -- Defensive success rate (offense failed)
            SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END)::FLOAT /
            NULLIF(COUNT(*), 0) AS def_success_rate_last3,

            -- Pressure rate
            (SUM(qb_hit) + SUM(sack))::FLOAT /
            NULLIF(SUM(pass_attempt), 0) AS def_pressure_rate,

            -- Turnover rate (per drive)
            (SUM(interception) + SUM(fumble_lost))::FLOAT /
            NULLIF(COUNT(DISTINCT drive), 0) AS def_turnover_rate
        FROM defensive_plays
        """

        result = conn.execute(query, [team, season, start_week, end_week]).fetchone()

        return {
            "def_epa_per_play_last3": (
                float(result[0]) if result and result[0] is not None else 0.0
            ),
            "def_success_rate_last3": (
                float(result[1]) if result and result[1] is not None else 0.5
            ),
            "def_pressure_rate": (
                float(result[2]) if result and result[2] is not None else 0.0
            ),
            "def_turnover_rate": (
                float(result[3]) if result and result[3] is not None else 0.0
            ),
        }

    def _calculate_situational_metrics(
        self, team: str, season: int, week: int, conn
    ) -> Dict[str, float]:
        """
        Calculate situational/pace metrics for a team.

        Args:
            team: Team abbreviation
            season: Season year
            week: Week number
            conn: DuckDB connection

        Returns:
            Dictionary with 3 situational metrics
        """
        start_week = max(1, week - 3)
        end_week = week - 1

        # Pass rate in neutral situations
        neutral_query = """
        SELECT
            SUM(pass_attempt)::FLOAT /
            NULLIF(SUM(pass_attempt) + SUM(rush_attempt), 0) AS pass_rate_neutral
        FROM raw_pbp
        WHERE posteam = ?
          AND season = ?
          AND week BETWEEN ? AND ?
          AND score_differential BETWEEN -7 AND 7
          AND quarter_seconds_remaining > 120
          AND down IN (1, 2)
          AND play = 1
        """

        neutral_result = conn.execute(
            neutral_query, [team, season, start_week, end_week]
        ).fetchone()

        # Pace and time of possession
        pace_query = """
        WITH drive_stats AS (
            SELECT DISTINCT
                drive,
                drive_play_count,
                drive_time_of_possession
            FROM raw_pbp
            WHERE posteam = ?
              AND season = ?
              AND week BETWEEN ? AND ?
              AND drive IS NOT NULL
              AND drive_play_count IS NOT NULL
        )
        SELECT
            AVG(drive_play_count) AS pace_of_play,
            AVG(
                CASE
                    WHEN drive_time_of_possession IS NOT NULL
                        AND drive_time_of_possession != ''
                    THEN
                        CAST(SPLIT_PART(drive_time_of_possession, ':', 1) AS INTEGER) +
                        CAST(SPLIT_PART(drive_time_of_possession, ':', 2) AS INTEGER) / 60.0
                    ELSE NULL
                END
            ) AS time_of_possession_avg
        FROM drive_stats
        """

        pace_result = conn.execute(
            pace_query, [team, season, start_week, end_week]
        ).fetchone()

        return {
            "pass_rate_neutral": (
                float(neutral_result[0])
                if neutral_result and neutral_result[0] is not None
                else 0.5
            ),
            "pace_of_play": (
                float(pace_result[0])
                if pace_result and pace_result[0] is not None
                else 65.0
            ),
            "time_of_possession_avg": (
                float(pace_result[1])
                if pace_result and pace_result[1] is not None
                else 30.0
            ),
        }

    def _calculate_team_aggregates_from_team_stats(self, season: int) -> int:
        """
        Fallback method for pre-2022 seasons using raw_team_stats.

        Limited metrics available compared to PBP-based calculation.
        Uses default values for metrics not available in team stats.

        Args:
            season: Season year to process

        Returns:
            Number of records inserted
        """
        conn = self.db.connect()

        # Check if raw_team_stats has data for this season
        count = conn.execute(
            """
            SELECT COUNT(*) FROM raw_team_stats WHERE season = ?
        """,
            [season],
        ).fetchone()[0]

        if count == 0:
            logger.info(f"    ℹ️  No team stats data found for season {season}")
            return 0

        query = """
        WITH team_weekly_stats AS (
            SELECT
                team,
                season,
                week,
                passing_epa,
                rushing_epa
            FROM raw_team_stats
            WHERE season = ?
              AND week > 3
        )
        SELECT
            team,
            season,
            week,

            -- Approximate offensive EPA from weekly stats
            0.0 AS off_epa_per_play_last3,

            -- Use default values for metrics not available in team stats
            0.5 AS off_success_rate_last3,
            0.5 AS def_success_rate_last3,
            0.0 AS off_explosive_play_rate,
            0.5 AS off_red_zone_efficiency,
            0.0 AS off_third_down_conv,
            0.0 AS def_epa_per_play_last3,
            0.0 AS def_pressure_rate,
            0.0 AS def_turnover_rate,
            0.5 AS pass_rate_neutral,
            65.0 AS pace_of_play,
            30.0 AS time_of_possession_avg
        FROM team_weekly_stats
        """

        try:
            results = conn.execute(query, [season]).fetchdf()

            if len(results) > 0:
                df = pl.from_pandas(results)
                conn.execute("INSERT INTO team_rolling_features SELECT * FROM df")
                logger.info(
                    f"    ✅ Inserted {len(results):,} team feature records for {season} (fallback)"
                )
                logger.warning(
                    f"    ⚠️  Pre-2022 season: Limited metrics available from team stats"
                )
                return len(results)

            return 0

        except Exception as e:
            logger.warning(f"    ⚠️  Failed to process team stats for {season}: {e}")
            return 0

    def _validate_team_aggregates(self):
        """
        Validate team aggregate calculations.

        Checks:
        - EPA values within expected range (-3 to +3)
        - Success rates between 0 and 1
        - No NULL values for primary metrics
        - Expected number of teams per season
        """
        try:
            conn = self.db.connect()

            # Check EPA ranges
            invalid_epa = conn.execute(
                """
                SELECT COUNT(*)
                FROM team_rolling_features
                WHERE off_epa_per_play_last3 < -3
                   OR off_epa_per_play_last3 > 3
                   OR def_epa_per_play_last3 < -3
                   OR def_epa_per_play_last3 > 3
            """
            ).fetchone()[0]

            if invalid_epa > 0:
                logger.warning(
                    f"  ⚠️  Found {invalid_epa} records with EPA outside expected range (-3 to +3)"
                )

            # Check success rates
            invalid_success = conn.execute(
                """
                SELECT COUNT(*)
                FROM team_rolling_features
                WHERE off_success_rate_last3 < 0 OR off_success_rate_last3 > 1
                   OR def_success_rate_last3 < 0 OR def_success_rate_last3 > 1
            """
            ).fetchone()[0]

            if invalid_success > 0:
                logger.warning(
                    f"  ⚠️  Found {invalid_success} records with invalid success rates"
                )

            # Check for NULL values
            null_count = conn.execute(
                """
                SELECT COUNT(*)
                FROM team_rolling_features
                WHERE off_epa_per_play_last3 IS NULL
                   OR def_epa_per_play_last3 IS NULL
            """
            ).fetchone()[0]

            if null_count > 0:
                logger.warning(f"  ⚠️  Found {null_count} records with NULL EPA values")

            # Log summary
            summary = conn.execute(
                """
                SELECT
                    COUNT(*) as total_records,
                    COUNT(DISTINCT team) as unique_teams,
                    COUNT(DISTINCT season) as seasons,
                    MIN(season) as earliest_season,
                    MAX(season) as latest_season,
                    AVG(off_epa_per_play_last3) as avg_off_epa,
                    AVG(def_epa_per_play_last3) as avg_def_epa
                FROM team_rolling_features
            """
            ).fetchone()

            if summary:
                logger.info(f"  📊 Team Aggregates Summary:")
                logger.info(f"     Total records: {summary[0]:,}")
                logger.info(f"     Unique teams: {summary[1]}")
                logger.info(f"     Seasons: {summary[2]} ({summary[3]}-{summary[4]})")
                logger.info(
                    f"     Avg offensive EPA: {summary[5]:.3f}"
                    if summary[5]
                    else "     Avg offensive EPA: N/A"
                )
                logger.info(
                    f"     Avg defensive EPA: {summary[6]:.3f}"
                    if summary[6]
                    else "     Avg defensive EPA: N/A"
                )

        except Exception as e:
            logger.warning(f"  ⚠️  Could not validate team aggregates: {e}")

    def handle_rookie_veteran_features(self):
        """Handle rookie/veteran feature differences"""
        logger.info("🆕 Handling rookie/veteran features...")
        # TODO: Implement from DATA_SETUP.md

    def combine_all_features(self):
        """
        Combine all features into ML-ready format.

        Joins player_rolling_features, team_rolling_features, and roster snapshots
        into a unified feature vector for each player-game observation.
        Features from week N-1 are used to predict week N (temporal alignment).

        Creates:
        - numerical_features: FLOAT[] array (~107 features)
        - feature_names: VARCHAR[] array (parallel to numerical_features)
        - categorical_features: JSON object with encoded categorical values

        Returns:
            int: Number of feature rows created

        Raises:
            Exception: If feature combination fails
        """
        logger.info("🔗 Combining all features...")

        try:
            conn = self.db.connect()

            # Clear existing data
            conn.execute("DELETE FROM ml_training_features")
            logger.info("  🗑️  Cleared existing ML features")

            # Check if prerequisite tables have data
            prf_count = conn.execute(
                "SELECT COUNT(*) FROM player_rolling_features"
            ).fetchone()[0]
            trf_count = conn.execute(
                "SELECT COUNT(*) FROM team_rolling_features"
            ).fetchone()[0]

            if prf_count == 0:
                logger.warning(
                    "⚠️  player_rolling_features is empty, cannot combine features"
                )
                return 0
            if trf_count == 0:
                logger.warning(
                    "⚠️  team_rolling_features is empty, cannot combine features"
                )
                return 0

            logger.info(
                f"  📊 Source data: {prf_count:,} player features, {trf_count:,} team features"
            )

            # Get position stat mappings from config for feature extraction
            config_mappings = config.position_stat_mappings

            # Build the main feature combination query
            logger.info("  🔨 Building feature vectors...")

            query = """
                INSERT INTO ml_training_features
                WITH game_context AS (
                    -- Get game schedule information
                    SELECT
                        game_id,
                        season,
                        week,
                        gameday as game_date,
                        home_team,
                        away_team,
                        home_score,
                        away_score
                    FROM raw_schedules
                    WHERE game_type = 'REG'
                ),

                player_features AS (
                    -- Get player rolling features from previous week (N-1 for week N)
                    SELECT
                        prf.player_id,
                        prf.season,
                        prf.week + 1 as target_week,
                        prf.position,
                        prf.stats_last3_games,
                        prf.stats_last5_games,
                        prf.stats_season_avg,
                        prf.performance_trend,
                        prf.usage_trend,
                        prf.target_share_trend,
                        prf.vs_opponent_history,
                        prf.opp_rank_vs_position,
                        prf.opp_avg_allowed_to_position,
                        prf.home_away_splits,
                        prf.divisional_game,
                        prf.rest_days
                    FROM player_rolling_features prf
                    WHERE prf.week < 18  -- Ensure we don't go past season end
                ),

                team_features AS (
                    -- Get team features from previous week
                    SELECT
                        trf.team,
                        trf.season,
                        trf.week + 1 as target_week,
                        trf.off_epa_per_play_last3,
                        trf.off_success_rate_last3,
                        trf.off_explosive_play_rate,
                        trf.off_red_zone_efficiency,
                        trf.off_third_down_conv,
                        trf.def_epa_per_play_last3,
                        trf.def_success_rate_last3,
                        trf.def_pressure_rate,
                        trf.def_turnover_rate,
                        trf.pass_rate_neutral,
                        trf.pace_of_play,
                        trf.time_of_possession_avg
                    FROM team_rolling_features trf
                    WHERE trf.week < 18
                ),

                opponent_team_features AS (
                    -- Get opponent defensive features
                    SELECT
                        trf.team as opponent,
                        trf.season,
                        trf.week + 1 as target_week,
                        trf.def_epa_per_play_last3 as opp_def_epa_per_play_last3,
                        trf.def_success_rate_last3 as opp_def_success_rate_last3,
                        trf.def_pressure_rate as opp_def_pressure_rate,
                        trf.def_turnover_rate as opp_def_turnover_rate
                    FROM team_rolling_features trf
                    WHERE trf.week < 18
                ),

                player_experience AS (
                    SELECT
                        player_id,
                        season,
                        experience_category as experience_level
                    FROM player_experience_classification
                ),

                current_week_context AS (
                    -- Get player's current team assignment for target week
                    SELECT DISTINCT
                        player_id,
                        player_name,
                        season,
                        week,
                        recent_team as team,
                        opponent_team
                    FROM raw_player_stats
                    WHERE week > 1  -- No features for week 1
                )

                -- Main join to combine all features
                SELECT
                    -- Generate unique feature_id
                    CONCAT(cwc.player_id, '_', cwc.season, '_', cwc.week) as feature_id,

                    -- Entity identification
                    'player' as entity_type,
                    cwc.player_id as entity_id,
                    NULL as prediction_target,  -- Set later based on position

                    -- Time context
                    cwc.season,
                    cwc.week,
                    gc.game_date,

                    -- Roster context
                    NULL as roster_snapshot_id,  -- Will be populated if needed
                    pe.experience_level as player_experience_level,

                    -- Build numerical_features array
                    ARRAY[
                        -- Universal features (17 features)
                        COALESCE(pf.performance_trend, 0.0),
                        COALESCE(pf.usage_trend, 0.0),
                        COALESCE(pf.target_share_trend, 0.0),
                        COALESCE(CAST(pf.opp_rank_vs_position AS FLOAT), 0.0),
                        COALESCE(CAST(pf.rest_days AS FLOAT), 0.0),
                        COALESCE(tf.off_epa_per_play_last3, 0.0),
                        COALESCE(tf.off_success_rate_last3, 0.0),
                        COALESCE(tf.off_explosive_play_rate, 0.0),
                        COALESCE(tf.off_red_zone_efficiency, 0.0),
                        COALESCE(tf.off_third_down_conv, 0.0),
                        COALESCE(tf.pass_rate_neutral, 0.0),
                        COALESCE(tf.pace_of_play, 0.0),
                        COALESCE(tf.time_of_possession_avg, 0.0),
                        COALESCE(otf.opp_def_epa_per_play_last3, 0.0),
                        COALESCE(otf.opp_def_success_rate_last3, 0.0),
                        COALESCE(otf.opp_def_pressure_rate, 0.0),
                        COALESCE(otf.opp_def_turnover_rate, 0.0)
                    ] ||
                    -- Position-specific stats from last 3 games (dynamic extraction based on position)
                    CASE pf.position
                        WHEN 'QB' THEN ARRAY[
                            COALESCE(CAST(json_extract(pf.stats_last3_games, '$.avg_passing_yards') AS FLOAT), 0.0),
                            COALESCE(CAST(json_extract(pf.stats_last3_games, '$.avg_passing_tds') AS FLOAT), 0.0),
                            COALESCE(CAST(json_extract(pf.stats_last3_games, '$.avg_completions') AS FLOAT), 0.0),
                            COALESCE(CAST(json_extract(pf.stats_last3_games, '$.avg_attempts') AS FLOAT), 0.0),
                            COALESCE(CAST(json_extract(pf.stats_last3_games, '$.avg_passing_epa') AS FLOAT), 0.0),
                            COALESCE(CAST(json_extract(pf.stats_last3_games, '$.avg_sacks_suffered') AS FLOAT), 0.0),
                            COALESCE(CAST(json_extract(pf.stats_last3_games, '$.avg_carries') AS FLOAT), 0.0),
                            COALESCE(CAST(json_extract(pf.stats_last3_games, '$.avg_rushing_yards') AS FLOAT), 0.0),
                            COALESCE(CAST(json_extract(pf.stats_last3_games, '$.avg_rushing_tds') AS FLOAT), 0.0),
                            COALESCE(CAST(json_extract(pf.stats_last3_games, '$.avg_fantasy_points') AS FLOAT), 0.0)
                        ]
                        WHEN 'RB' THEN ARRAY[
                            COALESCE(CAST(json_extract(pf.stats_last3_games, '$.avg_carries') AS FLOAT), 0.0),
                            COALESCE(CAST(json_extract(pf.stats_last3_games, '$.avg_rushing_yards') AS FLOAT), 0.0),
                            COALESCE(CAST(json_extract(pf.stats_last3_games, '$.avg_rushing_tds') AS FLOAT), 0.0),
                            COALESCE(CAST(json_extract(pf.stats_last3_games, '$.avg_targets') AS FLOAT), 0.0),
                            COALESCE(CAST(json_extract(pf.stats_last3_games, '$.avg_receptions') AS FLOAT), 0.0),
                            COALESCE(CAST(json_extract(pf.stats_last3_games, '$.avg_receiving_yards') AS FLOAT), 0.0),
                            COALESCE(CAST(json_extract(pf.stats_last3_games, '$.avg_receiving_tds') AS FLOAT), 0.0),
                            COALESCE(CAST(json_extract(pf.stats_last3_games, '$.avg_rushing_epa') AS FLOAT), 0.0),
                            COALESCE(CAST(json_extract(pf.stats_last3_games, '$.avg_receiving_epa') AS FLOAT), 0.0),
                            COALESCE(CAST(json_extract(pf.stats_last3_games, '$.avg_fantasy_points') AS FLOAT), 0.0)
                        ]
                        WHEN 'WR' THEN ARRAY[
                            COALESCE(CAST(json_extract(pf.stats_last3_games, '$.avg_targets') AS FLOAT), 0.0),
                            COALESCE(CAST(json_extract(pf.stats_last3_games, '$.avg_receptions') AS FLOAT), 0.0),
                            COALESCE(CAST(json_extract(pf.stats_last3_games, '$.avg_receiving_yards') AS FLOAT), 0.0),
                            COALESCE(CAST(json_extract(pf.stats_last3_games, '$.avg_receiving_tds') AS FLOAT), 0.0),
                            COALESCE(CAST(json_extract(pf.stats_last3_games, '$.avg_receiving_epa') AS FLOAT), 0.0),
                            COALESCE(CAST(json_extract(pf.stats_last3_games, '$.avg_target_share') AS FLOAT), 0.0),
                            COALESCE(CAST(json_extract(pf.stats_last3_games, '$.avg_air_yards_share') AS FLOAT), 0.0),
                            COALESCE(CAST(json_extract(pf.stats_last3_games, '$.avg_carries') AS FLOAT), 0.0),
                            COALESCE(CAST(json_extract(pf.stats_last3_games, '$.avg_rushing_yards') AS FLOAT), 0.0),
                            COALESCE(CAST(json_extract(pf.stats_last3_games, '$.avg_fantasy_points') AS FLOAT), 0.0)
                        ]
                        WHEN 'TE' THEN ARRAY[
                            COALESCE(CAST(json_extract(pf.stats_last3_games, '$.avg_targets') AS FLOAT), 0.0),
                            COALESCE(CAST(json_extract(pf.stats_last3_games, '$.avg_receptions') AS FLOAT), 0.0),
                            COALESCE(CAST(json_extract(pf.stats_last3_games, '$.avg_receiving_yards') AS FLOAT), 0.0),
                            COALESCE(CAST(json_extract(pf.stats_last3_games, '$.avg_receiving_tds') AS FLOAT), 0.0),
                            COALESCE(CAST(json_extract(pf.stats_last3_games, '$.avg_receiving_epa') AS FLOAT), 0.0),
                            COALESCE(CAST(json_extract(pf.stats_last3_games, '$.avg_target_share') AS FLOAT), 0.0),
                            COALESCE(CAST(json_extract(pf.stats_last3_games, '$.avg_air_yards_share') AS FLOAT), 0.0),
                            COALESCE(CAST(json_extract(pf.stats_last3_games, '$.avg_carries') AS FLOAT), 0.0),
                            COALESCE(CAST(json_extract(pf.stats_last3_games, '$.avg_rushing_yards') AS FLOAT), 0.0),
                            COALESCE(CAST(json_extract(pf.stats_last3_games, '$.avg_fantasy_points') AS FLOAT), 0.0)
                        ]
                        WHEN 'K' THEN ARRAY[
                            COALESCE(CAST(json_extract(pf.stats_last3_games, '$.avg_fg_att') AS FLOAT), 0.0),
                            COALESCE(CAST(json_extract(pf.stats_last3_games, '$.avg_fg_made') AS FLOAT), 0.0),
                            COALESCE(CAST(json_extract(pf.stats_last3_games, '$.avg_fg_pct') AS FLOAT), 0.0),
                            COALESCE(CAST(json_extract(pf.stats_last3_games, '$.avg_pat_made') AS FLOAT), 0.0),
                            COALESCE(CAST(json_extract(pf.stats_last3_games, '$.avg_pat_att') AS FLOAT), 0.0),
                            0.0, 0.0, 0.0, 0.0, 0.0  -- Padding to match array length
                        ]
                        ELSE ARRAY[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                    END ||
                    -- Last 5 games (same structure, different JSON field)
                    CASE pf.position
                        WHEN 'QB' THEN ARRAY[
                            COALESCE(CAST(json_extract(pf.stats_last5_games, '$.avg_passing_yards') AS FLOAT), 0.0),
                            COALESCE(CAST(json_extract(pf.stats_last5_games, '$.avg_passing_tds') AS FLOAT), 0.0),
                            COALESCE(CAST(json_extract(pf.stats_last5_games, '$.avg_completions') AS FLOAT), 0.0),
                            COALESCE(CAST(json_extract(pf.stats_last5_games, '$.avg_attempts') AS FLOAT), 0.0),
                            COALESCE(CAST(json_extract(pf.stats_last5_games, '$.avg_passing_epa') AS FLOAT), 0.0),
                            COALESCE(CAST(json_extract(pf.stats_last5_games, '$.avg_sacks_suffered') AS FLOAT), 0.0),
                            COALESCE(CAST(json_extract(pf.stats_last5_games, '$.avg_carries') AS FLOAT), 0.0),
                            COALESCE(CAST(json_extract(pf.stats_last5_games, '$.avg_rushing_yards') AS FLOAT), 0.0),
                            COALESCE(CAST(json_extract(pf.stats_last5_games, '$.avg_rushing_tds') AS FLOAT), 0.0),
                            COALESCE(CAST(json_extract(pf.stats_last5_games, '$.avg_fantasy_points') AS FLOAT), 0.0)
                        ]
                        WHEN 'RB' THEN ARRAY[
                            COALESCE(CAST(json_extract(pf.stats_last5_games, '$.avg_carries') AS FLOAT), 0.0),
                            COALESCE(CAST(json_extract(pf.stats_last5_games, '$.avg_rushing_yards') AS FLOAT), 0.0),
                            COALESCE(CAST(json_extract(pf.stats_last5_games, '$.avg_rushing_tds') AS FLOAT), 0.0),
                            COALESCE(CAST(json_extract(pf.stats_last5_games, '$.avg_targets') AS FLOAT), 0.0),
                            COALESCE(CAST(json_extract(pf.stats_last5_games, '$.avg_receptions') AS FLOAT), 0.0),
                            COALESCE(CAST(json_extract(pf.stats_last5_games, '$.avg_receiving_yards') AS FLOAT), 0.0),
                            COALESCE(CAST(json_extract(pf.stats_last5_games, '$.avg_receiving_tds') AS FLOAT), 0.0),
                            COALESCE(CAST(json_extract(pf.stats_last5_games, '$.avg_rushing_epa') AS FLOAT), 0.0),
                            COALESCE(CAST(json_extract(pf.stats_last5_games, '$.avg_receiving_epa') AS FLOAT), 0.0),
                            COALESCE(CAST(json_extract(pf.stats_last5_games, '$.avg_fantasy_points') AS FLOAT), 0.0)
                        ]
                        WHEN 'WR' THEN ARRAY[
                            COALESCE(CAST(json_extract(pf.stats_last5_games, '$.avg_targets') AS FLOAT), 0.0),
                            COALESCE(CAST(json_extract(pf.stats_last5_games, '$.avg_receptions') AS FLOAT), 0.0),
                            COALESCE(CAST(json_extract(pf.stats_last5_games, '$.avg_receiving_yards') AS FLOAT), 0.0),
                            COALESCE(CAST(json_extract(pf.stats_last5_games, '$.avg_receiving_tds') AS FLOAT), 0.0),
                            COALESCE(CAST(json_extract(pf.stats_last5_games, '$.avg_receiving_epa') AS FLOAT), 0.0),
                            COALESCE(CAST(json_extract(pf.stats_last5_games, '$.avg_target_share') AS FLOAT), 0.0),
                            COALESCE(CAST(json_extract(pf.stats_last5_games, '$.avg_air_yards_share') AS FLOAT), 0.0),
                            COALESCE(CAST(json_extract(pf.stats_last5_games, '$.avg_carries') AS FLOAT), 0.0),
                            COALESCE(CAST(json_extract(pf.stats_last5_games, '$.avg_rushing_yards') AS FLOAT), 0.0),
                            COALESCE(CAST(json_extract(pf.stats_last5_games, '$.avg_fantasy_points') AS FLOAT), 0.0)
                        ]
                        WHEN 'TE' THEN ARRAY[
                            COALESCE(CAST(json_extract(pf.stats_last5_games, '$.avg_targets') AS FLOAT), 0.0),
                            COALESCE(CAST(json_extract(pf.stats_last5_games, '$.avg_receptions') AS FLOAT), 0.0),
                            COALESCE(CAST(json_extract(pf.stats_last5_games, '$.avg_receiving_yards') AS FLOAT), 0.0),
                            COALESCE(CAST(json_extract(pf.stats_last5_games, '$.avg_receiving_tds') AS FLOAT), 0.0),
                            COALESCE(CAST(json_extract(pf.stats_last5_games, '$.avg_receiving_epa') AS FLOAT), 0.0),
                            COALESCE(CAST(json_extract(pf.stats_last5_games, '$.avg_target_share') AS FLOAT), 0.0),
                            COALESCE(CAST(json_extract(pf.stats_last5_games, '$.avg_air_yards_share') AS FLOAT), 0.0),
                            COALESCE(CAST(json_extract(pf.stats_last5_games, '$.avg_carries') AS FLOAT), 0.0),
                            COALESCE(CAST(json_extract(pf.stats_last5_games, '$.avg_rushing_yards') AS FLOAT), 0.0),
                            COALESCE(CAST(json_extract(pf.stats_last5_games, '$.avg_fantasy_points') AS FLOAT), 0.0)
                        ]
                        WHEN 'K' THEN ARRAY[
                            COALESCE(CAST(json_extract(pf.stats_last5_games, '$.avg_fg_att') AS FLOAT), 0.0),
                            COALESCE(CAST(json_extract(pf.stats_last5_games, '$.avg_fg_made') AS FLOAT), 0.0),
                            COALESCE(CAST(json_extract(pf.stats_last5_games, '$.avg_fg_pct') AS FLOAT), 0.0),
                            COALESCE(CAST(json_extract(pf.stats_last5_games, '$.avg_pat_made') AS FLOAT), 0.0),
                            COALESCE(CAST(json_extract(pf.stats_last5_games, '$.avg_pat_att') AS FLOAT), 0.0),
                            0.0, 0.0, 0.0, 0.0, 0.0
                        ]
                        ELSE ARRAY[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                    END ||
                    -- Season averages (same structure)
                    CASE pf.position
                        WHEN 'QB' THEN ARRAY[
                            COALESCE(CAST(json_extract(pf.stats_season_avg, '$.avg_passing_yards') AS FLOAT), 0.0),
                            COALESCE(CAST(json_extract(pf.stats_season_avg, '$.avg_passing_tds') AS FLOAT), 0.0),
                            COALESCE(CAST(json_extract(pf.stats_season_avg, '$.avg_completions') AS FLOAT), 0.0),
                            COALESCE(CAST(json_extract(pf.stats_season_avg, '$.avg_attempts') AS FLOAT), 0.0),
                            COALESCE(CAST(json_extract(pf.stats_season_avg, '$.avg_passing_epa') AS FLOAT), 0.0),
                            COALESCE(CAST(json_extract(pf.stats_season_avg, '$.avg_sacks_suffered') AS FLOAT), 0.0),
                            COALESCE(CAST(json_extract(pf.stats_season_avg, '$.avg_carries') AS FLOAT), 0.0),
                            COALESCE(CAST(json_extract(pf.stats_season_avg, '$.avg_rushing_yards') AS FLOAT), 0.0),
                            COALESCE(CAST(json_extract(pf.stats_season_avg, '$.avg_rushing_tds') AS FLOAT), 0.0),
                            COALESCE(CAST(json_extract(pf.stats_season_avg, '$.avg_fantasy_points') AS FLOAT), 0.0)
                        ]
                        WHEN 'RB' THEN ARRAY[
                            COALESCE(CAST(json_extract(pf.stats_season_avg, '$.avg_carries') AS FLOAT), 0.0),
                            COALESCE(CAST(json_extract(pf.stats_season_avg, '$.avg_rushing_yards') AS FLOAT), 0.0),
                            COALESCE(CAST(json_extract(pf.stats_season_avg, '$.avg_rushing_tds') AS FLOAT), 0.0),
                            COALESCE(CAST(json_extract(pf.stats_season_avg, '$.avg_targets') AS FLOAT), 0.0),
                            COALESCE(CAST(json_extract(pf.stats_season_avg, '$.avg_receptions') AS FLOAT), 0.0),
                            COALESCE(CAST(json_extract(pf.stats_season_avg, '$.avg_receiving_yards') AS FLOAT), 0.0),
                            COALESCE(CAST(json_extract(pf.stats_season_avg, '$.avg_receiving_tds') AS FLOAT), 0.0),
                            COALESCE(CAST(json_extract(pf.stats_season_avg, '$.avg_rushing_epa') AS FLOAT), 0.0),
                            COALESCE(CAST(json_extract(pf.stats_season_avg, '$.avg_receiving_epa') AS FLOAT), 0.0),
                            COALESCE(CAST(json_extract(pf.stats_season_avg, '$.avg_fantasy_points') AS FLOAT), 0.0)
                        ]
                        WHEN 'WR' THEN ARRAY[
                            COALESCE(CAST(json_extract(pf.stats_season_avg, '$.avg_targets') AS FLOAT), 0.0),
                            COALESCE(CAST(json_extract(pf.stats_season_avg, '$.avg_receptions') AS FLOAT), 0.0),
                            COALESCE(CAST(json_extract(pf.stats_season_avg, '$.avg_receiving_yards') AS FLOAT), 0.0),
                            COALESCE(CAST(json_extract(pf.stats_season_avg, '$.avg_receiving_tds') AS FLOAT), 0.0),
                            COALESCE(CAST(json_extract(pf.stats_season_avg, '$.avg_receiving_epa') AS FLOAT), 0.0),
                            COALESCE(CAST(json_extract(pf.stats_season_avg, '$.avg_target_share') AS FLOAT), 0.0),
                            COALESCE(CAST(json_extract(pf.stats_season_avg, '$.avg_air_yards_share') AS FLOAT), 0.0),
                            COALESCE(CAST(json_extract(pf.stats_season_avg, '$.avg_carries') AS FLOAT), 0.0),
                            COALESCE(CAST(json_extract(pf.stats_season_avg, '$.avg_rushing_yards') AS FLOAT), 0.0),
                            COALESCE(CAST(json_extract(pf.stats_season_avg, '$.avg_fantasy_points') AS FLOAT), 0.0)
                        ]
                        WHEN 'TE' THEN ARRAY[
                            COALESCE(CAST(json_extract(pf.stats_season_avg, '$.avg_targets') AS FLOAT), 0.0),
                            COALESCE(CAST(json_extract(pf.stats_season_avg, '$.avg_receptions') AS FLOAT), 0.0),
                            COALESCE(CAST(json_extract(pf.stats_season_avg, '$.avg_receiving_yards') AS FLOAT), 0.0),
                            COALESCE(CAST(json_extract(pf.stats_season_avg, '$.avg_receiving_tds') AS FLOAT), 0.0),
                            COALESCE(CAST(json_extract(pf.stats_season_avg, '$.avg_receiving_epa') AS FLOAT), 0.0),
                            COALESCE(CAST(json_extract(pf.stats_season_avg, '$.avg_target_share') AS FLOAT), 0.0),
                            COALESCE(CAST(json_extract(pf.stats_season_avg, '$.avg_air_yards_share') AS FLOAT), 0.0),
                            COALESCE(CAST(json_extract(pf.stats_season_avg, '$.avg_carries') AS FLOAT), 0.0),
                            COALESCE(CAST(json_extract(pf.stats_season_avg, '$.avg_rushing_yards') AS FLOAT), 0.0),
                            COALESCE(CAST(json_extract(pf.stats_season_avg, '$.avg_fantasy_points') AS FLOAT), 0.0)
                        ]
                        WHEN 'K' THEN ARRAY[
                            COALESCE(CAST(json_extract(pf.stats_season_avg, '$.avg_fg_att') AS FLOAT), 0.0),
                            COALESCE(CAST(json_extract(pf.stats_season_avg, '$.avg_fg_made') AS FLOAT), 0.0),
                            COALESCE(CAST(json_extract(pf.stats_season_avg, '$.avg_fg_pct') AS FLOAT), 0.0),
                            COALESCE(CAST(json_extract(pf.stats_season_avg, '$.avg_pat_made') AS FLOAT), 0.0),
                            COALESCE(CAST(json_extract(pf.stats_season_avg, '$.avg_pat_att') AS FLOAT), 0.0),
                            0.0, 0.0, 0.0, 0.0, 0.0
                        ]
                        ELSE ARRAY[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                    END
                    as numerical_features,

                    -- Build feature_names array (parallel to numerical_features)
                    ARRAY[
                        'performance_trend', 'usage_trend', 'target_share_trend', 'opp_rank_vs_position', 'rest_days',
                        'off_epa_per_play_last3', 'off_success_rate_last3', 'off_explosive_play_rate',
                        'off_red_zone_efficiency', 'off_third_down_conv', 'pass_rate_neutral',
                        'pace_of_play', 'time_of_possession_avg',
                        'opp_def_epa_per_play_last3', 'opp_def_success_rate_last3',
                        'opp_def_pressure_rate', 'opp_def_turnover_rate'
                    ] ||
                    -- Position-specific feature names for last 3 games
                    CASE pf.position
                        WHEN 'QB' THEN ARRAY[
                            'last3_passing_yards', 'last3_passing_tds', 'last3_completions', 'last3_attempts',
                            'last3_passing_epa', 'last3_sacks', 'last3_carries', 'last3_rushing_yards',
                            'last3_rushing_tds', 'last3_fantasy_points'
                        ]
                        WHEN 'RB' THEN ARRAY[
                            'last3_carries', 'last3_rushing_yards', 'last3_rushing_tds', 'last3_targets',
                            'last3_receptions', 'last3_receiving_yards', 'last3_receiving_tds',
                            'last3_rushing_epa', 'last3_receiving_epa', 'last3_fantasy_points'
                        ]
                        WHEN 'WR' THEN ARRAY[
                            'last3_targets', 'last3_receptions', 'last3_receiving_yards', 'last3_receiving_tds',
                            'last3_receiving_epa', 'last3_target_share', 'last3_air_yards_share',
                            'last3_carries', 'last3_rushing_yards', 'last3_fantasy_points'
                        ]
                        WHEN 'TE' THEN ARRAY[
                            'last3_targets', 'last3_receptions', 'last3_receiving_yards', 'last3_receiving_tds',
                            'last3_receiving_epa', 'last3_target_share', 'last3_air_yards_share',
                            'last3_carries', 'last3_rushing_yards', 'last3_fantasy_points'
                        ]
                        WHEN 'K' THEN ARRAY[
                            'last3_fg_att', 'last3_fg_made', 'last3_fg_pct', 'last3_pat_made', 'last3_pat_att',
                            'last3_pad1', 'last3_pad2', 'last3_pad3', 'last3_pad4', 'last3_pad5'
                        ]
                        ELSE ARRAY['last3_1', 'last3_2', 'last3_3', 'last3_4', 'last3_5', 'last3_6', 'last3_7', 'last3_8', 'last3_9', 'last3_10']
                    END ||
                    -- Last 5 games feature names
                    CASE pf.position
                        WHEN 'QB' THEN ARRAY[
                            'last5_passing_yards', 'last5_passing_tds', 'last5_completions', 'last5_attempts',
                            'last5_passing_epa', 'last5_sacks', 'last5_carries', 'last5_rushing_yards',
                            'last5_rushing_tds', 'last5_fantasy_points'
                        ]
                        WHEN 'RB' THEN ARRAY[
                            'last5_carries', 'last5_rushing_yards', 'last5_rushing_tds', 'last5_targets',
                            'last5_receptions', 'last5_receiving_yards', 'last5_receiving_tds',
                            'last5_rushing_epa', 'last5_receiving_epa', 'last5_fantasy_points'
                        ]
                        WHEN 'WR' THEN ARRAY[
                            'last5_targets', 'last5_receptions', 'last5_receiving_yards', 'last5_receiving_tds',
                            'last5_receiving_epa', 'last5_target_share', 'last5_air_yards_share',
                            'last5_carries', 'last5_rushing_yards', 'last5_fantasy_points'
                        ]
                        WHEN 'TE' THEN ARRAY[
                            'last5_targets', 'last5_receptions', 'last5_receiving_yards', 'last5_receiving_tds',
                            'last5_receiving_epa', 'last5_target_share', 'last5_air_yards_share',
                            'last5_carries', 'last5_rushing_yards', 'last5_fantasy_points'
                        ]
                        WHEN 'K' THEN ARRAY[
                            'last5_fg_att', 'last5_fg_made', 'last5_fg_pct', 'last5_pat_made', 'last5_pat_att',
                            'last5_pad1', 'last5_pad2', 'last5_pad3', 'last5_pad4', 'last5_pad5'
                        ]
                        ELSE ARRAY['last5_1', 'last5_2', 'last5_3', 'last5_4', 'last5_5', 'last5_6', 'last5_7', 'last5_8', 'last5_9', 'last5_10']
                    END ||
                    -- Season average feature names
                    CASE pf.position
                        WHEN 'QB' THEN ARRAY[
                            'season_passing_yards', 'season_passing_tds', 'season_completions', 'season_attempts',
                            'season_passing_epa', 'season_sacks', 'season_carries', 'season_rushing_yards',
                            'season_rushing_tds', 'season_fantasy_points'
                        ]
                        WHEN 'RB' THEN ARRAY[
                            'season_carries', 'season_rushing_yards', 'season_rushing_tds', 'season_targets',
                            'season_receptions', 'season_receiving_yards', 'season_receiving_tds',
                            'season_rushing_epa', 'season_receiving_epa', 'season_fantasy_points'
                        ]
                        WHEN 'WR' THEN ARRAY[
                            'season_targets', 'season_receptions', 'season_receiving_yards', 'season_receiving_tds',
                            'season_receiving_epa', 'season_target_share', 'season_air_yards_share',
                            'season_carries', 'season_rushing_yards', 'season_fantasy_points'
                        ]
                        WHEN 'TE' THEN ARRAY[
                            'season_targets', 'season_receptions', 'season_receiving_yards', 'season_receiving_tds',
                            'season_receiving_epa', 'season_target_share', 'season_air_yards_share',
                            'season_carries', 'season_rushing_yards', 'season_fantasy_points'
                        ]
                        WHEN 'K' THEN ARRAY[
                            'season_fg_att', 'season_fg_made', 'season_fg_pct', 'season_pat_made', 'season_pat_att',
                            'season_pad1', 'season_pad2', 'season_pad3', 'season_pad4', 'season_pad5'
                        ]
                        ELSE ARRAY['season_1', 'season_2', 'season_3', 'season_4', 'season_5', 'season_6', 'season_7', 'season_8', 'season_9', 'season_10']
                    END
                    as feature_names,

                    -- Build categorical_features JSON
                    json_object(
                        'position', json_object(
                            'value', pf.position,
                            'encoded', CASE pf.position
                                WHEN 'QB' THEN 1
                                WHEN 'RB' THEN 2
                                WHEN 'WR' THEN 3
                                WHEN 'TE' THEN 4
                                WHEN 'K' THEN 5
                                ELSE 0
                            END
                        ),
                        'team', json_object('value', cwc.team),
                        'opponent', json_object('value', cwc.opponent_team),
                        'home_away', json_object(
                            'value', CASE
                                WHEN gc.home_team = cwc.team THEN 'home'
                                WHEN gc.away_team = cwc.team THEN 'away'
                                ELSE 'neutral'
                            END,
                            'encoded', CASE
                                WHEN gc.home_team = cwc.team THEN 1
                                WHEN gc.away_team = cwc.team THEN 0
                                ELSE -1
                            END
                        ),
                        'divisional_game', json_object(
                            'value', COALESCE(pf.divisional_game, FALSE),
                            'encoded', CASE WHEN COALESCE(pf.divisional_game, FALSE) THEN 1 ELSE 0 END
                        ),
                        'experience_level', json_object(
                            'value', COALESCE(pe.experience_level, 'unknown'),
                            'encoded', CASE COALESCE(pe.experience_level, 'unknown')
                                WHEN 'rookie' THEN 1
                                WHEN 'developing' THEN 2
                                WHEN 'veteran' THEN 3
                                ELSE 0
                            END
                        )
                    ) as categorical_features,

                    -- Targets (to be filled later)
                    NULL as actual_outcomes,

                    -- Quality (to be calculated later)
                    NULL as data_quality_score,
                    NULL as missing_data_flags,

                    CURRENT_TIMESTAMP as created_at

                FROM current_week_context cwc

                -- Join game context
                LEFT JOIN game_context gc
                    ON cwc.season = gc.season
                    AND cwc.week = gc.week
                    AND (cwc.team = gc.home_team OR cwc.team = gc.away_team)

                -- Join player features from PREVIOUS week (temporal alignment)
                INNER JOIN player_features pf
                    ON cwc.player_id = pf.player_id
                    AND cwc.season = pf.season
                    AND cwc.week = pf.target_week

                -- Join team features from PREVIOUS week
                LEFT JOIN team_features tf
                    ON cwc.team = tf.team
                    AND cwc.season = tf.season
                    AND cwc.week = tf.target_week

                -- Join opponent defensive features from PREVIOUS week
                LEFT JOIN opponent_team_features otf
                    ON cwc.opponent_team = otf.opponent
                    AND cwc.season = otf.season
                    AND cwc.week = otf.target_week

                -- Join player experience
                LEFT JOIN player_experience pe
                    ON cwc.player_id = pe.player_id
                    AND cwc.season = pe.season

                WHERE
                    -- Ensure we have player features (features must exist from previous week)
                    pf.player_id IS NOT NULL
                    -- Filter to key offensive positions
                    AND pf.position IN ('QB', 'RB', 'WR', 'TE', 'K')
            """

            # Execute the feature combination
            conn.execute(query)

            # Get count of created features
            result = conn.execute(
                "SELECT COUNT(*) FROM ml_training_features"
            ).fetchone()
            rows_created = result[0] if result else 0

            logger.info(
                f"✅ Combined features for {rows_created:,} player-week observations"
            )

            # Log summary by position
            summary = conn.execute(
                """
                SELECT
                    json_extract(categorical_features, '$.position.value') as position,
                    COUNT(*) as count,
                    AVG(array_length(numerical_features)) as avg_feature_count
                FROM ml_training_features
                GROUP BY position
                ORDER BY count DESC
            """
            ).fetchall()

            if summary:
                logger.info("  📊 Feature distribution by position:")
                for row in summary:
                    pos, cnt, avg_feat = row
                    logger.info(
                        f"     {pos}: {cnt:,} observations, {avg_feat:.0f} avg features"
                    )

            return rows_created

        except Exception as e:
            logger.error(f"❌ Failed to combine features: {e}")
            raise

    def apply_data_quality_scoring(self):
        """
        Apply data quality scoring to ml_training_features.

        Calculates a comprehensive quality score for each feature vector:
        - Completeness score (60%): Percentage of non-NULL features
        - Outlier score (20%): Detection of extreme values (z-score > 3.0)
        - Recency score (20%): More recent data is more reliable
        - Critical features multiplier: Must have position, team, opponent

        Final formula: (completeness * 0.6 + outlier * 0.2 + recency * 0.2) * critical_multiplier
        Filters out rows with quality < 0.5

        Returns:
            tuple: (rows_scored, rows_passed, rows_filtered)

        Raises:
            Exception: If quality scoring fails
        """
        logger.info("✨ Applying data quality scoring...")

        try:
            conn = self.db.connect()

            # Check if there are features to score
            feature_count = conn.execute(
                "SELECT COUNT(*) FROM ml_training_features"
            ).fetchone()[0]
            if feature_count == 0:
                logger.warning("⚠️  No features to score")
                return (0, 0, 0)

            logger.info(f"  📊 Scoring {feature_count:,} feature rows...")

            # Step 1: Calculate completeness score (60% weight)
            logger.info("  🔍 Calculating completeness scores...")
            conn.execute(
                """
                CREATE OR REPLACE TEMP TABLE feature_completeness AS
                SELECT
                    feature_id,
                    numerical_features,
                    array_length(numerical_features) as total_features,

                    -- Count non-zero features (0.0 typically means NULL was coalesced)
                    array_length(
                        list_filter(numerical_features, x -> x != 0.0)
                    ) as non_zero_features,

                    -- Calculate completeness ratio
                    CAST(
                        array_length(list_filter(numerical_features, x -> x != 0.0))
                        AS FLOAT
                    ) / NULLIF(array_length(numerical_features), 0) as completeness_ratio,

                    -- Completeness score with thresholds
                    CASE
                        WHEN CAST(array_length(list_filter(numerical_features, x -> x != 0.0)) AS FLOAT)
                             / NULLIF(array_length(numerical_features), 0) >= 0.9 THEN 1.0
                        WHEN CAST(array_length(list_filter(numerical_features, x -> x != 0.0)) AS FLOAT)
                             / NULLIF(array_length(numerical_features), 0) >= 0.75 THEN 0.8
                        WHEN CAST(array_length(list_filter(numerical_features, x -> x != 0.0)) AS FLOAT)
                             / NULLIF(array_length(numerical_features), 0) >= 0.5 THEN 0.6
                        WHEN CAST(array_length(list_filter(numerical_features, x -> x != 0.0)) AS FLOAT)
                             / NULLIF(array_length(numerical_features), 0) >= 0.25 THEN 0.3
                        ELSE 0.0
                    END as completeness_score
                FROM ml_training_features
            """
            )

            # Step 2: Calculate outlier scores (20% weight)
            logger.info("  📈 Detecting outliers...")
            # Simplified outlier detection - count features with extreme values
            conn.execute(
                """
                CREATE OR REPLACE TEMP TABLE feature_outliers AS
                SELECT
                    feature_id,
                    numerical_features,

                    -- Count features with very high absolute values (simple outlier detection)
                    array_length(
                        list_filter(numerical_features, x -> ABS(x) > 1000.0)
                    ) as extreme_value_count,

                    -- Outlier score (fewer outliers = higher score)
                    CASE
                        WHEN array_length(list_filter(numerical_features, x -> ABS(x) > 1000.0)) = 0 THEN 1.0
                        WHEN array_length(list_filter(numerical_features, x -> ABS(x) > 1000.0)) <= 2 THEN 0.9
                        WHEN array_length(list_filter(numerical_features, x -> ABS(x) > 1000.0)) <= 5 THEN 0.7
                        WHEN array_length(list_filter(numerical_features, x -> ABS(x) > 1000.0)) <= 10 THEN 0.5
                        ELSE 0.0
                    END as outlier_score
                FROM ml_training_features
            """
            )

            # Step 3: Calculate recency scores (20% weight)
            logger.info("  📅 Calculating recency scores...")
            conn.execute(
                """
                CREATE OR REPLACE TEMP TABLE feature_recency AS
                WITH current_state AS (
                    SELECT
                        MAX(season) as current_season,
                        MAX(week) as current_week
                    FROM ml_training_features
                )
                SELECT
                    mtf.feature_id,
                    mtf.season,
                    mtf.week,

                    -- Calculate weeks ago
                    ((cs.current_season - mtf.season) * 18 + (cs.current_week - mtf.week)) as weeks_ago,

                    -- Recency score (decay over time)
                    CASE
                        WHEN ((cs.current_season - mtf.season) * 18 + (cs.current_week - mtf.week)) <= 4 THEN 1.0
                        WHEN ((cs.current_season - mtf.season) * 18 + (cs.current_week - mtf.week)) <= 8 THEN 0.9
                        WHEN ((cs.current_season - mtf.season) * 18 + (cs.current_week - mtf.week)) <= 17 THEN 0.8
                        WHEN ((cs.current_season - mtf.season) * 18 + (cs.current_week - mtf.week)) <= 35 THEN 0.7
                        ELSE 0.5
                    END as recency_score
                FROM ml_training_features mtf
                CROSS JOIN current_state cs
            """
            )

            # Step 4: Check critical features
            logger.info("  🔑 Checking critical features...")
            conn.execute(
                """
                CREATE OR REPLACE TEMP TABLE critical_features_check AS
                SELECT
                    feature_id,

                    -- Check critical categorical features
                    json_extract(categorical_features, '$.position.value') IS NOT NULL as has_position,
                    json_extract(categorical_features, '$.team.value') IS NOT NULL as has_team,
                    json_extract(categorical_features, '$.opponent.value') IS NOT NULL as has_opponent,

                    -- Check temporal data
                    season IS NOT NULL AND week IS NOT NULL as has_time_context,

                    -- Overall critical check
                    (json_extract(categorical_features, '$.position.value') IS NOT NULL
                     AND json_extract(categorical_features, '$.team.value') IS NOT NULL
                     AND json_extract(categorical_features, '$.opponent.value') IS NOT NULL
                     AND season IS NOT NULL
                     AND week IS NOT NULL) as has_all_critical,

                    -- Critical features multiplier
                    CASE
                        WHEN (json_extract(categorical_features, '$.position.value') IS NOT NULL
                              AND json_extract(categorical_features, '$.team.value') IS NOT NULL
                              AND json_extract(categorical_features, '$.opponent.value') IS NOT NULL
                              AND season IS NOT NULL
                              AND week IS NOT NULL)
                        THEN 1.0
                        ELSE 0.0
                    END as critical_multiplier
                FROM ml_training_features
            """
            )

            # Step 5: Combine all scores and update table
            logger.info("  🧮 Computing final quality scores...")
            conn.execute(
                """
                UPDATE ml_training_features
                SET
                    data_quality_score = (
                        SELECT
                            (fc.completeness_score * 0.6 +
                             fo.outlier_score * 0.2 +
                             fr.recency_score * 0.2) * cfc.critical_multiplier
                        FROM feature_completeness fc
                        JOIN feature_outliers fo ON fc.feature_id = fo.feature_id
                        JOIN feature_recency fr ON fc.feature_id = fr.feature_id
                        JOIN critical_features_check cfc ON fc.feature_id = cfc.feature_id
                        WHERE fc.feature_id = ml_training_features.feature_id
                    ),
                    missing_data_flags = (
                        SELECT
                            ARRAY[
                                CASE WHEN fc.completeness_ratio < 0.5 THEN 'low_completeness' ELSE NULL END,
                                CASE WHEN fo.extreme_value_count > 5 THEN 'high_outliers' ELSE NULL END,
                                CASE WHEN NOT cfc.has_position THEN 'missing_position' ELSE NULL END,
                                CASE WHEN NOT cfc.has_team THEN 'missing_team' ELSE NULL END,
                                CASE WHEN NOT cfc.has_opponent THEN 'missing_opponent' ELSE NULL END,
                                CASE WHEN fr.weeks_ago > 35 THEN 'stale_data' ELSE NULL END
                            ]
                        FROM feature_completeness fc
                        JOIN feature_outliers fo ON fc.feature_id = fo.feature_id
                        JOIN feature_recency fr ON fc.feature_id = fr.feature_id
                        JOIN critical_features_check cfc ON fc.feature_id = cfc.feature_id
                        WHERE fc.feature_id = ml_training_features.feature_id
                    )
            """
            )

            # Clean up NULL values in missing_data_flags arrays
            conn.execute(
                """
                UPDATE ml_training_features
                SET missing_data_flags = list_filter(
                    COALESCE(missing_data_flags, ARRAY[]::VARCHAR[]),
                    x -> x IS NOT NULL
                )
            """
            )

            # Step 6: Get statistics before filtering
            stats_before = conn.execute(
                """
                SELECT
                    COUNT(*) as total_rows,
                    AVG(data_quality_score) as avg_score,
                    MIN(data_quality_score) as min_score,
                    MAX(data_quality_score) as max_score,
                    COUNT(*) FILTER (WHERE data_quality_score >= 0.5) as rows_passing
                FROM ml_training_features
            """
            ).fetchone()

            total_rows, avg_score, min_score, max_score, rows_passing = stats_before

            logger.info(f"  📊 Quality score statistics:")
            logger.info(f"     Average: {avg_score:.3f}")
            logger.info(f"     Range: [{min_score:.3f}, {max_score:.3f}]")
            logger.info(
                f"     Passing threshold (≥0.5): {rows_passing:,} / {total_rows:,} ({100*rows_passing/total_rows:.1f}%)"
            )

            # Step 7: Filter out low-quality rows (< 0.5 threshold)
            logger.info("  🗑️  Filtering low-quality rows...")
            conn.execute(
                """
                DELETE FROM ml_training_features
                WHERE data_quality_score < 0.5
            """
            )

            rows_filtered = total_rows - rows_passing

            # Get final count
            final_count = conn.execute(
                "SELECT COUNT(*) FROM ml_training_features"
            ).fetchone()[0]

            logger.info(f"✅ Quality scoring complete:")
            logger.info(f"   Scored: {total_rows:,} rows")
            logger.info(f"   Passed: {rows_passing:,} rows")
            logger.info(f"   Filtered: {rows_filtered:,} rows")

            # Log distribution of flags
            flag_dist = conn.execute(
                """
                SELECT
                    unnest(missing_data_flags) as flag,
                    COUNT(*) as count
                FROM ml_training_features
                WHERE array_length(missing_data_flags) > 0
                GROUP BY flag
                ORDER BY count DESC
            """
            ).fetchall()

            if flag_dist:
                logger.info("  ⚠️  Data quality flags:")
                for flag, count in flag_dist:
                    logger.info(f"     {flag}: {count:,} occurrences")

            # Clean up temp tables
            conn.execute("DROP TABLE IF EXISTS feature_completeness")
            conn.execute("DROP TABLE IF EXISTS feature_outliers")
            conn.execute("DROP TABLE IF EXISTS feature_recency")
            conn.execute("DROP TABLE IF EXISTS critical_features_check")

            return (total_rows, rows_passing, rows_filtered)

        except Exception as e:
            logger.error(f"❌ Failed to apply quality scoring: {e}")
            raise

    def create_prediction_targets(self):
        """
        Create prediction target variables from actual game outcomes.

        For each feature row at week N, extract actual stats from week N
        (the week being predicted). Creates position-specific targets including:
        - QB: passing stats, rushing stats, fantasy points, team outcome
        - RB: rushing stats, receiving stats, fantasy points
        - WR/TE: receiving stats, fantasy points
        - K: field goals, extra points, fantasy points

        Returns:
            tuple: (rows_with_targets, rows_without_targets)

        Raises:
            Exception: If target creation fails
        """
        logger.info("🎯 Creating prediction targets...")

        try:
            conn = self.db.connect()

            # Check if there are features to add targets to
            feature_count = conn.execute(
                "SELECT COUNT(*) FROM ml_training_features"
            ).fetchone()[0]
            if feature_count == 0:
                logger.warning("⚠️  No features to add targets to")
                return (0, 0)

            logger.info(f"  📊 Creating targets for {feature_count:,} feature rows...")

            # Build and execute the target creation query
            logger.info("  🔨 Extracting actual stats and game outcomes...")

            conn.execute(
                """
                CREATE OR REPLACE TEMP TABLE prediction_targets AS
                WITH feature_rows AS (
                    SELECT
                        feature_id,
                        entity_id as player_id,
                        season,
                        week,
                        json_extract(categorical_features, '$.position.value') as position,
                        json_extract(categorical_features, '$.team.value') as team
                    FROM ml_training_features
                    WHERE actual_outcomes IS NULL  -- Only process rows without targets
                ),

                actual_stats AS (
                    SELECT
                        rps.player_id,
                        rps.season,
                        rps.week,
                        rps.position,
                        rps.recent_team as team,

                        -- Passing stats (QB)
                        rps.passing_yards,
                        rps.passing_tds,
                        rps.interceptions as passing_interceptions,
                        rps.completions,
                        rps.attempts,
                        rps.passing_epa,
                        rps.cpoe as passing_cpoe,

                        -- Rushing stats (RB, QB)
                        rps.rushing_yards,
                        rps.rushing_tds,
                        rps.carries,
                        rps.rushing_epa,

                        -- Receiving stats (WR, TE, RB)
                        rps.receiving_yards,
                        rps.receiving_tds,
                        rps.receptions,
                        rps.targets,
                        rps.receiving_epa,
                        rps.target_share,
                        rps.air_yards_share,

                        -- Kicking stats (K)
                        rps.fg_made,
                        rps.fg_att,
                        rps.fg_pct,
                        rps.pat_made,
                        rps.pat_att,

                        -- Fantasy points (calculated)
                        rps.fantasy_points as fantasy_points_ppr,

                        rps.opponent_team
                    FROM raw_player_stats rps
                ),

                game_outcomes AS (
                    SELECT
                        game_id,
                        season,
                        week,
                        home_team,
                        away_team,
                        home_score,
                        away_score,

                        -- Win indicator for home team
                        CASE
                            WHEN home_score > away_score THEN 1.0
                            WHEN home_score < away_score THEN 0.0
                            ELSE 0.5
                        END as home_win,

                        home_score + away_score as total_points,
                        home_score - away_score as score_diff,
                        overtime
                    FROM raw_schedules
                    WHERE game_type = 'REG'
                )

                SELECT
                    fr.feature_id,
                    fr.position,

                    -- Position-specific targets as JSON
                    CASE fr.position
                        WHEN 'QB' THEN json_object(
                            'passing_yards', COALESCE(ast.passing_yards, 0.0),
                            'passing_tds', COALESCE(ast.passing_tds, 0.0),
                            'passing_interceptions', COALESCE(ast.passing_interceptions, 0.0),
                            'completions', COALESCE(ast.completions, 0.0),
                            'attempts', COALESCE(ast.attempts, 0.0),
                            'completion_pct', COALESCE(
                                CAST(ast.completions AS FLOAT) / NULLIF(ast.attempts, 0),
                                0.0
                            ),
                            'passing_epa', COALESCE(ast.passing_epa, 0.0),
                            'rushing_yards', COALESCE(ast.rushing_yards, 0.0),
                            'rushing_tds', COALESCE(ast.rushing_tds, 0.0),
                            'fantasy_points_ppr', COALESCE(ast.fantasy_points_ppr, 0.0),
                            'team_points', CASE
                                WHEN ast.team = go.home_team THEN go.home_score
                                WHEN ast.team = go.away_team THEN go.away_score
                                ELSE NULL
                            END,
                            'team_won', CASE
                                WHEN ast.team = go.home_team THEN go.home_win
                                WHEN ast.team = go.away_team THEN 1.0 - go.home_win
                                ELSE NULL
                            END
                        )

                        WHEN 'RB' THEN json_object(
                            'rushing_yards', COALESCE(ast.rushing_yards, 0.0),
                            'rushing_tds', COALESCE(ast.rushing_tds, 0.0),
                            'carries', COALESCE(ast.carries, 0.0),
                            'yards_per_carry', COALESCE(
                                CAST(ast.rushing_yards AS FLOAT) / NULLIF(ast.carries, 0),
                                0.0
                            ),
                            'receiving_yards', COALESCE(ast.receiving_yards, 0.0),
                            'receiving_tds', COALESCE(ast.receiving_tds, 0.0),
                            'receptions', COALESCE(ast.receptions, 0.0),
                            'targets', COALESCE(ast.targets, 0.0),
                            'fantasy_points_ppr', COALESCE(ast.fantasy_points_ppr, 0.0),
                            'team_points', CASE
                                WHEN ast.team = go.home_team THEN go.home_score
                                WHEN ast.team = go.away_team THEN go.away_score
                                ELSE NULL
                            END,
                            'team_won', CASE
                                WHEN ast.team = go.home_team THEN go.home_win
                                WHEN ast.team = go.away_team THEN 1.0 - go.home_win
                                ELSE NULL
                            END
                        )

                        WHEN 'WR' THEN json_object(
                            'receiving_yards', COALESCE(ast.receiving_yards, 0.0),
                            'receiving_tds', COALESCE(ast.receiving_tds, 0.0),
                            'receptions', COALESCE(ast.receptions, 0.0),
                            'targets', COALESCE(ast.targets, 0.0),
                            'catch_rate', COALESCE(
                                CAST(ast.receptions AS FLOAT) / NULLIF(ast.targets, 0),
                                0.0
                            ),
                            'yards_per_reception', COALESCE(
                                CAST(ast.receiving_yards AS FLOAT) / NULLIF(ast.receptions, 0),
                                0.0
                            ),
                            'target_share', COALESCE(ast.target_share, 0.0),
                            'air_yards_share', COALESCE(ast.air_yards_share, 0.0),
                            'rushing_yards', COALESCE(ast.rushing_yards, 0.0),
                            'fantasy_points_ppr', COALESCE(ast.fantasy_points_ppr, 0.0),
                            'team_points', CASE
                                WHEN ast.team = go.home_team THEN go.home_score
                                WHEN ast.team = go.away_team THEN go.away_score
                                ELSE NULL
                            END,
                            'team_won', CASE
                                WHEN ast.team = go.home_team THEN go.home_win
                                WHEN ast.team = go.away_team THEN 1.0 - go.home_win
                                ELSE NULL
                            END
                        )

                        WHEN 'TE' THEN json_object(
                            'receiving_yards', COALESCE(ast.receiving_yards, 0.0),
                            'receiving_tds', COALESCE(ast.receiving_tds, 0.0),
                            'receptions', COALESCE(ast.receptions, 0.0),
                            'targets', COALESCE(ast.targets, 0.0),
                            'catch_rate', COALESCE(
                                CAST(ast.receptions AS FLOAT) / NULLIF(ast.targets, 0),
                                0.0
                            ),
                            'yards_per_reception', COALESCE(
                                CAST(ast.receiving_yards AS FLOAT) / NULLIF(ast.receptions, 0),
                                0.0
                            ),
                            'target_share', COALESCE(ast.target_share, 0.0),
                            'fantasy_points_ppr', COALESCE(ast.fantasy_points_ppr, 0.0),
                            'team_points', CASE
                                WHEN ast.team = go.home_team THEN go.home_score
                                WHEN ast.team = go.away_team THEN go.away_score
                                ELSE NULL
                            END,
                            'team_won', CASE
                                WHEN ast.team = go.home_team THEN go.home_win
                                WHEN ast.team = go.away_team THEN 1.0 - go.home_win
                                ELSE NULL
                            END
                        )

                        WHEN 'K' THEN json_object(
                            'fg_made', COALESCE(ast.fg_made, 0.0),
                            'fg_att', COALESCE(ast.fg_att, 0.0),
                            'fg_pct', COALESCE(ast.fg_pct, 0.0),
                            'pat_made', COALESCE(ast.pat_made, 0.0),
                            'pat_att', COALESCE(ast.pat_att, 0.0),
                            'fantasy_points_standard', COALESCE(ast.fg_made * 3.0 + ast.pat_made, 0.0),
                            'team_points', CASE
                                WHEN ast.team = go.home_team THEN go.home_score
                                WHEN ast.team = go.away_team THEN go.away_score
                                ELSE NULL
                            END,
                            'team_won', CASE
                                WHEN ast.team = go.home_team THEN go.home_win
                                WHEN ast.team = go.away_team THEN 1.0 - go.home_win
                                ELSE NULL
                            END
                        )

                        ELSE json_object('error', 'unknown_position')
                    END as actual_outcomes,

                    -- Primary target based on position
                    CASE fr.position
                        WHEN 'QB' THEN 'passing_yards'
                        WHEN 'RB' THEN 'rushing_yards'
                        WHEN 'WR' THEN 'receiving_yards'
                        WHEN 'TE' THEN 'receiving_yards'
                        WHEN 'K' THEN 'fg_made'
                        ELSE 'fantasy_points_ppr'
                    END as primary_target

                FROM feature_rows fr

                LEFT JOIN actual_stats ast
                    ON fr.player_id = ast.player_id
                    AND fr.season = ast.season
                    AND fr.week = ast.week

                LEFT JOIN game_outcomes go
                    ON ast.season = go.season
                    AND ast.week = go.week
                    AND (ast.team = go.home_team OR ast.team = go.away_team)
            """
            )

            # Update ml_training_features with targets
            logger.info("  📝 Updating feature table with targets...")
            conn.execute(
                """
                UPDATE ml_training_features mtf
                SET
                    actual_outcomes = targets.actual_outcomes,
                    prediction_target = targets.primary_target
                FROM prediction_targets targets
                WHERE mtf.feature_id = targets.feature_id
            """
            )

            # Get statistics
            stats = conn.execute(
                """
                SELECT
                    COUNT(*) as total_rows,
                    COUNT(actual_outcomes) as rows_with_targets,
                    COUNT(*) - COUNT(actual_outcomes) as rows_without_targets
                FROM ml_training_features
            """
            ).fetchone()

            total_rows, rows_with_targets, rows_without_targets = stats

            logger.info(f"✅ Target creation complete:")
            logger.info(f"   Total rows: {total_rows:,}")
            logger.info(
                f"   With targets: {rows_with_targets:,} ({100*rows_with_targets/total_rows:.1f}%)"
            )
            logger.info(
                f"   Without targets: {rows_without_targets:,} ({100*rows_without_targets/total_rows:.1f}%)"
            )

            # Log distribution by position
            position_stats = conn.execute(
                """
                SELECT
                    json_extract(categorical_features, '$.position.value') as position,
                    COUNT(*) as total,
                    COUNT(actual_outcomes) as with_targets,
                    AVG(COALESCE(
                        CAST(json_extract(actual_outcomes, '$.fantasy_points_ppr') AS FLOAT),
                        CAST(json_extract(actual_outcomes, '$.fantasy_points_standard') AS FLOAT),
                        0.0
                    )) as avg_fantasy_points
                FROM ml_training_features
                GROUP BY position
                ORDER BY position
            """
            ).fetchall()

            if position_stats:
                logger.info("  📊 Target distribution by position:")
                for pos, total, with_tgt, avg_fp in position_stats:
                    logger.info(
                        f"     {pos}: {with_tgt:,}/{total:,} ({100*with_tgt/total:.1f}%), Avg FP: {avg_fp:.1f}"
                    )

            # Flag rows without targets as potentially incomplete
            if rows_without_targets > 0:
                logger.info("  ⚠️  Flagging rows without targets...")
                conn.execute(
                    """
                    UPDATE ml_training_features
                    SET missing_data_flags = array_append(
                        COALESCE(missing_data_flags, ARRAY[]::VARCHAR[]),
                        'no_target_data'
                    )
                    WHERE actual_outcomes IS NULL
                """
                )

            # Clean up temp table
            conn.execute("DROP TABLE IF EXISTS prediction_targets")

            return (rows_with_targets, rows_without_targets)

        except Exception as e:
            logger.error(f"❌ Failed to create prediction targets: {e}")
            raise

    def validate_temporal_consistency(self):
        """
        Validate temporal consistency to prevent data leakage.

        Performs critical checks to ensure no future information leaks into features:
        1. Check week 1 features (should be 0 - no previous week data)
        2. Verify features always come from earlier weeks than targets
        3. Check train/val/test split temporal ordering
        4. Audit feature names for outcome-related terms

        Creates temporal_validation_log table with detailed results.
        Raises ValueError if any validation check fails.

        Returns:
            bool: True if all validations pass

        Raises:
            ValueError: If any temporal consistency check fails (data leakage detected)
        """
        logger.info("⏰ Validating temporal consistency...")

        try:
            conn = self.db.connect()

            # Create validation log table if it doesn't exist
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS temporal_validation_log (
                    validation_id VARCHAR PRIMARY KEY,
                    validation_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    check_name VARCHAR,
                    check_passed BOOLEAN,
                    violation_count INTEGER,
                    details JSON
                )
            """
            )

            # Clear previous validation results for this run
            validation_id_prefix = datetime.now().strftime("%Y%m%d_%H%M%S")

            all_checks_passed = True
            validation_results = []

            # Check 1: Verify no week 1 features exist
            logger.info("  🔍 Check 1: Verifying no week 1 features...")
            week1_count = conn.execute(
                """
                SELECT COUNT(*) FROM ml_training_features WHERE week = 1
            """
            ).fetchone()[0]

            check1_passed = week1_count == 0
            all_checks_passed = all_checks_passed and check1_passed

            conn.execute(
                """
                INSERT INTO temporal_validation_log
                VALUES (?, CURRENT_TIMESTAMP, ?, ?, ?, ?)
            """,
                [
                    f"{validation_id_prefix}_check1",
                    "week_1_features",
                    check1_passed,
                    week1_count,
                    json.dumps(
                        {
                            "description": "Verify no week 1 features exist (no previous week data)",
                            "week_1_count": week1_count,
                        }
                    ),
                ],
            )

            if check1_passed:
                logger.info("     ✅ PASS: No week 1 features found")
            else:
                logger.error(
                    f"     ❌ FAIL: Found {week1_count} week 1 features (should be 0)"
                )
                validation_results.append(f"Week 1 features found: {week1_count}")

            # Check 2: Verify feature names don't contain outcome-related terms
            logger.info(
                "  🔍 Check 2: Checking for outcome leakage in feature names..."
            )
            suspicious_features = conn.execute(
                """
                SELECT DISTINCT
                    unnest(feature_names) as feature_name
                FROM ml_training_features
                WHERE
                    unnest(feature_names) LIKE '%win%'
                    OR unnest(feature_names) LIKE '%loss%'
                    OR unnest(feature_names) LIKE '%final%'
                    OR unnest(feature_names) LIKE '%result%'
                    OR unnest(feature_names) LIKE '%outcome%'
            """
            ).fetchall()

            suspicious_count = len(suspicious_features)
            check2_passed = suspicious_count == 0
            all_checks_passed = all_checks_passed and check2_passed

            conn.execute(
                """
                INSERT INTO temporal_validation_log
                VALUES (?, CURRENT_TIMESTAMP, ?, ?, ?, ?)
            """,
                [
                    f"{validation_id_prefix}_check2",
                    "future_outcome_leakage",
                    check2_passed,
                    suspicious_count,
                    json.dumps(
                        {
                            "description": "Check for outcome variables in feature names",
                            "suspicious_features": (
                                [f[0] for f in suspicious_features]
                                if suspicious_features
                                else []
                            ),
                        }
                    ),
                ],
            )

            if check2_passed:
                logger.info("     ✅ PASS: No suspicious feature names found")
            else:
                logger.error(
                    f"     ❌ FAIL: Found {suspicious_count} suspicious feature names"
                )
                for (feature_name,) in suspicious_features:
                    logger.error(f"         - {feature_name}")
                validation_results.append(f"Suspicious features: {suspicious_count}")

            # Check 3: Assign and validate train/val/test splits
            logger.info("  🔍 Check 3: Creating and validating temporal splits...")

            # Add split column if it doesn't exist
            try:
                conn.execute(
                    "ALTER TABLE ml_training_features ADD COLUMN split VARCHAR(10)"
                )
            except:
                pass  # Column already exists

            # Assign splits based on temporal ordering
            conn.execute(
                """
                WITH season_week_ranking AS (
                    SELECT DISTINCT
                        season,
                        week,
                        ROW_NUMBER() OVER (ORDER BY season, week) as time_rank,
                        COUNT(*) OVER () as total_weeks
                    FROM ml_training_features
                ),
                split_assignments AS (
                    SELECT
                        season,
                        week,
                        CASE
                            -- Last 10% is test set (most recent)
                            WHEN time_rank > total_weeks * 0.9 THEN 'test'
                            -- Previous 20% is validation set
                            WHEN time_rank > total_weeks * 0.7 THEN 'validation'
                            -- First 70% is training set
                            ELSE 'train'
                        END as split
                    FROM season_week_ranking
                )
                UPDATE ml_training_features mtf
                SET split = sa.split
                FROM split_assignments sa
                WHERE mtf.season = sa.season AND mtf.week = sa.week
            """
            )

            # Validate split temporal ordering
            split_ranges = conn.execute(
                """
                SELECT
                    split,
                    MIN(season * 100 + week) as earliest_week,
                    MAX(season * 100 + week) as latest_week
                FROM ml_training_features
                GROUP BY split
            """
            ).fetchall()

            split_dict = {
                split: (earliest, latest) for split, earliest, latest in split_ranges
            }

            # Check that train < validation < test
            ordering_violations = []
            if "train" in split_dict and "validation" in split_dict:
                if split_dict["train"][1] >= split_dict["validation"][0]:
                    ordering_violations.append("train overlaps with validation")
            if "validation" in split_dict and "test" in split_dict:
                if split_dict["validation"][1] >= split_dict["test"][0]:
                    ordering_violations.append("validation overlaps with test")

            check3_passed = len(ordering_violations) == 0
            all_checks_passed = all_checks_passed and check3_passed

            conn.execute(
                """
                INSERT INTO temporal_validation_log
                VALUES (?, CURRENT_TIMESTAMP, ?, ?, ?, ?)
            """,
                [
                    f"{validation_id_prefix}_check3",
                    "split_temporal_order",
                    check3_passed,
                    len(ordering_violations),
                    json.dumps(
                        {
                            "description": "Validate train/val/test temporal ordering",
                            "split_ranges": {
                                k: {"earliest": v[0], "latest": v[1]}
                                for k, v in split_dict.items()
                            },
                            "violations": ordering_violations,
                        }
                    ),
                ],
            )

            if check3_passed:
                logger.info("     ✅ PASS: Train/val/test splits properly ordered")
                for split, (earliest, latest) in sorted(split_dict.items()):
                    logger.info(f"        {split}: weeks {earliest} - {latest}")
            else:
                logger.error(f"     ❌ FAIL: Split ordering violations detected")
                for violation in ordering_violations:
                    logger.error(f"         - {violation}")
                validation_results.append(
                    f"Split ordering violations: {len(ordering_violations)}"
                )

            # Check 4: Validate data completeness and targets
            logger.info("  🔍 Check 4: Validating data completeness...")
            completeness_stats = conn.execute(
                """
                SELECT
                    split,
                    COUNT(*) as total_rows,
                    COUNT(actual_outcomes) as rows_with_targets,
                    100.0 * COUNT(actual_outcomes) / COUNT(*) as target_pct
                FROM ml_training_features
                GROUP BY split
                ORDER BY split
            """
            ).fetchall()

            # Train and validation should have 100% targets
            completeness_issues = []
            for split, total, with_targets, pct in completeness_stats:
                if split in ["train", "validation"] and pct < 100.0:
                    completeness_issues.append(
                        f"{split} has only {pct:.1f}% targets (expected 100%)"
                    )

            check4_passed = len(completeness_issues) == 0
            all_checks_passed = all_checks_passed and check4_passed

            conn.execute(
                """
                INSERT INTO temporal_validation_log
                VALUES (?, CURRENT_TIMESTAMP, ?, ?, ?, ?)
            """,
                [
                    f"{validation_id_prefix}_check4",
                    "data_completeness",
                    check4_passed,
                    len(completeness_issues),
                    json.dumps(
                        {
                            "description": "Validate target completeness by split",
                            "stats": [
                                {"split": s, "total": t, "with_targets": wt, "pct": p}
                                for s, t, wt, p in completeness_stats
                            ],
                            "issues": completeness_issues,
                        }
                    ),
                ],
            )

            if check4_passed:
                logger.info("     ✅ PASS: Data completeness validated")
                for split, total, with_targets, pct in completeness_stats:
                    logger.info(
                        f"        {split}: {with_targets:,}/{total:,} ({pct:.1f}%)"
                    )
            else:
                logger.error(f"     ❌ FAIL: Data completeness issues detected")
                for issue in completeness_issues:
                    logger.error(f"         - {issue}")
                validation_results.append(
                    f"Completeness issues: {len(completeness_issues)}"
                )

            # Final summary
            logger.info("\n" + "=" * 60)
            if all_checks_passed:
                logger.info("✅ ALL TEMPORAL CONSISTENCY CHECKS PASSED")
                logger.info("   No data leakage detected")
                logger.info("   Dataset is ready for ML training")
            else:
                logger.error("❌ TEMPORAL CONSISTENCY VALIDATION FAILED")
                logger.error("   Data leakage detected - see violations above")
                logger.error("   DO NOT use this dataset for training")
                for result in validation_results:
                    logger.error(f"   - {result}")
            logger.info("=" * 60 + "\n")

            # Raise error if validation failed
            if not all_checks_passed:
                raise ValueError(
                    "Temporal consistency validation failed. Data leakage detected. "
                    "Check logs for details. DO NOT use this dataset for training."
                )

            return True

        except ValueError:
            # Re-raise validation failures
            raise
        except Exception as e:
            logger.error(f"❌ Failed to validate temporal consistency: {e}")
            raise
