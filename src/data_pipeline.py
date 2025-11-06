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
        conn.execute("""
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
        """)

        # Update for home teams
        conn.execute("""
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
        """)

        logger.info("     ✅ Rest days and divisional info updated")

    def _update_home_away_splits(self, conn):
        """Calculate and update home/away performance splits."""
        conn.execute("""
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
        """)

        logger.info("     ✅ Home/away splits calculated")

    def _update_opponent_history(self, conn):
        """Build opponent matchup history (last 3 games vs same opponent)."""
        conn.execute("""
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
        """)

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
            conn.execute(f"""
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
            """)

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

            conn.execute(f"""
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
            """)

        logger.info("     ✅ Opponent stats allowed calculated")

    def _log_matchup_features_summary(self, conn):
        """Log summary statistics for matchup features."""
        try:
            # Count records with each feature populated
            summary = conn.execute("""
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
            """).fetchone()

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

            logger.info(f"  📊 Found {len(seasons_with_pbp)} seasons with PBP data: {seasons_with_pbp}")
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
                logger.info(f"  📊 Processing season {season} with team stats fallback...")
                records = self._calculate_team_aggregates_from_team_stats(season)
                total_records += records

            # Step 4: Validate results
            self._validate_team_aggregates()

            logger.info(f"✅ Team aggregates created successfully: {total_records:,} total records")

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

            result = conn.execute("""
                SELECT DISTINCT season
                FROM raw_pbp
                WHERE season >= 2022
                  AND season IS NOT NULL
                ORDER BY season
            """).fetchall()

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
            result = conn.execute("""
                SELECT DISTINCT season
                FROM raw_player_stats
                WHERE season IS NOT NULL
                ORDER BY season
            """).fetchall()

            if result:
                return [row[0] for row in result]

            # Fallback to raw_team_stats
            result = conn.execute("""
                SELECT DISTINCT season
                FROM raw_team_stats
                WHERE season IS NOT NULL
                ORDER BY season
            """).fetchall()

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
        teams_weeks = conn.execute("""
            SELECT DISTINCT posteam AS team, week
            FROM raw_pbp
            WHERE season = ?
              AND posteam IS NOT NULL
              AND week IS NOT NULL
              AND season_type = 'REG'
            ORDER BY team, week
        """, [season]).fetchall()

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
                off_metrics = self._calculate_offensive_metrics(team, season, week, conn)

                # Calculate defensive metrics
                def_metrics = self._calculate_defensive_metrics(team, season, week, conn)

                # Calculate situational metrics
                sit_metrics = self._calculate_situational_metrics(team, season, week, conn)

                # Combine all metrics
                team_features = {
                    'team': team,
                    'season': season,
                    'week': week,
                    **off_metrics,
                    **def_metrics,
                    **sit_metrics
                }

                results.append(team_features)

            except Exception as e:
                logger.warning(f"    ⚠️  Failed to calculate metrics for {team} week {week}: {e}")
                continue

        # Bulk insert results
        if results:
            df = pl.DataFrame(results)
            conn.execute("INSERT INTO team_rolling_features SELECT * FROM df")
            logger.info(f"    ✅ Inserted {len(results):,} team feature records for {season}")
            return len(results)

        return 0

    def _calculate_offensive_metrics(self, team: str, season: int, week: int, conn) -> Dict[str, float]:
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

        rz_result = conn.execute(rz_query, [team, season, start_week, end_week]).fetchone()

        return {
            'off_epa_per_play_last3': float(result[0]) if result and result[0] is not None else 0.0,
            'off_success_rate_last3': float(result[1]) if result and result[1] is not None else 0.5,
            'off_explosive_play_rate': float(result[2]) if result and result[2] is not None else 0.0,
            'off_third_down_conv': float(result[3]) if result and result[3] is not None else 0.0,
            'off_red_zone_efficiency': float(rz_result[0]) if rz_result and rz_result[0] is not None else 0.5
        }

    def _calculate_defensive_metrics(self, team: str, season: int, week: int, conn) -> Dict[str, float]:
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
            'def_epa_per_play_last3': float(result[0]) if result and result[0] is not None else 0.0,
            'def_success_rate_last3': float(result[1]) if result and result[1] is not None else 0.5,
            'def_pressure_rate': float(result[2]) if result and result[2] is not None else 0.0,
            'def_turnover_rate': float(result[3]) if result and result[3] is not None else 0.0
        }

    def _calculate_situational_metrics(self, team: str, season: int, week: int, conn) -> Dict[str, float]:
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

        neutral_result = conn.execute(neutral_query, [team, season, start_week, end_week]).fetchone()

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

        pace_result = conn.execute(pace_query, [team, season, start_week, end_week]).fetchone()

        return {
            'pass_rate_neutral': float(neutral_result[0]) if neutral_result and neutral_result[0] is not None else 0.5,
            'pace_of_play': float(pace_result[0]) if pace_result and pace_result[0] is not None else 65.0,
            'time_of_possession_avg': float(pace_result[1]) if pace_result and pace_result[1] is not None else 30.0
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
        count = conn.execute("""
            SELECT COUNT(*) FROM raw_team_stats WHERE season = ?
        """, [season]).fetchone()[0]

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
                logger.info(f"    ✅ Inserted {len(results):,} team feature records for {season} (fallback)")
                logger.warning(f"    ⚠️  Pre-2022 season: Limited metrics available from team stats")
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
            invalid_epa = conn.execute("""
                SELECT COUNT(*)
                FROM team_rolling_features
                WHERE off_epa_per_play_last3 < -3
                   OR off_epa_per_play_last3 > 3
                   OR def_epa_per_play_last3 < -3
                   OR def_epa_per_play_last3 > 3
            """).fetchone()[0]

            if invalid_epa > 0:
                logger.warning(f"  ⚠️  Found {invalid_epa} records with EPA outside expected range (-3 to +3)")

            # Check success rates
            invalid_success = conn.execute("""
                SELECT COUNT(*)
                FROM team_rolling_features
                WHERE off_success_rate_last3 < 0 OR off_success_rate_last3 > 1
                   OR def_success_rate_last3 < 0 OR def_success_rate_last3 > 1
            """).fetchone()[0]

            if invalid_success > 0:
                logger.warning(f"  ⚠️  Found {invalid_success} records with invalid success rates")

            # Check for NULL values
            null_count = conn.execute("""
                SELECT COUNT(*)
                FROM team_rolling_features
                WHERE off_epa_per_play_last3 IS NULL
                   OR def_epa_per_play_last3 IS NULL
            """).fetchone()[0]

            if null_count > 0:
                logger.warning(f"  ⚠️  Found {null_count} records with NULL EPA values")

            # Log summary
            summary = conn.execute("""
                SELECT
                    COUNT(*) as total_records,
                    COUNT(DISTINCT team) as unique_teams,
                    COUNT(DISTINCT season) as seasons,
                    MIN(season) as earliest_season,
                    MAX(season) as latest_season,
                    AVG(off_epa_per_play_last3) as avg_off_epa,
                    AVG(def_epa_per_play_last3) as avg_def_epa
                FROM team_rolling_features
            """).fetchone()

            if summary:
                logger.info(f"  📊 Team Aggregates Summary:")
                logger.info(f"     Total records: {summary[0]:,}")
                logger.info(f"     Unique teams: {summary[1]}")
                logger.info(f"     Seasons: {summary[2]} ({summary[3]}-{summary[4]})")
                logger.info(f"     Avg offensive EPA: {summary[5]:.3f}" if summary[5] else "     Avg offensive EPA: N/A")
                logger.info(f"     Avg defensive EPA: {summary[6]:.3f}" if summary[6] else "     Avg defensive EPA: N/A")

        except Exception as e:
            logger.warning(f"  ⚠️  Could not validate team aggregates: {e}")

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
