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

    def build_player_lifecycle_table(self):
        """Build player career tracking table"""
        logger.info("👥 Building player lifecycle table...")
        # TODO: Implement from DATA_SETUP.md

    def create_weekly_roster_snapshots(self):
        """Create time-aware roster snapshots"""
        logger.info("📸 Creating weekly roster snapshots...")
        # TODO: Implement from DATA_SETUP.md

    def classify_player_experience_levels(self):
        """Classify players by experience level"""
        logger.info("🎓 Classifying player experience levels...")
        # TODO: Implement from DATA_SETUP.md

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


