"""
Batch Processing and Hash Checking for NFL Data Pipeline
Implements efficient batch streaming with duplicate detection
"""

import polars as pl
import hashlib
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass

from .database import NFLDatabase
from .config import config

logger = logging.getLogger(__name__)


@dataclass
class BatchResult:
    """Result of a batch processing operation"""
    table_name: str
    total_rows: int
    new_rows: int
    updated_rows: int
    skipped_rows: int
    processing_time: float
    batch_hash: str


class DataHasher:
    """Handles data hashing for duplicate detection"""

    @staticmethod
    def hash_dataframe(df: pl.DataFrame, key_columns: List[str]) -> pl.DataFrame:
        """Add hash column to dataframe based on key columns"""

        hash_data = df.select(key_columns).to_pandas()

        row_hashes = []
        for _, row in hash_data.iterrows():
            row_dict = {k: v for k, v in row.to_dict().items() if v is not None}
            row_str = json.dumps(row_dict, sort_keys=True, default=str)
            row_hash = hashlib.md5(row_str.encode()).hexdigest()
            row_hashes.append(row_hash)

        return df.with_columns(pl.Series("row_hash", row_hashes))

    @staticmethod
    def hash_batch(df: pl.DataFrame) -> str:
        """Generate hash for entire batch"""        
        content_str = str(df.shape) + str(df.columns) + str(df.height * df.width)
        return hashlib.sha256(content_str.encode()).hexdigest()[:16]


class BatchProcessor:
    """Handles batch processing with hash checking and streaming"""

    def __init__(self, db: NFLDatabase, batch_size: int = None):
        self.db = db
        self.batch_size = batch_size or config.data_collection_config["batch_size"]
        self.hasher = DataHasher()

        self.key_columns = {
            'raw_player_stats': ['player_id', 'season', 'week', 'season_type'],
            'raw_team_stats': ['team', 'season', 'week', 'season_type'],
            'raw_depth_charts': ['team', 'season', 'gsis_id', 'pos_name'],
            'raw_rosters_weekly': ['player_id', 'team', 'season', 'week'],
            'raw_schedules': ['game_id'],
            'raw_nextgen_passing': ['player_id', 'season', 'week', 'season_type'],
            'raw_nextgen_rushing': ['player_id', 'season', 'week', 'season_type'],
            'raw_nextgen_receiving': ['player_id', 'season', 'week', 'season_type'],
            'raw_snap_counts': ['pfr_player_id', 'season', 'week'],
            'raw_pbp': ['play_id', 'game_id'],
            'raw_players': ['player_id'],
            'raw_ftn_charting': ['game_id', 'play_id'],
            'raw_participation': ['nflverse_game_id', 'play_id', 'gsis_player_id'],
            'raw_draft_picks': ['season', 'round', 'pick'],
            'raw_combine': ['season', 'pfr_id']
        }

        logger.info(f"BatchProcessor initialized with batch_size={self.batch_size}")

    def process_dataframe_to_table(self, df: pl.DataFrame, table_name: str,
                                  operation_name: str = "data_load") -> BatchResult:
        """
        Process dataframe in batches with hash checking and streaming to database

        Args:
            df: Source dataframe
            table_name: Target table name
            operation_name: Description for logging

        Returns:
            BatchResult with processing statistics
        """
        start_time = datetime.now()

        batch_hash = self.hasher.hash_batch(df)

        logger.info(f"🔄 Processing {operation_name}: {len(df):,} rows → {table_name}")
        logger.info(f"📦 Batch hash: {batch_hash}")

        key_cols = self.key_columns.get(table_name, ['player_id', 'season', 'week'])

        available_key_cols = [col for col in key_cols if col in df.columns]
        if not available_key_cols:
            available_key_cols = df.columns[:3]
            logger.warning(f"⚠️  Using fallback key columns for {table_name}: {available_key_cols}")

        df_with_hash = self.hasher.hash_dataframe(df, available_key_cols)

        existing_hashes = self._get_existing_hashes(table_name)

        new_data, updated_data, skipped_count = self._categorize_data(
            df_with_hash, existing_hashes, available_key_cols
        )

        total_new = 0
        total_updated = 0

        if not new_data.is_empty():
            total_new = self._stream_batches(new_data, table_name, "INSERT", operation_name)

        if not updated_data.is_empty():
            total_updated = self._stream_batches(updated_data, table_name, "UPDATE", operation_name)

        processing_time = (datetime.now() - start_time).total_seconds()

        result = BatchResult(
            table_name=table_name,
            total_rows=len(df),
            new_rows=total_new,
            updated_rows=total_updated,
            skipped_rows=skipped_count,
            processing_time=processing_time,
            batch_hash=batch_hash
        )

        self._log_batch_result(result, operation_name)
        return result

    def _get_existing_hashes(self, table_name: str) -> set:
        """Get existing row hashes from database"""
        try:
            conn = self.db.connect()

            tables = conn.execute("SHOW TABLES").fetchall()
            table_exists = any(table[0] == table_name for table in tables)

            if not table_exists:
                return set()

            columns = conn.execute(f"DESCRIBE {table_name}").fetchall()
            has_hash_col = any(col[0] == 'row_hash' for col in columns)

            if not has_hash_col:
                conn.execute(f"ALTER TABLE {table_name} ADD COLUMN IF NOT EXISTS row_hash VARCHAR")
                return set()

            result = conn.execute(f"SELECT DISTINCT row_hash FROM {table_name} WHERE row_hash IS NOT NULL").fetchall()
            return {row[0] for row in result}

        except Exception as e:
            logger.warning(f"⚠️  Could not get existing hashes for {table_name}: {e}")
            return set()

    def _categorize_data(self, df_with_hash: pl.DataFrame, existing_hashes: set,
                        key_cols: List[str]) -> Tuple[pl.DataFrame, pl.DataFrame, int]:
        """Categorize data into new, updated, and skipped"""

        if not existing_hashes:
            return df_with_hash, pl.DataFrame(), 0

        new_mask = ~df_with_hash['row_hash'].is_in(list(existing_hashes))

        new_data = df_with_hash.filter(new_mask)
        potential_updates = df_with_hash.filter(~new_mask)

        skipped_count = len(potential_updates)

        return new_data, pl.DataFrame(), skipped_count

    def _stream_batches(self, df: pl.DataFrame, table_name: str,
                       operation: str, operation_name: str) -> int:
        """Stream data to database in batches"""

        if df.is_empty():
            return 0

        total_processed = 0
        total_batches = (len(df) + self.batch_size - 1) // self.batch_size

        for i in range(0, len(df), self.batch_size):
            batch_num = (i // self.batch_size) + 1
            batch_df = df.slice(i, self.batch_size)

            try:
                conn = self.db.connect()
                columns = conn.execute(f"DESCRIBE {table_name}").fetchall()
                table_has_hash_col = any(col[0] == 'row_hash' for col in columns)

                if 'row_hash' in batch_df.columns and not table_has_hash_col:
                    db_batch = batch_df.drop('row_hash')
                else:
                    db_batch = batch_df

                rows_inserted = self.db.store_dataframe(db_batch, table_name)
                total_processed += rows_inserted

                if batch_num % 10 == 0 or batch_num == total_batches:
                    progress = (batch_num / total_batches) * 100
                    logger.info(f"  📈 {operation_name} progress: {progress:.1f}% ({batch_num}/{total_batches} batches)")

            except Exception as e:
                logger.error(f"❌ Batch {batch_num} failed for {table_name}: {e}")
                continue

        return total_processed

    def _log_batch_result(self, result: BatchResult, operation_name: str):
        """Log batch processing results"""
        efficiency = ((result.new_rows + result.updated_rows) / result.total_rows * 100) if result.total_rows > 0 else 0

        logger.info(f"✅ {operation_name} completed:")
        logger.info(f"  📊 Total: {result.total_rows:,} | New: {result.new_rows:,} | Updated: {result.updated_rows:,} | Skipped: {result.skipped_rows:,}")
        logger.info(f"  ⚡ Efficiency: {efficiency:.1f}% | Time: {result.processing_time:.2f}s")

    def get_table_statistics(self, table_name: str) -> Dict[str, Any]:
        """Get current table statistics"""
        try:
            conn = self.db.connect()

            count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]

            hash_stats = {}
            try:
                hash_count = conn.execute(f"SELECT COUNT(DISTINCT row_hash) FROM {table_name} WHERE row_hash IS NOT NULL").fetchone()[0]
                hash_stats = {
                    'unique_hashes': hash_count,
                    'hash_coverage': (hash_count / count * 100) if count > 0 else 0
                }
            except:
                pass

            return {
                'table_name': table_name,
                'total_rows': count,
                **hash_stats
            }

        except Exception as e:
            logger.warning(f"⚠️  Could not get stats for {table_name}: {e}")
            return {'table_name': table_name, 'total_rows': 0}


class ProgressTracker:
    """Tracks progress across multiple table operations"""

    def __init__(self):
        self.operations: List[BatchResult] = []
        self.start_time = datetime.now()

    def add_result(self, result: BatchResult):
        """Add a batch result to tracking"""
        self.operations.append(result)

    def get_summary(self) -> Dict[str, Any]:
        """Get overall progress summary"""
        if not self.operations:
            return {'status': 'no_operations'}

        total_rows = sum(op.total_rows for op in self.operations)
        total_new = sum(op.new_rows for op in self.operations)
        total_updated = sum(op.updated_rows for op in self.operations)
        total_skipped = sum(op.skipped_rows for op in self.operations)
        total_time = (datetime.now() - self.start_time).total_seconds()

        efficiency = ((total_new + total_updated) / total_rows * 100) if total_rows > 0 else 0

        return {
            'total_operations': len(self.operations),
            'total_rows_processed': total_rows,
            'total_new_rows': total_new,
            'total_updated_rows': total_updated,
            'total_skipped_rows': total_skipped,
            'overall_efficiency': efficiency,
            'total_processing_time': total_time,
            'operations_per_second': len(self.operations) / total_time if total_time > 0 else 0,
            'tables_processed': [op.table_name for op in self.operations]
        }

    def log_summary(self, operation_name: str = "Pipeline"):
        """Log overall progress summary"""
        summary = self.get_summary()

        if summary.get('status') == 'no_operations':
            logger.info(f"📋 {operation_name}: No operations completed")
            return

        logger.info(f"📋 {operation_name} Summary:")
        logger.info(f"  🎯 Operations: {summary['total_operations']}")
        logger.info(f"  📊 Rows: {summary['total_rows_processed']:,} total | {summary['total_new_rows']:,} new | {summary['total_updated_rows']:,} updated")
        logger.info(f"  ⚡ Efficiency: {summary['overall_efficiency']:.1f}% | Time: {summary['total_processing_time']:.2f}s")
        logger.info(f"  📋 Tables: {', '.join(summary['tables_processed'])}")


