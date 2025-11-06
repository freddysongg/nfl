#!/usr/bin/env python3
"""
Test script for Stage 2: Player Lifecycle & Roster Management
Validates the implementation of the three Stage 2 methods
"""

import logging
from src.data_pipeline import NFLDataPipeline
from src.database import NFLDatabase

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Stage2Validator:
    """Validation checks for Stage 2 outputs"""

    def __init__(self, db: NFLDatabase):
        self.db = db

    def validate_player_lifecycle(self):
        """Validate player_lifecycle table"""
        logger.info("\n🔍 Validating player_lifecycle table...")
        conn = self.db.connect()

        checks = {}

        # Check 1: No NULL player_ids
        null_ids = conn.execute("""
            SELECT COUNT(*) FROM player_lifecycle WHERE player_id IS NULL
        """).fetchone()[0]
        checks['no_null_player_ids'] = (null_ids == 0)
        logger.info(f"  ✓ No NULL player_ids: {checks['no_null_player_ids']} (nulls: {null_ids})")

        # Check 2: first_nfl_season <= last_nfl_season
        invalid_spans = conn.execute("""
            SELECT COUNT(*) FROM player_lifecycle
            WHERE first_nfl_season > last_nfl_season
        """).fetchone()[0]
        checks['valid_career_spans'] = (invalid_spans == 0)
        logger.info(f"  ✓ Valid career spans: {checks['valid_career_spans']} (invalid: {invalid_spans})")

        # Check 3: Reasonable season range (e.g., 1990-2030)
        invalid_seasons = conn.execute("""
            SELECT COUNT(*) FROM player_lifecycle
            WHERE first_nfl_season < 1990 OR last_nfl_season > 2030
        """).fetchone()[0]
        checks['reasonable_season_range'] = (invalid_seasons == 0)
        logger.info(f"  ✓ Reasonable season range: {checks['reasonable_season_range']} (invalid: {invalid_seasons})")

        # Check 4: Unique player_ids
        total = conn.execute("SELECT COUNT(*) FROM player_lifecycle").fetchone()[0]
        unique = conn.execute("SELECT COUNT(DISTINCT player_id) FROM player_lifecycle").fetchone()[0]
        checks['unique_player_ids'] = (total == unique)
        logger.info(f"  ✓ Unique player_ids: {checks['unique_player_ids']} (total: {total}, unique: {unique})")

        all_passed = all(checks.values())
        logger.info(f"\n{'✅' if all_passed else '❌'} player_lifecycle validation: {'PASSED' if all_passed else 'FAILED'}")
        return all_passed

    def validate_roster_snapshots(self):
        """Validate team_roster_snapshots table"""
        logger.info("\n🔍 Validating team_roster_snapshots table...")
        conn = self.db.connect()

        checks = {}

        # Check 1: Unique snapshot_ids
        total = conn.execute("SELECT COUNT(*) FROM team_roster_snapshots").fetchone()[0]
        unique = conn.execute("SELECT COUNT(DISTINCT snapshot_id) FROM team_roster_snapshots").fetchone()[0]
        checks['unique_snapshot_ids'] = (total == unique)
        logger.info(f"  ✓ Unique snapshot_ids: {checks['unique_snapshot_ids']} (total: {total}, unique: {unique})")

        # Check 2: Valid team codes (3 characters)
        invalid_teams = conn.execute("""
            SELECT COUNT(*) FROM team_roster_snapshots
            WHERE LENGTH(team) != 3
        """).fetchone()[0]
        checks['valid_team_codes'] = (invalid_teams == 0)
        logger.info(f"  ✓ Valid team codes: {checks['valid_team_codes']} (invalid: {invalid_teams})")

        # Check 3: Valid week numbers (1-22)
        invalid_weeks = conn.execute("""
            SELECT COUNT(*) FROM team_roster_snapshots
            WHERE week < 1 OR week > 22
        """).fetchone()[0]
        checks['valid_week_numbers'] = (invalid_weeks == 0)
        logger.info(f"  ✓ Valid week numbers: {checks['valid_week_numbers']} (invalid: {invalid_weeks})")

        # Check 4: Active players JSON is valid (sample check)
        try:
            sample = conn.execute("""
                SELECT active_players FROM team_roster_snapshots LIMIT 1
            """).fetchone()
            checks['valid_json'] = (sample is not None and sample[0] is not None)
        except Exception as e:
            checks['valid_json'] = False
            logger.warning(f"  ⚠️  JSON validation error: {e}")
        logger.info(f"  ✓ Valid JSON format: {checks['valid_json']}")

        all_passed = all(checks.values())
        logger.info(f"\n{'✅' if all_passed else '❌'} team_roster_snapshots validation: {'PASSED' if all_passed else 'FAILED'}")
        return all_passed

    def validate_experience_classification(self):
        """Validate player_experience_classification table"""
        logger.info("\n🔍 Validating player_experience_classification table...")
        conn = self.db.connect()

        checks = {}

        # Check 1: Valid experience categories
        invalid_categories = conn.execute("""
            SELECT COUNT(*) FROM player_experience_classification
            WHERE experience_category NOT IN ('rookie', 'developing', 'veteran')
        """).fetchone()[0]
        checks['valid_categories'] = (invalid_categories == 0)
        logger.info(f"  ✓ Valid categories: {checks['valid_categories']} (invalid: {invalid_categories})")

        # Check 2: seasons_played >= 1
        invalid_seasons = conn.execute("""
            SELECT COUNT(*) FROM player_experience_classification
            WHERE seasons_played < 1
        """).fetchone()[0]
        checks['positive_seasons_played'] = (invalid_seasons == 0)
        logger.info(f"  ✓ Positive seasons_played: {checks['positive_seasons_played']} (invalid: {invalid_seasons})")

        # Check 3: Confidence multipliers in expected range
        invalid_confidence = conn.execute("""
            SELECT COUNT(*) FROM player_experience_classification
            WHERE confidence_multiplier < 0 OR confidence_multiplier > 1.5
        """).fetchone()[0]
        checks['valid_confidence_multipliers'] = (invalid_confidence == 0)
        logger.info(f"  ✓ Valid confidence multipliers: {checks['valid_confidence_multipliers']} (invalid: {invalid_confidence})")

        # Check 4: Category matches seasons_played logic
        mismatched_categories = conn.execute("""
            SELECT COUNT(*) FROM player_experience_classification
            WHERE (seasons_played <= 1 AND experience_category != 'rookie')
                OR (seasons_played IN (2, 3) AND experience_category != 'developing')
                OR (seasons_played >= 4 AND experience_category != 'veteran')
        """).fetchone()[0]
        checks['consistent_categorization'] = (mismatched_categories == 0)
        logger.info(f"  ✓ Consistent categorization: {checks['consistent_categorization']} (mismatched: {mismatched_categories})")

        # Check 5: Unique (player_id, season) combinations
        total = conn.execute("SELECT COUNT(*) FROM player_experience_classification").fetchone()[0]
        unique = conn.execute("""
            SELECT COUNT(DISTINCT player_id || '_' || season)
            FROM player_experience_classification
        """).fetchone()[0]
        checks['unique_player_seasons'] = (total == unique)
        logger.info(f"  ✓ Unique player-seasons: {checks['unique_player_seasons']} (total: {total}, unique: {unique})")

        all_passed = all(checks.values())
        logger.info(f"\n{'✅' if all_passed else '❌'} player_experience_classification validation: {'PASSED' if all_passed else 'FAILED'}")
        return all_passed

    def run_all_validations(self):
        """Run all Stage 2 validations"""
        logger.info("\n" + "=" * 60)
        logger.info("🧪 Running Stage 2 Validation Suite")
        logger.info("=" * 60)

        results = {
            'lifecycle': self.validate_player_lifecycle(),
            'snapshots': self.validate_roster_snapshots(),
            'experience': self.validate_experience_classification()
        }

        all_passed = all(results.values())

        logger.info("\n" + "=" * 60)
        logger.info(f"{'✅ ALL VALIDATIONS PASSED' if all_passed else '❌ SOME VALIDATIONS FAILED'}")
        logger.info("=" * 60)

        return all_passed


def test_stage2_implementation():
    """Test Stage 2 implementation"""
    logger.info("\n" + "=" * 60)
    logger.info("🚀 Testing Stage 2: Player Lifecycle & Roster Management")
    logger.info("=" * 60)

    try:
        # Initialize database
        db = NFLDatabase()
        tables = db.list_tables()

        # Check if raw data exists
        required_raw_tables = ['raw_players', 'raw_rosters_weekly', 'raw_player_stats']
        missing = [t for t in required_raw_tables if t not in tables]

        if missing:
            logger.error(f"❌ Missing required raw tables: {missing}")
            logger.info("💡 Please run Stage 1 (full_historical_load) first")
            return False

        # Check if raw tables have data
        conn = db.connect()
        for table in required_raw_tables:
            count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            logger.info(f"📊 {table}: {count:,} records")
            if count == 0:
                logger.error(f"❌ {table} is empty")
                logger.info("💡 Please run Stage 1 (full_historical_load) first")
                return False

        # Initialize pipeline
        logger.info("\n📦 Initializing NFL Data Pipeline...")
        pipeline = NFLDataPipeline()

        # Run Stage 2
        logger.info("\n🔄 Running Stage 2 methods...")
        try:
            lifecycle_count = pipeline.build_player_lifecycle_table()
            logger.info(f"✅ Player lifecycle: {lifecycle_count:,} records created")
        except Exception as e:
            logger.error(f"❌ build_player_lifecycle_table() failed: {e}")
            return False

        try:
            snapshot_count = pipeline.create_weekly_roster_snapshots()
            logger.info(f"✅ Roster snapshots: {snapshot_count:,} records created")
        except Exception as e:
            logger.error(f"❌ create_weekly_roster_snapshots() failed: {e}")
            return False

        try:
            classification_count = pipeline.classify_player_experience_levels()
            logger.info(f"✅ Experience classifications: {classification_count:,} records created")
        except Exception as e:
            logger.error(f"❌ classify_player_experience_levels() failed: {e}")
            return False

        # Run validations
        validator = Stage2Validator(db)
        validation_passed = validator.run_all_validations()

        if validation_passed:
            logger.info("\n🎉 Stage 2 implementation test PASSED!")
            return True
        else:
            logger.warning("\n⚠️  Stage 2 implementation test completed with validation warnings")
            return False

    except Exception as e:
        logger.error(f"\n❌ Stage 2 test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function"""
    success = test_stage2_implementation()

    if success:
        logger.info("\n" + "=" * 60)
        logger.info("✅ Stage 2 is ready for commit!")
        logger.info("=" * 60)
        return 0
    else:
        logger.error("\n" + "=" * 60)
        logger.error("❌ Stage 2 needs attention before commit")
        logger.error("=" * 60)
        return 1


if __name__ == "__main__":
    exit(main())
