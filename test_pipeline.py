#!/usr/bin/env python3
"""
Test script for NFL Data Pipeline - Stage 1 validation
Tests data collection from nflreadr with limited scope
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.data_pipeline import NFLDataPipeline
from src.database import NFLDatabase
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_stage1_limited():
    """Test Stage 1 with limited 2025 data only"""
    logger.info("🧪 Testing Stage 1 Data Collection (Limited)")
    logger.info("=" * 50)

    test_db_file = "test_nfl_pipeline.duckdb"

    import os
    if os.path.exists(test_db_file):
        os.remove(test_db_file)

    pipeline = NFLDataPipeline(test_db_file)

    test_seasons = [2025]

    try:
        import sys
        sys.path.append('.')
        from setup_database import NFLDatabaseSetup
        db_setup = NFLDatabaseSetup(test_db_file)
        db_setup.setup_database()
        db_setup.close()

        with NFLDatabase(test_db_file) as db:
            before_summary = db.get_database_summary()
            logger.info(f"📊 Before: {before_summary['total_rows']} total rows")

        logger.info("🚀 Starting Stage 1 data collection...")
        pipeline.full_historical_load(test_seasons)

        with NFLDatabase(test_db_file) as db:
            after_summary = db.get_database_summary()
            logger.info(f"📊 After: {after_summary['total_rows']} total rows")

            print("\n📋 Data Collection Results:")
            print("-" * 40)

            for table_name, table_info in after_summary['tables'].items():
                if table_info['row_count'] > 0:
                    print(f"✅ {table_name:<30} {table_info['row_count']:>8,} rows")
                else:
                    print(f"⚪ {table_name:<30} {table_info['row_count']:>8,} rows")

        logger.info("✅ Stage 1 test completed successfully!")
        return True

    except Exception as e:
        logger.error(f"❌ Stage 1 test failed: {e}")
        return False


def test_specific_function():
    """Test a specific nflreadr function"""
    logger.info("🔧 Testing specific nflreadr function")

    try:
        import nflreadpy as nfl

        logger.info("📊 Testing load_player_stats(2025)...")
        stats = nfl.load_player_stats(seasons=2025)

        if not stats.is_empty():
            logger.info(f"✅ Retrieved {len(stats):,} player stat records")
            logger.info(f"📅 Columns: {len(stats.columns)}")

            sample_cols = stats.columns[:10]
            logger.info(f"🔍 Sample columns: {', '.join(sample_cols)}")

            teams = stats.select("team").unique().sort("team")
            logger.info(f"🏈 Teams found: {len(teams)} teams")

        else:
            logger.warning("⚠️  No data returned from load_player_stats")

        return True

    except Exception as e:
        logger.error(f"❌ Function test failed: {e}")
        return False


def test_database_connectivity():
    """Test basic database operations"""
    logger.info("🔗 Testing database connectivity")

    try:
        with NFLDatabase() as db:
            result = db.execute("SELECT 1 as test_value")
            logger.info(f"✅ Database query successful: {result}")

            tables = db.list_tables()
            logger.info(f"📊 Found {len(tables)} tables in database")

            if "raw_player_stats" in tables:
                info = db.get_table_info("raw_player_stats")
                logger.info(f"📋 raw_player_stats: {info['row_count']} rows, {info['column_count']} columns")

        return True

    except Exception as e:
        logger.error(f"❌ Database test failed: {e}")
        return False


def validate_schemas():
    """Validate all required schemas exist"""
    logger.info("🔍 Validating database schemas")

    required_tables = [
        "raw_player_stats",
        "raw_team_stats",
        "raw_depth_charts",
        "raw_rosters_weekly",
        "raw_schedules"
    ]

    try:
        with NFLDatabase() as db:
            existing_tables = db.list_tables()

            all_exist = True
            for table in required_tables:
                if table in existing_tables:
                    logger.info(f"✅ {table} exists")
                else:
                    logger.error(f"❌ {table} missing")
                    all_exist = False

        return all_exist

    except Exception as e:
        logger.error(f"❌ Schema validation failed: {e}")
        return False


def main():
    """Run all tests"""
    print("🧪 NFL Data Pipeline Test Suite")
    print("=" * 40)

    tests = [
        ("Database Connectivity", test_database_connectivity),
        ("Schema Validation", validate_schemas),
        ("nflreadr Function Test", test_specific_function),
        ("Stage 1 Limited Test", test_stage1_limited),
    ]

    results = {}
    for test_name, test_func in tests:
        print(f"\n🔍 Running: {test_name}")
        print("-" * 30)

        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"❌ {test_name} crashed: {e}")
            results[test_name] = False

    print("\n📊 Test Results Summary")
    print("=" * 30)

    passed = 0
    for test_name, passed_test in results.items():
        status = "✅ PASS" if passed_test else "❌ FAIL"
        print(f"{test_name:<25} {status}")
        if passed_test:
            passed += 1

    print(f"\n📈 Overall: {passed}/{len(tests)} tests passed")

    if passed == len(tests):
        print("🎉 All tests passed! Stage 1 is ready for production.")
    else:
        print("⚠️  Some tests failed. Check implementation.")

    return passed == len(tests)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)