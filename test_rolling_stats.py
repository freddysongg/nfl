"""
Test script for Stage 3a: Rolling Statistics Calculator
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data_pipeline import NFLDataPipeline
from src.database import NFLDatabase
import json


def ensure_test_data():
    """Ensure test database has data from Stage 1"""
    print("\n📋 Checking for test data...")

    test_db_file = "test_nfl_pipeline.duckdb"
    db_exists = os.path.exists(test_db_file)

    if db_exists:
        with NFLDatabase(test_db_file) as db:
            count = db.execute("SELECT COUNT(*) FROM raw_player_stats").fetchone()[0]
            if count > 0:
                print(f"✓ Found existing test database with {count:,} records")
                return test_db_file

    print("\n⚠️  No test data found. Running Stage 1 first...")
    print("   This will fetch 2025 data from nflverse (takes ~2-3 minutes)")

    response = input("\n   Continue? [y/N]: ")
    if response.lower() != "y":
        print("\n❌ Cancelled. Please run Stage 1 manually:")
        print("   python test_pipeline.py")
        return None

    # Run Stage 1 to create test data
    from setup_database import NFLDatabaseSetup

    print(f"\n🗄️  Setting up test database: {test_db_file}")
    db_setup = NFLDatabaseSetup(test_db_file)
    db_setup.setup_database()
    db_setup.close()

    print("\n📥 Fetching 2025 player data...")
    pipeline = NFLDataPipeline(test_db_file)
    pipeline.full_historical_load(seasons=[2025])

    print("\n✅ Test data created successfully")
    return test_db_file


def test_rolling_statistics():
    """Test rolling statistics calculation"""
    print("=" * 60)
    print("Testing Stage 3a: Rolling Statistics Calculator")
    print("=" * 60)

    # Ensure we have test data
    test_db_file = ensure_test_data()
    if not test_db_file:
        return False, None

    # Initialize pipeline with test database
    pipeline = NFLDataPipeline(test_db_file)

    # Check if raw_player_stats has data
    with NFLDatabase(test_db_file) as db:
        count = db.execute("SELECT COUNT(*) FROM raw_player_stats").fetchone()[0]
        print(f"\n✓ Found {count:,} records in raw_player_stats")

        if count == 0:
            print("\n❌ No data in raw_player_stats")
            return False, None

        # Check position distribution
        positions = db.execute(
            """
            SELECT position, COUNT(*) as count
            FROM raw_player_stats
            WHERE season_type = 'REG' AND player_id IS NOT NULL
            GROUP BY position
            ORDER BY count DESC
        """
        ).fetchall()

        print(f"\n📊 Position distribution in raw_player_stats:")
        for pos, cnt in positions[:10]:
            print(f"   {pos}: {cnt:,} records")

    # Run rolling statistics calculation
    print(f"\n🚀 Running rolling statistics calculation...")
    print("=" * 60)

    try:
        pipeline.calculate_rolling_statistics()
        print("\n✅ Rolling statistics calculation completed successfully!")

    except Exception as e:
        print(f"\n❌ Rolling statistics calculation failed: {e}")
        import traceback

        traceback.print_exc()
        return False, test_db_file

    # Verify results
    print("\n" + "=" * 60)
    print("Verifying results...")
    print("=" * 60)

    with NFLDatabase(test_db_file) as db:
        # Check total records
        count = db.execute("SELECT COUNT(*) FROM player_rolling_features").fetchone()[0]
        print(f"\n✓ Created {count:,} rolling feature records")

        if count == 0:
            print("❌ No records created!")
            return False, test_db_file

        # Check by position
        pos_dist = db.execute(
            """
            SELECT position, COUNT(*) as count
            FROM player_rolling_features
            GROUP BY position
            ORDER BY count DESC
        """
        ).fetchall()

        print(f"\n📊 Records by position:")
        for pos, cnt in pos_dist:
            print(f"   {pos}: {cnt:,} records")

        # Check a sample record
        sample = db.execute(
            """
            SELECT
                player_id,
                season,
                week,
                position,
                stats_last3_games,
                stats_last5_games,
                performance_trend,
                usage_trend
            FROM player_rolling_features
            WHERE stats_last3_games IS NOT NULL
            LIMIT 1
        """
        ).fetchone()

        if sample:
            print(f"\n📋 Sample record:")
            print(f"   Player: {sample[0]}")
            print(f"   Season: {sample[1]}, Week: {sample[2]}")
            print(f"   Position: {sample[3]}")

            # Parse JSON columns
            try:
                stats_3 = json.loads(sample[4])
                print(f"\n   Stats (last 3 games):")
                print(f"      Window size: {stats_3.get('window_size')}")
                print(f"      Games in window: {stats_3.get('games_in_window')}")

                # Show first few stats
                stat_keys = [
                    k
                    for k in stats_3.keys()
                    if k not in ["window_size", "games_in_window"]
                ]
                for key in stat_keys[:5]:
                    print(f"      {key}: {stats_3[key]}")

                if len(stat_keys) > 5:
                    print(f"      ... and {len(stat_keys) - 5} more stats")

            except json.JSONDecodeError as e:
                print(f"   ⚠️  Could not parse JSON: {e}")

            print(f"\n   Performance trend: {sample[6]}")
            print(f"   Usage trend: {sample[7]}")

        # Check for NULL stats (should have some for weeks with insufficient history)
        null_stats = db.execute(
            """
            SELECT COUNT(*)
            FROM player_rolling_features
            WHERE stats_last3_games IS NULL
        """
        ).fetchone()[0]

        print(f"\n✓ Records with NULL stats (insufficient history): {null_stats:,}")

        # Check data quality
        valid_records = db.execute(
            """
            SELECT COUNT(*)
            FROM player_rolling_features
            WHERE stats_last3_games IS NOT NULL
               OR stats_last5_games IS NOT NULL
               OR stats_season_avg IS NOT NULL
        """
        ).fetchone()[0]

        print(f"✓ Records with at least one rolling stat: {valid_records:,}")
        print(f"✓ Quality ratio: {valid_records/count*100:.1f}%")

    print("\n" + "=" * 60)
    print("✅ All tests passed!")
    print("=" * 60)

    return True, test_db_file


def test_json_structure(test_db_file="test_nfl_pipeline.duckdb"):
    """Test that JSON structure matches expected schema"""
    print("\n" + "=" * 60)
    print("Testing JSON structure validation...")
    print("=" * 60)

    with NFLDatabase(test_db_file) as db:
        # Get samples for each position
        positions = ["QB", "RB", "WR", "TE", "K", "DEF"]

        for position in positions:
            sample = db.execute(
                f"""
                SELECT stats_last5_games
                FROM player_rolling_features
                WHERE position = '{position}'
                  AND stats_last5_games IS NOT NULL
                LIMIT 1
            """
            ).fetchone()

            if sample:
                try:
                    stats = json.loads(sample[0])
                    required_keys = ["window_size", "games_in_window"]

                    has_required = all(k in stats for k in required_keys)
                    has_stats = (
                        len([k for k in stats.keys() if k not in required_keys]) > 0
                    )

                    if has_required and has_stats:
                        print(f"✓ {position}: Valid JSON structure")
                    else:
                        print(f"⚠️  {position}: Missing required keys or no stats")
                        return False

                except json.JSONDecodeError as e:
                    print(f"❌ {position}: Invalid JSON: {e}")
                    return False
            else:
                print(f"ℹ️  {position}: No data found")

    print("\n✅ JSON structure validation passed!")
    return True


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Stage 3a: Rolling Statistics Test Suite")
    print("=" * 60)

    # Run tests
    test1_passed, test_db_file = test_rolling_statistics()

    if test1_passed and test_db_file:
        test2_passed = test_json_structure(test_db_file)
    else:
        print("\n⚠️  Skipping JSON structure tests due to previous failures")
        test2_passed = False

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"Rolling statistics calculation: {'✅ PASS' if test1_passed else '❌ FAIL'}")
    print(f"JSON structure validation: {'✅ PASS' if test2_passed else '❌ FAIL'}")

    if test1_passed and test2_passed:
        print("\n🎉 All tests passed! Implementation is ready.")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Please review the errors above.")
        sys.exit(1)
