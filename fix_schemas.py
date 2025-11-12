"""
Script to help fix schema mismatches
Run this locally where you have working data downloads
"""

import nflreadpy as nfl
import duckdb

def compare_schemas(table_name, df, conn):
    """Compare actual data schema with database schema"""
    print(f"\n{'='*70}")
    print(f"Analyzing: {table_name}")
    print(f"{'='*70}")

    # Get actual columns from data
    actual_cols = df.columns
    actual_types = {col: str(df[col].dtype) for col in actual_cols}

    print(f"Actual data columns: {len(actual_cols)}")

    # Get database schema
    try:
        result = conn.execute(f"DESCRIBE {table_name}").fetchall()
        db_cols = [row[0] for row in result]
        db_types = {row[0]: row[1] for row in result}
        print(f"Database columns: {len(db_cols)}")

        # Find differences
        missing_in_db = set(actual_cols) - set(db_cols)
        extra_in_db = set(db_cols) - set(actual_cols)

        if missing_in_db:
            print(f"\n⚠️  Missing in database ({len(missing_in_db)} columns):")
            for col in sorted(missing_in_db):
                print(f"   + {col:40s} {actual_types[col]}")

        if extra_in_db:
            print(f"\n⚠️  Extra in database ({len(extra_in_db)} columns):")
            for col in sorted(extra_in_db):
                print(f"   - {col:40s} {db_types[col]}")

        # Check type mismatches for common columns
        common_cols = set(actual_cols) & set(db_cols)
        type_mismatches = []
        for col in common_cols:
            actual_type = actual_types[col]
            db_type = db_types[col]

            # Map Polars types to DuckDB types for comparison
            type_map = {
                'Int64': 'BIGINT',
                'Int32': 'INTEGER',
                'Float64': 'DOUBLE',
                'Float32': 'FLOAT',
                'Utf8': 'VARCHAR',
                'Boolean': 'BOOLEAN',
                'Date': 'DATE',
                'Datetime': 'TIMESTAMP'
            }

            expected_db_type = type_map.get(actual_type, actual_type)
            if expected_db_type.upper() not in db_type.upper() and actual_type not in db_type:
                type_mismatches.append((col, actual_type, db_type))

        if type_mismatches:
            print(f"\n⚠️  Type mismatches ({len(type_mismatches)} columns):")
            for col, actual, db in type_mismatches:
                print(f"   ~ {col:40s} {actual:15s} -> {db}")

        if not missing_in_db and not extra_in_db and not type_mismatches:
            print("\n✅ Schema matches!")

    except Exception as e:
        print(f"\n❌ Error checking database: {e}")

    return {
        'table': table_name,
        'actual_columns': len(actual_cols),
        'db_columns': len(db_cols) if 'db_cols' in locals() else 0,
        'missing_in_db': list(missing_in_db) if 'missing_in_db' in locals() else [],
        'extra_in_db': list(extra_in_db) if 'extra_in_db' in locals() else []
    }

if __name__ == "__main__":
    print("Schema Mismatch Analyzer")
    print("="*70)

    # Connect to database
    conn = duckdb.connect('nfl_predictions.duckdb')
    season = 2021

    issues = []

    try:
        # Check raw_player_stats
        print("\n\n1. Checking raw_player_stats...")
        df_player = nfl.load_player_stats(seasons=season)
        issues.append(compare_schemas('raw_player_stats', df_player, conn))

        # Check raw_team_stats
        print("\n\n2. Checking raw_team_stats...")
        df_team = nfl.load_team_stats(seasons=season)
        issues.append(compare_schemas('raw_team_stats', df_team, conn))

        # Check raw_schedules
        print("\n\n3. Checking raw_schedules...")
        df_sched = nfl.load_schedules(seasons=season)
        issues.append(compare_schemas('raw_schedules', df_sched, conn))

        # Check raw_snap_counts
        print("\n\n4. Checking raw_snap_counts...")
        df_snap = nfl.load_snap_counts(seasons=season)
        issues.append(compare_schemas('raw_snap_counts', df_snap, conn))

        # Check Next Gen Stats
        print("\n\n5. Checking raw_nextgen_passing...")
        df_ng_pass = nfl.load_nextgen_stats(stat_type='passing', seasons=season)
        issues.append(compare_schemas('raw_nextgen_passing', df_ng_pass, conn))

        print("\n\n6. Checking raw_nextgen_rushing...")
        df_ng_rush = nfl.load_nextgen_stats(stat_type='rushing', seasons=season)
        issues.append(compare_schemas('raw_nextgen_rushing', df_ng_rush, conn))

        print("\n\n7. Checking raw_nextgen_receiving...")
        df_ng_rec = nfl.load_nextgen_stats(stat_type='receiving', seasons=season)
        issues.append(compare_schemas('raw_nextgen_receiving', df_ng_rec, conn))

        # Summary
        print(f"\n\n{'='*70}")
        print("SUMMARY")
        print(f"{'='*70}")
        for issue in issues:
            if issue['missing_in_db'] or issue['extra_in_db']:
                print(f"\n{issue['table']}:")
                print(f"  Actual: {issue['actual_columns']} cols | Database: {issue['db_columns']} cols")
                if issue['missing_in_db']:
                    print(f"  Missing: {len(issue['missing_in_db'])} columns")
                if issue['extra_in_db']:
                    print(f"  Extra: {len(issue['extra_in_db'])} columns")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        conn.close()
