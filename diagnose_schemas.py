"""
Diagnostic script to inspect actual nflverse data schemas
Run this on your local machine where data downloads work
"""

import nflreadpy as nfl
import polars as pl
import json

def inspect_schema(name, df):
    """Print schema information for a dataframe"""
    print(f"\n{'='*70}")
    print(f"Table: {name}")
    print(f"{'='*70}")
    print(f"Column count: {len(df.columns)}")
    print(f"\nColumns and types:")
    for col in df.columns:
        dtype = df[col].dtype
        print(f"  {col:40s} {str(dtype):20s}")

    # Save to JSON for easy reference
    schema_dict = {
        'table': name,
        'column_count': len(df.columns),
        'columns': {col: str(df[col].dtype) for col in df.columns}
    }
    return schema_dict

if __name__ == "__main__":
    schemas = {}
    season = 2021  # Use 2021 as reference

    print("Inspecting nflverse data schemas...")
    print(f"Reference season: {season}")

    try:
        # Player stats
        print("\n\nFetching player stats...")
        df_player = nfl.load_player_stats(seasons=season)
        schemas['raw_player_stats'] = inspect_schema('raw_player_stats', df_player)

        # Team stats
        print("\n\nFetching team stats...")
        df_team = nfl.load_team_stats(seasons=season)
        schemas['raw_team_stats'] = inspect_schema('raw_team_stats', df_team)

        # Schedules
        print("\n\nFetching schedules...")
        df_sched = nfl.load_schedules(seasons=season)
        schemas['raw_schedules'] = inspect_schema('raw_schedules', df_sched)

        # Rosters
        print("\n\nFetching weekly rosters...")
        df_roster = nfl.load_rosters(seasons=season)
        schemas['raw_rosters_weekly'] = inspect_schema('raw_rosters_weekly', df_roster)

        # Depth charts
        print("\n\nFetching depth charts...")
        df_depth = nfl.load_depth_charts(seasons=season)
        schemas['raw_depth_charts'] = inspect_schema('raw_depth_charts', df_depth)

        # Next Gen Stats
        print("\n\nFetching Next Gen passing stats...")
        df_ng_pass = nfl.load_nextgen_stats(stat_type='passing', seasons=season)
        schemas['raw_nextgen_passing'] = inspect_schema('raw_nextgen_passing', df_ng_pass)

        print("\n\nFetching Next Gen rushing stats...")
        df_ng_rush = nfl.load_nextgen_stats(stat_type='rushing', seasons=season)
        schemas['raw_nextgen_rushing'] = inspect_schema('raw_nextgen_rushing', df_ng_rush)

        print("\n\nFetching Next Gen receiving stats...")
        df_ng_rec = nfl.load_nextgen_stats(stat_type='receiving', seasons=season)
        schemas['raw_nextgen_receiving'] = inspect_schema('raw_nextgen_receiving', df_ng_rec)

        # Snap counts
        print("\n\nFetching snap counts...")
        df_snap = nfl.load_snap_counts(seasons=season)
        schemas['raw_snap_counts'] = inspect_schema('raw_snap_counts', df_snap)

        # Players metadata
        print("\n\nFetching players metadata...")
        df_players = nfl.load_players()
        schemas['raw_players'] = inspect_schema('raw_players', df_players)

        # Save all schemas to JSON
        with open('actual_schemas.json', 'w') as f:
            json.dump(schemas, f, indent=2)

        print(f"\n\n{'='*70}")
        print("✅ Schema inspection complete!")
        print(f"{'='*70}")
        print(f"\nSchemas saved to: actual_schemas.json")
        print(f"Tables inspected: {len(schemas)}")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
