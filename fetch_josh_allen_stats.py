#!/usr/bin/env python3
"""
Script to fetch Josh Allen's 2025 player stats.
Josh Allen is the quarterback for the Buffalo Bills.
"""

import nflreadpy as nfl
import polars as pl


def main():
    """Fetch and display Josh Allen's 2025 stats."""

    print("Fetching 2025 player stats...")

    # Load player stats for 2025 season
    # Note: 2025 season may not be available yet if running before the season starts
    # The library will use the current season if 2025 is not available
    try:
        # Load player stats for 2025
        stats_df = nfl.load_player_stats(seasons=2025)

        # Filter for Josh Allen of the Buffalo Bills
        # QB position, Buffalo Bills team
        # Note: player_name might be "J.Allen" or "Josh Allen"
        josh_allen_stats = stats_df.filter(
            ((pl.col("player_name") == "Josh Allen") |
             (pl.col("player_name") == "J.Allen")) &
            (pl.col("team") == "BUF") &
            (pl.col("position") == "QB")
        )

        if josh_allen_stats.is_empty():
            print("\nNo stats found for Josh Allen in 2025.")
            print("This could mean:")
            print("- The 2025 season hasn't started yet")
            print("- Josh Allen hasn't played any games yet")
            print("\nLet's check what players are available in the dataset:")

            # Show available QBs for Buffalo
            buf_qbs = stats_df.filter(
                (pl.col("team") == "BUF") &
                (pl.col("position") == "QB")
            ).select(["player_name", "team", "position"]).unique()

            if not buf_qbs.is_empty():
                print("\nBuffalo Bills QBs in the dataset:")
                print(buf_qbs)

            # Also check if there are any Josh Allens in the data
            all_josh_allens = stats_df.filter(
                pl.col("player_name").str.contains("Josh Allen")
            ).select(["player_name", "team", "position"]).unique()

            if not all_josh_allens.is_empty():
                print("\nAll players named Josh Allen in the dataset:")
                print(all_josh_allens)
        else:
            print(f"\nFound {len(josh_allen_stats)} records for Josh Allen (BUF, QB) in 2025")

            # Select and display key stats columns
            key_stats_cols = [
                "player_name",
                "team",
                "position",
                "season",
                "week",
                "season_type",
                "opponent_team",
                "completions",
                "attempts",
                "passing_yards",
                "passing_tds",
                "passing_interceptions",
                "sacks_suffered",
                "sack_yards_lost",
                "passing_air_yards",
                "passing_yards_after_catch",
                "passing_first_downs",
                "passing_epa",
                "passing_2pt_conversions",
                "carries",
                "rushing_yards",
                "rushing_tds",
                "rushing_first_downs",
                "rushing_epa",
                "targets",
                "receptions",
                "receiving_yards",
                "receiving_tds",
                "fantasy_points",
                "fantasy_points_ppr"
            ]

            # Filter to only existing columns
            available_cols = [col for col in key_stats_cols if col in josh_allen_stats.columns]

            # Display the stats
            print("\n" + "="*80)
            print("JOSH ALLEN 2025 STATS")
            print("="*80)

            # Show week by week stats
            josh_allen_display = josh_allen_stats.select(available_cols).sort("week")
            print(josh_allen_display)

            # Calculate season totals for key passing stats
            if len(josh_allen_stats) > 0:
                print("\n" + "="*80)
                print("2025 SEASON TOTALS")
                print("="*80)

                season_totals = josh_allen_stats.select([
                    pl.sum("completions").alias("Total Completions"),
                    pl.sum("attempts").alias("Total Attempts"),
                    (pl.sum("completions") / pl.sum("attempts") * 100).alias("Completion %"),
                    pl.sum("passing_yards").alias("Total Passing Yards"),
                    pl.sum("passing_tds").alias("Total Passing TDs"),
                    pl.sum("passing_interceptions").alias("Total INTs"),
                    pl.sum("sacks_suffered").alias("Total Sacks"),
                    pl.sum("rushing_yards").alias("Total Rushing Yards"),
                    pl.sum("rushing_tds").alias("Total Rushing TDs"),
                    pl.mean("fantasy_points").alias("Avg Fantasy Points"),
                    pl.mean("fantasy_points_ppr").alias("Avg Fantasy Points PPR")
                ])

                # Display totals
                for row in season_totals.iter_rows(named=True):
                    for stat_name, value in row.items():
                        if value is not None:
                            if isinstance(value, float):
                                print(f"{stat_name}: {value:.2f}")
                            else:
                                print(f"{stat_name}: {value}")

    except Exception as e:
        print(f"\nError fetching stats: {e}")
        print("\nNote: The 2025 season may not be available yet.")
        print("The current season is determined by the library based on today's date.")

        # Try to get the current season
        current_season = nfl.get_current_season()
        print(f"\nCurrent NFL season according to the library: {current_season}")

        if current_season < 2025:
            print(f"\nSince we're currently in the {current_season} season,")
            print("2025 data is not available yet. You may want to fetch")
            print(f"stats for {current_season} instead.")


if __name__ == "__main__":
    main()