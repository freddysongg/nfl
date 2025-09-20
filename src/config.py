"""
Configuration management for NFL Prediction System.
"""

from pathlib import Path
from typing import Dict, Any


class NFLConfig:
    """Configuration manager for the NFL prediction system"""

    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.db_file = self.project_root / "nfl_predictions.duckdb"

    @property
    def database_config(self) -> Dict[str, Any]:
        """Database configuration"""
        return {
            "db_file": str(self.db_file),
            "enable_wal": True,
            "memory_limit": "4GB",
            "threads": 4
        }

    @property
    def data_collection_config(self) -> Dict[str, Any]:
        """Data collection configuration"""
        return {
            "seasons_to_collect": list(range(2021, 2026)),
            "batch_size": 1000,
            "retry_attempts": 3,
            "rate_limit_delay": 1.0  # seconds between API calls
        }

    @property
    def feature_engineering_config(self) -> Dict[str, Any]:
        """Feature engineering configuration"""
        return {
            "rolling_windows": [3, 5, 10],
            "position_groups": {
                "QB": ["QB"],
                "RB": ["RB", "FB"],
                "WR": ["WR"],
                "TE": ["TE"],
                "OL": ["C", "G", "T"],
                "DL": ["DE", "DT", "NT"],
                "LB": ["LB", "ILB", "OLB"],
                "DB": ["CB", "S", "FS", "SS"],
                "ST": ["K", "P", "LS"]
            },
            "experience_thresholds": {
                "rookie": 1,
                "developing": [2, 3],
                "veteran": 4
            }
        }

    @property
    def position_stat_mappings(self) -> Dict[str, Dict[str, list]]:
        """Position-specific stat mappings from DATA_SETUP.md"""
        return {
            "QB": {
                "passing": [
                    "attempts", "completions", "passing_yards", "passing_tds",
                    "passing_interceptions", "passing_first_downs", "passing_air_yards",
                    "passing_yards_after_catch", "passing_epa", "passing_cpoe",
                    "pacr", "passing_2pt_conversions"
                ],
                "pressure": [
                    "sacks_suffered", "sack_yards_lost", "sack_fumbles", "sack_fumbles_lost"
                ],
                "rushing": [
                    "carries", "rushing_yards", "rushing_tds", "rushing_fumbles",
                    "rushing_fumbles_lost", "rushing_first_downs", "rushing_epa",
                    "rushing_2pt_conversions"
                ],
                "recovery": ["fumble_recovery_own", "fumble_recovery_yards_own", "fumble_recovery_tds"],
                "universal": ["misc_yards", "penalties", "penalty_yards"]
            },
            "WR": {
                "receiving": [
                    "targets", "receptions", "receiving_yards", "receiving_tds",
                    "receiving_fumbles", "receiving_fumbles_lost", "receiving_air_yards",
                    "receiving_yards_after_catch", "receiving_first_downs", "receiving_epa",
                    "racr", "target_share", "air_yards_share", "wopr", "receiving_2pt_conversions"
                ],
                "rushing": [
                    "carries", "rushing_yards", "rushing_tds", "rushing_fumbles",
                    "rushing_fumbles_lost", "rushing_first_downs", "rushing_epa",
                    "rushing_2pt_conversions"
                ],
                "special_teams": [
                    "punt_returns", "punt_return_yards", "kickoff_returns",
                    "kickoff_return_yards", "special_teams_tds"
                ],
                "recovery": ["fumble_recovery_own", "fumble_recovery_yards_own", "fumble_recovery_tds"],
                "universal": ["misc_yards", "penalties", "penalty_yards"]
            },
            "RB": {
                "rushing": [
                    "carries", "rushing_yards", "rushing_tds", "rushing_fumbles",
                    "rushing_fumbles_lost", "rushing_first_downs", "rushing_epa",
                    "rushing_2pt_conversions"
                ],
                "receiving": [
                    "targets", "receptions", "receiving_yards", "receiving_tds",
                    "receiving_fumbles", "receiving_fumbles_lost", "receiving_air_yards",
                    "receiving_yards_after_catch", "receiving_first_downs", "receiving_epa",
                    "racr", "target_share", "air_yards_share", "wopr", "receiving_2pt_conversions"
                ],
                "special_teams": ["kickoff_returns", "kickoff_return_yards", "special_teams_tds"],
                "recovery": ["fumble_recovery_own", "fumble_recovery_yards_own", "fumble_recovery_tds"],
                "universal": ["misc_yards", "penalties", "penalty_yards"]
            },
            "TE": {
                "receiving": [
                    "targets", "receptions", "receiving_yards", "receiving_tds",
                    "receiving_fumbles", "receiving_fumbles_lost", "receiving_air_yards",
                    "receiving_yards_after_catch", "receiving_first_downs", "receiving_epa",
                    "racr", "target_share", "air_yards_share", "wopr", "receiving_2pt_conversions"
                ],
                "rushing": [
                    "carries", "rushing_yards", "rushing_tds", "rushing_fumbles",
                    "rushing_fumbles_lost", "rushing_first_downs", "rushing_epa",
                    "rushing_2pt_conversions"
                ],
                "special_teams": ["kickoff_returns", "kickoff_return_yards", "special_teams_tds"],
                "recovery": ["fumble_recovery_own", "fumble_recovery_yards_own", "fumble_recovery_tds"],
                "universal": ["misc_yards", "penalties", "penalty_yards"]
            },
            "K": {
                "field_goals": [
                    "fg_att", "fg_made", "fg_missed", "fg_blocked", "fg_pct", "fg_long",
                    "fg_made_0_19", "fg_made_20_29", "fg_made_30_39", "fg_made_40_49",
                    "fg_made_50_59", "fg_made_60_", "fg_missed_0_19", "fg_missed_20_29",
                    "fg_missed_30_39", "fg_missed_40_49", "fg_missed_50_59", "fg_missed_60_",
                    "fg_made_list", "fg_missed_list", "fg_blocked_list",
                    "fg_made_distance", "fg_missed_distance", "fg_blocked_distance"
                ],
                "extra_points": ["pat_att", "pat_made", "pat_missed", "pat_blocked", "pat_pct"],
                "game_winning": ["gwfg_att", "gwfg_made", "gwfg_missed", "gwfg_blocked", "gwfg_distance"],
                "universal": ["misc_yards", "penalties", "penalty_yards"]
            },
            "DEF": {
                "tackling": [
                    "def_tackles_solo", "def_tackles_with_assist", "def_tackle_assists",
                    "def_tackles_for_loss", "def_tackles_for_loss_yards"
                ],
                "pass_rush": ["def_sacks", "def_sack_yards", "def_qb_hits"],
                "coverage": ["def_interceptions", "def_interception_yards", "def_pass_defended"],
                "turnovers": ["def_fumbles_forced"],
                "scoring": ["def_tds", "def_safeties"],
                "recovery": [
                    "fumble_recovery_own", "fumble_recovery_yards_own",
                    "fumble_recovery_opp", "fumble_recovery_yards_opp", "fumble_recovery_tds"
                ],
                "own_fumbles": ["def_fumbles"],
                "special_teams": [
                    "punt_returns", "punt_return_yards", "kickoff_returns",
                    "kickoff_return_yards", "special_teams_tds"
                ],
                "universal": ["misc_yards", "penalties", "penalty_yards"]
            }
        }

    @property
    def ml_config(self) -> Dict[str, Any]:
        """Machine learning configuration"""
        return {
            "prediction_targets": [
                "player_stats",
                "team_total_points",
                "win_probability",
                "quarter_scores"
            ],
            "confidence_thresholds": {
                "rookie": 0.6,
                "developing": 0.8,
                "veteran": 1.0
            },
            "validation_split": 0.2,
            "test_split": 0.1,
            "random_seed": 42
        }

    def get_position_stats(self, position: str) -> list:
        """Get all relevant stats for a position"""
        position_map = self.position_stat_mappings.get(position, {})
        all_stats = []
        for category, stats in position_map.items():
            all_stats.extend(stats)
        return all_stats

    def is_offensive_position(self, position: str) -> bool:
        """Check if position is offensive"""
        offensive_positions = ["QB", "RB", "FB", "WR", "TE", "C", "G", "T"]
        return position in offensive_positions

    def is_defensive_position(self, position: str) -> bool:
        """Check if position is defensive"""
        defensive_positions = ["DE", "DT", "NT", "LB", "ILB", "OLB", "CB", "S", "FS", "SS"]
        return position in defensive_positions

    def get_experience_category(self, seasons_played: int) -> str:
        """Classify player experience level"""
        thresholds = self.feature_engineering_config["experience_thresholds"]

        if seasons_played <= thresholds["rookie"]:
            return "rookie"
        elif seasons_played in thresholds["developing"]:
            return "developing"
        else:
            return "veteran"


config = NFLConfig()


def main():
    """Test configuration"""
    print("🔧 NFL Prediction System Configuration")
    print("=" * 45)

    print(f"📁 Project root: {config.project_root}")
    print(f"💾 Database file: {config.db_file}")
    print(f"📅 Seasons to collect: {config.data_collection_config['seasons_to_collect']}")

    qb_stats = config.get_position_stats("QB")
    print(f"🏈 QB stats count: {len(qb_stats)}")

    wr_stats = config.get_position_stats("WR")
    print(f"🏃 WR stats count: {len(wr_stats)}")

    print(f"👶 Rookie classification: {config.get_experience_category(1)}")
    print(f"🚀 Veteran classification: {config.get_experience_category(5)}")

    print("✅ Configuration loaded successfully")


if __name__ == "__main__":
    main()