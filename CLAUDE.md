# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NFL Predictor - A Python-based data pipeline and prediction system for analyzing and predicting NFL game outcomes. The project uses DuckDB for efficient data storage and Polars for high-performance data processing.

## Tech Stack

- **Python 3.10+**: Core language
- **Package Manager**: `uv` (modern, fast Python package manager)
- **Database**: DuckDB (embedded analytical database)
- **Data Processing**: Polars (fast DataFrame library)
- **Data Source**: nflreadpy (Python wrapper for nflverse data)
- **Additional Libraries**: numpy, pandas, requests, beautifulsoup4

## Essential Commands

### Environment Setup
```bash
# Install all dependencies
uv sync

# Install with dev dependencies (pytest, black, isort, flake8)
uv sync --group dev
```

### Database Setup
```bash
# Initialize database with complete schema
python setup_database.py
```

### Testing
```bash
# Run full test suite
python test_pipeline.py

# Run specific test functions (edit test_pipeline.py main() to select tests)
pytest  # if using pytest directly
```

### Code Formatting
```bash
# Format code with black (line length: 88)
black src/

# Sort imports with isort (black profile)
isort src/

# Lint with flake8
flake8 src/
```

## Architecture

### 4-Stage Data Pipeline

The system implements a comprehensive data pipeline with distinct stages (defined in `src/data_pipeline.py`):

**Stage 1: Raw Data Collection (`full_historical_load`)**
- Loads raw NFL data from nflreadpy API for specified seasons
- Organized into: player data, team data, roster data, advanced metrics
- Uses batch processing with hash-based duplicate detection
- Data stored in `raw_*` tables with temporal consistency

**Stage 2: Roster Snapshots (`process_roster_snapshots`)**
- Creates time-aware roster snapshots for each team/week
- Tracks player lifecycle and career progression
- Classifies players by experience level (rookie/developing/veteran)

**Stage 3: Feature Engineering (`engineer_features`)**
- Calculates rolling statistics (3, 5, 10 game windows)
- Builds matchup-specific features
- Creates team aggregates
- Handles rookie vs veteran feature differences

**Stage 4: ML Dataset Creation (`build_ml_dataset`)**
- Combines all features into ML-ready format
- Applies data quality scoring
- Creates prediction targets (player stats, team points, win probability)
- Validates temporal consistency (no data leakage)

### Key Modules

**`src/config.py`** - Centralized configuration management
- `NFLConfig` class: All system configuration in one place
- Position-specific stat mappings for each position (QB, WR, RB, TE, K, DEF)
- Feature engineering config (rolling windows, position groups)
- ML config (targets, thresholds, splits)
- Database config (connection settings, memory limits)

**`src/database.py`** - Database operations and utilities
- `NFLDatabase`: Core connection and query operations
- `PlayerStatsManager`: Player-specific queries and aggregations
- `FeatureManager`: Feature calculation and storage
- `DataQualityManager`: Data completeness and outlier detection

**`src/batch_processor.py`** - Efficient batch processing with deduplication
- `BatchProcessor`: Streaming batch processor with hash-based duplicate detection
- `DataHasher`: MD5 row hashing for change detection
- `ProgressTracker`: Cross-operation progress tracking
- Processes data in configurable batch sizes (default: 1000 rows)
- Returns detailed `BatchResult` with new/updated/skipped row counts

**`src/data_pipeline.py`** - Main orchestration pipeline
- `NFLDataPipeline`: Master controller for all 4 stages
- Methods for each data category: `load_player_data`, `load_team_data`, `load_roster_data`, `load_advanced_data`
- Handles season-specific data availability (e.g., Next Gen stats from 2016+, PBP from 2022+)
- Special handling for 2025+ depth chart timestamps (ISO8601 format)
- Retry logic with exponential backoff

**`src/table_schemas.py`** - DuckDB schema definitions
- Complete schema definitions for all raw tables (114+ columns for player_stats)
- Schema matches actual 2025 nflverse data structure
- Includes feature engineering tables (rolling features, ML training features)
- Index definitions for query performance

**`setup_database.py`** - Database initialization
- Creates all tables: raw data, player lifecycle, roster management, feature engineering
- Sets up performance indexes on key columns
- Creates helpful views (current_season_players, latest_team_rosters, season_summary)
- Enables Polars integration in DuckDB

**`test_pipeline.py`** - Test suite
- Tests Stage 1 data collection with limited scope (2025 only)
- Database connectivity and schema validation tests
- Specific nflreadr function tests
- Returns detailed results summary

## Database Structure

### Raw Data Tables (Stage 1)
- `raw_player_stats`: Weekly player statistics (all positions)
- `raw_team_stats`: Weekly team statistics
- `raw_schedules`: Game schedules and results
- `raw_rosters_weekly`: Weekly roster snapshots
- `raw_depth_charts`: Team depth charts by week
- `raw_nextgen_{passing,rushing,receiving}`: Next Gen Stats (2016+)
- `raw_snap_counts`: Player snap counts (2012+)
- `raw_pbp`: Play-by-play data (2022+)
- `raw_players`: Player metadata (bio, draft info)
- `raw_ftn_charting`: FTN advanced charting (2022+)
- `raw_participation`: Play-level participation (pre-2025)
- `raw_draft_picks`: Historical draft picks
- `raw_combine`: NFL Combine results

### Processed Tables (Stages 2-4)
- `player_lifecycle`: Player career tracking
- `player_experience_classification`: Experience level classification
- `team_roster_snapshots`: Time-aware roster snapshots
- `player_rolling_features`: Rolling stats and trends
- `team_rolling_features`: Team-level rolling metrics
- `ml_training_features`: ML-ready feature vectors with targets

## Data Collection Configuration

Default seasons: 2021-2025 (configurable in `config.py`)
- Batch size: 1000 rows
- Retry attempts: 3 with exponential backoff
- Rate limit delay: 1.0 second between API calls

Position groups defined in config:
- Offensive: QB, RB, WR, TE, OL
- Defensive: DL, LB, DB
- Special Teams: K, P, LS

## Important Implementation Details

### Hash-Based Deduplication
The batch processor uses MD5 hashing of key columns to detect duplicates and changes:
- Each row gets a `row_hash` based on key columns (player_id, season, week, etc.)
- On insert, compares against existing hashes to categorize as new/updated/skipped
- Provides efficiency metrics (% of rows that are truly new)

### 2025+ Depth Chart Handling
Starting in 2025, depth charts use ISO8601 timestamps instead of week numbers:
- Must parse timestamp and derive week number from ordinal day
- Preseason games (months ≤ 8) mapped to week 1
- Regular season: `(ordinal_day - 244) / 7 + 1`
- See `process_2025_depth_charts()` in data_pipeline.py

### Temporal Consistency
The pipeline enforces strict temporal ordering to prevent data leakage:
- Features only use data from previous weeks
- No future information in training data
- Validated in Stage 4 (`validate_temporal_consistency()`)

### Position-Specific Stats
Different positions have different relevant statistics (defined in `config.py`):
- QB: passing, pressure, rushing stats
- WR/TE: receiving, rushing, special teams
- RB: rushing, receiving, special teams
- K: field goals, extra points, game-winning kicks
- DEF: tackling, pass rush, coverage, turnovers

## Common Development Tasks

### Adding a New Data Source
1. Define schema in `src/table_schemas.py`
2. Add table creation function
3. Add to `create_all_raw_tables()`
4. Implement loader in `src/data_pipeline.py`
5. Add to appropriate stage (usually Stage 1)
6. Update batch processor key columns if needed

### Running a Data Collection
```python
from src.data_pipeline import NFLDataPipeline

pipeline = NFLDataPipeline()
pipeline.full_historical_load(seasons=[2024, 2025])  # or
pipeline.run_full_pipeline()  # Runs all 4 stages
```

### Querying the Database
```python
from src.database import NFLDatabase

with NFLDatabase() as db:
    # Get table info
    summary = db.get_database_summary()

    # Query directly
    results = db.execute("SELECT * FROM raw_player_stats LIMIT 10")

    # Use managers for specialized queries
    from src.database import PlayerStatsManager
    player_mgr = PlayerStatsManager(db)
    stats = player_mgr.get_player_stats("player_id", season=2024)
```

### Testing Changes
1. Use `test_pipeline.py` for limited testing (2025 data only)
2. Creates temporary test database (`test_nfl_pipeline.duckdb`)
3. Validates data collection and schema
4. Clean up test database after validation

## Project File Structure

```
.
├── src/
│   ├── __init__.py
│   ├── config.py              # Configuration management
│   ├── database.py            # Database operations
│   ├── batch_processor.py     # Batch processing with deduplication
│   ├── data_pipeline.py       # Main pipeline orchestration
│   └── table_schemas.py       # DuckDB schema definitions
├── docs/                      # nflreadpy documentation
├── setup_database.py          # Database initialization script
├── test_pipeline.py           # Test suite
├── pyproject.toml            # Project metadata and dependencies
├── uv.lock                   # Locked dependency versions
└── nfl_predictions.duckdb    # Main database file (created on first run)
```

## Notes

- The project uses `uv` instead of pip/poetry for faster dependency management
- DuckDB database file grows with data; expect ~GB size range for full historical data
- Polars is preferred over pandas for new data processing code (better performance)
- All data fetching goes through nflreadpy, which handles caching automatically
- The batch processor provides detailed progress tracking - use for large operations
- Stages 2-4 have placeholder implementations (TODO) - Stage 1 is fully functional
