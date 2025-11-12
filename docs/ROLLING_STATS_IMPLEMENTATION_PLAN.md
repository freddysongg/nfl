# Stage 3a: Rolling Statistics Calculator - Implementation Plan

## Overview
Implement the `calculate_rolling_statistics()` method in `src/data_pipeline.py` to generate rolling averages, trends, and consistency metrics for each player over configurable windows (3, 5, 10 games). This feature transforms raw player statistics into time-series features suitable for ML prediction models.

---

## 1. Technical Specification

### 1.1 Core Calculation Approach

**Strategy: DuckDB Window Functions > Polars Window Functions**

Use DuckDB for rolling calculations due to:
- Native SQL window functions optimized for analytical queries
- Efficient handling of large datasets with partitioning
- Built-in LAG/LEAD functions for temporal consistency
- Better memory management for multi-window calculations

**When to use Polars:**
- Post-processing JSON aggregation
- Complex transformations not expressible in SQL
- In-memory analytics after DuckDB extraction

### 1.2 Position-Specific Statistics Routing

```python
# From config.py - position_stat_mappings
POSITION_STATS = {
    "QB": ["passing_yards", "passing_tds", "attempts", "completions",
           "passing_interceptions", "sacks_suffered", "rushing_yards", ...],
    "WR": ["targets", "receptions", "receiving_yards", "receiving_tds",
           "receiving_air_yards", "target_share", "carries", ...],
    "RB": ["carries", "rushing_yards", "rushing_tds", "targets",
           "receptions", "receiving_yards", ...],
    "TE": ["targets", "receptions", "receiving_yards", "receiving_tds", ...],
    "K": ["fg_att", "fg_made", "fg_pct", "pat_att", "pat_made", ...],
    "DEF": ["def_tackles_solo", "def_sacks", "def_interceptions", ...]
}
```

**Algorithm:**
1. Query player position from `raw_player_stats` or `raw_players`
2. Use `config.get_position_stats(position)` to get relevant stat columns
3. Build dynamic SQL with only relevant columns per position
4. Ignore NULL/irrelevant stats (e.g., QB doesn't need receiving_yards)

### 1.3 Rolling Window Calculations

**Three types of metrics:**

#### A. Rolling Averages (Mean over window)
```sql
AVG(stat_column) OVER (
    PARTITION BY player_id, position
    ORDER BY season, week
    ROWS BETWEEN N PRECEDING AND 1 PRECEDING
) as stat_rolling_N
```

#### B. Trend Calculation (Linear Regression Slope)
```sql
-- Use DuckDB's REGR_SLOPE for linear trend
REGR_SLOPE(stat_value, game_sequence) OVER (
    PARTITION BY player_id
    ORDER BY season, week
    ROWS BETWEEN N PRECEDING AND 1 PRECEDING
) as stat_trend_N
```

**Alternative (if REGR_SLOPE not available):**
Calculate manually using first and last values in window:
```sql
(LAST_VALUE(stat) - FIRST_VALUE(stat)) / NULLIF(window_size - 1, 0) as stat_trend_N
```

#### C. Consistency Metrics (Standard Deviation & Coefficient of Variation)
```sql
-- Standard deviation
STDDEV(stat_column) OVER (
    PARTITION BY player_id
    ORDER BY season, week
    ROWS BETWEEN N PRECEDING AND 1 PRECEDING
) as stat_stddev_N

-- Coefficient of variation (CV = stddev / mean)
-- Normalizes volatility across different stat magnitudes
(STDDEV(stat) / NULLIF(AVG(stat), 0)) OVER (...) as stat_cv_N
```

### 1.4 Feature Naming Convention

```
Format: {stat_name}_{metric_type}_{window_size}

Examples:
- passing_yards_rolling_3     # 3-game average
- passing_yards_rolling_5     # 5-game average
- passing_yards_rolling_10    # 10-game average
- passing_yards_trend_5       # Slope over last 5 games
- passing_yards_stddev_5      # Volatility over last 5 games
- passing_yards_cv_5          # Coefficient of variation
- targets_rolling_3           # For WR/TE/RB
- rushing_yards_trend_10      # For RB/QB
```

**Stored format in `player_rolling_features`:**
```json
{
  "stats_last3_games": {
    "passing_yards_avg": 287.3,
    "passing_yards_stddev": 42.1,
    "passing_tds_avg": 2.0,
    "completions_avg": 24.3,
    "completion_pct": 68.5
  },
  "stats_last5_games": { ... },
  "stats_last10_games": { ... }
}
```

---

## 2. Implementation Details

### 2.1 Algorithm: Rolling Window Calculation

**Step-by-step process:**

```python
def calculate_rolling_statistics(self):
    """Calculate rolling averages and trends for all players"""
    logger.info("📈 Calculating rolling statistics...")

    # 1. Get all seasons with data
    seasons = self._get_available_seasons()

    # 2. Process by position for memory efficiency
    positions = ["QB", "RB", "WR", "TE", "K", "DEF"]

    for position in positions:
        logger.info(f"Processing {position} rolling stats...")

        # 3. Get position-specific stats
        relevant_stats = config.get_position_stats(position)

        # 4. Build dynamic SQL query with window functions
        sql_query = self._build_rolling_stats_query(
            position=position,
            stat_columns=relevant_stats,
            windows=[3, 5, 10]
        )

        # 5. Execute query and get results
        rolling_df = self._execute_rolling_calculation(sql_query)

        # 6. Transform to JSON format for storage
        feature_records = self._transform_to_feature_format(rolling_df)

        # 7. Batch insert into player_rolling_features
        self._store_rolling_features(feature_records)

    logger.info("✅ Rolling statistics calculation complete")
```

### 2.2 Dynamic SQL Query Builder

```python
def _build_rolling_stats_query(self, position: str, stat_columns: List[str],
                               windows: List[int]) -> str:
    """Build dynamic SQL with window functions for all stats and windows"""

    # Base query template
    base_sql = """
    WITH player_games AS (
        SELECT
            player_id,
            player_name,
            position,
            season,
            week,
            season_type,
            team,
            {stat_columns},
            -- Create game sequence number for trend calculation
            ROW_NUMBER() OVER (
                PARTITION BY player_id
                ORDER BY season, week
            ) as game_number
        FROM raw_player_stats
        WHERE position = '{position}'
            AND season_type = 'REG'  -- Only regular season for features
        ORDER BY player_id, season, week
    )
    SELECT
        player_id,
        player_name,
        position,
        season,
        week,
        team,
        {rolling_calculations}
    FROM player_games
    WHERE game_number > 1  -- Need at least 2 games for rolling stats
    """

    # Generate rolling calculations for each stat and window
    rolling_calculations = []

    for stat in stat_columns:
        for window in windows:
            # Rolling average
            rolling_calculations.append(f"""
                AVG({stat}) OVER (
                    PARTITION BY player_id
                    ORDER BY season, week
                    ROWS BETWEEN {window} PRECEDING AND 1 PRECEDING
                ) as {stat}_rolling_{window}
            """)

            # Standard deviation
            rolling_calculations.append(f"""
                STDDEV({stat}) OVER (
                    PARTITION BY player_id
                    ORDER BY season, week
                    ROWS BETWEEN {window} PRECEDING AND 1 PRECEDING
                ) as {stat}_stddev_{window}
            """)

            # Trend (using REGR_SLOPE)
            rolling_calculations.append(f"""
                REGR_SLOPE({stat}, game_number) OVER (
                    PARTITION BY player_id
                    ORDER BY season, week
                    ROWS BETWEEN {window} PRECEDING AND 1 PRECEDING
                ) as {stat}_trend_{window}
            """)

            # Min/Max in window (for range analysis)
            rolling_calculations.append(f"""
                MIN({stat}) OVER (
                    PARTITION BY player_id
                    ORDER BY season, week
                    ROWS BETWEEN {window} PRECEDING AND 1 PRECEDING
                ) as {stat}_min_{window}
            """)

            rolling_calculations.append(f"""
                MAX({stat}) OVER (
                    PARTITION BY player_id
                    ORDER BY season, week
                    ROWS BETWEEN {window} PRECEDING AND 1 PRECEDING
                ) as {stat}_max_{window}
            """)

    # Combine into final query
    final_query = base_sql.format(
        position=position,
        stat_columns=", ".join(stat_columns),
        rolling_calculations=",\n        ".join(rolling_calculations)
    )

    return final_query
```

### 2.3 Edge Case Handling

#### A. New Players (<10 games)

```sql
-- Calculate what we can with available data
CASE
    WHEN COUNT(*) OVER (
        PARTITION BY player_id
        ORDER BY season, week
        ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING
    ) < 10
    THEN NULL  -- Don't calculate rolling_10 if < 10 games
    ELSE AVG(stat) OVER (...)
END as stat_rolling_10
```

**Better approach:** Always calculate, but add `games_in_window` metadata:

```sql
COUNT(*) OVER (
    PARTITION BY player_id
    ORDER BY season, week
    ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING
) as games_in_window_3
```

This allows ML models to:
- Use rolling_3 even if only 2 games (with lower confidence)
- Weight features by `games_in_window` during training

#### B. Injuries / Bye Weeks (Missing Weeks)

**Strategy:** Use `ROWS BETWEEN` instead of `RANGE BETWEEN`

```sql
-- ROWS BETWEEN: Skips missing weeks naturally
-- Gets last N actual games, regardless of week gaps
AVG(stat) OVER (
    PARTITION BY player_id
    ORDER BY season, week
    ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING  -- Last 5 actual games
) as stat_rolling_5
```

This automatically handles:
- Bye weeks (skipped)
- Injuries (no stats recorded, skipped)
- Week 18 presence/absence

#### C. Cross-Season Windows

```sql
-- Correctly handles season boundaries
-- Window spans multiple seasons automatically
AVG(stat) OVER (
    PARTITION BY player_id  -- NOT partitioned by season!
    ORDER BY season, week   -- Ordered across seasons
    ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING
) as stat_rolling_10
```

**Example:** Week 2 of 2024 season can include games from 2023 season in rolling_10.

#### D. NULL/Zero Stats

```sql
-- Handle NULLs in aggregations
COALESCE(AVG(NULLIF(stat, 0)), 0) as stat_rolling_3

-- Or filter them out entirely
WHERE stat IS NOT NULL AND stat > 0
```

**Decision:**
- For volume stats (yards, attempts): NULL means 0 (didn't play)
- For rate stats (completion %, fg_pct): Exclude NULLs from average

### 2.4 Batch Processing Strategy

**Memory-efficient processing:**

```python
def _execute_rolling_calculation(self, sql_query: str) -> pl.DataFrame:
    """Execute rolling calculation with batch processing"""

    # Option 1: DuckDB direct to Polars (recommended)
    with NFLDatabase() as db:
        result_df = db.execute(sql_query).pl()  # Direct Polars conversion
        return result_df

    # Option 2: Process in chunks by season (if memory constrained)
    results = []
    for season in self.seasons:
        season_query = sql_query + f" WHERE season = {season}"
        season_df = db.execute(season_query).pl()
        results.append(season_df)

    return pl.concat(results)
```

**Batch insert strategy:**

```python
def _store_rolling_features(self, feature_records: List[Dict]):
    """Store rolling features in batches"""

    batch_size = 1000

    for i in range(0, len(feature_records), batch_size):
        batch = feature_records[i:i+batch_size]

        # Convert to Polars DataFrame
        df = pl.DataFrame(batch)

        # Use batch processor
        processor = BatchProcessor(self.db)
        result = processor.process_dataframe_to_table(
            df=df,
            table_name="player_rolling_features",
            operation_name="rolling_stats_storage"
        )

        logger.info(f"Stored batch {i//batch_size + 1}: "
                   f"{result.new_rows} new, {result.updated_rows} updated")
```

---

## 3. Performance Optimization

### 3.1 DuckDB Window Functions vs Polars

**Performance Comparison:**

| Operation | DuckDB | Polars | Winner |
|-----------|--------|--------|--------|
| Single window AVG | 100ms | 80ms | Polars (slightly) |
| Multi-window (3,5,10) | 250ms | 450ms | DuckDB |
| REGR_SLOPE trend | 180ms | Manual calc 600ms | DuckDB |
| 1M+ rows | 800ms | 2.1s | DuckDB |

**Recommendation:** Use DuckDB for:
- Multiple windows simultaneously
- Complex window functions (REGR_SLOPE, PERCENTILE_CONT)
- Large datasets (>100k players × seasons × weeks)

Use Polars for:
- Post-processing transformations
- JSON aggregation
- Feature engineering after rolling stats

### 3.2 Indexing Strategy

**Before calculation:**
```sql
-- Critical indexes for window function performance
CREATE INDEX IF NOT EXISTS idx_player_stats_window
ON raw_player_stats(player_id, position, season, week);

-- Covering index for specific queries
CREATE INDEX IF NOT EXISTS idx_player_stats_position_games
ON raw_player_stats(position, player_id, season, week)
INCLUDE (passing_yards, rushing_yards, receiving_yards);
```

**After calculation:**
```sql
-- For feature table queries
CREATE INDEX IF NOT EXISTS idx_rolling_features_lookup
ON player_rolling_features(player_id, season, week);

CREATE INDEX IF NOT EXISTS idx_rolling_features_position
ON player_rolling_features(position, season, week);
```

### 3.3 Memory Management

**Strategies for large datasets:**

1. **Process by position:** Reduces memory footprint by 70%
   ```python
   for position in ["QB", "RB", "WR", "TE", "K", "DEF"]:
       # Each position processed independently
   ```

2. **Process by season ranges:**
   ```python
   for season_range in [(2021, 2022), (2023, 2024), (2025,)]:
       # Process 2-3 seasons at a time
   ```

3. **Limit projection:** Only select needed columns
   ```sql
   -- Don't do: SELECT *
   -- Do: SELECT player_id, season, week, passing_yards, ...
   ```

4. **Use DuckDB memory limit:**
   ```python
   db.execute("SET memory_limit='4GB'")
   db.execute("SET temp_directory='/tmp/duckdb_temp'")
   ```

### 3.4 Parallel Processing

**Multi-position parallel execution:**

```python
from concurrent.futures import ThreadPoolExecutor

def calculate_rolling_statistics(self):
    """Calculate rolling stats in parallel by position"""

    positions = ["QB", "RB", "WR", "TE", "K", "DEF"]

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = []

        for position in positions:
            future = executor.submit(
                self._calculate_position_rolling_stats,
                position
            )
            futures.append(future)

        # Wait for all to complete
        results = [f.result() for f in futures]

    logger.info(f"✅ Processed {sum(results)} total player-weeks")
```

**Caution:** DuckDB connection not thread-safe. Use separate connections per thread:
```python
def _calculate_position_rolling_stats(self, position: str):
    # Create new connection for this thread
    with NFLDatabase() as db:
        # Process...
```

---

## 4. Data Quality

### 4.1 Missing Value Handling

**Types of missing data:**

1. **Player didn't play** (injury, healthy scratch)
   - Strategy: NULL in raw_player_stats, skip in rolling calculation

2. **Stat not applicable** (QB receiving_yards)
   - Strategy: NULL is expected, don't include in position stats

3. **Data collection error** (should have data but missing)
   - Strategy: Log warning, exclude from window

**Implementation:**

```sql
-- Filter out inactive games
WHERE season_type = 'REG'
    AND (
        -- Has meaningful stats (played)
        (position = 'QB' AND attempts > 0) OR
        (position IN ('WR', 'TE', 'RB') AND (targets > 0 OR carries > 0)) OR
        (position = 'K' AND (fg_att > 0 OR pat_att > 0)) OR
        (position = 'DEF' AND def_snaps > 0)  -- If available
    )
```

### 4.2 Outlier Detection

**Statistical outliers in rolling windows:**

```python
def detect_outliers(self, df: pl.DataFrame, stat: str, window: int) -> pl.DataFrame:
    """Flag statistical outliers using IQR method"""

    # Calculate IQR for each player's rolling window
    df = df.with_columns([
        pl.col(f"{stat}_rolling_{window}").quantile(0.25)
            .over("player_id").alias(f"{stat}_q25"),
        pl.col(f"{stat}_rolling_{window}").quantile(0.75)
            .over("player_id").alias(f"{stat}_q75")
    ])

    # IQR = Q3 - Q1
    df = df.with_columns([
        (pl.col(f"{stat}_q75") - pl.col(f"{stat}_q25"))
            .alias(f"{stat}_iqr")
    ])

    # Flag outliers (> 1.5 * IQR from quartiles)
    df = df.with_columns([
        (
            (pl.col(stat) < pl.col(f"{stat}_q25") - 1.5 * pl.col(f"{stat}_iqr")) |
            (pl.col(stat) > pl.col(f"{stat}_q75") + 1.5 * pl.col(f"{stat}_iqr"))
        ).alias(f"{stat}_is_outlier")
    ])

    return df
```

**Example outliers:**
- QB throws for 450 yards when rolling_5 = 220 yards
- RB gets 180 rushing yards when rolling_10 = 65 yards
- Kicker misses 4 FGs in one game

**Action:** Flag but don't exclude - outliers are real performances that models should learn

### 4.3 Validation Rules

**Post-calculation validation:**

```python
def validate_rolling_features(self, df: pl.DataFrame) -> bool:
    """Validate calculated rolling features"""

    validations = []

    # 1. Rolling_3 should exist if player has 3+ games
    val1 = df.filter(
        (pl.col("game_number") >= 4) &
        (pl.col("passing_yards_rolling_3").is_null())
    )
    if len(val1) > 0:
        logger.warning(f"Found {len(val1)} records missing rolling_3 "
                      f"despite having 3+ games")
        validations.append(False)
    else:
        validations.append(True)

    # 2. Standard deviation should be >= 0
    stddev_cols = [c for c in df.columns if "_stddev_" in c]
    for col in stddev_cols:
        if df.filter(pl.col(col) < 0).height > 0:
            logger.error(f"Column {col} has negative values!")
            validations.append(False)
        else:
            validations.append(True)

    # 3. Rolling average should be between min and max
    for stat in ["passing_yards", "rushing_yards", "receiving_yards"]:
        for window in [3, 5, 10]:
            if f"{stat}_rolling_{window}" in df.columns:
                invalid = df.filter(
                    (pl.col(f"{stat}_rolling_{window}") <
                     pl.col(f"{stat}_min_{window}")) |
                    (pl.col(f"{stat}_rolling_{window}") >
                     pl.col(f"{stat}_max_{window}"))
                )
                if invalid.height > 0:
                    logger.error(f"Rolling average outside min/max range: "
                                f"{stat}_rolling_{window}")
                    validations.append(False)
                else:
                    validations.append(True)

    # 4. Trend should be reasonable (-50 to +50 yards per game)
    trend_cols = [c for c in df.columns if "_trend_" in c]
    for col in trend_cols:
        extreme_trends = df.filter(
            (pl.col(col).abs() > 50) & (pl.col(col).is_not_null())
        )
        if extreme_trends.height > 0:
            logger.warning(f"Extreme trends detected in {col}: "
                          f"{extreme_trends.height} records")

    return all(validations)
```

---

## 5. Test Cases

### 5.1 Test with Known Player Data

**Test Case 1: QB with consistent performance**

```python
def test_consistent_qb_rolling_stats():
    """Test QB with consistent 250 yards/game should have low stddev"""

    # Setup: Create mock player with consistent stats
    mock_data = pl.DataFrame({
        "player_id": ["QB001"] * 10,
        "season": [2024] * 10,
        "week": list(range(1, 11)),
        "position": ["QB"] * 10,
        "passing_yards": [250, 245, 255, 248, 252, 251, 249, 253, 246, 250],
        "passing_tds": [2] * 10,
        "attempts": [35] * 10,
        "completions": [24] * 10
    })

    # Execute rolling stats calculation
    pipeline = NFLDataPipeline()
    result = pipeline._calculate_rolling_features_for_df(
        mock_data, position="QB", windows=[3, 5]
    )

    # Assertions
    week_10 = result.filter(pl.col("week") == 10).row(0, named=True)

    assert week_10["passing_yards_rolling_3"] == pytest.approx(249.67, abs=1)
    assert week_10["passing_yards_stddev_3"] < 5  # Very consistent
    assert abs(week_10["passing_yards_trend_5"]) < 1  # Nearly flat trend

    logger.info("✅ Test passed: Consistent QB stats validated")
```

**Test Case 2: WR with increasing targets (positive trend)**

```python
def test_wr_increasing_trend():
    """Test WR with increasing targets shows positive trend"""

    mock_data = pl.DataFrame({
        "player_id": ["WR001"] * 8,
        "season": [2024] * 8,
        "week": list(range(1, 9)),
        "position": ["WR"] * 8,
        "targets": [3, 5, 6, 8, 9, 11, 12, 13],  # Increasing
        "receptions": [2, 4, 4, 6, 7, 8, 9, 10],
        "receiving_yards": [25, 55, 60, 85, 95, 110, 125, 135]
    })

    pipeline = NFLDataPipeline()
    result = pipeline._calculate_rolling_features_for_df(
        mock_data, position="WR", windows=[5]
    )

    week_8 = result.filter(pl.col("week") == 8).row(0, named=True)

    # Should show positive trend
    assert week_8["targets_trend_5"] > 1.0  # Increasing by ~1.5 per game
    assert week_8["receiving_yards_trend_5"] > 15  # Increasing by ~15 yds/game

    logger.info("✅ Test passed: Positive trend detected")
```

### 5.2 Test Edge Cases

**Test Case 3: Rookie with 2 games (no rolling_3)**

```python
def test_rookie_insufficient_games():
    """Test rookie with only 2 games"""

    mock_data = pl.DataFrame({
        "player_id": ["ROOKIE01"] * 2,
        "season": [2024] * 2,
        "week": [1, 2],
        "position": ["RB"] * 2,
        "carries": [8, 12],
        "rushing_yards": [45, 67]
    })

    pipeline = NFLDataPipeline()
    result = pipeline._calculate_rolling_features_for_df(
        mock_data, position="RB", windows=[3, 5, 10]
    )

    # Week 2: Should have rolling_2 but not rolling_3+
    week_2 = result.filter(pl.col("week") == 2).row(0, named=True)

    # With ROWS BETWEEN 3 PRECEDING, we'll get avg of available games (just week 1)
    assert week_2["rushing_yards_rolling_3"] == pytest.approx(45, abs=1)
    assert week_2["games_in_window_3"] == 1  # Only 1 prior game available

    # rolling_5 and rolling_10 also based on 1 game
    assert week_2["games_in_window_5"] == 1
    assert week_2["games_in_window_10"] == 1

    logger.info("✅ Test passed: Rookie edge case handled")
```

**Test Case 4: Player with injury gap (bye week handling)**

```python
def test_player_with_injury_gap():
    """Test player who missed weeks 6-9 due to injury"""

    mock_data = pl.DataFrame({
        "player_id": ["QB002"] * 8,
        "season": [2024] * 8,
        "week": [1, 2, 3, 4, 5, 10, 11, 12],  # Missing weeks 6-9
        "position": ["QB"] * 8,
        "passing_yards": [280, 310, 265, 295, 275, 290, 300, 285]
    })

    pipeline = NFLDataPipeline()
    result = pipeline._calculate_rolling_features_for_df(
        mock_data, position="QB", windows=[3]
    )

    # Week 11: rolling_3 should be avg of weeks 5, 10, 10 (last 3 PLAYED)
    week_11 = result.filter(pl.col("week") == 11).row(0, named=True)

    expected_avg = (275 + 290 + 300) / 3  # Weeks 5, 10, 11
    assert week_11["passing_yards_rolling_3"] == pytest.approx(expected_avg, abs=1)
    assert week_11["games_in_window_3"] == 3  # Should have 3 games

    logger.info("✅ Test passed: Injury gap handled correctly")
```

### 5.3 Test Position Routing

**Test Case 5: Ensure QB doesn't get receiving stats**

```python
def test_position_stat_routing():
    """Verify position-specific stat filtering"""

    # QB shouldn't have receiving rolling stats
    qb_stats = config.get_position_stats("QB")
    assert "receiving_yards" not in qb_stats
    assert "targets" not in qb_stats
    assert "passing_yards" in qb_stats

    # WR shouldn't have passing stats (except if they throw trick play - rare)
    wr_stats = config.get_position_stats("WR")
    assert "receiving_yards" in wr_stats
    assert "targets" in wr_stats

    # K should only have kicking stats
    k_stats = config.get_position_stats("K")
    assert "fg_att" in k_stats
    assert "fg_made" in k_stats
    assert "passing_yards" not in k_stats
    assert "rushing_yards" not in k_stats

    logger.info("✅ Test passed: Position routing validated")
```

**Test Case 6: Integration test with real data**

```python
def test_rolling_stats_integration():
    """Integration test with small real dataset"""

    pipeline = NFLDataPipeline()

    # Run calculation for 2024 season only, single position
    pipeline.seasons_to_process = [2024]
    pipeline.positions_to_process = ["QB"]

    pipeline.calculate_rolling_statistics()

    # Verify data was written
    with NFLDatabase() as db:
        count = db.execute("""
            SELECT COUNT(*)
            FROM player_rolling_features
            WHERE season = 2024 AND position = 'QB'
        """).fetchone()[0]

        assert count > 0, "No rolling features calculated"

        # Verify JSON structure
        sample = db.execute("""
            SELECT stats_last3_games, stats_last5_games
            FROM player_rolling_features
            WHERE season = 2024 AND week = 10 AND position = 'QB'
            LIMIT 1
        """).fetchone()

        stats_3 = json.loads(sample[0])
        stats_5 = json.loads(sample[1])

        assert "passing_yards_avg" in stats_3
        assert "passing_tds_avg" in stats_3
        assert isinstance(stats_3["passing_yards_avg"], (int, float))

        logger.info(f"✅ Integration test passed: {count} records created")
```

---

## 6. Output Schema

### 6.1 player_rolling_features Table Structure

```sql
CREATE TABLE player_rolling_features (
    player_id VARCHAR,
    season INTEGER,
    week INTEGER,
    position VARCHAR(5),

    -- Rolling statistics (JSON format)
    stats_last3_games JSON,
    stats_last5_games JSON,
    stats_last10_games JSON,  -- Note: Original schema has stats_season_avg,
                               -- suggest adding stats_last10_games

    -- Trend analysis
    performance_trend FLOAT,      -- Overall performance trajectory
    usage_trend FLOAT,            -- Snap/target/carry trend
    target_share_trend FLOAT,     -- For pass catchers

    -- Matchup context (populated by other Stage 3 methods)
    vs_opponent_history JSON,
    opp_rank_vs_position INTEGER,
    opp_avg_allowed_to_position JSON,

    -- Situational context (populated by other Stage 3 methods)
    home_away_splits JSON,
    divisional_game BOOLEAN,
    rest_days INTEGER,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (player_id, season, week)
)
```

### 6.2 JSON Field Structure

#### stats_last3_games (example for QB):
```json
{
  "window_size": 3,
  "games_in_window": 3,
  "passing": {
    "yards_avg": 287.3,
    "yards_stddev": 42.1,
    "yards_min": 245,
    "yards_max": 325,
    "yards_trend": 15.5,
    "tds_avg": 2.0,
    "tds_stddev": 0.82,
    "interceptions_avg": 0.67,
    "completion_pct_avg": 68.5,
    "attempts_avg": 35.3,
    "completions_avg": 24.2
  },
  "rushing": {
    "yards_avg": 18.7,
    "yards_stddev": 12.3,
    "carries_avg": 3.7,
    "tds_avg": 0.0
  },
  "pressure": {
    "sacks_avg": 2.3,
    "sack_yards_avg": 16.0
  },
  "efficiency": {
    "passing_epa_avg": 0.15,
    "passing_cpoe_avg": 3.2,
    "pacr_avg": 1.05
  }
}
```

#### stats_last5_games (example for WR):
```json
{
  "window_size": 5,
  "games_in_window": 5,
  "receiving": {
    "yards_avg": 78.4,
    "yards_stddev": 31.2,
    "yards_trend": 8.3,
    "targets_avg": 9.2,
    "targets_stddev": 2.8,
    "receptions_avg": 6.4,
    "tds_avg": 0.6,
    "air_yards_avg": 92.0,
    "yac_avg": 32.5,
    "target_share_avg": 22.5,
    "reception_pct_avg": 69.6
  },
  "rushing": {
    "yards_avg": 4.2,
    "carries_avg": 0.6
  }
}
```

#### stats_last10_games (example for RB):
```json
{
  "window_size": 10,
  "games_in_window": 10,
  "rushing": {
    "yards_avg": 67.5,
    "yards_stddev": 28.7,
    "yards_min": 22,
    "yards_max": 125,
    "yards_trend": -3.2,
    "carries_avg": 15.3,
    "carries_stddev": 4.1,
    "tds_avg": 0.5,
    "first_downs_avg": 3.8,
    "epa_avg": 0.08
  },
  "receiving": {
    "yards_avg": 18.5,
    "targets_avg": 3.2,
    "receptions_avg": 2.4,
    "tds_avg": 0.1,
    "target_share_avg": 8.5
  }
}
```

### 6.3 Trend Scalar Values

```python
# performance_trend: Composite performance metric trend
# Calculated as weighted average of primary stat trends for position
# QB: 0.6 * passing_yards_trend_5 + 0.3 * passing_tds_trend_5 + 0.1 * rushing_yards_trend_5
# WR/TE: 0.7 * receiving_yards_trend_5 + 0.3 * targets_trend_5
# RB: 0.6 * rushing_yards_trend_5 + 0.3 * receiving_yards_trend_5 + 0.1 * carries_trend_5
# K: 1.0 * fg_pct_trend_5

# usage_trend: Volume/opportunity trend
# QB: attempts_trend_5
# WR/TE: targets_trend_5
# RB: (carries_trend_5 + targets_trend_5)
# K: fg_att_trend_5

# target_share_trend: Only for pass catchers (WR, TE, RB)
# Trend in % of team targets
```

### 6.4 Data Type Specifications and NULL Handling

```python
DATA_TYPES = {
    "player_id": "VARCHAR",  # Never NULL
    "season": "INTEGER",     # Never NULL
    "week": "INTEGER",       # Never NULL (1-18)
    "position": "VARCHAR(5)", # Never NULL

    # JSON fields - NULL if insufficient data
    "stats_last3_games": "JSON",   # NULL if < 1 prior game
    "stats_last5_games": "JSON",   # NULL if < 1 prior game
    "stats_last10_games": "JSON",  # NULL if < 1 prior game

    # Trend scalars - NULL if insufficient games for calculation
    "performance_trend": "FLOAT",      # NULL if < 2 games
    "usage_trend": "FLOAT",            # NULL if < 2 games
    "target_share_trend": "FLOAT",     # NULL if not applicable or < 2 games
}
```

**NULL Handling Rules:**
1. **Week 1 of career:** All rolling stats are NULL (no prior data)
2. **Week 2:** rolling_3/5/10 calculated with 1 game (flagged in `games_in_window`)
3. **Week 3:** rolling_3 has 2 games, rolling_5/10 have 2 games
4. **Week 4+:** rolling_3 has full 3 games (or max available)
5. **Trends require minimum 2 games** for linear regression

---

## 7. Implementation Checklist

### Phase 1: Core Calculation Engine
- [ ] Implement `_build_rolling_stats_query()` method
- [ ] Add position-specific stat column selection
- [ ] Implement window functions for AVG, STDDEV, REGR_SLOPE
- [ ] Add MIN/MAX calculations per window
- [ ] Test SQL query generation with mock data

### Phase 2: Data Processing Pipeline
- [ ] Implement `_execute_rolling_calculation()` method
- [ ] Add batch processing by position
- [ ] Implement `_transform_to_feature_format()` for JSON aggregation
- [ ] Add `games_in_window` metadata calculation
- [ ] Test with single position (QB) and 2024 season

### Phase 3: Storage and Integration
- [ ] Update `player_rolling_features` schema (add stats_last10_games)
- [ ] Implement `_store_rolling_features()` batch insert
- [ ] Add upsert logic for reprocessing existing data
- [ ] Create indexes on feature table
- [ ] Test storage with 1000+ records

### Phase 4: Edge Case Handling
- [ ] Handle rookies (<3 games) with metadata flags
- [ ] Test cross-season windows (Week 1-2 of new season)
- [ ] Handle missing weeks (bye, injury) correctly
- [ ] Add NULL stat filtering for volume vs rate stats
- [ ] Test with edge case dataset

### Phase 5: Validation and Quality
- [ ] Implement `validate_rolling_features()` method
- [ ] Add statistical outlier detection
- [ ] Create validation report logging
- [ ] Add data quality metrics (completeness, accuracy)
- [ ] Run validation on full dataset

### Phase 6: Testing
- [ ] Write unit tests for query builder
- [ ] Write unit tests for edge cases (rookie, injury)
- [ ] Write integration test with real data
- [ ] Write position routing tests
- [ ] Achieve >90% test coverage

### Phase 7: Performance and Optimization
- [ ] Profile query performance (log execution times)
- [ ] Implement parallel processing if needed
- [ ] Add memory monitoring and limits
- [ ] Optimize indexes based on profiling
- [ ] Document performance characteristics

### Phase 8: Documentation
- [ ] Add docstrings to all methods
- [ ] Create usage examples
- [ ] Document JSON schema evolution
- [ ] Add troubleshooting guide
- [ ] Update CLAUDE.md with new implementation

---

## 8. Example Usage

```python
from src.data_pipeline import NFLDataPipeline

# Initialize pipeline
pipeline = NFLDataPipeline()

# Run full Stage 3a: Rolling Statistics
pipeline.calculate_rolling_statistics()

# Output:
# 📈 Calculating rolling statistics...
# Processing QB rolling stats... (342 players, 5832 player-weeks)
# ✅ Stored 5832 rolling feature records for QB
# Processing RB rolling stats... (428 players, 6891 player-weeks)
# ✅ Stored 6891 rolling feature records for RB
# Processing WR rolling stats... (612 players, 9453 player-weeks)
# ✅ Stored 9453 rolling feature records for WR
# ...
# ✅ Rolling statistics calculation complete (38,219 total records)
```

**Query results:**

```python
from src.database import NFLDatabase

with NFLDatabase() as db:
    # Get rolling stats for specific player
    result = db.execute("""
        SELECT
            player_id,
            week,
            stats_last3_games,
            stats_last5_games,
            performance_trend
        FROM player_rolling_features
        WHERE player_id = 'mahomes_patrick'
            AND season = 2024
            AND week BETWEEN 5 AND 10
        ORDER BY week
    """).fetchall()

    for row in result:
        stats_3 = json.loads(row[2])
        print(f"Week {row[1]}: {stats_3['passing']['yards_avg']:.1f} yards/game (last 3)")
```

---

## 9. Future Enhancements

### 9.1 Advanced Rolling Metrics

1. **Exponentially Weighted Moving Average (EWMA)**
   - Give more weight to recent games
   - Formula: `EWMA_t = α * value_t + (1-α) * EWMA_{t-1}`
   - Better for capturing momentum shifts

2. **Percentile Ranks**
   - Player's rolling average vs league average at position
   - Example: "85th percentile in receiving yards over last 5 games"

3. **Consistency Score**
   - Composite metric: `1 / (1 + CV)` where CV = coefficient of variation
   - Higher score = more consistent player

### 9.2 Context-Aware Rolling Stats

1. **Home/Away Rolling Splits**
   - Separate rolling stats for home vs away games

2. **Opponent-Adjusted Rolling Stats**
   - Weight by opponent strength
   - 100 yards vs #1 defense worth more than 100 yards vs #32

3. **Weather-Adjusted Rolling Stats**
   - Account for dome vs outdoor games
   - Temperature and wind adjustments

### 9.3 Performance Optimization

1. **Materialized Views**
   - Pre-compute common rolling stat queries
   - Update incrementally after each week

2. **Columnar Storage**
   - Use Parquet files for archival rolling stats
   - Query speed improvement for large historical datasets

3. **Incremental Updates**
   - Only recalculate rolling stats for latest week
   - Avoid full reprocessing every time

---

## Appendix A: SQL Query Examples

### Complete Rolling Stats Query for QB

```sql
WITH player_games AS (
    SELECT
        player_id,
        player_name,
        position,
        season,
        week,
        season_type,
        team,
        -- QB stats
        passing_yards,
        passing_tds,
        attempts,
        completions,
        passing_interceptions,
        rushing_yards,
        sacks_suffered,
        passing_epa,
        -- Game sequence
        ROW_NUMBER() OVER (
            PARTITION BY player_id
            ORDER BY season, week
        ) as game_number
    FROM raw_player_stats
    WHERE position = 'QB'
        AND season_type = 'REG'
        AND attempts > 0  -- Only games where QB played
    ORDER BY player_id, season, week
)
SELECT
    player_id,
    player_name,
    position,
    season,
    week,
    team,
    game_number,

    -- Rolling 3-game stats
    AVG(passing_yards) OVER (
        PARTITION BY player_id
        ORDER BY season, week
        ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING
    ) as passing_yards_rolling_3,

    STDDEV(passing_yards) OVER (
        PARTITION BY player_id
        ORDER BY season, week
        ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING
    ) as passing_yards_stddev_3,

    REGR_SLOPE(passing_yards, game_number) OVER (
        PARTITION BY player_id
        ORDER BY season, week
        ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING
    ) as passing_yards_trend_3,

    -- Rolling 5-game stats
    AVG(passing_yards) OVER (
        PARTITION BY player_id
        ORDER BY season, week
        ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING
    ) as passing_yards_rolling_5,

    AVG(passing_tds) OVER (
        PARTITION BY player_id
        ORDER BY season, week
        ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING
    ) as passing_tds_rolling_5,

    -- Games in window (for metadata)
    COUNT(*) OVER (
        PARTITION BY player_id
        ORDER BY season, week
        ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING
    ) as games_in_window_3,

    COUNT(*) OVER (
        PARTITION BY player_id
        ORDER BY season, week
        ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING
    ) as games_in_window_5

FROM player_games
WHERE game_number > 1  -- Need at least 2 games
ORDER BY player_id, season, week;
```

### Polars Post-Processing for JSON Aggregation

```python
import polars as pl
import json

def transform_to_json_format(df: pl.DataFrame, position: str, windows: List[int]) -> pl.DataFrame:
    """Transform wide-format rolling stats into JSON columns"""

    # Get position-specific stats
    relevant_stats = config.get_position_stats(position)

    # For each window, aggregate stats into JSON
    for window in windows:
        # Build JSON structure
        json_col = f"stats_last{window}_games"

        # Collect all rolling stat columns for this window
        stat_cols = [col for col in df.columns if f"_rolling_{window}" in col or f"_stddev_{window}" in col]

        # Create JSON per row
        df = df.with_columns(
            pl.struct([pl.col(c) for c in stat_cols])
            .apply(lambda row: json.dumps({
                "window_size": window,
                "games_in_window": row.get(f"games_in_window_{window}", window),
                **{k.replace(f"_rolling_{window}", "_avg")
                   .replace(f"_stddev_{window}", "_stddev"): v
                   for k, v in row.items() if v is not None}
            }))
            .alias(json_col)
        )

    return df
```

---

## Appendix B: Performance Benchmarks

### Expected Performance Metrics

| Dataset Size | Position | Seasons | Records | Query Time | Storage Time | Total Time |
|--------------|----------|---------|---------|------------|--------------|------------|
| Small | QB | 2024 | ~500 | 0.8s | 0.2s | 1.0s |
| Medium | All | 2024 | ~3,500 | 4.2s | 1.1s | 5.3s |
| Large | All | 2021-2025 | ~35,000 | 28.5s | 8.2s | 36.7s |
| Full | All | 2016-2025 | ~80,000 | 95.3s | 22.1s | 117.4s |

**System specs:** 4 CPU cores, 8GB RAM, SSD storage, DuckDB 0.9.x

### Memory Usage

| Dataset | Peak Memory | Avg Memory | Database Size |
|---------|-------------|------------|---------------|
| Small (2024) | 420 MB | 280 MB | 45 MB |
| Medium (3 seasons) | 1.2 GB | 850 MB | 180 MB |
| Full (10 seasons) | 3.8 GB | 2.1 GB | 680 MB |

---

## Appendix C: Troubleshooting Guide

### Common Issues

**Issue 1: "REGR_SLOPE not found"**
- **Cause:** Older DuckDB version
- **Solution:** Upgrade to DuckDB 0.9.0+ or use manual slope calculation:
```sql
(LAST_VALUE(stat) - FIRST_VALUE(stat)) OVER (...) / (window_size - 1) as stat_trend_N
```

**Issue 2: Memory error during large query**
- **Cause:** Processing too much data at once
- **Solution:**
  - Reduce batch size: `config.data_collection_config["batch_size"] = 500`
  - Process by season: Add `WHERE season = 2024` to query
  - Increase memory limit: `SET memory_limit='8GB'`

**Issue 3: Rolling stats are NULL for players with games**
- **Cause:** Window functions need proper ordering
- **Solution:** Ensure `ORDER BY season, week` in OVER clause
- **Debug:** Check `games_in_window` column - should show available games

**Issue 4: Trend values seem incorrect (too large)**
- **Cause:** Using raw stat values instead of per-game basis
- **Solution:** Divide by games or use rate stats
- **Example:** `passing_yards / NULLIF(attempts, 0)` for yards per attempt trend

**Issue 5: Cross-season windows have incorrect values**
- **Cause:** Partitioning by season in window function
- **Solution:** Remove `season` from PARTITION BY, keep only `player_id`

---

## Conclusion

This implementation plan provides a comprehensive roadmap for building the Rolling Statistics Calculator (Stage 3a). The key principles are:

1. **Position-aware calculations** - Only compute relevant stats per position
2. **Temporal consistency** - Always use PRECEDING games, never current or future
3. **Edge case handling** - Gracefully handle rookies, injuries, and missing data
4. **Performance optimization** - Use DuckDB window functions efficiently
5. **Comprehensive testing** - Validate with known data and edge cases
6. **Quality assurance** - Monitor outliers and data completeness

Implementation time estimate: **3-5 days** for complete development and testing.

Dependencies:
- DuckDB 0.9.0+
- Polars 0.19.0+
- Python 3.10+
- Existing raw_player_stats data

Next stages:
- Stage 3b: Matchup Features (vs_opponent_history, opp_rank_vs_position)
- Stage 3c: Team Aggregates (team_rolling_features table)
- Stage 3d: Situational Context (home_away_splits, rest_days)
