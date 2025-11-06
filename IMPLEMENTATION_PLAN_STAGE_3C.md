# Implementation Plan: Stage 3c - Team Aggregates & Advanced Metrics

## Overview
Implement `create_team_aggregates()` method in `/home/user/nfl/src/data_pipeline.py` (lines 465-468) to populate the `team_rolling_features` table with team-level offensive, defensive, and situational metrics.

---

## 1. Technical Specification

### 1.1 Data Sources

#### Primary Source (2022+): raw_pbp
- **EPA per play**: `AVG(epa)` grouped by `posteam` (offensive) or `defteam` (defensive)
- **Success rate**: `SUM(success) / COUNT(*)` where `success = 1`
- **Explosive plays**: Count where `yards_gained > 10` / total plays
- **Red zone opportunities**: Count drives where `yardline_100 <= 20` and `posteam` possesses ball
- **Red zone TDs**: Count where `touchdown = 1` and `yardline_100 <= 20`
- **Third down conversions**: `SUM(third_down_converted) / (SUM(third_down_converted) + SUM(third_down_failed))`
- **Pressure rate**: `(COUNT(qb_hit = 1) + COUNT(sack = 1)) / COUNT(pass_attempt = 1)` from defensive perspective
- **Turnover rate**: `(COUNT(interception = 1) + COUNT(fumble_lost = 1)) / COUNT(DISTINCT drive)`
- **Pass rate (neutral)**: `COUNT(pass_attempt = 1) / COUNT(*)` where `score_differential BETWEEN -7 AND 7` and `quarter_seconds_remaining > 120`
- **Pace metrics**: `AVG(drive_play_count)` for plays per possession, time between plays
- **Time of possession**: Parse and average `drive_time_of_possession` (format: "MM:SS")

#### Fallback Source (pre-2022): raw_team_stats
- Use aggregated stats: `passing_epa`, `rushing_epa`, `receiving_epa`
- Defensive metrics from opponent team's offensive stats
- Limited EPA data, primarily box score statistics

#### Supplementary Source: raw_schedules
- Game context: home/away, weather, roof type
- Final scores for validation

### 1.2 Key Calculations

#### Offensive Metrics

**1. EPA per Play (Last 3 Games)**
```sql
AVG(epa)
WHERE posteam = [team]
  AND season = [season]
  AND week BETWEEN [current_week - 3] AND [current_week - 1]
  AND play_type IN ('pass', 'run')
  AND epa IS NOT NULL
```

**2. Success Rate (Last 3 Games)**
```sql
SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) / COUNT(*)
WHERE posteam = [team]
  AND week BETWEEN [current_week - 3] AND [current_week - 1]
  AND play_type IN ('pass', 'run')
```

**3. Explosive Play Rate**
```sql
SUM(CASE WHEN yards_gained > 10 THEN 1 ELSE 0 END) / COUNT(*)
WHERE posteam = [team]
  AND play_type IN ('pass', 'run')
  AND week BETWEEN [current_week - 3] AND [current_week - 1]
```

**4. Red Zone Efficiency**
```sql
-- Red zone TDs
WITH red_zone_drives AS (
  SELECT DISTINCT drive, posteam, touchdown
  FROM raw_pbp
  WHERE yardline_100 <= 20
    AND posteam = [team]
    AND week BETWEEN [current_week - 3] AND [current_week - 1]
)
SELECT
  SUM(CASE WHEN touchdown = 1 THEN 1 ELSE 0 END)::FLOAT /
  COUNT(DISTINCT drive) AS rz_efficiency
FROM red_zone_drives
```

**5. Third Down Conversion Rate**
```sql
SUM(third_down_converted) /
NULLIF(SUM(third_down_converted) + SUM(third_down_failed), 0)
WHERE posteam = [team]
  AND week BETWEEN [current_week - 3] AND [current_week - 1]
  AND down = 3
```

#### Defensive Metrics

**6. Defensive EPA per Play (Last 3 Games)**
```sql
-- Lower is better for defense
AVG(epa)
WHERE defteam = [team]
  AND week BETWEEN [current_week - 3] AND [current_week - 1]
  AND play_type IN ('pass', 'run')
  AND epa IS NOT NULL
```

**7. Defensive Success Rate**
```sql
-- For defense, success = offensive play failed (success = 0)
SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END) / COUNT(*)
WHERE defteam = [team]
  AND week BETWEEN [current_week - 3] AND [current_week - 1]
  AND play_type IN ('pass', 'run')
```

**8. Pressure Rate**
```sql
(SUM(qb_hit) + SUM(sack)) / NULLIF(SUM(pass_attempt), 0)
WHERE defteam = [team]
  AND week BETWEEN [current_week - 3] AND [current_week - 1]
  AND pass_attempt = 1
```

**9. Turnover Rate**
```sql
(SUM(interception) + SUM(fumble_lost)) /
NULLIF(COUNT(DISTINCT drive), 0)
WHERE defteam = [team]
  AND week BETWEEN [current_week - 3] AND [current_week - 1]
  AND posteam IS NOT NULL
```

#### Situational Metrics

**10. Pass Rate (Neutral Game Script)**
```sql
SUM(pass_attempt) / NULLIF(SUM(pass_attempt) + SUM(rush_attempt), 0)
WHERE posteam = [team]
  AND week BETWEEN [current_week - 3] AND [current_week - 1]
  AND score_differential BETWEEN -7 AND 7
  AND quarter_seconds_remaining > 120  -- Not end of quarter
  AND down IN (1, 2)  -- First/second down
```

**11. Pace of Play (Plays per Game)**
```sql
-- Use drive play count
AVG(drive_play_count)
WHERE posteam = [team]
  AND week BETWEEN [current_week - 3] AND [current_week - 1]
GROUP BY game_id, week
```

**12. Time of Possession (Average per Drive)**
```sql
-- Parse MM:SS format to seconds, then average
AVG(
  CAST(SPLIT_PART(drive_time_of_possession, ':', 1) AS INTEGER) * 60 +
  CAST(SPLIT_PART(drive_time_of_possession, ':', 2) AS INTEGER)
) / 60.0  -- Convert back to minutes
WHERE posteam = [team]
  AND week BETWEEN [current_week - 3] AND [current_week - 1]
  AND drive_time_of_possession IS NOT NULL
```

---

## 2. Implementation Details

### 2.1 Method Structure

```python
def create_team_aggregates(self):
    """
    Create team aggregate features from play-by-play and team stats
    Populates team_rolling_features table
    """
    logger.info("🏈 Creating team aggregates...")

    # Step 1: Determine available seasons
    seasons_with_pbp = self._get_seasons_with_pbp()  # 2022+
    all_seasons = self._get_all_seasons()  # All seasons in database

    # Step 2: Process seasons with PBP data (preferred)
    for season in seasons_with_pbp:
        logger.info(f"  Processing season {season} with PBP data...")
        self._calculate_team_aggregates_from_pbp(season)

    # Step 3: Process pre-2022 seasons with team stats fallback
    legacy_seasons = [s for s in all_seasons if s not in seasons_with_pbp]
    for season in legacy_seasons:
        logger.info(f"  Processing season {season} with team stats fallback...")
        self._calculate_team_aggregates_from_team_stats(season)

    # Step 4: Validate results
    self._validate_team_aggregates()

    logger.info("✅ Team aggregates created successfully")
```

### 2.2 Core Calculation Method (PBP-based)

```python
def _calculate_team_aggregates_from_pbp(self, season: int):
    """Calculate team aggregates from play-by-play data"""

    conn = self.db.connect()

    # Get all teams and weeks for this season
    teams_weeks = conn.execute("""
        SELECT DISTINCT posteam AS team, week
        FROM raw_pbp
        WHERE season = ?
          AND posteam IS NOT NULL
          AND week IS NOT NULL
        ORDER BY team, week
    """, [season]).fetchall()

    results = []

    for team, week in teams_weeks:
        if week <= 3:
            # Not enough history for rolling stats
            continue

        # Calculate offensive metrics
        off_metrics = self._calculate_offensive_metrics(team, season, week, conn)

        # Calculate defensive metrics
        def_metrics = self._calculate_defensive_metrics(team, season, week, conn)

        # Calculate situational metrics
        sit_metrics = self._calculate_situational_metrics(team, season, week, conn)

        # Combine all metrics
        team_features = {
            'team': team,
            'season': season,
            'week': week,
            **off_metrics,
            **def_metrics,
            **sit_metrics
        }

        results.append(team_features)

    # Bulk insert results
    if results:
        df = pl.DataFrame(results)
        rows_inserted = self.db.store_dataframe(df, 'team_rolling_features')
        logger.info(f"  ✅ Inserted {rows_inserted:,} team feature records for {season}")
```

### 2.3 Offensive Metrics Calculation

```python
def _calculate_offensive_metrics(self, team: str, season: int, week: int, conn) -> dict:
    """Calculate offensive metrics for a team"""

    query = """
    WITH offensive_plays AS (
        SELECT
            epa,
            success,
            yards_gained,
            yardline_100,
            touchdown,
            drive,
            third_down_converted,
            third_down_failed,
            down
        FROM raw_pbp
        WHERE posteam = ?
          AND season = ?
          AND week BETWEEN ? AND ?
          AND play_type IN ('pass', 'run')
          AND play = 1  -- Exclude penalties, timeouts, etc.
    ),
    red_zone_drives AS (
        SELECT DISTINCT
            drive,
            MAX(touchdown) AS scored_td
        FROM raw_pbp
        WHERE posteam = ?
          AND season = ?
          AND week BETWEEN ? AND ?
          AND yardline_100 <= 20
        GROUP BY drive
    )
    SELECT
        -- EPA per play
        AVG(op.epa) AS off_epa_per_play_last3,

        -- Success rate
        SUM(CASE WHEN op.success = 1 THEN 1 ELSE 0 END)::FLOAT /
        NULLIF(COUNT(*), 0) AS off_success_rate_last3,

        -- Explosive play rate (>10 yards)
        SUM(CASE WHEN op.yards_gained > 10 THEN 1 ELSE 0 END)::FLOAT /
        NULLIF(COUNT(*), 0) AS off_explosive_play_rate,

        -- Third down conversion rate
        SUM(op.third_down_converted)::FLOAT /
        NULLIF(SUM(op.third_down_converted) + SUM(op.third_down_failed), 0)
        AS off_third_down_conv
    FROM offensive_plays op
    """

    start_week = max(1, week - 3)
    end_week = week - 1

    result = conn.execute(query, [
        team, season, start_week, end_week,
        team, season, start_week, end_week
    ]).fetchone()

    # Red zone efficiency (separate query due to DISTINCT)
    rz_query = """
    WITH red_zone_drives AS (
        SELECT DISTINCT
            drive,
            MAX(touchdown) AS scored_td
        FROM raw_pbp
        WHERE posteam = ?
          AND season = ?
          AND week BETWEEN ? AND ?
          AND yardline_100 <= 20
          AND drive IS NOT NULL
        GROUP BY drive
    )
    SELECT
        SUM(scored_td)::FLOAT / NULLIF(COUNT(*), 0) AS rz_efficiency
    FROM red_zone_drives
    """

    rz_result = conn.execute(rz_query, [team, season, start_week, end_week]).fetchone()

    return {
        'off_epa_per_play_last3': result[0] if result[0] is not None else 0.0,
        'off_success_rate_last3': result[1] if result[1] is not None else 0.5,
        'off_explosive_play_rate': result[2] if result[2] is not None else 0.0,
        'off_third_down_conv': result[3] if result[3] is not None else 0.0,
        'off_red_zone_efficiency': rz_result[0] if rz_result and rz_result[0] is not None else 0.5
    }
```

### 2.4 Defensive Metrics Calculation

```python
def _calculate_defensive_metrics(self, team: str, season: int, week: int, conn) -> dict:
    """Calculate defensive metrics for a team"""

    query = """
    WITH defensive_plays AS (
        SELECT
            epa,
            success,
            qb_hit,
            sack,
            pass_attempt,
            interception,
            fumble_lost,
            drive
        FROM raw_pbp
        WHERE defteam = ?
          AND season = ?
          AND week BETWEEN ? AND ?
          AND play_type IN ('pass', 'run')
          AND play = 1
    )
    SELECT
        -- Defensive EPA (lower is better)
        AVG(epa) AS def_epa_per_play_last3,

        -- Defensive success rate (offense failed)
        SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END)::FLOAT /
        NULLIF(COUNT(*), 0) AS def_success_rate_last3,

        -- Pressure rate
        (SUM(qb_hit) + SUM(sack))::FLOAT /
        NULLIF(SUM(pass_attempt), 0) AS def_pressure_rate,

        -- Turnover rate (per drive)
        (SUM(interception) + SUM(fumble_lost))::FLOAT /
        NULLIF(COUNT(DISTINCT drive), 0) AS def_turnover_rate
    FROM defensive_plays
    """

    start_week = max(1, week - 3)
    end_week = week - 1

    result = conn.execute(query, [team, season, start_week, end_week]).fetchone()

    return {
        'def_epa_per_play_last3': result[0] if result[0] is not None else 0.0,
        'def_success_rate_last3': result[1] if result[1] is not None else 0.5,
        'def_pressure_rate': result[2] if result[2] is not None else 0.0,
        'def_turnover_rate': result[3] if result[3] is not None else 0.0
    }
```

### 2.5 Situational Metrics Calculation

```python
def _calculate_situational_metrics(self, team: str, season: int, week: int, conn) -> dict:
    """Calculate situational/pace metrics for a team"""

    query = """
    WITH neutral_script_plays AS (
        SELECT
            pass_attempt,
            rush_attempt,
            drive,
            drive_play_count,
            drive_time_of_possession
        FROM raw_pbp
        WHERE posteam = ?
          AND season = ?
          AND week BETWEEN ? AND ?
          AND score_differential BETWEEN -7 AND 7
          AND quarter_seconds_remaining > 120
          AND down IN (1, 2)
          AND play = 1
    ),
    all_drives AS (
        SELECT DISTINCT
            drive,
            drive_play_count,
            drive_time_of_possession
        FROM raw_pbp
        WHERE posteam = ?
          AND season = ?
          AND week BETWEEN ? AND ?
          AND drive IS NOT NULL
    )
    SELECT
        -- Pass rate in neutral situations
        SUM(nsp.pass_attempt)::FLOAT /
        NULLIF(SUM(nsp.pass_attempt) + SUM(nsp.rush_attempt), 0) AS pass_rate_neutral,

        -- Plays per drive (pace)
        AVG(ad.drive_play_count) AS pace_of_play,

        -- Time of possession (convert MM:SS to minutes)
        AVG(
            CASE
                WHEN ad.drive_time_of_possession IS NOT NULL AND ad.drive_time_of_possession != ''
                THEN
                    CAST(SPLIT_PART(ad.drive_time_of_possession, ':', 1) AS INTEGER) +
                    CAST(SPLIT_PART(ad.drive_time_of_possession, ':', 2) AS INTEGER) / 60.0
                ELSE NULL
            END
        ) AS time_of_possession_avg
    FROM neutral_script_plays nsp
    CROSS JOIN all_drives ad
    """

    start_week = max(1, week - 3)
    end_week = week - 1

    result = conn.execute(query, [
        team, season, start_week, end_week,
        team, season, start_week, end_week
    ]).fetchone()

    return {
        'pass_rate_neutral': result[0] if result[0] is not None else 0.5,
        'pace_of_play': result[1] if result[1] is not None else 65.0,
        'time_of_possession_avg': result[2] if result[2] is not None else 30.0
    }
```

### 2.6 Fallback Method (Team Stats)

```python
def _calculate_team_aggregates_from_team_stats(self, season: int):
    """
    Fallback method for pre-2022 seasons using raw_team_stats
    Limited metrics available
    """

    conn = self.db.connect()

    query = """
    WITH team_weekly_stats AS (
        SELECT
            team,
            season,
            week,
            passing_epa,
            rushing_epa,
            receiving_epa,
            def_sacks,
            def_interceptions,
            attempts + carries AS total_plays
        FROM raw_team_stats
        WHERE season = ?
    )
    SELECT
        tws.team,
        tws.season,
        tws.week,

        -- Offensive EPA (approximate from weekly stats)
        AVG(tws.passing_epa + tws.rushing_epa) OVER (
            PARTITION BY tws.team, tws.season
            ORDER BY tws.week
            ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING
        ) AS off_epa_per_play_last3,

        -- Defensive stats from opponent's offensive performance
        -- (requires joining with opponent team stats - complex)
        0.0 AS def_epa_per_play_last3,

        -- Use default values for metrics not available in team stats
        0.5 AS off_success_rate_last3,
        0.5 AS def_success_rate_last3,
        0.0 AS off_explosive_play_rate,
        0.0 AS off_red_zone_efficiency,
        0.0 AS off_third_down_conv,
        0.0 AS def_pressure_rate,
        0.0 AS def_turnover_rate,
        0.5 AS pass_rate_neutral,
        65.0 AS pace_of_play,
        30.0 AS time_of_possession_avg
    FROM team_weekly_stats tws
    WHERE week > 3  -- Need rolling window
    """

    results = conn.execute(query, [season]).fetchdf()

    if len(results) > 0:
        df = pl.from_pandas(results)
        rows_inserted = self.db.store_dataframe(df, 'team_rolling_features')
        logger.info(f"  ✅ Inserted {rows_inserted:,} team feature records for {season} (fallback)")
        logger.warning(f"  ⚠️  Pre-2022 season: Limited metrics available from team stats")
```

---

## 3. Advanced Metrics

### 3.1 Opponent-Adjusted EPA

For more sophisticated analysis, adjust EPA based on opponent strength:

```sql
WITH opponent_epa_allowed AS (
    SELECT
        defteam,
        season,
        AVG(epa) AS avg_epa_allowed
    FROM raw_pbp
    WHERE week BETWEEN [start_week] AND [end_week]
      AND play_type IN ('pass', 'run')
    GROUP BY defteam, season
)
SELECT
    p.posteam,
    AVG(p.epa) AS raw_epa,
    AVG(p.epa) - AVG(oa.avg_epa_allowed) AS opponent_adjusted_epa
FROM raw_pbp p
LEFT JOIN opponent_epa_allowed oa
    ON p.defteam = oa.defteam AND p.season = oa.season
WHERE p.posteam = [team]
GROUP BY p.posteam
```

### 3.2 Situational EPA

Break down EPA by game situation:

- **Early down EPA**: `down IN (1, 2)`
- **Passing EPA**: `pass_attempt = 1`
- **Rushing EPA**: `rush_attempt = 1`
- **Late-game EPA**: `quarter = 4 AND score_differential BETWEEN -8 AND 8`

### 3.3 Drive Efficiency Metrics

```sql
WITH drive_outcomes AS (
    SELECT
        drive,
        posteam,
        MAX(CASE
            WHEN touchdown = 1 THEN 'TD'
            WHEN field_goal_attempt = 1 AND field_goal_result = 'made' THEN 'FG'
            WHEN punt_attempt = 1 THEN 'Punt'
            WHEN interception = 1 OR fumble_lost = 1 THEN 'Turnover'
            ELSE 'Other'
        END) AS outcome
    FROM raw_pbp
    WHERE week BETWEEN [start] AND [end]
    GROUP BY drive, posteam
)
SELECT
    posteam,
    SUM(CASE WHEN outcome = 'TD' THEN 1 ELSE 0 END)::FLOAT / COUNT(*) AS td_rate,
    SUM(CASE WHEN outcome IN ('TD', 'FG') THEN 1 ELSE 0 END)::FLOAT / COUNT(*) AS scoring_rate,
    SUM(CASE WHEN outcome = 'Turnover' THEN 1 ELSE 0 END)::FLOAT / COUNT(*) AS turnover_rate
FROM drive_outcomes
GROUP BY posteam
```

---

## 4. Data Quality

### 4.1 Handling Missing PBP Data

```python
def _get_seasons_with_pbp(self) -> list:
    """Get seasons that have play-by-play data available"""
    conn = self.db.connect()

    result = conn.execute("""
        SELECT DISTINCT season
        FROM raw_pbp
        WHERE season >= 2022
        ORDER BY season
    """).fetchall()

    return [row[0] for row in result]
```

### 4.2 Validation Rules

```python
def _validate_team_aggregates(self):
    """Validate team aggregate calculations"""
    conn = self.db.connect()

    # Check EPA ranges (-3 to +3 is typical)
    invalid_epa = conn.execute("""
        SELECT COUNT(*)
        FROM team_rolling_features
        WHERE off_epa_per_play_last3 < -3
           OR off_epa_per_play_last3 > 3
           OR def_epa_per_play_last3 < -3
           OR def_epa_per_play_last3 > 3
    """).fetchone()[0]

    if invalid_epa > 0:
        logger.warning(f"  ⚠️  Found {invalid_epa} records with EPA outside expected range (-3 to +3)")

    # Check success rates (0 to 1)
    invalid_success = conn.execute("""
        SELECT COUNT(*)
        FROM team_rolling_features
        WHERE off_success_rate_last3 < 0 OR off_success_rate_last3 > 1
           OR def_success_rate_last3 < 0 OR def_success_rate_last3 > 1
    """).fetchone()[0]

    if invalid_success > 0:
        logger.warning(f"  ⚠️  Found {invalid_success} records with invalid success rates")

    # Check for NULL values
    null_count = conn.execute("""
        SELECT COUNT(*)
        FROM team_rolling_features
        WHERE off_epa_per_play_last3 IS NULL
           OR def_epa_per_play_last3 IS NULL
    """).fetchone()[0]

    if null_count > 0:
        logger.warning(f"  ⚠️  Found {null_count} records with NULL EPA values")

    # Log summary
    total_records = conn.execute("SELECT COUNT(*) FROM team_rolling_features").fetchone()[0]
    logger.info(f"  ✅ Validated {total_records:,} team aggregate records")
```

### 4.3 Handling Edge Cases

1. **Insufficient History (Week 1-3)**:
   - Skip or use available data only
   - Flag with `data_quality_score` if implementing quality scoring

2. **Missing Drive Data**:
   - Some plays may have NULL drive IDs
   - Filter with `drive IS NOT NULL` before counting

3. **Preseason Games**:
   - Exclude with `season_type = 'REG'` filter

4. **Special Teams Plays**:
   - Exclude from EPA calculations: `play_type IN ('pass', 'run')`

---

## 5. Test Cases

### 5.1 High-Powered Offense Test (Kansas City Chiefs)

```python
def test_high_powered_offense():
    """Test that Chiefs have high offensive EPA in recent seasons"""
    conn = self.db.connect()

    result = conn.execute("""
        SELECT
            AVG(off_epa_per_play_last3) AS avg_epa,
            AVG(off_success_rate_last3) AS avg_success_rate
        FROM team_rolling_features
        WHERE team = 'KC'
          AND season >= 2022
          AND week > 3
    """).fetchone()

    assert result[0] > 0.1, "Chiefs should have positive EPA per play"
    assert result[1] > 0.45, "Chiefs should have above-average success rate"
    logger.info("✅ High-powered offense test passed")
```

### 5.2 Defensive Metrics Test

```python
def test_defensive_metrics():
    """Test defensive metrics calculation"""
    conn = self.db.connect()

    result = conn.execute("""
        SELECT
            MIN(def_pressure_rate) AS min_pressure,
            MAX(def_pressure_rate) AS max_pressure,
            AVG(def_pressure_rate) AS avg_pressure
        FROM team_rolling_features
        WHERE season >= 2022
    """).fetchone()

    assert result[0] >= 0, "Pressure rate should be non-negative"
    assert result[1] <= 1, "Pressure rate should not exceed 100%"
    assert 0.15 <= result[2] <= 0.35, "Average pressure rate should be reasonable (15-35%)"
    logger.info("✅ Defensive metrics test passed")
```

### 5.3 Rolling Aggregation Test

```python
def test_rolling_aggregation():
    """Test that week 5 uses weeks 2-4 data"""
    conn = self.db.connect()

    # Calculate manually for a specific team/season/week
    manual_epa = conn.execute("""
        SELECT AVG(epa)
        FROM raw_pbp
        WHERE posteam = 'KC'
          AND season = 2024
          AND week BETWEEN 2 AND 4
          AND play_type IN ('pass', 'run')
    """).fetchone()[0]

    stored_epa = conn.execute("""
        SELECT off_epa_per_play_last3
        FROM team_rolling_features
        WHERE team = 'KC'
          AND season = 2024
          AND week = 5
    """).fetchone()[0]

    # Allow small floating point differences
    assert abs(manual_epa - stored_epa) < 0.01, \
        f"Rolling EPA mismatch: manual={manual_epa}, stored={stored_epa}"
    logger.info("✅ Rolling aggregation test passed")
```

### 5.4 Missing PBP Data Test

```python
def test_missing_pbp_fallback():
    """Test that pre-2022 seasons use team stats fallback"""
    conn = self.db.connect()

    # Check 2021 season exists (if loaded)
    count_2021 = conn.execute("""
        SELECT COUNT(*)
        FROM team_rolling_features
        WHERE season = 2021
    """).fetchone()[0]

    if count_2021 > 0:
        logger.info(f"  Found {count_2021} records for 2021 (fallback method)")

        # Check that fallback defaults are used
        result = conn.execute("""
            SELECT
                AVG(off_success_rate_last3),
                AVG(def_pressure_rate)
            FROM team_rolling_features
            WHERE season = 2021
        """).fetchone()

        # Fallback method uses default 0.5 for success rate, 0.0 for pressure
        assert result[0] == 0.5, "Fallback should use default success rate"
        assert result[1] == 0.0, "Fallback should use default pressure rate"

    logger.info("✅ Missing PBP fallback test passed")
```

### 5.5 Data Completeness Test

```python
def test_data_completeness():
    """Test that all teams/weeks have feature records"""
    conn = self.db.connect()

    # Count distinct teams per season
    result = conn.execute("""
        SELECT season, COUNT(DISTINCT team) AS team_count
        FROM team_rolling_features
        WHERE season >= 2022
        GROUP BY season
        ORDER BY season
    """).fetchall()

    for season, team_count in result:
        assert team_count == 32, \
            f"Season {season} should have 32 teams, found {team_count}"

    logger.info("✅ Data completeness test passed")
```

---

## 6. Output Schema

### 6.1 team_rolling_features Table Columns

| Column | Type | Description | Expected Range | Source |
|--------|------|-------------|----------------|--------|
| `team` | VARCHAR(3) | Team abbreviation (e.g., 'KC', 'BUF') | N/A | Primary key |
| `season` | INTEGER | Season year | 2021-2025 | Primary key |
| `week` | INTEGER | Week number | 4-18 | Primary key |
| `off_epa_per_play_last3` | FLOAT | Offensive EPA per play (last 3 games) | -0.5 to 0.5 | raw_pbp.epa |
| `off_success_rate_last3` | FLOAT | Offensive success rate (last 3 games) | 0.35 to 0.55 | raw_pbp.success |
| `off_explosive_play_rate` | FLOAT | % of plays gaining 10+ yards | 0.10 to 0.25 | raw_pbp.yards_gained |
| `off_red_zone_efficiency` | FLOAT | TDs / red zone possessions | 0.40 to 0.70 | raw_pbp (drive-based) |
| `off_third_down_conv` | FLOAT | Third down conversion rate | 0.30 to 0.50 | raw_pbp.third_down_* |
| `def_epa_per_play_last3` | FLOAT | Defensive EPA allowed (last 3 games) | -0.5 to 0.5 | raw_pbp.epa (defteam) |
| `def_success_rate_last3` | FLOAT | Defensive success rate (last 3 games) | 0.45 to 0.65 | raw_pbp.success (inverse) |
| `def_pressure_rate` | FLOAT | QB pressures / pass attempts | 0.15 to 0.35 | raw_pbp.qb_hit + sack |
| `def_turnover_rate` | FLOAT | Turnovers forced / drive | 0.05 to 0.20 | raw_pbp.interception + fumble_lost |
| `pass_rate_neutral` | FLOAT | Pass rate in neutral situations | 0.45 to 0.65 | raw_pbp.pass_attempt |
| `pace_of_play` | FLOAT | Average plays per drive | 5.0 to 7.5 | raw_pbp.drive_play_count |
| `time_of_possession_avg` | FLOAT | Average TOP per drive (minutes) | 2.0 to 3.5 | raw_pbp.drive_time_of_possession |
| `created_at` | TIMESTAMP | Record creation timestamp | N/A | Auto-generated |

### 6.2 Example Record

```json
{
  "team": "KC",
  "season": 2024,
  "week": 10,
  "off_epa_per_play_last3": 0.24,
  "off_success_rate_last3": 0.52,
  "off_explosive_play_rate": 0.18,
  "off_red_zone_efficiency": 0.65,
  "off_third_down_conv": 0.44,
  "def_epa_per_play_last3": -0.08,
  "def_success_rate_last3": 0.53,
  "def_pressure_rate": 0.28,
  "def_turnover_rate": 0.12,
  "pass_rate_neutral": 0.58,
  "pace_of_play": 6.8,
  "time_of_possession_avg": 2.9,
  "created_at": "2025-11-06 12:34:56"
}
```

---

## 7. Implementation Checklist

### Phase 1: Core Implementation
- [ ] Add `_get_seasons_with_pbp()` helper method
- [ ] Add `_get_all_seasons()` helper method
- [ ] Implement `_calculate_offensive_metrics()` with EPA, success rate, explosive plays, 3rd down
- [ ] Implement `_calculate_defensive_metrics()` with EPA, success rate, pressure, turnovers
- [ ] Implement `_calculate_situational_metrics()` with pass rate, pace, TOP
- [ ] Implement `_calculate_team_aggregates_from_pbp()` main loop
- [ ] Implement `create_team_aggregates()` orchestration method

### Phase 2: Fallback & Validation
- [ ] Implement `_calculate_team_aggregates_from_team_stats()` for pre-2022
- [ ] Implement `_validate_team_aggregates()` with EPA range checks
- [ ] Add NULL value handling
- [ ] Add logging for data quality warnings

### Phase 3: Testing
- [ ] Write unit test for high-powered offense (Chiefs)
- [ ] Write unit test for defensive metrics
- [ ] Write unit test for rolling window calculation
- [ ] Write unit test for pre-2022 fallback
- [ ] Write integration test for data completeness

### Phase 4: Documentation & Polish
- [ ] Add docstrings to all methods
- [ ] Add inline comments for complex SQL
- [ ] Update CLAUDE.md with usage examples
- [ ] Add performance benchmarking logs

---

## 8. Performance Considerations

### 8.1 Query Optimization

1. **Indexes** (already created in setup_database.py):
   ```sql
   CREATE INDEX idx_pbp_season_week ON raw_pbp(season, week);
   CREATE INDEX idx_pbp_posteam_season ON raw_pbp(posteam, season);
   CREATE INDEX idx_pbp_defteam_season ON raw_pbp(defteam, season);
   ```

2. **Batch Processing**:
   - Process all teams for a season in one pass
   - Use CTEs to avoid repeated subqueries
   - Materialize intermediate results if needed

3. **Memory Management**:
   - DuckDB configured with 4GB memory limit (config.py)
   - Chunk large seasons into smaller batches if needed

### 8.2 Estimated Performance

- **Single season (2024)**: ~30 seconds for 32 teams × 18 weeks = 576 records
- **Full pipeline (2022-2025)**: ~2 minutes for ~1,900 records
- **Legacy fallback (2021)**: ~5 seconds (simpler aggregations)

---

## 9. Future Enhancements

### 9.1 Additional Metrics (Future)
- **EPA variance**: Measure consistency
- **Home/away splits**: Adjust for venue
- **Weather adjustments**: Account for dome vs outdoor
- **Rest days**: Factor in bye weeks and short rest
- **Injury impact**: Weight by starter availability

### 9.2 Machine Learning Features
- **EPA trends**: First derivative (improving vs declining)
- **Matchup-specific EPA**: vs specific position groups
- **Clutch performance**: Late-game EPA in close games
- **Play-calling tendencies**: Formation-specific EPA

---

## 10. SQL Query Examples

### 10.1 Complete Offensive Metrics Query

```sql
WITH offensive_plays AS (
    SELECT
        epa,
        success,
        yards_gained,
        yardline_100,
        touchdown,
        drive,
        third_down_converted,
        third_down_failed,
        down,
        play_type
    FROM raw_pbp
    WHERE posteam = 'KC'
      AND season = 2024
      AND week BETWEEN 7 AND 9  -- Weeks 7-9 for week 10 rolling
      AND play_type IN ('pass', 'run')
      AND play = 1
      AND epa IS NOT NULL
),
red_zone_drives AS (
    SELECT DISTINCT
        drive,
        MAX(touchdown) AS scored_td
    FROM raw_pbp
    WHERE posteam = 'KC'
      AND season = 2024
      AND week BETWEEN 7 AND 9
      AND yardline_100 <= 20
      AND drive IS NOT NULL
    GROUP BY drive
)
SELECT
    'KC' AS team,
    2024 AS season,
    10 AS week,

    -- Offensive EPA per play
    AVG(op.epa) AS off_epa_per_play_last3,

    -- Success rate
    SUM(CASE WHEN op.success = 1 THEN 1 ELSE 0 END)::FLOAT /
    COUNT(*) AS off_success_rate_last3,

    -- Explosive play rate
    SUM(CASE WHEN op.yards_gained > 10 THEN 1 ELSE 0 END)::FLOAT /
    COUNT(*) AS off_explosive_play_rate,

    -- Third down conversion
    SUM(CASE WHEN op.down = 3 THEN op.third_down_converted ELSE 0 END)::FLOAT /
    NULLIF(
        SUM(CASE WHEN op.down = 3 THEN op.third_down_converted ELSE 0 END) +
        SUM(CASE WHEN op.down = 3 THEN op.third_down_failed ELSE 0 END),
        0
    ) AS off_third_down_conv,

    -- Red zone efficiency
    (SELECT SUM(scored_td)::FLOAT / NULLIF(COUNT(*), 0) FROM red_zone_drives) AS off_red_zone_efficiency

FROM offensive_plays op
```

### 10.2 Complete Defensive Metrics Query

```sql
WITH defensive_plays AS (
    SELECT
        epa,
        success,
        qb_hit,
        sack,
        pass_attempt,
        interception,
        fumble_lost,
        drive,
        play_type
    FROM raw_pbp
    WHERE defteam = 'SF'
      AND season = 2024
      AND week BETWEEN 7 AND 9
      AND play_type IN ('pass', 'run')
      AND play = 1
)
SELECT
    'SF' AS team,
    2024 AS season,
    10 AS week,

    -- Defensive EPA (negative is good)
    AVG(epa) AS def_epa_per_play_last3,

    -- Defensive success rate (inverse of offensive success)
    SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END)::FLOAT /
    COUNT(*) AS def_success_rate_last3,

    -- Pressure rate
    (SUM(qb_hit) + SUM(sack))::FLOAT /
    NULLIF(SUM(pass_attempt), 0) AS def_pressure_rate,

    -- Turnover rate (per drive)
    (SUM(interception) + SUM(fumble_lost))::FLOAT /
    NULLIF(COUNT(DISTINCT drive), 0) AS def_turnover_rate

FROM defensive_plays
```

### 10.3 Complete Situational Metrics Query

```sql
WITH neutral_plays AS (
    SELECT
        pass_attempt,
        rush_attempt,
        drive,
        drive_play_count
    FROM raw_pbp
    WHERE posteam = 'BUF'
      AND season = 2024
      AND week BETWEEN 7 AND 9
      AND score_differential BETWEEN -7 AND 7
      AND quarter_seconds_remaining > 120
      AND down IN (1, 2)
      AND play = 1
),
all_drives AS (
    SELECT DISTINCT
        drive,
        drive_play_count,
        drive_time_of_possession
    FROM raw_pbp
    WHERE posteam = 'BUF'
      AND season = 2024
      AND week BETWEEN 7 AND 9
      AND drive IS NOT NULL
)
SELECT
    'BUF' AS team,
    2024 AS season,
    10 AS week,

    -- Pass rate neutral script
    SUM(np.pass_attempt)::FLOAT /
    NULLIF(SUM(np.pass_attempt) + SUM(np.rush_attempt), 0) AS pass_rate_neutral,

    -- Pace (plays per drive)
    AVG(ad.drive_play_count) AS pace_of_play,

    -- Time of possession (minutes)
    AVG(
        CASE
            WHEN ad.drive_time_of_possession IS NOT NULL
            THEN
                CAST(SPLIT_PART(ad.drive_time_of_possession, ':', 1) AS INTEGER) +
                CAST(SPLIT_PART(ad.drive_time_of_possession, ':', 2) AS INTEGER) / 60.0
            ELSE NULL
        END
    ) AS time_of_possession_avg

FROM neutral_plays np
CROSS JOIN all_drives ad
```

---

## 11. Expected Output Summary

After implementation, running `create_team_aggregates()` should produce:

```
🏈 Creating team aggregates...
  Processing season 2022 with PBP data...
  ✅ Inserted 512 team feature records for 2022
  Processing season 2023 with PBP data...
  ✅ Inserted 544 team feature records for 2023
  Processing season 2024 with PBP data...
  ✅ Inserted 416 team feature records for 2024
  Processing season 2025 with PBP data...
  ✅ Inserted 128 team feature records for 2025
  Processing season 2021 with team stats fallback...
  ✅ Inserted 480 team feature records for 2021 (fallback)
  ⚠️  Pre-2022 season: Limited metrics available from team stats
  ✅ Validated 2,080 team aggregate records
✅ Team aggregates created successfully
```

Database table `team_rolling_features` should contain approximately:
- **2021**: 480 records (fallback, 32 teams × 15 weeks)
- **2022**: 512 records (32 teams × 16 weeks)
- **2023**: 544 records (32 teams × 17 weeks)
- **2024**: 416 records (32 teams × 13 weeks through November)
- **2025**: 128 records (32 teams × 4 weeks)
- **Total**: ~2,080 records

---

## End of Implementation Plan

This plan provides a complete specification for implementing Stage 3c: Team Aggregates & Advanced Metrics. The implementation should prioritize PBP-based calculations for 2022+ seasons while providing graceful fallback for earlier seasons.
