# Stage 3c: Team Aggregates - Quick Reference

## Core SQL Queries

### 1. Offensive EPA per Play (Last 3 Games)
```sql
SELECT AVG(epa) AS off_epa_per_play_last3
FROM raw_pbp
WHERE posteam = ?
  AND season = ?
  AND week BETWEEN ? AND ?  -- (current_week - 3) to (current_week - 1)
  AND play_type IN ('pass', 'run')
  AND play = 1
  AND epa IS NOT NULL
```
**Expected Range**: -0.5 to 0.5 (elite offenses > 0.2)

### 2. Success Rate (Last 3 Games)
```sql
SELECT SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END)::FLOAT / COUNT(*)
FROM raw_pbp
WHERE posteam = ? AND week BETWEEN ? AND ?
  AND play_type IN ('pass', 'run') AND play = 1
```
**Expected Range**: 0.35 to 0.55 (average ~0.45)

### 3. Explosive Play Rate
```sql
SELECT SUM(CASE WHEN yards_gained > 10 THEN 1 ELSE 0 END)::FLOAT / COUNT(*)
FROM raw_pbp
WHERE posteam = ? AND week BETWEEN ? AND ?
  AND play_type IN ('pass', 'run')
```
**Expected Range**: 0.10 to 0.25 (big-play offenses > 0.18)

### 4. Red Zone Efficiency
```sql
WITH red_zone_drives AS (
    SELECT DISTINCT drive, MAX(touchdown) AS scored_td
    FROM raw_pbp
    WHERE posteam = ? AND week BETWEEN ? AND ?
      AND yardline_100 <= 20 AND drive IS NOT NULL
    GROUP BY drive
)
SELECT SUM(scored_td)::FLOAT / COUNT(*) FROM red_zone_drives
```
**Expected Range**: 0.40 to 0.70 (elite ~0.65+)

### 5. Third Down Conversion Rate
```sql
SELECT SUM(third_down_converted)::FLOAT /
       NULLIF(SUM(third_down_converted) + SUM(third_down_failed), 0)
FROM raw_pbp
WHERE posteam = ? AND week BETWEEN ? AND ?
  AND down = 3
```
**Expected Range**: 0.30 to 0.50 (average ~0.40)

### 6. Defensive EPA Allowed
```sql
SELECT AVG(epa) AS def_epa_per_play_last3
FROM raw_pbp
WHERE defteam = ? AND week BETWEEN ? AND ?
  AND play_type IN ('pass', 'run') AND play = 1
```
**Note**: Lower is better for defense (negative EPA = good defense)

### 7. Defensive Success Rate
```sql
SELECT SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END)::FLOAT / COUNT(*)
FROM raw_pbp
WHERE defteam = ? AND week BETWEEN ? AND ?
  AND play_type IN ('pass', 'run')
```
**Expected Range**: 0.45 to 0.65 (elite defenses > 0.55)

### 8. Pressure Rate
```sql
SELECT (SUM(qb_hit) + SUM(sack))::FLOAT / NULLIF(SUM(pass_attempt), 0)
FROM raw_pbp
WHERE defteam = ? AND week BETWEEN ? AND ?
  AND pass_attempt = 1
```
**Expected Range**: 0.15 to 0.35 (elite pass rushes > 0.28)

### 9. Turnover Rate
```sql
SELECT (SUM(interception) + SUM(fumble_lost))::FLOAT /
       NULLIF(COUNT(DISTINCT drive), 0)
FROM raw_pbp
WHERE defteam = ? AND week BETWEEN ? AND ?
```
**Expected Range**: 0.05 to 0.20 (opportunistic defenses > 0.12)

### 10. Pass Rate (Neutral Script)
```sql
SELECT SUM(pass_attempt)::FLOAT /
       NULLIF(SUM(pass_attempt) + SUM(rush_attempt), 0)
FROM raw_pbp
WHERE posteam = ? AND week BETWEEN ? AND ?
  AND score_differential BETWEEN -7 AND 7
  AND quarter_seconds_remaining > 120
  AND down IN (1, 2)
```
**Expected Range**: 0.45 to 0.65 (pass-heavy teams > 0.58)

### 11. Pace of Play
```sql
SELECT AVG(drive_play_count)
FROM (
    SELECT DISTINCT drive, drive_play_count
    FROM raw_pbp
    WHERE posteam = ? AND week BETWEEN ? AND ?
      AND drive IS NOT NULL
)
```
**Expected Range**: 5.0 to 7.5 plays per drive

### 12. Time of Possession
```sql
SELECT AVG(
    CAST(SPLIT_PART(drive_time_of_possession, ':', 1) AS INTEGER) +
    CAST(SPLIT_PART(drive_time_of_possession, ':', 2) AS INTEGER) / 60.0
)
FROM (
    SELECT DISTINCT drive, drive_time_of_possession
    FROM raw_pbp
    WHERE posteam = ? AND week BETWEEN ? AND ?
      AND drive IS NOT NULL
      AND drive_time_of_possession IS NOT NULL
)
```
**Expected Range**: 2.0 to 3.5 minutes per drive

---

## Implementation Checklist

### Core Methods to Implement
1. `create_team_aggregates()` - Main orchestration
2. `_get_seasons_with_pbp()` - Check PBP availability
3. `_calculate_offensive_metrics()` - All offensive calculations
4. `_calculate_defensive_metrics()` - All defensive calculations
5. `_calculate_situational_metrics()` - Pace/TOP calculations
6. `_calculate_team_aggregates_from_pbp()` - Main processing loop
7. `_calculate_team_aggregates_from_team_stats()` - Pre-2022 fallback
8. `_validate_team_aggregates()` - Data quality checks

### Key Filters
- **Play type**: `play_type IN ('pass', 'run')` and `play = 1`
- **Season type**: `season_type = 'REG'` (exclude preseason)
- **Rolling window**: `week BETWEEN (current_week - 3) AND (current_week - 1)`
- **Neutral script**: `score_differential BETWEEN -7 AND 7`
- **Red zone**: `yardline_100 <= 20`

### Common Pitfalls
1. **Division by zero**: Always use `NULLIF(denominator, 0)`
2. **NULL EPA**: Filter with `epa IS NOT NULL`
3. **Drive counting**: Use `COUNT(DISTINCT drive)` for per-drive metrics
4. **Time parsing**: Handle NULL and empty strings in `drive_time_of_possession`
5. **Week 1-3**: Skip or handle insufficient rolling window data

### Validation Rules
- EPA should be in range [-3, 3]
- Success rates in range [0, 1]
- No NULL values for primary metrics
- All 32 teams present for each season
- Records only for weeks > 3 (rolling window requirement)

---

## Example Test Query

```sql
-- Verify Chiefs have elite offense in 2024
SELECT
    team,
    season,
    AVG(off_epa_per_play_last3) AS avg_epa,
    AVG(off_success_rate_last3) AS avg_success,
    AVG(off_explosive_play_rate) AS avg_explosive
FROM team_rolling_features
WHERE team = 'KC'
  AND season = 2024
  AND week > 3
GROUP BY team, season
```

**Expected Result**:
- `avg_epa` > 0.15 (top 5 offense)
- `avg_success` > 0.48 (above average)
- `avg_explosive` > 0.15 (big plays)

---

## Performance Tips

1. **Use CTEs**: Break complex queries into readable chunks
2. **Index usage**: Queries leverage existing indexes on season/week/team
3. **Batch by season**: Process all teams for a season in one pass
4. **Materialize drives**: Create temp table for drive-level metrics if slow

---

## File Locations

- **Implementation**: `/home/user/nfl/src/data_pipeline.py` lines 465-468
- **Output table**: `team_rolling_features` defined in `/home/user/nfl/setup_database.py` lines 135-162
- **Source data**: `raw_pbp` (2022+), `raw_team_stats` (pre-2022)
- **Config**: `/home/user/nfl/src/config.py` (rolling windows, experience thresholds)

---

## Estimated Metrics by Team Type

| Team Type | Off EPA | Off Success | Explosive | RZ Eff | 3rd Down |
|-----------|---------|-------------|-----------|--------|----------|
| Elite Offense (KC, BUF) | 0.20+ | 0.50+ | 0.18+ | 0.65+ | 0.45+ |
| Average Offense | 0.00 | 0.45 | 0.14 | 0.55 | 0.40 |
| Poor Offense | -0.15 | 0.40 | 0.10 | 0.45 | 0.35 |

| Team Type | Def EPA | Def Success | Pressure | Turnover |
|-----------|---------|-------------|----------|----------|
| Elite Defense (SF, BAL) | -0.10 | 0.55+ | 0.28+ | 0.12+ |
| Average Defense | 0.00 | 0.50 | 0.22 | 0.08 |
| Poor Defense | 0.10+ | 0.45 | 0.16 | 0.05 |
