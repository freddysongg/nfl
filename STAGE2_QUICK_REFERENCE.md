# Stage 2 Quick Reference Guide

**Full Documentation:** See `/home/user/nfl/STAGE2_IMPLEMENTATION_PLAN.md`

---

## Quick Start

### Execution Order
```python
pipeline = NFLDataPipeline()

# Stage 2 runs in this order:
pipeline.process_roster_snapshots()
  ├─ 1. build_player_lifecycle_table()         # ~30 seconds
  ├─ 2. create_weekly_roster_snapshots()       # ~60 seconds
  └─ 3. classify_player_experience_levels()    # ~15 seconds
```

### Pre-requisites
- Stage 1 completed (raw tables populated)
- Tables needed: `raw_players`, `raw_rosters_weekly`, `raw_depth_charts`, `raw_player_stats`, `raw_schedules`

---

## Method 1: `build_player_lifecycle_table()`

**Purpose:** Track player career progression

**Key Logic:**
```python
# Calculate career span
first_nfl_season = MIN(season) from rosters/stats
last_nfl_season = MAX(season) from rosters/stats
career_teams = DISTINCT teams played for

# Retirement status
if status == 'RET': 'retired'
elif last_season < current_season - 1: 'inactive'
else: 'active'
```

**Output Table:** `player_lifecycle`
- Primary key: `player_id`
- Key fields: `first_nfl_season`, `last_nfl_season`, `career_teams[]`, `retirement_status`

**Edge Cases:**
- Missing draft info → Allow NULL (undrafted players)
- Conflicting career spans → Use MIN/MAX across sources
- Missing gsis_id → Use player_id only

---

## Method 2: `create_weekly_roster_snapshots()`

**Purpose:** Time-aware roster snapshots per team/week

**Key Logic:**
```python
for each (team, season, week):
    active_players = roster where status IN ('ACT', 'RES')
    depth_chart = depth chart data for team/week

    if week > 1:
        key_changes = {
            'new_players': current - previous,
            'removed_players': previous - current,
            'position_changes': position != prev_position
        }
```

**Output Table:** `team_roster_snapshots`
- Primary key: `snapshot_id` (e.g., "KC_2024_W5")
- JSON fields: `active_players`, `depth_chart`, `key_changes`

**Edge Cases:**
- Week 1 → No previous week comparison (empty key_changes)
- Bye week → snapshot_date is NULL
- Missing depth chart → Empty JSON array

---

## Method 3: `classify_player_experience_levels()`

**Purpose:** Classify players by experience for ML confidence

**Key Logic:**
```python
seasons_played = current_season - first_nfl_season + 1

if seasons_played <= 1:
    category = 'rookie'
    confidence = 0.6
    strategy = 'high_variance_model'
elif seasons_played in [2, 3]:
    category = 'developing'
    confidence = 0.8
    strategy = 'mixed_model'
else:  # >= 4
    category = 'veteran'
    confidence = 1.0
    strategy = 'historical_trend_model'
```

**Output Table:** `player_experience_classification`
- Primary key: `(player_id, season)`
- Key fields: `experience_category`, `seasons_played`, `confidence_multiplier`

**Edge Cases:**
- Missing first_nfl_season → Exclude from classification
- Negative seasons_played → Data error, skip and log

---

## Validation Checklist

```python
from src.data_pipeline import Stage2Validator

validator = Stage2Validator(db)
results = validator.run_all_validations()

# Check results
if results['all_validations_passed']:
    print("✅ Stage 2 validation passed")
else:
    print("❌ Validation failed")
    for table, result in results['results'].items():
        for check, passed in result['checks'].items():
            if not passed:
                print(f"  ❌ {table}: {check}")
```

**Key Validations:**
- No NULL player_ids
- first_nfl_season <= last_nfl_season
- Unique snapshot_ids
- Valid JSON in roster snapshots
- Experience categories match seasons_played logic

---

## Common SQL Queries

### Check lifecycle coverage
```sql
SELECT
    COUNT(*) as total_players,
    COUNT(*) FILTER (WHERE retirement_status = 'active') as active,
    COUNT(*) FILTER (WHERE retirement_status = 'retired') as retired,
    COUNT(*) FILTER (WHERE retirement_status = 'inactive') as inactive
FROM player_lifecycle;
```

### View roster snapshot for specific team/week
```sql
SELECT
    snapshot_id,
    snapshot_date,
    JSON_ARRAY_LENGTH(active_players) as roster_size,
    JSON_EXTRACT(key_changes, '$.new_players') as new_players
FROM team_roster_snapshots
WHERE team = 'KC' AND season = 2024 AND week = 5;
```

### Experience distribution by season
```sql
SELECT
    season,
    experience_category,
    COUNT(*) as player_count,
    AVG(confidence_multiplier) as avg_confidence
FROM player_experience_classification
GROUP BY season, experience_category
ORDER BY season DESC, experience_category;
```

---

## Performance Benchmarks

| Method | Expected Time | Optimization |
|--------|--------------|--------------|
| `build_player_lifecycle_table()` | < 30 sec (5000 players) | Use Polars, avoid row-by-row |
| `create_weekly_roster_snapshots()` | < 60 sec (5 seasons) | Batch processing, pre-aggregate |
| `classify_player_experience_levels()` | < 15 sec (10k player-seasons) | Vectorized operations |

**If slower than expected:**
1. Check indexes exist (created by `setup_database.py`)
2. Use `EXPLAIN` to analyze query plans
3. Enable batch processing for large datasets
4. Use Polars lazy evaluation for memory efficiency

---

## Test Execution

```bash
# Run Stage 2 tests
pytest test_stage2.py -v

# Run specific test class
pytest test_stage2.py::TestPlayerLifecycle -v

# Run with coverage
pytest test_stage2.py --cov=src.data_pipeline
```

**Test files to create:**
- `test_stage2.py` - Unit tests for all three methods
- Sample test data fixtures in `tests/fixtures/`

---

## Troubleshooting

| Error | Likely Cause | Solution |
|-------|--------------|----------|
| "Missing required raw tables" | Stage 1 not run | Run `pipeline.full_historical_load()` first |
| "player_lifecycle doesn't exist" | Setup not run | Run `setup_database.py` |
| KeyError on column | Schema mismatch | Check raw table schemas match expected |
| JSON parsing error | Invalid JSON format | Validate JSON before storing |
| Memory error | Dataset too large | Use batch processing |

---

## Configuration

**Experience Thresholds** (`src/config.py`):
```python
"experience_thresholds": {
    "rookie": 1,              # <= 1 season
    "developing": [2, 3],     # 2-3 seasons
    "veteran": 4              # >= 4 seasons
}
```

**Confidence Multipliers** (`src/config.py`):
```python
"confidence_thresholds": {
    "rookie": 0.6,      # 60% confidence
    "developing": 0.8,  # 80% confidence
    "veteran": 1.0      # 100% confidence
}
```

---

## Success Criteria

Stage 2 is **DONE** when:

- ✅ All three methods implemented
- ✅ All unit tests pass (100%)
- ✅ Integration test passes
- ✅ Data validations pass
- ✅ Performance benchmarks met
- ✅ Documentation complete

---

## Next Stage

After Stage 2 completes successfully, you can proceed to:
- **Stage 3:** Feature Engineering (rolling stats, matchup features)
- **Stage 4:** ML Dataset Creation (combine features, create targets)

---

**Quick Reference Version:** 1.0
**See Full Plan:** `/home/user/nfl/STAGE2_IMPLEMENTATION_PLAN.md` (1793 lines)
