# Stage 2 Implementation Plan: Player Lifecycle & Roster Management

**Status:** Design Document
**Target Files:** `/home/user/nfl/src/data_pipeline.py` (lines 440-453)
**Created:** 2025-11-06

---

## Executive Summary

This document provides a comprehensive implementation plan for **Stage 2** of the NFL Data Pipeline, which focuses on player lifecycle tracking, weekly roster snapshots, and experience classification. Stage 2 bridges raw data collection (Stage 1) and feature engineering (Stage 3) by creating time-aware roster management tables.

### Methods to Implement
1. `build_player_lifecycle_table()` - Track player career progression
2. `create_weekly_roster_snapshots()` - Time-aware roster snapshots per team/week
3. `classify_player_experience_levels()` - Categorize players by experience level

---

## 1. Technical Specifications

### 1.1 Method: `build_player_lifecycle_table()`

**Purpose:** Create a comprehensive player career tracking table that consolidates player metadata, draft information, and career timeline.

#### Input Data Sources
```python
# Primary source: raw_players table
"""
Columns needed:
- player_id (PRIMARY KEY)
- gsis_id, espn_id (alternate IDs)
- position, position_group
- draft_year, draft_round, draft_pick, draft_ovr
- college
- status
"""

# Secondary source: raw_rosters_weekly for career span
"""
Columns needed:
- gsis_id
- season (to determine first_nfl_season, last_nfl_season)
- team (to build career_teams array)
"""

# Tertiary source: raw_player_stats for validation
"""
Columns needed:
- player_id
- season (cross-validation for career span)
- team (cross-validation for teams played for)
"""
```

#### SQL Query Pattern

```sql
-- Step 1: Get player metadata from raw_players
WITH player_metadata AS (
    SELECT
        player_id,
        gsis_id,
        espn_id,
        position,
        position_group,
        draft_year,
        draft_round,
        draft_pick,
        draft_ovr,
        college,
        status
    FROM raw_players
    WHERE player_id IS NOT NULL
),

-- Step 2: Calculate career span from rosters_weekly
career_span AS (
    SELECT
        gsis_id,
        MIN(season) as first_nfl_season,
        MAX(season) as last_nfl_season,
        ARRAY_AGG(DISTINCT team ORDER BY team) as career_teams
    FROM raw_rosters_weekly
    WHERE gsis_id IS NOT NULL
        AND season IS NOT NULL
        AND team IS NOT NULL
    GROUP BY gsis_id
),

-- Step 3: Get additional career data from player_stats
stats_career AS (
    SELECT
        player_id,
        MIN(season) as stats_first_season,
        MAX(season) as stats_last_season,
        ARRAY_AGG(DISTINCT team ORDER BY team) as stats_teams
    FROM raw_player_stats
    WHERE player_id IS NOT NULL
        AND season IS NOT NULL
        AND team IS NOT NULL
    GROUP BY player_id
),

-- Step 4: Combine all sources with coalesce for best data
combined_lifecycle AS (
    SELECT
        pm.player_id,
        pm.gsis_id,
        pm.espn_id,

        -- Career span: prefer rosters_weekly, fallback to stats
        COALESCE(cs.first_nfl_season, sc.stats_first_season) as first_nfl_season,
        COALESCE(cs.last_nfl_season, sc.stats_last_season) as last_nfl_season,

        -- Career teams: merge both sources
        CASE
            WHEN cs.career_teams IS NOT NULL AND sc.stats_teams IS NOT NULL
            THEN ARRAY_DISTINCT(ARRAY_CONCAT(cs.career_teams, sc.stats_teams))
            ELSE COALESCE(cs.career_teams, sc.stats_teams)
        END as career_teams,

        -- Position: use position_group as primary, position as fallback
        COALESCE(pm.position_group, pm.position) as primary_position,

        -- Draft info
        pm.draft_year,
        pm.draft_round,
        pm.draft_pick,

        -- College
        pm.college,

        -- Retirement status logic
        CASE
            WHEN pm.status = 'RET' THEN 'retired'
            WHEN COALESCE(cs.last_nfl_season, sc.stats_last_season) < (SELECT MAX(season) FROM raw_player_stats) - 1
            THEN 'inactive'
            ELSE 'active'
        END as retirement_status,

        CURRENT_TIMESTAMP as created_at,
        CURRENT_TIMESTAMP as updated_at

    FROM player_metadata pm
    LEFT JOIN career_span cs ON pm.gsis_id = cs.gsis_id
    LEFT JOIN stats_career sc ON pm.player_id = sc.player_id
)

-- Step 5: Insert or replace into player_lifecycle
INSERT OR REPLACE INTO player_lifecycle
SELECT * FROM combined_lifecycle;
```

#### Polars Alternative Implementation

```python
def build_player_lifecycle_table(self):
    """Build player career tracking table using Polars"""
    logger.info("👥 Building player lifecycle table...")

    conn = self.db.connect()

    # Load raw data as Polars DataFrames
    players_df = pl.from_pandas(
        conn.execute("SELECT * FROM raw_players").fetchdf()
    )

    rosters_df = pl.from_pandas(
        conn.execute("""
            SELECT gsis_id, season, team
            FROM raw_rosters_weekly
            WHERE gsis_id IS NOT NULL
        """).fetchdf()
    )

    stats_df = pl.from_pandas(
        conn.execute("""
            SELECT player_id, season, team
            FROM raw_player_stats
            WHERE player_id IS NOT NULL
        """).fetchdf()
    )

    # Calculate career span from rosters
    career_span = rosters_df.group_by("gsis_id").agg([
        pl.col("season").min().alias("first_nfl_season"),
        pl.col("season").max().alias("last_nfl_season"),
        pl.col("team").unique().sort().alias("career_teams")
    ])

    # Calculate career span from stats
    stats_career = stats_df.group_by("player_id").agg([
        pl.col("season").min().alias("stats_first_season"),
        pl.col("season").max().alias("stats_last_season"),
        pl.col("team").unique().sort().alias("stats_teams")
    ])

    # Get current max season for retirement status
    max_season = stats_df.select(pl.col("season").max()).item()

    # Join and combine
    lifecycle = (
        players_df
        .join(career_span, left_on="gsis_id", right_on="gsis_id", how="left")
        .join(stats_career, left_on="player_id", right_on="player_id", how="left")
        .with_columns([
            # Coalesce career span
            pl.coalesce(["first_nfl_season", "stats_first_season"]).alias("first_nfl_season"),
            pl.coalesce(["last_nfl_season", "stats_last_season"]).alias("last_nfl_season"),

            # Primary position
            pl.coalesce(["position_group", "position"]).alias("primary_position"),

            # Retirement status
            pl.when(pl.col("status") == "RET")
                .then(pl.lit("retired"))
                .when(
                    pl.coalesce(["last_nfl_season", "stats_last_season"]) < (max_season - 1)
                )
                .then(pl.lit("inactive"))
                .otherwise(pl.lit("active"))
                .alias("retirement_status"),

            # Timestamps
            pl.lit(datetime.now()).alias("created_at"),
            pl.lit(datetime.now()).alias("updated_at")
        ])
        .select([
            "player_id", "gsis_id", "espn_id",
            "first_nfl_season", "last_nfl_season",
            "career_teams",
            "primary_position",
            "draft_year", "draft_round", "draft_pick",
            "college",
            "retirement_status",
            "created_at", "updated_at"
        ])
    )

    # Store in database
    rows_inserted = self.db.store_dataframe(
        lifecycle,
        "player_lifecycle",
        if_exists="replace"
    )

    logger.info(f"✅ Built player lifecycle for {rows_inserted:,} players")

    # Log summary statistics
    lifecycle_stats = lifecycle.select([
        pl.col("player_id").count().alias("total_players"),
        pl.col("retirement_status").value_counts().alias("status_breakdown"),
        pl.col("first_nfl_season").min().alias("earliest_season"),
        pl.col("last_nfl_season").max().alias("latest_season")
    ])

    logger.info(f"  📊 Career span: {lifecycle_stats['earliest_season'][0]} - {lifecycle_stats['latest_season'][0]}")

    return rows_inserted
```

#### Edge Cases & Handling

| Edge Case | Detection | Resolution |
|-----------|-----------|------------|
| **Missing gsis_id** | `WHERE gsis_id IS NULL` | Use player_id from raw_players only, join on player_name as fallback |
| **Conflicting career spans** | Compare rosters vs stats seasons | Use MIN of first_season, MAX of last_season across both sources |
| **Duplicate team entries** | Same team multiple stints | Use `ARRAY_DISTINCT()` or `pl.unique()` |
| **Missing draft info** | `draft_year IS NULL` | Accept NULL values, indicates undrafted free agent |
| **Position changes** | Multiple positions in career | Use most recent position_group from raw_players |
| **Player ID mismatches** | Different IDs across sources | Primary key on player_id, use gsis_id for joining rosters |

---

### 1.2 Method: `create_weekly_roster_snapshots()`

**Purpose:** Create time-aware roster snapshots for each team/week combination, capturing active players, depth chart positions, and roster changes.

#### Input Data Sources

```python
# Primary: raw_rosters_weekly
"""
Columns needed:
- team, season, week
- gsis_id, full_name, position, depth_chart_position
- status, status_description_abbr
- years_exp
"""

# Secondary: raw_depth_charts
"""
Columns needed:
- club_code (team), season, week
- gsis_id, position, formation, depth_team
"""

# Tertiary: raw_schedules for snapshot_date
"""
Columns needed:
- season, week, gameday
- home_team, away_team (to get game dates per team)
"""
```

#### Algorithm Overview

```
For each (team, season, week) combination:
  1. Query active roster from raw_rosters_weekly
  2. Query depth chart from raw_depth_charts
  3. Get game date from raw_schedules
  4. Compare to previous week's snapshot (if exists)
  5. Identify key changes (additions, removals, position changes)
  6. Store snapshot with JSON fields
```

#### SQL Query Pattern

```sql
-- Step 1: Build base roster snapshots
WITH roster_snapshots AS (
    SELECT
        -- Generate unique snapshot ID
        team || '_' || season::VARCHAR || '_W' || week::VARCHAR as snapshot_id,
        team,
        season,
        week,

        -- Get game date from schedule
        (
            SELECT MIN(gameday)
            FROM raw_schedules s
            WHERE s.season = rw.season
                AND s.week = rw.week
                AND (s.home_team = rw.team OR s.away_team = rw.team)
        ) as snapshot_date,

        -- Build active players JSON
        JSON_GROUP_ARRAY(
            JSON_OBJECT(
                'gsis_id', gsis_id,
                'name', full_name,
                'position', position,
                'depth_position', depth_chart_position,
                'jersey_number', jersey_number,
                'status', status,
                'years_exp', years_exp,
                'height', height,
                'weight', weight
            )
        ) as active_players

    FROM raw_rosters_weekly rw
    WHERE status IN ('ACT', 'RES')  -- Active or Reserve
        AND gsis_id IS NOT NULL
    GROUP BY team, season, week
),

-- Step 2: Add depth chart information
depth_info AS (
    SELECT
        club_code as team,
        season,
        week,
        JSON_GROUP_ARRAY(
            JSON_OBJECT(
                'gsis_id', gsis_id,
                'position', position,
                'formation', formation,
                'depth_team', depth_team,
                'name', first_name || ' ' || last_name
            )
        ) as depth_chart
    FROM raw_depth_charts
    WHERE gsis_id IS NOT NULL
    GROUP BY club_code, season, week
),

-- Step 3: Calculate key changes from previous week
roster_changes AS (
    SELECT
        curr.snapshot_id,
        curr.team,
        curr.season,
        curr.week,

        JSON_OBJECT(
            'new_players', (
                -- Players in current week but not in previous week
                SELECT JSON_GROUP_ARRAY(gsis_id)
                FROM raw_rosters_weekly curr_r
                WHERE curr_r.team = curr.team
                    AND curr_r.season = curr.season
                    AND curr_r.week = curr.week
                    AND curr_r.gsis_id NOT IN (
                        SELECT prev_r.gsis_id
                        FROM raw_rosters_weekly prev_r
                        WHERE prev_r.team = curr.team
                            AND prev_r.season = curr.season
                            AND prev_r.week = curr.week - 1
                    )
            ),
            'removed_players', (
                -- Players in previous week but not in current week
                SELECT JSON_GROUP_ARRAY(gsis_id)
                FROM raw_rosters_weekly prev_r
                WHERE prev_r.team = curr.team
                    AND prev_r.season = curr.season
                    AND prev_r.week = curr.week - 1
                    AND prev_r.gsis_id NOT IN (
                        SELECT curr_r.gsis_id
                        FROM raw_rosters_weekly curr_r
                        WHERE curr_r.team = curr.team
                            AND curr_r.season = curr.season
                            AND curr_r.week = curr.week
                    )
            ),
            'position_changes', (
                -- Players whose position changed
                SELECT JSON_GROUP_ARRAY(
                    JSON_OBJECT(
                        'gsis_id', curr_r.gsis_id,
                        'old_position', prev_r.position,
                        'new_position', curr_r.position
                    )
                )
                FROM raw_rosters_weekly curr_r
                JOIN raw_rosters_weekly prev_r
                    ON curr_r.gsis_id = prev_r.gsis_id
                    AND prev_r.team = curr.team
                    AND prev_r.season = curr.season
                    AND prev_r.week = curr.week - 1
                WHERE curr_r.team = curr.team
                    AND curr_r.season = curr.season
                    AND curr_r.week = curr.week
                    AND curr_r.position != prev_r.position
            )
        ) as key_changes

    FROM roster_snapshots curr
)

-- Step 4: Combine all data
INSERT INTO team_roster_snapshots
SELECT
    rs.snapshot_id,
    rs.team,
    rs.season,
    rs.week,
    rs.snapshot_date,
    rs.active_players,
    COALESCE(di.depth_chart, JSON_ARRAY()) as depth_chart,
    COALESCE(rc.key_changes, JSON_OBJECT()) as key_changes,
    CURRENT_TIMESTAMP as created_at
FROM roster_snapshots rs
LEFT JOIN depth_info di
    ON rs.team = di.team
    AND rs.season = di.season
    AND rs.week = di.week
LEFT JOIN roster_changes rc
    ON rs.snapshot_id = rc.snapshot_id;
```

#### Polars Implementation

```python
def create_weekly_roster_snapshots(self):
    """Create time-aware roster snapshots using Polars"""
    logger.info("📸 Creating weekly roster snapshots...")

    conn = self.db.connect()

    # Load base roster data
    rosters_df = pl.from_pandas(
        conn.execute("""
            SELECT team, season, week, gsis_id, full_name,
                   position, depth_chart_position, jersey_number,
                   status, years_exp, height, weight
            FROM raw_rosters_weekly
            WHERE gsis_id IS NOT NULL
        """).fetchdf()
    )

    # Load depth charts
    depth_df = pl.from_pandas(
        conn.execute("""
            SELECT club_code as team, season, week, gsis_id,
                   position, formation, depth_team,
                   first_name, last_name
            FROM raw_depth_charts
            WHERE gsis_id IS NOT NULL
        """).fetchdf()
    )

    # Load schedules for dates
    schedules_df = pl.from_pandas(
        conn.execute("""
            SELECT season, week, gameday, home_team, away_team
            FROM raw_schedules
        """).fetchdf()
    )

    # Get unique team/season/week combinations
    team_weeks = rosters_df.select(["team", "season", "week"]).unique()

    snapshots = []

    for row in team_weeks.iter_rows(named=True):
        team = row["team"]
        season = row["season"]
        week = row["week"]

        snapshot_id = f"{team}_{season}_W{week}"

        # Get snapshot date from schedule
        game_dates = schedules_df.filter(
            (pl.col("season") == season) &
            (pl.col("week") == week) &
            ((pl.col("home_team") == team) | (pl.col("away_team") == team))
        ).select("gameday")

        snapshot_date = game_dates[0, "gameday"] if len(game_dates) > 0 else None

        # Get active players for this snapshot
        active_players = rosters_df.filter(
            (pl.col("team") == team) &
            (pl.col("season") == season) &
            (pl.col("week") == week) &
            pl.col("status").is_in(["ACT", "RES"])
        ).to_dicts()

        # Get depth chart
        depth_chart = depth_df.filter(
            (pl.col("team") == team) &
            (pl.col("season") == season) &
            (pl.col("week") == week)
        ).to_dicts()

        # Calculate key changes from previous week
        if week > 1:
            prev_players = rosters_df.filter(
                (pl.col("team") == team) &
                (pl.col("season") == season) &
                (pl.col("week") == week - 1)
            )

            curr_player_ids = set(rosters_df.filter(
                (pl.col("team") == team) &
                (pl.col("season") == season) &
                (pl.col("week") == week)
            )["gsis_id"].to_list())

            prev_player_ids = set(prev_players["gsis_id"].to_list())

            new_players = list(curr_player_ids - prev_player_ids)
            removed_players = list(prev_player_ids - curr_player_ids)

            # Position changes (simplified)
            position_changes = []

            key_changes = {
                "new_players": new_players,
                "removed_players": removed_players,
                "position_changes": position_changes
            }
        else:
            key_changes = {
                "new_players": [],
                "removed_players": [],
                "position_changes": []
            }

        snapshots.append({
            "snapshot_id": snapshot_id,
            "team": team,
            "season": season,
            "week": week,
            "snapshot_date": snapshot_date,
            "active_players": json.dumps(active_players),
            "depth_chart": json.dumps(depth_chart),
            "key_changes": json.dumps(key_changes),
            "created_at": datetime.now()
        })

    # Convert to DataFrame and store
    snapshots_df = pl.DataFrame(snapshots)

    rows_inserted = self.db.store_dataframe(
        snapshots_df,
        "team_roster_snapshots",
        if_exists="replace"
    )

    logger.info(f"✅ Created {rows_inserted:,} roster snapshots")
    logger.info(f"  📊 Teams: {snapshots_df['team'].n_unique()}")
    logger.info(f"  📊 Seasons: {snapshots_df['season'].n_unique()}")
    logger.info(f"  📊 Weeks: {snapshots_df['week'].n_unique()}")

    return rows_inserted
```

#### Edge Cases & Handling

| Edge Case | Detection | Resolution |
|-----------|-----------|------------|
| **Week 1 (no previous week)** | `week == 1` | Set key_changes to empty JSON object |
| **Missing depth chart** | No depth chart data for team/week | Set depth_chart to empty JSON array |
| **Bye week** | No game in schedule | Set snapshot_date to NULL, still create snapshot |
| **Multiple games in week** | Rare (e.g., Thursday + Sunday) | Use MIN(gameday) for snapshot_date |
| **Player status changes** | Same player, different status | Include status in active_players JSON |
| **Season boundary** | Week 1 of new season | Don't compare to previous season's last week |

---

### 1.3 Method: `classify_player_experience_levels()`

**Purpose:** Classify each player's experience level for every season they played, used for ML confidence adjustments.

#### Input Data Sources

```python
# Primary: player_lifecycle table (must run build_player_lifecycle_table first)
"""
Columns needed:
- player_id
- first_nfl_season
"""

# Secondary: raw_rosters_weekly for years_exp validation
"""
Columns needed:
- gsis_id
- season
- years_exp (from NFL data)
"""

# Configuration: config.py
"""
experience_thresholds:
  rookie: 1           # seasons_played <= 1
  developing: [2, 3]  # seasons_played in [2, 3]
  veteran: 4          # seasons_played >= 4

confidence_thresholds:
  rookie: 0.6
  developing: 0.8
  veteran: 1.0
"""
```

#### SQL Query Pattern

```sql
-- Step 1: Calculate seasons played for each player in each season
WITH player_seasons AS (
    SELECT DISTINCT
        ps.player_id,
        ps.season,
        pl.first_nfl_season,

        -- Calculate seasons played (current season - first season + 1)
        ps.season - pl.first_nfl_season + 1 as seasons_played

    FROM raw_player_stats ps
    INNER JOIN player_lifecycle pl ON ps.player_id = pl.player_id
    WHERE ps.player_id IS NOT NULL
        AND ps.season IS NOT NULL
        AND pl.first_nfl_season IS NOT NULL
),

-- Step 2: Classify experience level based on thresholds
experience_classification AS (
    SELECT
        player_id,
        season,
        seasons_played,

        -- Experience category
        CASE
            WHEN seasons_played <= 1 THEN 'rookie'
            WHEN seasons_played IN (2, 3) THEN 'developing'
            ELSE 'veteran'
        END as experience_category,

        -- Prediction strategy based on experience
        CASE
            WHEN seasons_played <= 1 THEN 'high_variance_model'
            WHEN seasons_played IN (2, 3) THEN 'mixed_model'
            ELSE 'historical_trend_model'
        END as prediction_strategy,

        -- Confidence multiplier from config
        CASE
            WHEN seasons_played <= 1 THEN 0.6
            WHEN seasons_played IN (2, 3) THEN 0.8
            ELSE 1.0
        END as confidence_multiplier,

        CURRENT_TIMESTAMP as created_at

    FROM player_seasons
)

-- Step 3: Insert into classification table
INSERT OR REPLACE INTO player_experience_classification
SELECT * FROM experience_classification;
```

#### Polars Implementation

```python
def classify_player_experience_levels(self):
    """Classify players by experience level using Polars"""
    logger.info("🎓 Classifying player experience levels...")

    conn = self.db.connect()

    # Load player lifecycle (must exist)
    lifecycle_df = pl.from_pandas(
        conn.execute("""
            SELECT player_id, first_nfl_season
            FROM player_lifecycle
            WHERE first_nfl_season IS NOT NULL
        """).fetchdf()
    )

    # Load all player-season combinations from stats
    player_seasons_df = pl.from_pandas(
        conn.execute("""
            SELECT DISTINCT player_id, season
            FROM raw_player_stats
            WHERE player_id IS NOT NULL AND season IS NOT NULL
        """).fetchdf()
    )

    # Join to get first_nfl_season for each player-season
    experience_df = player_seasons_df.join(
        lifecycle_df,
        on="player_id",
        how="inner"
    )

    # Calculate seasons played
    experience_df = experience_df.with_columns([
        (pl.col("season") - pl.col("first_nfl_season") + 1).alias("seasons_played")
    ])

    # Get thresholds from config
    thresholds = config.feature_engineering_config["experience_thresholds"]
    confidence = config.ml_config["confidence_thresholds"]

    # Classify experience
    experience_df = experience_df.with_columns([
        # Experience category
        pl.when(pl.col("seasons_played") <= thresholds["rookie"])
            .then(pl.lit("rookie"))
            .when(pl.col("seasons_played").is_in(thresholds["developing"]))
            .then(pl.lit("developing"))
            .otherwise(pl.lit("veteran"))
            .alias("experience_category"),

        # Prediction strategy
        pl.when(pl.col("seasons_played") <= thresholds["rookie"])
            .then(pl.lit("high_variance_model"))
            .when(pl.col("seasons_played").is_in(thresholds["developing"]))
            .then(pl.lit("mixed_model"))
            .otherwise(pl.lit("historical_trend_model"))
            .alias("prediction_strategy"),

        # Confidence multiplier
        pl.when(pl.col("seasons_played") <= thresholds["rookie"])
            .then(pl.lit(confidence["rookie"]))
            .when(pl.col("seasons_played").is_in(thresholds["developing"]))
            .then(pl.lit(confidence["developing"]))
            .otherwise(pl.lit(confidence["veteran"]))
            .alias("confidence_multiplier"),

        # Timestamp
        pl.lit(datetime.now()).alias("created_at")
    ])

    # Select final columns
    final_df = experience_df.select([
        "player_id",
        "season",
        "experience_category",
        "seasons_played",
        "prediction_strategy",
        "confidence_multiplier",
        "created_at"
    ])

    # Store in database
    rows_inserted = self.db.store_dataframe(
        final_df,
        "player_experience_classification",
        if_exists="replace"
    )

    logger.info(f"✅ Classified {rows_inserted:,} player-season combinations")

    # Log distribution
    distribution = final_df.group_by("experience_category").agg([
        pl.col("player_id").count().alias("count"),
        pl.col("confidence_multiplier").first().alias("confidence")
    ])

    for row in distribution.iter_rows(named=True):
        logger.info(f"  📊 {row['experience_category']}: {row['count']:,} records (confidence: {row['confidence']})")

    return rows_inserted
```

#### Edge Cases & Handling

| Edge Case | Detection | Resolution |
|-----------|-----------|------------|
| **Missing first_nfl_season** | `first_nfl_season IS NULL` in lifecycle | Exclude from classification (log warning) |
| **Negative seasons_played** | `season < first_nfl_season` | Data error - log and skip, investigate raw data |
| **Inconsistent years_exp** | roster years_exp != calculated seasons | Trust calculated value from first_nfl_season |
| **Mid-season roster additions** | Player appears mid-season as rookie | Still count as season 1 if first_nfl_season matches |
| **Practice squad players** | May not have stats but in rosters | Include if in raw_rosters_weekly, exclude if no games played |

---

## 2. Implementation Details

### 2.1 Method Signatures

```python
def build_player_lifecycle_table(self) -> int:
    """
    Build player career tracking table.

    Combines data from raw_players, raw_rosters_weekly, and raw_player_stats
    to create comprehensive player lifecycle records.

    Returns:
        int: Number of player records created

    Raises:
        ValueError: If required raw tables don't exist
        DatabaseError: If database operation fails

    Side Effects:
        - Replaces all data in player_lifecycle table
        - Logs summary statistics
    """
    pass

def create_weekly_roster_snapshots(self) -> int:
    """
    Create time-aware roster snapshots for each team/week.

    Generates snapshot records with active players, depth chart info,
    and roster changes compared to previous week.

    Returns:
        int: Number of snapshot records created

    Raises:
        ValueError: If required raw tables don't exist
        DatabaseError: If database operation fails

    Side Effects:
        - Replaces all data in team_roster_snapshots table
        - Logs snapshot creation progress
    """
    pass

def classify_player_experience_levels(self) -> int:
    """
    Classify players by experience level for each season.

    Uses player_lifecycle.first_nfl_season to calculate seasons_played
    and classify into rookie/developing/veteran categories.

    Returns:
        int: Number of player-season classifications created

    Raises:
        ValueError: If player_lifecycle table doesn't exist
        DatabaseError: If database operation fails

    Pre-conditions:
        - build_player_lifecycle_table() must be run first

    Side Effects:
        - Replaces all data in player_experience_classification table
        - Logs experience distribution
    """
    pass
```

### 2.2 Execution Order & Dependencies

```
STAGE 1: Raw Data Collection (must complete first)
    ↓
    ├── raw_players
    ├── raw_rosters_weekly
    ├── raw_depth_charts
    ├── raw_player_stats
    └── raw_schedules

STAGE 2: Player Lifecycle & Roster Management
    ↓
    ├── Step 1: build_player_lifecycle_table()
    │   └── Depends on: raw_players, raw_rosters_weekly, raw_player_stats
    │   └── Creates: player_lifecycle
    │
    ├── Step 2: create_weekly_roster_snapshots()
    │   └── Depends on: raw_rosters_weekly, raw_depth_charts, raw_schedules
    │   └── Creates: team_roster_snapshots
    │
    └── Step 3: classify_player_experience_levels()
        └── Depends on: player_lifecycle (from Step 1), raw_player_stats
        └── Creates: player_experience_classification

STAGE 3: Feature Engineering (future)
    └── Uses: player_lifecycle, team_roster_snapshots, player_experience_classification
```

### 2.3 Database Operations

#### Pre-flight Checks

```python
def _validate_raw_data_exists(self) -> None:
    """Validate required raw tables exist before Stage 2"""
    required_tables = [
        'raw_players',
        'raw_rosters_weekly',
        'raw_depth_charts',
        'raw_player_stats',
        'raw_schedules'
    ]

    existing_tables = self.db.list_tables()

    missing = [t for t in required_tables if t not in existing_tables]

    if missing:
        raise ValueError(
            f"Missing required raw tables: {missing}. "
            f"Please run Stage 1 (full_historical_load) first."
        )

    # Check for data
    for table in required_tables:
        count = self.db.execute_one(f"SELECT COUNT(*) FROM {table}")[0]
        if count == 0:
            logger.warning(f"⚠️  Table {table} exists but is empty")
```

#### Transaction Handling

```python
def process_roster_snapshots(self):
    """
    Stage 2: Create time-aware roster snapshots with transaction safety
    """
    logger.info("🗂️  Stage 2: Processing roster snapshots")

    # Pre-flight checks
    self._validate_raw_data_exists()

    try:
        # Execute methods in order
        logger.info("Step 1/3: Building player lifecycle...")
        lifecycle_count = self.build_player_lifecycle_table()

        logger.info("Step 2/3: Creating roster snapshots...")
        snapshot_count = self.create_weekly_roster_snapshots()

        logger.info("Step 3/3: Classifying experience levels...")
        classification_count = self.classify_player_experience_levels()

        # Summary
        logger.info("✅ Roster snapshot processing completed")
        logger.info(f"  📊 Player lifecycle: {lifecycle_count:,} records")
        logger.info(f"  📊 Roster snapshots: {snapshot_count:,} records")
        logger.info(f"  📊 Experience classifications: {classification_count:,} records")

    except Exception as e:
        logger.error(f"❌ Stage 2 failed: {e}")
        raise
```

---

## 3. Data Quality

### 3.1 Validation Checks

```python
class Stage2Validator:
    """Validation checks for Stage 2 outputs"""

    def __init__(self, db: NFLDatabase):
        self.db = db

    def validate_player_lifecycle(self) -> Dict[str, Any]:
        """Validate player_lifecycle table"""
        conn = self.db.connect()

        checks = {}

        # Check 1: No NULL player_ids
        null_ids = conn.execute("""
            SELECT COUNT(*) FROM player_lifecycle WHERE player_id IS NULL
        """).fetchone()[0]
        checks['no_null_player_ids'] = (null_ids == 0)

        # Check 2: first_nfl_season <= last_nfl_season
        invalid_spans = conn.execute("""
            SELECT COUNT(*) FROM player_lifecycle
            WHERE first_nfl_season > last_nfl_season
        """).fetchone()[0]
        checks['valid_career_spans'] = (invalid_spans == 0)

        # Check 3: Reasonable season range (e.g., 1990-2025)
        invalid_seasons = conn.execute("""
            SELECT COUNT(*) FROM player_lifecycle
            WHERE first_nfl_season < 1990 OR last_nfl_season > 2030
        """).fetchone()[0]
        checks['reasonable_season_range'] = (invalid_seasons == 0)

        # Check 4: career_teams is not empty for active players
        empty_teams = conn.execute("""
            SELECT COUNT(*) FROM player_lifecycle
            WHERE retirement_status = 'active'
                AND (career_teams IS NULL OR LENGTH(career_teams) = 0)
        """).fetchone()[0]
        checks['active_players_have_teams'] = (empty_teams == 0)

        # Check 5: Unique player_ids
        total = conn.execute("SELECT COUNT(*) FROM player_lifecycle").fetchone()[0]
        unique = conn.execute("SELECT COUNT(DISTINCT player_id) FROM player_lifecycle").fetchone()[0]
        checks['unique_player_ids'] = (total == unique)

        return {
            'table': 'player_lifecycle',
            'checks': checks,
            'all_passed': all(checks.values())
        }

    def validate_roster_snapshots(self) -> Dict[str, Any]:
        """Validate team_roster_snapshots table"""
        conn = self.db.connect()

        checks = {}

        # Check 1: Unique snapshot_ids
        total = conn.execute("SELECT COUNT(*) FROM team_roster_snapshots").fetchone()[0]
        unique = conn.execute("SELECT COUNT(DISTINCT snapshot_id) FROM team_roster_snapshots").fetchone()[0]
        checks['unique_snapshot_ids'] = (total == unique)

        # Check 2: Valid team codes (3 characters)
        invalid_teams = conn.execute("""
            SELECT COUNT(*) FROM team_roster_snapshots
            WHERE LENGTH(team) != 3
        """).fetchone()[0]
        checks['valid_team_codes'] = (invalid_teams == 0)

        # Check 3: Valid week numbers (1-18 for regular season, 1-22 total)
        invalid_weeks = conn.execute("""
            SELECT COUNT(*) FROM team_roster_snapshots
            WHERE week < 1 OR week > 22
        """).fetchone()[0]
        checks['valid_week_numbers'] = (invalid_weeks == 0)

        # Check 4: Active players JSON is valid
        invalid_json = conn.execute("""
            SELECT COUNT(*) FROM team_roster_snapshots
            WHERE active_players IS NOT NULL
                AND JSON_VALID(active_players) = 0
        """).fetchone()[0]
        checks['valid_active_players_json'] = (invalid_json == 0)

        # Check 5: At least some players in each snapshot
        empty_rosters = conn.execute("""
            SELECT COUNT(*) FROM team_roster_snapshots
            WHERE JSON_ARRAY_LENGTH(active_players) = 0
        """).fetchone()[0]
        checks['non_empty_rosters'] = (empty_rosters < total * 0.01)  # Allow <1% empty

        return {
            'table': 'team_roster_snapshots',
            'checks': checks,
            'all_passed': all(checks.values())
        }

    def validate_experience_classification(self) -> Dict[str, Any]:
        """Validate player_experience_classification table"""
        conn = self.db.connect()

        checks = {}

        # Check 1: Valid experience categories
        invalid_categories = conn.execute("""
            SELECT COUNT(*) FROM player_experience_classification
            WHERE experience_category NOT IN ('rookie', 'developing', 'veteran')
        """).fetchone()[0]
        checks['valid_categories'] = (invalid_categories == 0)

        # Check 2: seasons_played >= 1
        invalid_seasons = conn.execute("""
            SELECT COUNT(*) FROM player_experience_classification
            WHERE seasons_played < 1
        """).fetchone()[0]
        checks['positive_seasons_played'] = (invalid_seasons == 0)

        # Check 3: Confidence multipliers in expected range
        invalid_confidence = conn.execute("""
            SELECT COUNT(*) FROM player_experience_classification
            WHERE confidence_multiplier < 0 OR confidence_multiplier > 1.5
        """).fetchone()[0]
        checks['valid_confidence_multipliers'] = (invalid_confidence == 0)

        # Check 4: Category matches seasons_played logic
        mismatched_categories = conn.execute("""
            SELECT COUNT(*) FROM player_experience_classification
            WHERE (seasons_played <= 1 AND experience_category != 'rookie')
                OR (seasons_played IN (2, 3) AND experience_category != 'developing')
                OR (seasons_played >= 4 AND experience_category != 'veteran')
        """).fetchone()[0]
        checks['consistent_categorization'] = (mismatched_categories == 0)

        # Check 5: Unique (player_id, season) combinations
        total = conn.execute("SELECT COUNT(*) FROM player_experience_classification").fetchone()[0]
        unique = conn.execute("""
            SELECT COUNT(DISTINCT player_id || '_' || season)
            FROM player_experience_classification
        """).fetchone()[0]
        checks['unique_player_seasons'] = (total == unique)

        return {
            'table': 'player_experience_classification',
            'checks': checks,
            'all_passed': all(checks.values())
        }

    def run_all_validations(self) -> Dict[str, Any]:
        """Run all Stage 2 validations"""
        results = {
            'lifecycle': self.validate_player_lifecycle(),
            'snapshots': self.validate_roster_snapshots(),
            'experience': self.validate_experience_classification()
        }

        all_passed = all(r['all_passed'] for r in results.values())

        return {
            'stage': 'Stage 2: Player Lifecycle & Roster Management',
            'all_validations_passed': all_passed,
            'results': results
        }
```

### 3.2 Error Handling Strategy

```python
# Pattern for robust error handling
def build_player_lifecycle_table(self):
    """Build player career tracking table with robust error handling"""
    logger.info("👥 Building player lifecycle table...")

    try:
        # Validate inputs
        self._validate_raw_data_exists()

        conn = self.db.connect()

        # Check for data in source tables
        player_count = conn.execute("SELECT COUNT(*) FROM raw_players").fetchone()[0]
        if player_count == 0:
            logger.warning("⚠️  raw_players is empty, skipping lifecycle build")
            return 0

        # Execute main logic (from section 1.1)
        # ... implementation ...

        # Validate output
        validator = Stage2Validator(self.db)
        validation = validator.validate_player_lifecycle()

        if not validation['all_passed']:
            logger.warning("⚠️  Lifecycle validation failed:")
            for check, passed in validation['checks'].items():
                if not passed:
                    logger.warning(f"  ❌ {check}")

        return rows_inserted

    except KeyError as e:
        logger.error(f"❌ Missing expected column: {e}")
        logger.info("💡 Tip: Ensure raw data schema matches expected structure")
        raise

    except pl.exceptions.ComputeError as e:
        logger.error(f"❌ Polars computation error: {e}")
        logger.info("💡 Tip: Check for NULL values or type mismatches")
        raise

    except Exception as e:
        logger.error(f"❌ Unexpected error building lifecycle: {e}")
        logger.info("💡 Tip: Check database logs for more details")
        raise
```

### 3.3 Logging Requirements

```python
# Logging pattern for Stage 2 methods

# At method start
logger.info("👥 Building player lifecycle table...")

# Progress indicators for long operations
logger.info("  📊 Loading player metadata...")
logger.info("  📊 Calculating career spans...")
logger.info("  📊 Combining data sources...")

# Success with statistics
logger.info(f"✅ Built player lifecycle for {rows_inserted:,} players")
logger.info(f"  📊 Career span: {earliest_season} - {latest_season}")
logger.info(f"  📊 Active: {active_count:,} | Retired: {retired_count:,} | Inactive: {inactive_count:,}")

# Warnings for data issues (non-fatal)
logger.warning(f"⚠️  {missing_draft_info_count} players missing draft information")
logger.warning(f"⚠️  {conflicting_data_count} players have conflicting career span data")

# Errors (fatal)
logger.error(f"❌ Failed to build lifecycle table: {error_message}")
logger.info("💡 Tip: [helpful suggestion for user]")
```

---

## 4. Test Cases

### 4.1 Unit Test Structure

```python
# test_stage2.py

import pytest
import polars as pl
from datetime import datetime
from src.data_pipeline import NFLDataPipeline
from src.database import NFLDatabase

@pytest.fixture
def test_db():
    """Create test database with sample data"""
    db = NFLDatabase("test_stage2.duckdb")
    conn = db.connect()

    # Create tables
    # ... (use setup_database.py functions)

    yield db

    # Cleanup
    db.close()
    os.remove("test_stage2.duckdb")

class TestPlayerLifecycle:
    """Tests for build_player_lifecycle_table()"""

    def test_basic_lifecycle_creation(self, test_db):
        """Test basic lifecycle table creation"""
        # Insert test data
        test_players = pl.DataFrame({
            'player_id': ['P001', 'P002'],
            'gsis_id': ['G001', 'G002'],
            'position': ['QB', 'WR'],
            'draft_year': [2020, 2019]
        })
        test_db.store_dataframe(test_players, 'raw_players')

        # Run pipeline method
        pipeline = NFLDataPipeline(db_file="test_stage2.duckdb")
        result = pipeline.build_player_lifecycle_table()

        # Assertions
        assert result == 2

        lifecycle = test_db.execute("SELECT * FROM player_lifecycle")
        assert len(lifecycle) == 2

    def test_missing_draft_info_handling(self, test_db):
        """Test handling of players without draft information (UDFAs)"""
        test_players = pl.DataFrame({
            'player_id': ['P003'],
            'gsis_id': ['G003'],
            'position': ['RB'],
            'draft_year': [None],  # Undrafted
            'draft_round': [None],
            'draft_pick': [None]
        })
        test_db.store_dataframe(test_players, 'raw_players')

        pipeline = NFLDataPipeline(db_file="test_stage2.duckdb")
        result = pipeline.build_player_lifecycle_table()

        lifecycle = test_db.execute("SELECT * FROM player_lifecycle WHERE player_id = 'P003'")
        assert lifecycle[0]['draft_year'] is None  # Should accept NULL

    def test_career_span_calculation(self, test_db):
        """Test correct calculation of first/last NFL season"""
        # Player with multi-season career
        test_rosters = pl.DataFrame({
            'gsis_id': ['G001', 'G001', 'G001'],
            'season': [2020, 2021, 2022],
            'team': ['KC', 'KC', 'TB']
        })
        test_db.store_dataframe(test_rosters, 'raw_rosters_weekly')

        test_players = pl.DataFrame({
            'player_id': ['P001'],
            'gsis_id': ['G001'],
            'position': ['QB']
        })
        test_db.store_dataframe(test_players, 'raw_players')

        pipeline = NFLDataPipeline(db_file="test_stage2.duckdb")
        pipeline.build_player_lifecycle_table()

        lifecycle = test_db.execute("SELECT * FROM player_lifecycle WHERE player_id = 'P001'")[0]
        assert lifecycle['first_nfl_season'] == 2020
        assert lifecycle['last_nfl_season'] == 2022
        assert 'KC' in lifecycle['career_teams'] and 'TB' in lifecycle['career_teams']

    def test_retirement_status_logic(self, test_db):
        """Test retirement status classification"""
        # Active player (played in latest season)
        # Inactive player (last played 2+ years ago)
        # Retired player (status = 'RET')
        # ... implementation
        pass

class TestRosterSnapshots:
    """Tests for create_weekly_roster_snapshots()"""

    def test_snapshot_creation(self, test_db):
        """Test basic snapshot creation"""
        test_rosters = pl.DataFrame({
            'team': ['KC', 'KC'],
            'season': [2024, 2024],
            'week': [1, 1],
            'gsis_id': ['G001', 'G002'],
            'full_name': ['Patrick Mahomes', 'Travis Kelce'],
            'position': ['QB', 'TE'],
            'status': ['ACT', 'ACT']
        })
        test_db.store_dataframe(test_rosters, 'raw_rosters_weekly')

        pipeline = NFLDataPipeline(db_file="test_stage2.duckdb")
        result = pipeline.create_weekly_roster_snapshots()

        snapshot = test_db.execute("""
            SELECT * FROM team_roster_snapshots
            WHERE snapshot_id = 'KC_2024_W1'
        """)[0]

        assert snapshot['team'] == 'KC'
        assert snapshot['season'] == 2024
        assert snapshot['week'] == 1
        assert len(json.loads(snapshot['active_players'])) == 2

    def test_week1_no_previous_comparison(self, test_db):
        """Test that Week 1 doesn't try to compare to non-existent Week 0"""
        # ... implementation
        pass

    def test_roster_changes_detection(self, test_db):
        """Test detection of added/removed players between weeks"""
        # Week 1: Players A, B, C
        # Week 2: Players A, B, D (C removed, D added)
        # ... implementation
        pass

    def test_bye_week_handling(self, test_db):
        """Test snapshot creation during bye week (no game in schedule)"""
        # ... implementation
        pass

class TestExperienceClassification:
    """Tests for classify_player_experience_levels()"""

    def test_rookie_classification(self, test_db):
        """Test rookie classification (seasons_played == 1)"""
        # Player with first_nfl_season = 2024, playing in 2024
        test_lifecycle = pl.DataFrame({
            'player_id': ['P001'],
            'first_nfl_season': [2024]
        })
        test_db.store_dataframe(test_lifecycle, 'player_lifecycle')

        test_stats = pl.DataFrame({
            'player_id': ['P001'],
            'season': [2024]
        })
        test_db.store_dataframe(test_stats, 'raw_player_stats')

        pipeline = NFLDataPipeline(db_file="test_stage2.duckdb")
        pipeline.classify_player_experience_levels()

        classification = test_db.execute("""
            SELECT * FROM player_experience_classification
            WHERE player_id = 'P001' AND season = 2024
        """)[0]

        assert classification['experience_category'] == 'rookie'
        assert classification['seasons_played'] == 1
        assert classification['confidence_multiplier'] == 0.6
        assert classification['prediction_strategy'] == 'high_variance_model'

    def test_developing_classification(self, test_db):
        """Test developing classification (seasons_played in [2, 3])"""
        # ... implementation
        pass

    def test_veteran_classification(self, test_db):
        """Test veteran classification (seasons_played >= 4)"""
        # ... implementation
        pass

    def test_multi_season_progression(self, test_db):
        """Test player progressing through experience levels"""
        # Player across multiple seasons should show progression
        # 2020: rookie (1 season)
        # 2021: developing (2 seasons)
        # 2022: developing (3 seasons)
        # 2023: veteran (4 seasons)
        # ... implementation
        pass
```

### 4.2 Integration Test

```python
def test_full_stage2_pipeline(test_db):
    """Test complete Stage 2 execution"""
    # Setup: Load sample raw data
    load_sample_raw_data(test_db)

    # Execute Stage 2
    pipeline = NFLDataPipeline(db_file="test_stage2.duckdb")
    pipeline.process_roster_snapshots()

    # Validate all tables were created
    tables = test_db.list_tables()
    assert 'player_lifecycle' in tables
    assert 'team_roster_snapshots' in tables
    assert 'player_experience_classification' in tables

    # Validate data consistency
    lifecycle_count = test_db.execute("SELECT COUNT(*) FROM player_lifecycle")[0][0]
    experience_players = test_db.execute("""
        SELECT COUNT(DISTINCT player_id) FROM player_experience_classification
    """)[0][0]

    # Every player in experience_classification should be in lifecycle
    assert experience_players <= lifecycle_count

    # Run validations
    validator = Stage2Validator(test_db)
    results = validator.run_all_validations()
    assert results['all_validations_passed']
```

### 4.3 Sample Data Inputs/Outputs

```python
# Sample input: raw_players
"""
player_id | gsis_id | position | draft_year | draft_round | college
P001      | G001    | QB       | 2017       | 1           | Texas Tech
P002      | G002    | TE       | 2013       | 3           | Cincinnati
P003      | G003    | WR       | 2024       | 1           | Ohio State
"""

# Sample input: raw_rosters_weekly
"""
gsis_id | season | week | team | years_exp
G001    | 2017   | 1    | KC   | 0
G001    | 2024   | 5    | KC   | 7
G002    | 2013   | 1    | KC   | 0
G002    | 2024   | 5    | KC   | 11
G003    | 2024   | 1    | MIN  | 0
"""

# Expected output: player_lifecycle
"""
player_id | first_nfl_season | last_nfl_season | career_teams | primary_position | retirement_status
P001      | 2017            | 2024            | ['KC']       | QB               | active
P002      | 2013            | 2024            | ['KC']       | TE               | active
P003      | 2024            | 2024            | ['MIN']      | WR               | active
"""

# Expected output: player_experience_classification
"""
player_id | season | experience_category | seasons_played | confidence_multiplier
P001      | 2017   | rookie             | 1              | 0.6
P001      | 2024   | veteran            | 8              | 1.0
P002      | 2013   | rookie             | 1              | 0.6
P002      | 2014   | developing         | 2              | 0.8
P002      | 2024   | veteran            | 12             | 1.0
P003      | 2024   | rookie             | 1              | 0.6
"""
```

### 4.4 Success Criteria

Stage 2 is considered **DONE** when:

1. ✅ All three methods are implemented and functional
2. ✅ All unit tests pass (100% pass rate)
3. ✅ Integration test passes
4. ✅ Data validation checks all pass (Stage2Validator)
5. ✅ No errors logged during execution on test data
6. ✅ Performance benchmarks met:
   - `build_player_lifecycle_table()`: < 30 seconds for 5000 players
   - `create_weekly_roster_snapshots()`: < 60 seconds for 5 seasons of data
   - `classify_player_experience_levels()`: < 15 seconds for 10,000 player-seasons
7. ✅ Documentation updated with examples
8. ✅ Code reviewed and merged to main branch

---

## 5. Dependencies

### 5.1 Required Raw Tables

Must exist and contain data before Stage 2 can run:

| Table | Purpose | Criticality | Minimum Data Required |
|-------|---------|-------------|----------------------|
| `raw_players` | Player metadata, draft info | **CRITICAL** | At least 1 player record |
| `raw_rosters_weekly` | Weekly roster status, years_exp | **CRITICAL** | At least 1 season of rosters |
| `raw_player_stats` | Player statistics for validation | **HIGH** | At least 1 season of stats |
| `raw_depth_charts` | Depth chart positions | **MEDIUM** | Can create snapshots without depth charts (empty JSON) |
| `raw_schedules` | Game dates for snapshot_date | **MEDIUM** | Can create snapshots without dates (NULL snapshot_date) |

### 5.2 Configuration Requirements

```python
# config.py must contain:

feature_engineering_config = {
    "experience_thresholds": {
        "rookie": 1,              # int
        "developing": [2, 3],     # list[int]
        "veteran": 4              # int
    }
}

ml_config = {
    "confidence_thresholds": {
        "rookie": 0.6,            # float
        "developing": 0.8,        # float
        "veteran": 1.0            # float
    }
}
```

### 5.3 External Library Requirements

```toml
# pyproject.toml dependencies (already present)
dependencies = [
    "duckdb>=0.9.0",           # Database engine
    "polars>=0.19.0",          # DataFrame operations
    "nflreadpy>=0.2.0",        # Data fetching (Stage 1)
    "numpy>=1.24.0",           # Numerical operations
    "pandas>=2.0.0",           # Compatibility layer
]
```

### 5.4 Pre-execution Checklist

Before running Stage 2:

- [ ] Database file exists (created by `setup_database.py`)
- [ ] Stage 1 completed successfully (raw tables populated)
- [ ] `player_lifecycle`, `team_roster_snapshots`, `player_experience_classification` tables exist (created by setup)
- [ ] Configuration values in `config.py` are correct
- [ ] Database has sufficient disk space (estimate: ~100MB per season)
- [ ] Database connection is working (`db.connect()` succeeds)

---

## 6. Performance Considerations

### 6.1 Query Optimization

```sql
-- Use indexes for common joins
CREATE INDEX IF NOT EXISTS idx_player_lifecycle_gsis
    ON player_lifecycle(gsis_id);

CREATE INDEX IF NOT EXISTS idx_roster_snapshots_team_season_week
    ON team_roster_snapshots(team, season, week);

CREATE INDEX IF NOT EXISTS idx_experience_player_season
    ON player_experience_classification(player_id, season);

-- Use EXPLAIN to analyze query plans
EXPLAIN SELECT * FROM player_lifecycle WHERE gsis_id = 'G001';
```

### 6.2 Batch Processing Strategy

```python
# For large datasets, process in batches
def create_weekly_roster_snapshots_batched(self, batch_size=100):
    """Process roster snapshots in batches for large datasets"""

    conn = self.db.connect()

    # Get all team-week combinations
    combinations = conn.execute("""
        SELECT DISTINCT team, season, week
        FROM raw_rosters_weekly
        ORDER BY season, week, team
    """).fetchall()

    total_batches = (len(combinations) + batch_size - 1) // batch_size

    for i in range(0, len(combinations), batch_size):
        batch = combinations[i:i+batch_size]
        logger.info(f"Processing batch {i//batch_size + 1}/{total_batches}")

        # Process batch...
        # ...
```

### 6.3 Memory Management

```python
# Use Polars lazy evaluation for large datasets
def build_player_lifecycle_lazy(self):
    """Memory-efficient version using Polars lazy evaluation"""

    # Scan without loading into memory
    players_lf = pl.scan_parquet("temp_players.parquet")  # if available
    # or use lazy CSV/IPC reading

    lifecycle_lf = (
        players_lf
        .join(career_span_lf, on="gsis_id", how="left")
        # ... other operations
        .collect(streaming=True)  # Streaming execution
    )
```

---

## 7. Implementation Timeline

### Phase 1: Core Implementation (Days 1-3)
- Day 1: Implement `build_player_lifecycle_table()`
- Day 2: Implement `create_weekly_roster_snapshots()`
- Day 3: Implement `classify_player_experience_levels()`

### Phase 2: Testing & Validation (Days 4-5)
- Day 4: Write and run unit tests, fix bugs
- Day 5: Write and run integration tests, data validation

### Phase 3: Documentation & Review (Day 6)
- Day 6: Update documentation, code review, merge

**Total Estimated Time:** 6 working days

---

## 8. Next Steps After Stage 2

Once Stage 2 is complete, the following becomes possible:

1. **Stage 3: Feature Engineering**
   - Calculate rolling statistics using player_lifecycle and roster snapshots
   - Build matchup features using team_roster_snapshots
   - Handle rookie vs veteran features using player_experience_classification

2. **Data Analysis**
   - Analyze player career trajectories
   - Study roster construction patterns
   - Identify experience distribution across positions

3. **ML Model Development**
   - Use confidence_multiplier for prediction weighting
   - Apply prediction_strategy for model selection
   - Implement rookie-specific models

---

## 9. References

### Code Files
- `/home/user/nfl/src/data_pipeline.py` - Pipeline orchestration
- `/home/user/nfl/src/database.py` - Database utilities
- `/home/user/nfl/src/config.py` - Configuration management
- `/home/user/nfl/setup_database.py` - Schema definitions

### Database Tables
- **Input:** `raw_players`, `raw_rosters_weekly`, `raw_depth_charts`, `raw_player_stats`, `raw_schedules`
- **Output:** `player_lifecycle`, `team_roster_snapshots`, `player_experience_classification`

### External Documentation
- DuckDB Documentation: https://duckdb.org/docs/
- Polars Documentation: https://pola-rs.github.io/polars/
- nflverse Data Dictionary: https://nflverse.nflverse.com/articles/dictionary.html

---

## Appendix A: SQL Reference Queries

### A.1 Check Player Lifecycle Coverage

```sql
-- See which players are in lifecycle vs raw tables
SELECT
    'raw_players' as source,
    COUNT(DISTINCT player_id) as player_count
FROM raw_players
UNION ALL
SELECT
    'raw_rosters_weekly',
    COUNT(DISTINCT gsis_id)
FROM raw_rosters_weekly
UNION ALL
SELECT
    'player_lifecycle',
    COUNT(DISTINCT player_id)
FROM player_lifecycle;
```

### A.2 Roster Snapshot Summary

```sql
-- Get roster snapshot counts by team and season
SELECT
    team,
    season,
    COUNT(*) as snapshot_count,
    MIN(week) as first_week,
    MAX(week) as last_week,
    AVG(JSON_ARRAY_LENGTH(active_players)) as avg_roster_size
FROM team_roster_snapshots
GROUP BY team, season
ORDER BY season DESC, team;
```

### A.3 Experience Distribution

```sql
-- See experience distribution across seasons
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

## Appendix B: Common Issues & Solutions

| Issue | Symptoms | Solution |
|-------|----------|----------|
| **Missing gsis_id** | Join between tables fails | Use LEFT JOIN and COALESCE for alternate IDs |
| **Duplicate player entries** | Lifecycle count > expected | Add DISTINCT in queries, validate primary keys |
| **JSON parsing errors** | Invalid JSON in snapshots | Validate JSON before storing, use try/except |
| **Memory overflow** | Process crashes on large datasets | Use batch processing, Polars lazy evaluation |
| **Slow queries** | Methods take >5 minutes | Add indexes, use EXPLAIN to analyze, optimize joins |
| **NULL snapshot_dates** | Bye weeks or missing schedule | Allow NULL, document in schema |
| **Experience category mismatch** | Rookie classified as veteran | Check first_nfl_season calculation, validate thresholds |

---

**Document Version:** 1.0
**Last Updated:** 2025-11-06
**Author:** Claude Code (based on codebase analysis)
**Status:** Ready for Implementation
