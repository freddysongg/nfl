# Stage 4: ML Dataset Assembly & Target Creation - Implementation Plan

## Overview
This document provides a detailed technical specification for implementing the 4 methods in Stage 4 of the NFL data pipeline. Stage 4 combines engineered features from Stages 2-3 into a single ML-ready dataset with prediction targets, quality scores, and temporal validation.

**Goal**: Transform feature engineering outputs into a complete ML dataset ready for model training, with no data leakage and high data quality guarantees.

---

## 1. Technical Specification: `combine_all_features()`

### Purpose
Join player rolling features, team rolling features, and roster snapshots into a unified feature vector for each player-game observation.

### SQL JOIN Strategy

```sql
WITH game_context AS (
    -- Get game schedule information to determine opponents and home/away
    SELECT
        game_id,
        season,
        week,
        gameday as game_date,
        home_team,
        away_team,
        home_score,
        away_score,
        overtime
    FROM raw_schedules
    WHERE game_type = 'REG'  -- Regular season only
),

player_features AS (
    -- Get player rolling features from previous week (N-1 for week N)
    SELECT
        prf.player_id,
        prf.season,
        prf.week + 1 as target_week,  -- Features from week N predict week N+1
        prf.position,
        prf.stats_last3_games,
        prf.stats_last5_games,
        prf.stats_season_avg,
        prf.performance_trend,
        prf.usage_trend,
        prf.target_share_trend,
        prf.vs_opponent_history,
        prf.opp_rank_vs_position,
        prf.opp_avg_allowed_to_position,
        prf.home_away_splits,
        prf.divisional_game,
        prf.rest_days
    FROM player_rolling_features prf
),

team_features AS (
    -- Get team features from previous week
    SELECT
        trf.team,
        trf.season,
        trf.week + 1 as target_week,
        trf.off_epa_per_play_last3,
        trf.off_success_rate_last3,
        trf.off_explosive_play_rate,
        trf.off_red_zone_efficiency,
        trf.off_third_down_conv,
        trf.def_epa_per_play_last3,
        trf.def_success_rate_last3,
        trf.def_pressure_rate,
        trf.def_turnover_rate,
        trf.pass_rate_neutral,
        trf.pace_of_play,
        trf.time_of_possession_avg
    FROM team_rolling_features trf
),

opponent_team_features AS (
    -- Get opponent defensive features (what defense player will face)
    SELECT
        trf.team as opponent,
        trf.season,
        trf.week + 1 as target_week,
        trf.def_epa_per_play_last3 as opp_def_epa_per_play_last3,
        trf.def_success_rate_last3 as opp_def_success_rate_last3,
        trf.def_pressure_rate as opp_def_pressure_rate,
        trf.def_turnover_rate as opp_def_turnover_rate
    FROM team_rolling_features trf
),

roster_context AS (
    -- Get roster snapshot for experience level
    SELECT
        trs.snapshot_id,
        trs.team,
        trs.season,
        trs.week,
        trs.active_players,
        trs.depth_chart,
        -- Extract player experience from active_players JSON
        -- (assumes active_players contains player_id and experience info)
        trs.active_players as roster_json
    FROM team_roster_snapshots trs
),

player_experience AS (
    -- Get player experience classification
    SELECT
        player_id,
        season,
        experience_level
    FROM player_experience_classification
),

current_week_context AS (
    -- Get player's current team assignment for target week
    SELECT DISTINCT
        player_id,
        player_name,
        season,
        week,
        team,
        opponent_team
    FROM raw_player_stats
)

-- Main join to combine all features
SELECT
    -- Generate unique feature_id
    CONCAT(cwc.player_id, '_', cwc.season, '_', cwc.week) as feature_id,

    -- Entity identification
    'player' as entity_type,
    cwc.player_id as entity_id,

    -- Time context
    cwc.season,
    cwc.week,
    gc.game_date,

    -- Roster context
    rc.snapshot_id as roster_snapshot_id,
    pe.experience_level as player_experience_level,

    -- Player metadata (categorical)
    pf.position,
    cwc.team,
    cwc.opponent_team,
    CASE
        WHEN gc.home_team = cwc.team THEN 'home'
        WHEN gc.away_team = cwc.team THEN 'away'
        ELSE NULL
    END as home_away,
    pf.divisional_game,

    -- Player rolling features (numerical - extract from JSON)
    pf.performance_trend,
    pf.usage_trend,
    pf.target_share_trend,
    pf.opp_rank_vs_position,
    pf.rest_days,

    -- Team offensive features (numerical)
    tf.off_epa_per_play_last3,
    tf.off_success_rate_last3,
    tf.off_explosive_play_rate,
    tf.off_red_zone_efficiency,
    tf.off_third_down_conv,
    tf.pass_rate_neutral,
    tf.pace_of_play,
    tf.time_of_possession_avg,

    -- Opponent defensive features (numerical)
    otf.opp_def_epa_per_play_last3,
    otf.opp_def_success_rate_last3,
    otf.opp_def_pressure_rate,
    otf.opp_def_turnover_rate,

    -- JSON feature objects (to be extracted)
    pf.stats_last3_games,
    pf.stats_last5_games,
    pf.stats_season_avg,
    pf.vs_opponent_history,
    pf.opp_avg_allowed_to_position,
    pf.home_away_splits

FROM current_week_context cwc

-- Join game context
LEFT JOIN game_context gc
    ON cwc.season = gc.season
    AND cwc.week = gc.week
    AND (cwc.team = gc.home_team OR cwc.team = gc.away_team)

-- Join player features from PREVIOUS week (temporal alignment)
LEFT JOIN player_features pf
    ON cwc.player_id = pf.player_id
    AND cwc.season = pf.season
    AND cwc.week = pf.target_week

-- Join team features from PREVIOUS week
LEFT JOIN team_features tf
    ON cwc.team = tf.team
    AND cwc.season = tf.season
    AND cwc.week = tf.target_week

-- Join opponent defensive features from PREVIOUS week
LEFT JOIN opponent_team_features otf
    ON cwc.opponent_team = otf.opponent
    AND cwc.season = otf.season
    AND cwc.week = otf.target_week

-- Join roster context
LEFT JOIN roster_context rc
    ON cwc.team = rc.team
    AND cwc.season = rc.season
    AND cwc.week = rc.week

-- Join player experience
LEFT JOIN player_experience pe
    ON cwc.player_id = pe.player_id
    AND cwc.season = pe.season

WHERE
    -- Ensure we have feature data (must have features from previous week)
    pf.player_id IS NOT NULL
    -- Exclude week 1 (no previous week features)
    AND cwc.week > 1

ORDER BY cwc.season, cwc.week, cwc.player_id;
```

### Feature Vector Assembly

#### Numerical Features Array Construction

The `numerical_features` column should be a FLOAT[] array containing all numerical features in a fixed order:

```sql
-- Extract numerical features in fixed order
ARRAY[
    -- Player trend metrics (3)
    COALESCE(pf.performance_trend, 0.0),
    COALESCE(pf.usage_trend, 0.0),
    COALESCE(pf.target_share_trend, 0.0),

    -- Matchup metrics (2)
    COALESCE(CAST(pf.opp_rank_vs_position AS FLOAT), 0.0),
    COALESCE(CAST(pf.rest_days AS FLOAT), 0.0),

    -- Team offensive metrics (8)
    COALESCE(tf.off_epa_per_play_last3, 0.0),
    COALESCE(tf.off_success_rate_last3, 0.0),
    COALESCE(tf.off_explosive_play_rate, 0.0),
    COALESCE(tf.off_red_zone_efficiency, 0.0),
    COALESCE(tf.off_third_down_conv, 0.0),
    COALESCE(tf.pass_rate_neutral, 0.0),
    COALESCE(tf.pace_of_play, 0.0),
    COALESCE(tf.time_of_possession_avg, 0.0),

    -- Opponent defensive metrics (4)
    COALESCE(otf.opp_def_epa_per_play_last3, 0.0),
    COALESCE(otf.opp_def_success_rate_last3, 0.0),
    COALESCE(otf.opp_def_pressure_rate, 0.0),
    COALESCE(otf.opp_def_turnover_rate, 0.0),

    -- Position-specific rolling stats from JSON (extracted dynamically)
    -- For QB: last 3 games avg passing_yards, passing_tds, completions, attempts, etc.
    -- For RB: last 3 games avg rushing_yards, rushing_tds, carries, receptions, etc.
    -- For WR: last 3 games avg receiving_yards, receiving_tds, targets, receptions, etc.
    -- This section is position-specific and extracted from stats_last3_games JSON

    -- Example for QB (add ~10-15 position-specific stats)
    COALESCE(CAST(json_extract(pf.stats_last3_games, '$.avg_passing_yards') AS FLOAT), 0.0),
    COALESCE(CAST(json_extract(pf.stats_last3_games, '$.avg_passing_tds') AS FLOAT), 0.0),
    COALESCE(CAST(json_extract(pf.stats_last3_games, '$.avg_completions') AS FLOAT), 0.0),
    COALESCE(CAST(json_extract(pf.stats_last3_games, '$.avg_attempts') AS FLOAT), 0.0),
    COALESCE(CAST(json_extract(pf.stats_last3_games, '$.avg_passing_epa') AS FLOAT), 0.0),
    -- ... (continue for all position-specific stats from config.py)

    -- Same for last 5 games
    COALESCE(CAST(json_extract(pf.stats_last5_games, '$.avg_passing_yards') AS FLOAT), 0.0),
    -- ...

    -- Same for season average
    COALESCE(CAST(json_extract(pf.stats_season_avg, '$.avg_passing_yards') AS FLOAT), 0.0)
    -- ...

] as numerical_features
```

#### Feature Names Array

Parallel array to document feature names:

```sql
ARRAY[
    'performance_trend',
    'usage_trend',
    'target_share_trend',
    'opp_rank_vs_position',
    'rest_days',
    'off_epa_per_play_last3',
    'off_success_rate_last3',
    'off_explosive_play_rate',
    'off_red_zone_efficiency',
    'off_third_down_conv',
    'pass_rate_neutral',
    'pace_of_play',
    'time_of_possession_avg',
    'opp_def_epa_per_play_last3',
    'opp_def_success_rate_last3',
    'opp_def_pressure_rate',
    'opp_def_turnover_rate',
    'last3_avg_passing_yards',
    'last3_avg_passing_tds',
    -- ... (all position-specific feature names)
] as feature_names
```

### Categorical Feature Encoding

Store categorical features as JSON with both raw values and encoded integers:

```sql
{
    "position": {
        "value": "QB",
        "encoded": 1  -- QB=1, RB=2, WR=3, TE=4, K=5, DEF=6
    },
    "team": {
        "value": "KC",
        "encoded": 12  -- Team ID mapping (alphabetical or by division)
    },
    "opponent": {
        "value": "LV",
        "encoded": 18
    },
    "home_away": {
        "value": "home",
        "encoded": 1  -- home=1, away=0
    },
    "divisional_game": {
        "value": true,
        "encoded": 1  -- yes=1, no=0
    },
    "experience_level": {
        "value": "veteran",
        "encoded": 3  -- rookie=1, developing=2, veteran=3
    }
} as categorical_features
```

### Position-Specific Feature Extraction

Since different positions have different relevant stats (from `config.py`), use CASE statements:

```sql
-- Example: Extract position-specific features
CASE pf.position
    WHEN 'QB' THEN
        ARRAY[
            COALESCE(CAST(json_extract(pf.stats_last3_games, '$.avg_passing_yards') AS FLOAT), 0.0),
            COALESCE(CAST(json_extract(pf.stats_last3_games, '$.avg_passing_tds') AS FLOAT), 0.0),
            COALESCE(CAST(json_extract(pf.stats_last3_games, '$.avg_completions') AS FLOAT), 0.0),
            -- ... all QB-specific stats from config.position_stat_mappings['QB']
        ]
    WHEN 'RB' THEN
        ARRAY[
            COALESCE(CAST(json_extract(pf.stats_last3_games, '$.avg_rushing_yards') AS FLOAT), 0.0),
            COALESCE(CAST(json_extract(pf.stats_last3_games, '$.avg_rushing_tds') AS FLOAT), 0.0),
            COALESCE(CAST(json_extract(pf.stats_last3_games, '$.avg_carries') AS FLOAT), 0.0),
            -- ... all RB-specific stats
        ]
    WHEN 'WR' THEN
        ARRAY[
            COALESCE(CAST(json_extract(pf.stats_last3_games, '$.avg_receiving_yards') AS FLOAT), 0.0),
            COALESCE(CAST(json_extract(pf.stats_last3_games, '$.avg_receiving_tds') AS FLOAT), 0.0),
            COALESCE(CAST(json_extract(pf.stats_last3_games, '$.avg_targets') AS FLOAT), 0.0),
            -- ... all WR-specific stats
        ]
    -- ... other positions
    ELSE ARRAY[]::FLOAT[]
END as position_specific_features
```

### Implementation Notes

1. **Temporal Alignment**: Critical that features from week N-1 are used to predict week N
2. **NULL Handling**: Use COALESCE to replace NULLs with 0.0 or appropriate defaults
3. **Position Filtering**: Consider filtering to key positions (QB, RB, WR, TE, K) for initial implementation
4. **Batch Processing**: Use batch processor for large inserts (1000 rows per batch)
5. **Indexing**: Create indexes on (season, week, player_id) for performance

---

## 2. Technical Specification: `apply_data_quality_scoring()`

### Purpose
Calculate a comprehensive data quality score for each feature vector to identify incomplete or unreliable data.

### Quality Score Components

#### Component 1: Completeness Score (60% weight)

Percentage of non-NULL features in the numerical_features array:

```sql
WITH feature_completeness AS (
    SELECT
        feature_id,
        numerical_features,
        feature_names,

        -- Count total features
        array_length(numerical_features) as total_features,

        -- Count non-zero features (0.0 typically means NULL was coalesced)
        -- More sophisticated: track which were originally NULL
        array_length(
            list_filter(numerical_features, x -> x != 0.0)
        ) as non_null_features,

        -- Calculate completeness ratio
        CAST(
            array_length(list_filter(numerical_features, x -> x != 0.0))
            AS FLOAT
        ) / NULLIF(array_length(numerical_features), 0) as completeness_ratio

    FROM ml_training_features
)
SELECT
    feature_id,
    completeness_ratio,
    CASE
        WHEN completeness_ratio >= 0.9 THEN 1.0  -- Excellent
        WHEN completeness_ratio >= 0.75 THEN 0.8  -- Good
        WHEN completeness_ratio >= 0.5 THEN 0.6   -- Acceptable
        WHEN completeness_ratio >= 0.25 THEN 0.3  -- Poor
        ELSE 0.0  -- Very poor
    END as completeness_score
FROM feature_completeness;
```

#### Component 2: Outlier Detection (20% weight)

Identify features with extreme z-scores (> 3.0):

```sql
WITH feature_stats AS (
    -- Calculate mean and stddev for each feature position in array
    SELECT
        i as feature_index,
        AVG(numerical_features[i]) as feature_mean,
        STDDEV(numerical_features[i]) as feature_stddev
    FROM ml_training_features,
         LATERAL (SELECT unnest(generate_series(1, array_length(numerical_features))) as i)
    WHERE numerical_features[i] IS NOT NULL
    GROUP BY i
),

z_scores AS (
    SELECT
        mtf.feature_id,
        mtf.numerical_features,

        -- Calculate z-score for each feature
        ARRAY[
            SELECT
                ABS(
                    (mtf.numerical_features[fs.feature_index] - fs.feature_mean)
                    / NULLIF(fs.feature_stddev, 0)
                )
            FROM feature_stats fs
            WHERE fs.feature_index <= array_length(mtf.numerical_features)
            ORDER BY fs.feature_index
        ] as feature_z_scores

    FROM ml_training_features mtf
),

outlier_counts AS (
    SELECT
        feature_id,
        array_length(
            list_filter(feature_z_scores, z -> z > 3.0)
        ) as outlier_count,
        array_length(feature_z_scores) as total_features,

        -- Calculate outlier ratio
        CAST(
            array_length(list_filter(feature_z_scores, z -> z > 3.0))
            AS FLOAT
        ) / NULLIF(array_length(feature_z_scores), 0) as outlier_ratio

    FROM z_scores
)
SELECT
    feature_id,
    outlier_ratio,
    CASE
        WHEN outlier_ratio = 0.0 THEN 1.0  -- No outliers
        WHEN outlier_ratio < 0.05 THEN 0.9  -- Very few outliers
        WHEN outlier_ratio < 0.1 THEN 0.7   -- Some outliers
        WHEN outlier_ratio < 0.2 THEN 0.5   -- Many outliers
        ELSE 0.0  -- Too many outliers
    END as outlier_score
FROM outlier_counts;
```

#### Component 3: Recency Score (20% weight)

Data from recent weeks is more reliable:

```sql
WITH current_state AS (
    SELECT MAX(season) as current_season,
           MAX(week) as current_week
    FROM ml_training_features
)
SELECT
    feature_id,
    season,
    week,

    -- Calculate weeks ago
    ((cs.current_season - season) * 18 + (cs.current_week - week)) as weeks_ago,

    -- Recency score (decay over time)
    CASE
        WHEN ((cs.current_season - season) * 18 + (cs.current_week - week)) <= 4 THEN 1.0
        WHEN ((cs.current_season - season) * 18 + (cs.current_week - week)) <= 8 THEN 0.9
        WHEN ((cs.current_season - season) * 18 + (cs.current_week - week)) <= 17 THEN 0.8
        WHEN ((cs.current_season - season) * 18 + (cs.current_week - week)) <= 35 THEN 0.7
        ELSE 0.5
    END as recency_score

FROM ml_training_features,
     current_state cs;
```

### Critical Features Identification

Certain features are absolutely required:

```sql
WITH critical_checks AS (
    SELECT
        feature_id,

        -- Check critical categorical features
        json_extract(categorical_features, '$.position.value') IS NOT NULL as has_position,
        json_extract(categorical_features, '$.team.value') IS NOT NULL as has_team,
        json_extract(categorical_features, '$.opponent.value') IS NOT NULL as has_opponent,

        -- Check roster context
        roster_snapshot_id IS NOT NULL as has_roster_snapshot,
        player_experience_level IS NOT NULL as has_experience,

        -- Check temporal data
        season IS NOT NULL AND week IS NOT NULL as has_time_context,

        -- Overall critical check
        (json_extract(categorical_features, '$.position.value') IS NOT NULL
         AND json_extract(categorical_features, '$.team.value') IS NOT NULL
         AND json_extract(categorical_features, '$.opponent.value') IS NOT NULL
         AND roster_snapshot_id IS NOT NULL
         AND season IS NOT NULL
         AND week IS NOT NULL) as has_all_critical

    FROM ml_training_features
)
SELECT
    feature_id,
    CASE WHEN has_all_critical THEN 1.0 ELSE 0.0 END as critical_features_score
FROM critical_checks;
```

### Combined Quality Score Formula

```sql
WITH all_scores AS (
    SELECT
        feature_id,
        completeness_score,
        outlier_score,
        recency_score,
        critical_features_score,

        -- Weighted combination
        (completeness_score * 0.6 +
         outlier_score * 0.2 +
         recency_score * 0.2) as base_quality_score

    FROM (/* previous CTEs */)
)
SELECT
    feature_id,

    -- Apply critical features multiplier (0 if missing critical features)
    base_quality_score * critical_features_score as final_quality_score,

    -- Individual component scores for debugging
    completeness_score,
    outlier_score,
    recency_score,
    critical_features_score

FROM all_scores;
```

### Missing Data Flags

Track specific missing feature categories:

```sql
ARRAY[
    SELECT flag FROM (VALUES
        (CASE WHEN completeness_ratio < 0.5 THEN 'low_completeness' ELSE NULL END),
        (CASE WHEN outlier_ratio > 0.1 THEN 'high_outliers' ELSE NULL END),
        (CASE WHEN json_extract(categorical_features, '$.position.value') IS NULL
              THEN 'missing_position' ELSE NULL END),
        (CASE WHEN json_extract(categorical_features, '$.team.value') IS NULL
              THEN 'missing_team' ELSE NULL END),
        (CASE WHEN roster_snapshot_id IS NULL THEN 'missing_roster' ELSE NULL END),
        (CASE WHEN player_experience_level IS NULL THEN 'missing_experience' ELSE NULL END),
        (CASE WHEN weeks_ago > 35 THEN 'stale_data' ELSE NULL END)
    ) AS t(flag)
    WHERE flag IS NOT NULL
] as missing_data_flags
```

### Quality Filtering Threshold

```sql
-- Update quality scores
UPDATE ml_training_features
SET
    data_quality_score = (/* final_quality_score from above */),
    missing_data_flags = (/* missing_data_flags from above */);

-- Filter out low-quality rows (can be soft delete or hard delete)
DELETE FROM ml_training_features
WHERE data_quality_score < 0.5;  -- Configurable threshold

-- Or use a flag for soft delete
ALTER TABLE ml_training_features ADD COLUMN quality_passed BOOLEAN;

UPDATE ml_training_features
SET quality_passed = (data_quality_score >= 0.5);
```

### Implementation Notes

1. **Run after feature combination**: Quality scoring happens after `combine_all_features()`
2. **Log statistics**: Track how many rows fail quality check
3. **Position-specific thresholds**: Consider different thresholds for different positions
4. **Seasonal adjustment**: Rookie data might have lower completeness but still be valuable

---

## 3. Technical Specification: `create_prediction_targets()`

### Purpose
Extract actual outcomes from future weeks to create prediction targets. For a feature vector at week N, extract stats from week N (the "current" week being predicted).

### Target Extraction Strategy

The key insight: Features are from week N-1, targets are from week N.

```sql
WITH feature_rows AS (
    -- Get all feature rows that need targets
    SELECT
        feature_id,
        entity_id as player_id,
        season,
        week,
        json_extract(categorical_features, '$.position.value') as position
    FROM ml_training_features
    WHERE
        quality_passed = TRUE  -- Only add targets to quality rows
        AND actual_outcomes IS NULL  -- Don't re-process
),

actual_stats AS (
    -- Get actual stats from the target week
    SELECT
        rps.player_id,
        rps.season,
        rps.week,
        rps.position,

        -- Passing stats (QB)
        rps.passing_yards,
        rps.passing_tds,
        rps.passing_interceptions,
        rps.completions,
        rps.attempts,
        rps.passing_epa,
        rps.passing_cpoe,

        -- Rushing stats (RB, QB)
        rps.rushing_yards,
        rps.rushing_tds,
        rps.carries,
        rps.rushing_epa,

        -- Receiving stats (WR, TE, RB)
        rps.receiving_yards,
        rps.receiving_tds,
        rps.receptions,
        rps.targets,
        rps.receiving_epa,
        rps.target_share,
        rps.air_yards_share,

        -- Kicking stats (K)
        rps.fg_made,
        rps.fg_att,
        rps.fg_pct,
        rps.pat_made,
        rps.pat_att,

        -- Defensive stats (DEF)
        rps.def_tackles_solo,
        rps.def_tackles_with_assist,
        rps.def_sacks,
        rps.def_interceptions,
        rps.def_fumbles_forced,
        rps.def_tds,

        -- Fantasy points (calculated)
        (
            COALESCE(rps.passing_yards, 0) * 0.04 +
            COALESCE(rps.passing_tds, 0) * 4 +
            COALESCE(rps.passing_interceptions, 0) * -2 +
            COALESCE(rps.rushing_yards, 0) * 0.1 +
            COALESCE(rps.rushing_tds, 0) * 6 +
            COALESCE(rps.receiving_yards, 0) * 0.1 +
            COALESCE(rps.receiving_tds, 0) * 6 +
            COALESCE(rps.receptions, 0) * 1
        ) as fantasy_points_ppr,

        -- Game outcome context
        rps.team,
        rps.opponent_team

    FROM raw_player_stats rps
),

game_outcomes AS (
    -- Get team-level outcomes for the target week
    SELECT
        game_id,
        season,
        week,
        home_team,
        away_team,
        home_score,
        away_score,

        -- Win probability (based on actual outcome)
        CASE
            WHEN home_score > away_score THEN 1.0
            WHEN home_score < away_score THEN 0.0
            ELSE 0.5  -- Tie (rare)
        END as home_win,

        -- Total points
        home_score + away_score as total_points,

        -- Score differential
        home_score - away_score as score_diff,

        overtime

    FROM raw_schedules
    WHERE game_type = 'REG'
)

-- Combine into target JSON
SELECT
    fr.feature_id,

    -- Position-specific targets
    CASE fr.position
        WHEN 'QB' THEN json_object(
            'passing_yards', COALESCE(ast.passing_yards, 0),
            'passing_tds', COALESCE(ast.passing_tds, 0),
            'passing_interceptions', COALESCE(ast.passing_interceptions, 0),
            'completions', COALESCE(ast.completions, 0),
            'attempts', COALESCE(ast.attempts, 0),
            'completion_pct', COALESCE(CAST(ast.completions AS FLOAT) / NULLIF(ast.attempts, 0), 0),
            'passing_epa', COALESCE(ast.passing_epa, 0),
            'rushing_yards', COALESCE(ast.rushing_yards, 0),
            'rushing_tds', COALESCE(ast.rushing_tds, 0),
            'fantasy_points_ppr', COALESCE(ast.fantasy_points_ppr, 0),
            'team_points', CASE
                WHEN ast.team = go.home_team THEN go.home_score
                WHEN ast.team = go.away_team THEN go.away_score
                ELSE NULL
            END,
            'team_won', CASE
                WHEN ast.team = go.home_team THEN go.home_win
                WHEN ast.team = go.away_team THEN 1.0 - go.home_win
                ELSE NULL
            END
        )

        WHEN 'RB' THEN json_object(
            'rushing_yards', COALESCE(ast.rushing_yards, 0),
            'rushing_tds', COALESCE(ast.rushing_tds, 0),
            'carries', COALESCE(ast.carries, 0),
            'yards_per_carry', COALESCE(CAST(ast.rushing_yards AS FLOAT) / NULLIF(ast.carries, 0), 0),
            'receiving_yards', COALESCE(ast.receiving_yards, 0),
            'receiving_tds', COALESCE(ast.receiving_tds, 0),
            'receptions', COALESCE(ast.receptions, 0),
            'targets', COALESCE(ast.targets, 0),
            'fantasy_points_ppr', COALESCE(ast.fantasy_points_ppr, 0),
            'team_points', CASE
                WHEN ast.team = go.home_team THEN go.home_score
                WHEN ast.team = go.away_team THEN go.away_score
                ELSE NULL
            END,
            'team_won', CASE
                WHEN ast.team = go.home_team THEN go.home_win
                WHEN ast.team = go.away_team THEN 1.0 - go.home_win
                ELSE NULL
            END
        )

        WHEN 'WR' THEN json_object(
            'receiving_yards', COALESCE(ast.receiving_yards, 0),
            'receiving_tds', COALESCE(ast.receiving_tds, 0),
            'receptions', COALESCE(ast.receptions, 0),
            'targets', COALESCE(ast.targets, 0),
            'catch_rate', COALESCE(CAST(ast.receptions AS FLOAT) / NULLIF(ast.targets, 0), 0),
            'yards_per_reception', COALESCE(CAST(ast.receiving_yards AS FLOAT) / NULLIF(ast.receptions, 0), 0),
            'target_share', COALESCE(ast.target_share, 0),
            'air_yards_share', COALESCE(ast.air_yards_share, 0),
            'rushing_yards', COALESCE(ast.rushing_yards, 0),
            'fantasy_points_ppr', COALESCE(ast.fantasy_points_ppr, 0),
            'team_points', CASE
                WHEN ast.team = go.home_team THEN go.home_score
                WHEN ast.team = go.away_team THEN go.away_score
                ELSE NULL
            END,
            'team_won', CASE
                WHEN ast.team = go.home_team THEN go.home_win
                WHEN ast.team = go.away_team THEN 1.0 - go.home_win
                ELSE NULL
            END
        )

        WHEN 'TE' THEN json_object(
            'receiving_yards', COALESCE(ast.receiving_yards, 0),
            'receiving_tds', COALESCE(ast.receiving_tds, 0),
            'receptions', COALESCE(ast.receptions, 0),
            'targets', COALESCE(ast.targets, 0),
            'catch_rate', COALESCE(CAST(ast.receptions AS FLOAT) / NULLIF(ast.targets, 0), 0),
            'yards_per_reception', COALESCE(CAST(ast.receiving_yards AS FLOAT) / NULLIF(ast.receptions, 0), 0),
            'target_share', COALESCE(ast.target_share, 0),
            'fantasy_points_ppr', COALESCE(ast.fantasy_points_ppr, 0),
            'team_points', CASE
                WHEN ast.team = go.home_team THEN go.home_score
                WHEN ast.team = go.away_team THEN go.away_score
                ELSE NULL
            END,
            'team_won', CASE
                WHEN ast.team = go.home_team THEN go.home_win
                WHEN ast.team = go.away_team THEN 1.0 - go.home_win
                ELSE NULL
            END
        )

        WHEN 'K' THEN json_object(
            'fg_made', COALESCE(ast.fg_made, 0),
            'fg_att', COALESCE(ast.fg_att, 0),
            'fg_pct', COALESCE(ast.fg_pct, 0),
            'pat_made', COALESCE(ast.pat_made, 0),
            'pat_att', COALESCE(ast.pat_att, 0),
            'fantasy_points_standard', COALESCE(ast.fg_made * 3 + ast.pat_made, 0),
            'team_points', CASE
                WHEN ast.team = go.home_team THEN go.home_score
                WHEN ast.team = go.away_team THEN go.away_score
                ELSE NULL
            END,
            'team_won', CASE
                WHEN ast.team = go.home_team THEN go.home_win
                WHEN ast.team = go.away_team THEN 1.0 - go.home_win
                ELSE NULL
            END
        )

        WHEN 'DEF' THEN json_object(
            'tackles_solo', COALESCE(ast.def_tackles_solo, 0),
            'tackles_total', COALESCE(ast.def_tackles_solo + ast.def_tackles_with_assist, 0),
            'sacks', COALESCE(ast.def_sacks, 0),
            'interceptions', COALESCE(ast.def_interceptions, 0),
            'fumbles_forced', COALESCE(ast.def_fumbles_forced, 0),
            'tds', COALESCE(ast.def_tds, 0),
            'fantasy_points_idp', COALESCE(
                ast.def_tackles_solo * 1 +
                ast.def_sacks * 2 +
                ast.def_interceptions * 3 +
                ast.def_tds * 6,
                0
            )
        )

        ELSE json_object('error', 'unknown_position')
    END as actual_outcomes

FROM feature_rows fr

LEFT JOIN actual_stats ast
    ON fr.player_id = ast.player_id
    AND fr.season = ast.season
    AND fr.week = ast.week

LEFT JOIN game_outcomes go
    ON ast.season = go.season
    AND ast.week = go.week
    AND (ast.team = go.home_team OR ast.team = go.away_team);
```

### Update Feature Table with Targets

```sql
-- Update ml_training_features with actual outcomes
UPDATE ml_training_features mtf
SET
    actual_outcomes = targets.actual_outcomes,
    prediction_target = targets.primary_target
FROM (
    -- Query from above
    SELECT feature_id, actual_outcomes,
           -- Set primary target based on position
           CASE position
               WHEN 'QB' THEN 'passing_yards'
               WHEN 'RB' THEN 'rushing_yards'
               WHEN 'WR' THEN 'receiving_yards'
               WHEN 'TE' THEN 'receiving_yards'
               WHEN 'K' THEN 'fg_made'
               WHEN 'DEF' THEN 'tackles_total'
               ELSE 'fantasy_points_ppr'
           END as primary_target
    FROM (/* CTE query above */)
) targets
WHERE mtf.feature_id = targets.feature_id;
```

### Multiple Prediction Target Support

Based on `config.ml_config["prediction_targets"]`:
- `player_stats`: Position-specific stats (implemented above)
- `team_total_points`: Team's total points in the game
- `win_probability`: Did the team win?
- `quarter_scores`: Points by quarter (future enhancement)

For team-level predictions, create separate rows with `entity_type = 'team'`:

```sql
INSERT INTO ml_training_features (
    feature_id,
    entity_type,
    entity_id,
    prediction_target,
    season,
    week,
    game_date,
    numerical_features,
    feature_names,
    categorical_features,
    actual_outcomes,
    data_quality_score
)
SELECT
    CONCAT('TEAM_', team, '_', season, '_', week) as feature_id,
    'team' as entity_type,
    team as entity_id,
    'team_total_points' as prediction_target,
    season,
    week,
    game_date,

    -- Team-level features (aggregate of team_rolling_features)
    ARRAY[
        off_epa_per_play_last3,
        off_success_rate_last3,
        off_explosive_play_rate,
        def_epa_per_play_last3,
        def_success_rate_last3
        -- ...
    ] as numerical_features,

    ARRAY[
        'off_epa_per_play_last3',
        'off_success_rate_last3',
        -- ...
    ] as feature_names,

    json_object(
        'team', team,
        'opponent', opponent,
        'home_away', home_away
    ) as categorical_features,

    json_object(
        'total_points', total_points,
        'won', won,
        'score_differential', score_diff
    ) as actual_outcomes,

    1.0 as data_quality_score  -- Team data typically complete

FROM (
    -- Join team_rolling_features with game outcomes
    -- ...
);
```

### Target Availability Check

Not all weeks will have target data (e.g., current week hasn't been played yet):

```sql
-- Flag rows without targets
UPDATE ml_training_features
SET missing_data_flags = array_append(
    COALESCE(missing_data_flags, ARRAY[]::VARCHAR[]),
    'no_target_data'
)
WHERE actual_outcomes IS NULL;

-- Optionally exclude from training
UPDATE ml_training_features
SET quality_passed = FALSE
WHERE actual_outcomes IS NULL;
```

### Implementation Notes

1. **Position-specific targets**: Use config.position_stat_mappings to determine relevant targets
2. **Fantasy points**: Include both standard and PPR scoring
3. **Team outcomes**: Always include team win/loss and total points
4. **NULL handling**: Missing stats should be 0, not NULL, in target JSON
5. **Validation**: Ensure target week data exists before creating features

---

## 4. Technical Specification: `validate_temporal_consistency()`

### Purpose
Ensure no data leakage by validating that features only use data from previous weeks, not future weeks.

### Validation Checks

#### Check 1: Feature Week < Target Week

```sql
-- Verify features are from week N-1 when predicting week N
WITH temporal_check AS (
    SELECT
        feature_id,
        entity_id,
        season,
        week as target_week,

        -- Extract the week from which features were calculated
        -- (This assumes feature generation logic stores source week info)
        -- For now, we validate by checking that week > 1 (can't have features for week 1)

        CASE
            WHEN week = 1 THEN 'INVALID: Week 1 has no prior week features'
            WHEN week > 1 THEN 'VALID'
            ELSE 'UNKNOWN'
        END as temporal_validity

    FROM ml_training_features
)
SELECT
    temporal_validity,
    COUNT(*) as row_count
FROM temporal_check
GROUP BY temporal_validity;
```

More sophisticated check using feature metadata:

```sql
-- If we track feature calculation week in a metadata column
ALTER TABLE ml_training_features ADD COLUMN feature_week INTEGER;

-- Check that feature_week < target_week
SELECT
    feature_id,
    feature_week,
    week as target_week,
    feature_week < week as is_valid
FROM ml_training_features
WHERE feature_week >= week;  -- Find invalid rows

-- Count violations
SELECT COUNT(*) as temporal_violations
FROM ml_training_features
WHERE feature_week >= week;
```

#### Check 2: No Future Game Outcomes in Features

Ensure game outcomes (wins, scores) aren't leaked into features:

```sql
-- Check that numerical_features don't contain outcome-related values
-- This is more of a design check - ensure feature engineering doesn't include:
-- - Final scores
-- - Win/loss flags
-- - Post-game stats

-- Audit feature names for suspicious terms
SELECT DISTINCT
    feature_id,
    unnest(feature_names) as feature_name
FROM ml_training_features
WHERE
    unnest(feature_names) LIKE '%win%'
    OR unnest(feature_names) LIKE '%loss%'
    OR unnest(feature_names) LIKE '%score%'
    OR unnest(feature_names) LIKE '%final%'
    OR unnest(feature_names) LIKE '%result%';

-- If found, these need investigation
```

#### Check 3: Train/Val/Test Split Integrity

Ensure splits are temporally ordered with no overlap:

```sql
-- Add split column
ALTER TABLE ml_training_features ADD COLUMN split VARCHAR(10);

WITH season_week_ranking AS (
    SELECT DISTINCT
        season,
        week,
        -- Rank chronologically
        ROW_NUMBER() OVER (ORDER BY season, week) as time_rank,
        COUNT(*) OVER () as total_weeks
    FROM ml_training_features
),

split_assignments AS (
    SELECT
        season,
        week,
        time_rank,
        total_weeks,

        -- Assign splits temporally
        CASE
            -- Last 10% is test set (most recent data)
            WHEN time_rank > total_weeks * 0.9 THEN 'test'

            -- Previous 20% is validation set
            WHEN time_rank > total_weeks * 0.7 THEN 'validation'

            -- First 70% is training set
            ELSE 'train'
        END as split

    FROM season_week_ranking
)

-- Update ml_training_features
UPDATE ml_training_features mtf
SET split = sa.split
FROM split_assignments sa
WHERE mtf.season = sa.season AND mtf.week = sa.week;
```

Validate no overlap:

```sql
-- Check that test set is strictly after validation set
WITH split_ranges AS (
    SELECT
        split,
        MIN(season * 100 + week) as earliest_week,
        MAX(season * 100 + week) as latest_week
    FROM ml_training_features
    GROUP BY split
)
SELECT
    train.split as set1,
    test.split as set2,
    train.latest_week < test.earliest_week as properly_ordered
FROM split_ranges train
CROSS JOIN split_ranges test
WHERE train.split = 'train' AND test.split IN ('validation', 'test')

UNION ALL

SELECT
    val.split as set1,
    test.split as set2,
    val.latest_week < test.earliest_week as properly_ordered
FROM split_ranges val
CROSS JOIN split_ranges test
WHERE val.split = 'validation' AND test.split = 'test';

-- All properly_ordered should be TRUE
```

#### Check 4: No Data from Same Week in Features and Targets

```sql
-- Verify that when predicting week N, no week N data is in features
-- This check requires tracking which weeks contributed to each feature

-- Example: Check that last3_games doesn't include target week
WITH feature_metadata AS (
    SELECT
        feature_id,
        week as target_week,

        -- Extract max week from stats_last3_games (if we store it)
        -- This is hypothetical - would need to be tracked during feature engineering
        CAST(json_extract(
            json_extract(categorical_features, '$.feature_metadata'),
            '$.max_feature_week'
        ) AS INTEGER) as max_feature_week

    FROM ml_training_features
)
SELECT
    COUNT(*) as leakage_violations
FROM feature_metadata
WHERE max_feature_week >= target_week;

-- Should be 0
```

### Validation Report Generation

```sql
CREATE TABLE IF NOT EXISTS temporal_validation_log (
    validation_id VARCHAR PRIMARY KEY,
    validation_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    check_name VARCHAR,
    check_passed BOOLEAN,
    violation_count INTEGER,
    details JSON
);

-- Insert validation results
INSERT INTO temporal_validation_log VALUES
(
    uuid(),
    CURRENT_TIMESTAMP,
    'week_1_features',
    (SELECT COUNT(*) = 0 FROM ml_training_features WHERE week = 1),
    (SELECT COUNT(*) FROM ml_training_features WHERE week = 1),
    json_object('description', 'Verify no week 1 features exist')
),
(
    uuid(),
    CURRENT_TIMESTAMP,
    'future_outcome_leakage',
    (SELECT COUNT(*) = 0 FROM /* future outcome check */),
    (SELECT COUNT(*) FROM /* violations */),
    json_object('description', 'Check for outcome variables in features')
),
(
    uuid(),
    CURRENT_TIMESTAMP,
    'split_temporal_order',
    (SELECT BOOL_AND(properly_ordered) FROM /* split order check */),
    (SELECT COUNT(*) FROM /* violations */),
    json_object('description', 'Validate train/val/test temporal ordering')
);
```

### Logging and Warnings

```python
# Python implementation logging
logger.info("⏰ Validating temporal consistency...")

# Run validation queries
violations = db.execute("""
    SELECT check_name, check_passed, violation_count, details
    FROM temporal_validation_log
    WHERE validation_timestamp > NOW() - INTERVAL '1 minute'
    ORDER BY validation_timestamp DESC
""")

all_passed = True
for check in violations:
    if not check['check_passed']:
        logger.warning(f"❌ TEMPORAL VIOLATION: {check['check_name']}")
        logger.warning(f"   Violations: {check['violation_count']}")
        logger.warning(f"   Details: {check['details']}")
        all_passed = False
    else:
        logger.info(f"✅ {check['check_name']}: PASSED")

if all_passed:
    logger.info("✅ All temporal consistency checks passed")
else:
    logger.error("❌ TEMPORAL CONSISTENCY VALIDATION FAILED")
    raise ValueError("Data leakage detected - fix temporal issues before proceeding")
```

### Implementation Notes

1. **Fail fast**: If any check fails, halt the pipeline
2. **Detailed logging**: Log specific violations with feature_ids
3. **Track metadata**: Store feature calculation week for auditing
4. **Automate checks**: Run on every pipeline execution
5. **Manual review**: Periodically audit a sample of features manually

---

## 5. Data Quality Checks

### Uniqueness Check

```sql
-- Ensure no duplicates on (player_id, season, week)
SELECT
    entity_id,
    season,
    week,
    COUNT(*) as duplicate_count
FROM ml_training_features
WHERE entity_type = 'player'
GROUP BY entity_id, season, week
HAVING COUNT(*) > 1;

-- Should return 0 rows
```

### Feature Alignment Validation

```sql
-- Verify numerical_features and feature_names arrays have same length
SELECT
    feature_id,
    array_length(numerical_features) as num_feature_count,
    array_length(feature_names) as name_count,
    array_length(numerical_features) = array_length(feature_names) as arrays_aligned
FROM ml_training_features
WHERE array_length(numerical_features) != array_length(feature_names);

-- Should return 0 rows
```

### Target Availability Check

```sql
-- Check percentage of rows with valid targets
SELECT
    split,
    COUNT(*) as total_rows,
    COUNT(actual_outcomes) as rows_with_targets,
    ROUND(100.0 * COUNT(actual_outcomes) / COUNT(*), 2) as target_coverage_pct
FROM ml_training_features
GROUP BY split;

-- Expect 100% for train/val, <100% for test (future weeks may not be played)
```

### Position Distribution Check

```sql
-- Ensure balanced representation of positions
SELECT
    json_extract(categorical_features, '$.position.value') as position,
    split,
    COUNT(*) as count,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (PARTITION BY split), 2) as percentage
FROM ml_training_features
GROUP BY position, split
ORDER BY split, count DESC;
```

### Quality Score Distribution

```sql
-- Check distribution of quality scores
SELECT
    split,
    MIN(data_quality_score) as min_score,
    AVG(data_quality_score) as avg_score,
    MAX(data_quality_score) as max_score,
    PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY data_quality_score) as q1,
    PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY data_quality_score) as median,
    PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY data_quality_score) as q3
FROM ml_training_features
WHERE quality_passed = TRUE
GROUP BY split;
```

---

## 6. Test Cases

### Test 1: Feature Join Completeness

```python
def test_feature_join():
    """Test that features from player + team + matchup are properly joined"""

    # Setup: Create sample data
    db.execute("""
        INSERT INTO player_rolling_features VALUES
        ('P1', 2024, 5, 'QB', '{"avg_passing_yards": 280}', '{"avg_passing_yards": 290}',
         '{"avg_passing_yards": 275}', 0.85, 0.90, 0.35, '{}', 3, '{}', '{}', false, 4);
    """)

    db.execute("""
        INSERT INTO team_rolling_features VALUES
        ('KC', 2024, 5, 0.15, 0.55, 0.12, 0.65, 0.45, -0.10, 0.48, 0.32, 0.15, 0.58, 65.0, 32.5);
    """)

    db.execute("""
        INSERT INTO raw_player_stats VALUES
        ('P1', 'Player 1', 'Player One', 'QB', 'offense', '', 2024, 6, 'REG', 'KC', 'LV', ...);
    """)

    # Execute combine_all_features()
    pipeline.combine_all_features()

    # Verify
    result = db.execute_one("""
        SELECT
            feature_id,
            array_length(numerical_features) as feature_count,
            json_extract(categorical_features, '$.position.value') as position
        FROM ml_training_features
        WHERE entity_id = 'P1' AND season = 2024 AND week = 6
    """)

    assert result is not None, "Feature row should exist"
    assert result[1] > 20, "Should have at least 20 numerical features"
    assert result[2] == 'QB', "Position should be QB"
```

### Test 2: Quality Scoring

```python
def test_quality_scoring():
    """Test that rows with 50% missing features score ~0.5"""

    # Create feature row with 50% zeros (missing)
    features_half_missing = [0.0] * 10 + [1.5, 2.3, 3.1, 4.2, 5.0] * 2  # 10 zeros, 10 values

    db.execute("""
        INSERT INTO ml_training_features
        (feature_id, entity_type, entity_id, season, week, numerical_features, feature_names)
        VALUES
        ('TEST1', 'player', 'P1', 2024, 6, ?,
         ARRAY['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10',
               'f11', 'f12', 'f13', 'f14', 'f15', 'f16', 'f17', 'f18', 'f19', 'f20'])
    """, [features_half_missing])

    # Apply quality scoring
    pipeline.apply_data_quality_scoring()

    # Check score
    score = db.execute_one("""
        SELECT data_quality_score
        FROM ml_training_features
        WHERE feature_id = 'TEST1'
    """)[0]

    assert 0.3 <= score <= 0.7, f"Score should be ~0.5 for 50% missing, got {score}"
```

### Test 3: Target Creation

```python
def test_target_creation():
    """Test that week 5 features get week 6 targets"""

    # Setup: Features for week 6 (from week 5 data)
    db.execute("""
        INSERT INTO ml_training_features
        (feature_id, entity_type, entity_id, season, week, categorical_features)
        VALUES
        ('TEST1', 'player', 'P1', 2024, 6, '{"position": {"value": "QB"}}')
    """)

    # Actual stats from week 6
    db.execute("""
        INSERT INTO raw_player_stats
        (player_id, season, week, passing_yards, passing_tds, position)
        VALUES
        ('P1', 2024, 6, 325, 3, 'QB')
    """)

    # Create targets
    pipeline.create_prediction_targets()

    # Verify target
    target = db.execute_one("""
        SELECT json_extract(actual_outcomes, '$.passing_yards') as yards
        FROM ml_training_features
        WHERE feature_id = 'TEST1'
    """)[0]

    assert target == 325, f"Target should be 325 yards, got {target}"
```

### Test 4: Temporal Validation

```python
def test_temporal_validation():
    """Test that week 10 features don't use week 11 data"""

    # This test validates the feature engineering process
    # Ensure features for week 10 only use data from weeks 1-9

    # Setup: Create features for week 10
    db.execute("""
        INSERT INTO ml_training_features
        (feature_id, entity_type, entity_id, season, week, feature_week)
        VALUES
        ('TEST1', 'player', 'P1', 2024, 10, 9)  -- Features from week 9
    """)

    # Validate
    is_valid = pipeline.validate_temporal_consistency()

    assert is_valid == True, "Temporal validation should pass"

    # Test failure case
    db.execute("""
        INSERT INTO ml_training_features
        (feature_id, entity_type, entity_id, season, week, feature_week)
        VALUES
        ('TEST2', 'player', 'P2', 2024, 10, 10)  -- INVALID: uses same week
    """)

    with pytest.raises(ValueError):
        pipeline.validate_temporal_consistency()  # Should raise error
```

### Test 5: Quality Filtering

```python
def test_quality_filtering():
    """Test that rows below quality threshold are removed"""

    # Create low quality row
    db.execute("""
        INSERT INTO ml_training_features
        (feature_id, entity_type, entity_id, season, week, data_quality_score)
        VALUES
        ('LOW_QUALITY', 'player', 'P1', 2024, 6, 0.3)  -- Below 0.5 threshold
    """)

    # Create high quality row
    db.execute("""
        INSERT INTO ml_training_features
        (feature_id, entity_type, entity_id, season, week, data_quality_score)
        VALUES
        ('HIGH_QUALITY', 'player', 'P2', 2024, 6, 0.85)
    """)

    # Apply filtering
    pipeline.apply_data_quality_scoring()  # Includes filtering

    # Check results
    low_exists = db.execute_one("""
        SELECT COUNT(*) FROM ml_training_features WHERE feature_id = 'LOW_QUALITY'
    """)[0]

    high_exists = db.execute_one("""
        SELECT COUNT(*) FROM ml_training_features WHERE feature_id = 'HIGH_QUALITY'
    """)[0]

    assert low_exists == 0, "Low quality row should be filtered out"
    assert high_exists == 1, "High quality row should remain"
```

---

## 7. Output Schema

### Exact Format for `numerical_features[]` Array

Fixed-order float array with all numerical features:

```
Position: [0-16] - Universal features (17 features)
  [0] performance_trend
  [1] usage_trend
  [2] target_share_trend
  [3] opp_rank_vs_position
  [4] rest_days
  [5] off_epa_per_play_last3
  [6] off_success_rate_last3
  [7] off_explosive_play_rate
  [8] off_red_zone_efficiency
  [9] off_third_down_conv
  [10] pass_rate_neutral
  [11] pace_of_play
  [12] time_of_possession_avg
  [13] opp_def_epa_per_play_last3
  [14] opp_def_success_rate_last3
  [15] opp_def_pressure_rate
  [16] opp_def_turnover_rate

Position: [17-46] - Last 3 games position-specific stats (30 features)
  For QB: passing_yards, passing_tds, completions, attempts, passing_epa, ...
  For RB: rushing_yards, rushing_tds, carries, receptions, targets, ...
  For WR: receiving_yards, receiving_tds, receptions, targets, air_yards, ...
  (Extracted from stats_last3_games JSON based on position)

Position: [47-76] - Last 5 games position-specific stats (30 features)
  Same structure as last 3 games

Position: [77-106] - Season average position-specific stats (30 features)
  Same structure as last 3 games

Total: ~107 features (varies slightly by position)
```

### Exact Format for `categorical_features` JSON

```json
{
  "position": {
    "value": "QB",
    "encoded": 1,
    "encoding_map": {"QB": 1, "RB": 2, "WR": 3, "TE": 4, "K": 5, "DEF": 6}
  },
  "team": {
    "value": "KC",
    "encoded": 12,
    "encoding_type": "alphabetical"
  },
  "opponent": {
    "value": "LV",
    "encoded": 18,
    "encoding_type": "alphabetical"
  },
  "home_away": {
    "value": "home",
    "encoded": 1,
    "encoding_map": {"home": 1, "away": 0}
  },
  "divisional_game": {
    "value": true,
    "encoded": 1,
    "encoding_map": {"true": 1, "false": 0}
  },
  "experience_level": {
    "value": "veteran",
    "encoded": 3,
    "encoding_map": {"rookie": 1, "developing": 2, "veteran": 3}
  },
  "season": {
    "value": 2024,
    "encoded": 4,
    "encoding_map": {"2021": 1, "2022": 2, "2023": 3, "2024": 4, "2025": 5}
  }
}
```

### JSON Structure for `actual_outcomes`

Position-specific outcome structure:

**For QB:**
```json
{
  "passing_yards": 325,
  "passing_tds": 3,
  "passing_interceptions": 1,
  "completions": 25,
  "attempts": 38,
  "completion_pct": 0.658,
  "passing_epa": 12.5,
  "rushing_yards": 22,
  "rushing_tds": 0,
  "fantasy_points_ppr": 28.6,
  "team_points": 31,
  "team_won": 1.0
}
```

**For RB:**
```json
{
  "rushing_yards": 112,
  "rushing_tds": 1,
  "carries": 22,
  "yards_per_carry": 5.09,
  "receiving_yards": 35,
  "receiving_tds": 0,
  "receptions": 4,
  "targets": 5,
  "fantasy_points_ppr": 22.7,
  "team_points": 28,
  "team_won": 1.0
}
```

**For WR:**
```json
{
  "receiving_yards": 87,
  "receiving_tds": 1,
  "receptions": 6,
  "targets": 9,
  "catch_rate": 0.667,
  "yards_per_reception": 14.5,
  "target_share": 0.24,
  "air_yards_share": 0.28,
  "rushing_yards": 0,
  "fantasy_points_ppr": 20.7,
  "team_points": 28,
  "team_won": 1.0
}
```

**For TE:**
```json
{
  "receiving_yards": 68,
  "receiving_tds": 1,
  "receptions": 5,
  "targets": 7,
  "catch_rate": 0.714,
  "yards_per_reception": 13.6,
  "target_share": 0.18,
  "fantasy_points_ppr": 17.8,
  "team_points": 28,
  "team_won": 1.0
}
```

**For K:**
```json
{
  "fg_made": 3,
  "fg_att": 3,
  "fg_pct": 100.0,
  "pat_made": 4,
  "pat_att": 4,
  "fantasy_points_standard": 13,
  "team_points": 31,
  "team_won": 1.0
}
```

**For DEF:**
```json
{
  "tackles_solo": 4,
  "tackles_total": 8,
  "sacks": 1.0,
  "interceptions": 0,
  "fumbles_forced": 1,
  "tds": 0,
  "fantasy_points_idp": 14
}
```

### Complete Row Example

```sql
SELECT * FROM ml_training_features WHERE feature_id = 'P12345_2024_6';

{
  "feature_id": "P12345_2024_6",
  "entity_type": "player",
  "entity_id": "P12345",
  "prediction_target": "passing_yards",

  "season": 2024,
  "week": 6,
  "game_date": "2024-10-13",

  "roster_snapshot_id": "KC_2024_6_20241010",
  "player_experience_level": "veteran",

  "numerical_features": [0.85, 0.90, 0.35, 3.0, 4.0, 0.15, 0.55, ...],  -- 107 floats
  "feature_names": ["performance_trend", "usage_trend", ...],  -- 107 names

  "categorical_features": {
    "position": {"value": "QB", "encoded": 1},
    "team": {"value": "KC", "encoded": 12},
    "opponent": {"value": "LV", "encoded": 18},
    "home_away": {"value": "home", "encoded": 1},
    "divisional_game": {"value": true, "encoded": 1},
    "experience_level": {"value": "veteran", "encoded": 3}
  },

  "actual_outcomes": {
    "passing_yards": 325,
    "passing_tds": 3,
    "passing_interceptions": 1,
    "completions": 25,
    "attempts": 38,
    "completion_pct": 0.658,
    "passing_epa": 12.5,
    "rushing_yards": 22,
    "rushing_tds": 0,
    "fantasy_points_ppr": 28.6,
    "team_points": 31,
    "team_won": 1.0
  },

  "data_quality_score": 0.92,
  "missing_data_flags": [],

  "quality_passed": true,
  "split": "train",

  "created_at": "2024-11-06 14:23:45"
}
```

---

## Implementation Execution Order

1. **Run `combine_all_features()`**
   - Join player_rolling_features + team_rolling_features + roster_snapshots
   - Assemble numerical_features array (position-specific)
   - Create categorical_features JSON
   - Insert into ml_training_features table
   - Log: "✅ Combined features for X players across Y weeks"

2. **Run `apply_data_quality_scoring()`**
   - Calculate completeness score
   - Detect outliers (z-score > 3)
   - Calculate recency score
   - Check critical features
   - Compute weighted quality score
   - Update data_quality_score and missing_data_flags
   - Filter rows with score < 0.5
   - Log: "✅ Quality scoring complete: X rows passed, Y rows filtered"

3. **Run `create_prediction_targets()`**
   - Extract actual stats from raw_player_stats (target week)
   - Extract game outcomes from raw_schedules
   - Create position-specific target JSON
   - Update actual_outcomes column
   - Set prediction_target column
   - Flag rows without targets
   - Log: "✅ Created targets for X rows, Y rows missing targets"

4. **Run `validate_temporal_consistency()`**
   - Check week 1 features (should be 0)
   - Verify feature_week < target_week
   - Check train/val/test temporal ordering
   - Audit for future outcome leakage
   - Insert validation results into temporal_validation_log
   - Raise error if any check fails
   - Log: "✅ Temporal validation passed" or "❌ TEMPORAL VIOLATIONS DETECTED"

5. **Final validation and logging**
   - Run data quality checks (uniqueness, alignment, etc.)
   - Generate summary statistics
   - Log dataset size and split distribution
   - Save ml_training_features table

---

## Summary

This implementation plan provides:

✅ **Complete SQL join strategies** for combining features
✅ **Quality scoring formula** with weighted components
✅ **Position-specific target creation** aligned with config.py
✅ **Comprehensive temporal validation** to prevent data leakage
✅ **Exact output schema** for ML-ready features
✅ **Test cases** for each component
✅ **Execution order** for the full pipeline

**Next Steps:**
1. Review this plan for technical accuracy
2. Implement each method sequentially in `src/data_pipeline.py`
3. Run test suite after each method
4. Validate on 2024-2025 data
5. Document any deviations or learnings

**Estimated LOC:** ~800-1000 lines of SQL + Python (across 4 methods)
**Complexity:** High (requires careful temporal logic and position-specific handling)
**Risk Areas:** JSON extraction from rolling features, temporal validation, outlier detection
