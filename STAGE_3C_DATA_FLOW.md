# Stage 3c: Team Aggregates - Data Flow Architecture

## High-Level Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                     create_team_aggregates()                     │
│                     Entry Point (line 465)                       │
└────────────────────────────┬────────────────────────────────────┘
                             │
                ┌────────────┴────────────┐
                │                         │
                ▼                         ▼
    ┌────────────────────┐    ┌────────────────────┐
    │  Check PBP Data    │    │ Check All Seasons  │
    │  Availability      │    │   in Database      │
    │  (2022+ seasons)   │    │   (2021-2025)      │
    └─────────┬──────────┘    └──────────┬─────────┘
              │                          │
              │                          │
              ▼                          ▼
    ┌──────────────────────────────────────────────┐
    │         Route by Season Availability          │
    └──────┬───────────────────────────────┬────────┘
           │                               │
           ▼                               ▼
┌──────────────────────┐        ┌──────────────────────┐
│   Seasons 2022+      │        │   Seasons pre-2022   │
│   (PBP Available)    │        │   (Team Stats Only)  │
└──────────┬───────────┘        └──────────┬───────────┘
           │                               │
           ▼                               ▼
┌──────────────────────┐        ┌──────────────────────┐
│  Process from PBP    │        │ Process from         │
│  (Detailed Metrics)  │        │ Team Stats           │
└──────────┬───────────┘        │ (Limited Metrics)    │
           │                    └──────────┬───────────┘
           │                               │
           └───────────┬───────────────────┘
                       │
                       ▼
           ┌─────────────────────┐
           │  Validate Results   │
           │  - EPA ranges       │
           │  - Success rates    │
           │  - NULL checks      │
           └─────────┬───────────┘
                     │
                     ▼
           ┌─────────────────────┐
           │ team_rolling_       │
           │ features Table      │
           │ (2,080 records)     │
           └─────────────────────┘
```

## Detailed PBP Processing Flow (2022+ Seasons)

```
┌──────────────────────────────────────────────────────────────┐
│         _calculate_team_aggregates_from_pbp(season)          │
└────────────────────────────┬─────────────────────────────────┘
                             │
                ┌────────────┴───────────────┐
                │  Get Teams × Weeks from    │
                │  raw_pbp for this season   │
                │  (32 teams × 18 weeks)     │
                └────────────┬───────────────┘
                             │
                ┌────────────▼────────────┐
                │  For each (team, week): │
                │  Skip if week <= 3      │
                └────────────┬────────────┘
                             │
          ┌──────────────────┼──────────────────┐
          │                  │                  │
          ▼                  ▼                  ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│   Calculate     │ │   Calculate     │ │   Calculate     │
│   Offensive     │ │   Defensive     │ │  Situational    │
│   Metrics       │ │   Metrics       │ │   Metrics       │
└────────┬────────┘ └────────┬────────┘ └────────┬────────┘
         │                   │                   │
         └───────────────────┼───────────────────┘
                             │
                    ┌────────▼─────────┐
                    │  Combine into    │
                    │  Single Record   │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │  Collect all     │
                    │  records into    │
                    │  DataFrame       │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │  Bulk Insert to  │
                    │  team_rolling_   │
                    │  features        │
                    └──────────────────┘
```

## Offensive Metrics Calculation Detail

```
┌─────────────────────────────────────────────────────────────┐
│          _calculate_offensive_metrics(team, week)            │
└───────────────────────────┬─────────────────────────────────┘
                            │
                ┌───────────▼──────────┐
                │  Filter raw_pbp:     │
                │  - posteam = team    │
                │  - week in [w-3,w-1] │
                │  - play_type: pass/run│
                └───────────┬──────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
┌───────────────┐  ┌───────────────┐  ┌───────────────┐
│  EPA          │  │  Success      │  │  Explosive    │
│  AVG(epa)     │  │  Rate         │  │  Plays        │
│               │  │  SUM(success)/│  │  COUNT(yds>10)│
│  Range:       │  │  COUNT(*)     │  │  /COUNT(*)    │
│  -0.5 to 0.5  │  │               │  │               │
│               │  │  Range:       │  │  Range:       │
│  Elite: >0.2  │  │  0.35 to 0.55 │  │  0.10 to 0.25 │
└───────────────┘  └───────────────┘  └───────────────┘

        ┌───────────────────┬───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
┌───────────────┐  ┌───────────────┐  ┌───────────────┐
│  Red Zone     │  │  Third Down   │  │  Return       │
│  Efficiency   │  │  Conversion   │  │  Combined     │
│               │  │               │  │  Dict         │
│  GROUP drives │  │  SUM(conv)/   │  │               │
│  WHERE yrd<=20│  │  (conv+fail)  │  │  All 5        │
│  SUM(TD)/     │  │               │  │  metrics      │
│  COUNT(drive) │  │  Range:       │  │               │
│               │  │  0.30 to 0.50 │  │               │
│  Range:       │  │               │  │               │
│  0.40 to 0.70 │  │               │  │               │
└───────────────┘  └───────────────┘  └───────────────┘
```

## Defensive Metrics Calculation Detail

```
┌─────────────────────────────────────────────────────────────┐
│          _calculate_defensive_metrics(team, week)            │
└───────────────────────────┬─────────────────────────────────┘
                            │
                ┌───────────▼──────────┐
                │  Filter raw_pbp:     │
                │  - defteam = team    │
                │  - week in [w-3,w-1] │
                │  - play_type: pass/run│
                └───────────┬──────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
┌───────────────┐  ┌───────────────┐  ┌───────────────┐
│  Def EPA      │  │  Def Success  │  │  Pressure     │
│  Allowed      │  │  Rate         │  │  Rate         │
│               │  │               │  │               │
│  AVG(epa)     │  │  SUM(success  │  │  (qb_hit +    │
│  (opponent)   │  │  = 0)/COUNT   │  │  sack) /      │
│               │  │               │  │  pass_attempt │
│  Lower is     │  │  Higher is    │  │               │
│  better!      │  │  better!      │  │  Range:       │
│               │  │               │  │  0.15 to 0.35 │
│  Range:       │  │  Range:       │  │               │
│  -0.5 to 0.5  │  │  0.45 to 0.65 │  │  Elite: >0.28 │
│               │  │               │  │               │
│  Elite: <-0.1 │  │  Elite: >0.55 │  │               │
└───────────────┘  └───────────────┘  └───────────────┘

        ┌───────────────────┐
        │                   │
        ▼                   ▼
┌───────────────┐  ┌───────────────┐
│  Turnover     │  │  Return       │
│  Rate         │  │  Combined     │
│               │  │  Dict         │
│  (INT +       │  │               │
│  fumble_lost) │  │  All 4        │
│  / drives     │  │  metrics      │
│               │  │               │
│  Range:       │  │               │
│  0.05 to 0.20 │  │               │
│               │  │               │
│  Elite: >0.12 │  │               │
└───────────────┘  └───────────────┘
```

## Situational Metrics Calculation Detail

```
┌─────────────────────────────────────────────────────────────┐
│         _calculate_situational_metrics(team, week)           │
└───────────────────────────┬─────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
┌───────────────┐  ┌───────────────┐  ┌───────────────┐
│  Pass Rate    │  │  Pace of Play │  │  Time of      │
│  (Neutral)    │  │               │  │  Possession   │
│               │  │  AVG(drive_   │  │               │
│  Filter:      │  │  play_count)  │  │  Parse MM:SS  │
│  - score_diff │  │               │  │  format       │
│    in [-7,7]  │  │  By drive     │  │               │
│  - qtr_sec    │  │               │  │  Convert to   │
│    > 120      │  │  Range:       │  │  minutes      │
│  - down 1 or 2│  │  5.0 to 7.5   │  │               │
│               │  │               │  │  AVG per      │
│  pass / total │  │  Fast: <5.5   │  │  drive        │
│               │  │  Slow: >7.0   │  │               │
│  Range:       │  │               │  │  Range:       │
│  0.45 to 0.65 │  │               │  │  2.0 to 3.5   │
│               │  │               │  │  minutes      │
│  Pass-heavy:  │  │               │  │               │
│  >0.58        │  │               │  │               │
└───────────────┘  └───────────────┘  └───────────────┘
```

## Database Schema Flow

```
┌─────────────────────────────────────────────────────────────┐
│                      INPUT TABLES                            │
└────────┬─────────────────────────────────┬──────────────────┘
         │                                 │
         ▼                                 ▼
┌──────────────────┐           ┌──────────────────┐
│   raw_pbp        │           │  raw_team_stats  │
│   (2022+)        │           │  (all seasons)   │
│                  │           │                  │
│  - 900+ columns  │           │  - 102 columns   │
│  - Play-level    │           │  - Weekly aggs   │
│  - EPA metrics   │           │  - Box scores    │
│  - Drive data    │           │                  │
│  - Pressure data │           │                  │
└──────────┬───────┘           └──────────┬───────┘
           │                              │
           └──────────────┬───────────────┘
                          │
                          ▼
           ┌──────────────────────────────┐
           │   PROCESSING LOGIC            │
           │                               │
           │  - Rolling 3-game windows     │
           │  - Offensive calculations     │
           │  - Defensive calculations     │
           │  - Situational calculations   │
           └──────────────┬────────────────┘
                          │
                          ▼
           ┌──────────────────────────────┐
           │     OUTPUT TABLE              │
           │  team_rolling_features        │
           │                               │
           │  Columns (15 total):          │
           │  - team, season, week         │
           │  - off_epa_per_play_last3     │
           │  - off_success_rate_last3     │
           │  - off_explosive_play_rate    │
           │  - off_red_zone_efficiency    │
           │  - off_third_down_conv        │
           │  - def_epa_per_play_last3     │
           │  - def_success_rate_last3     │
           │  - def_pressure_rate          │
           │  - def_turnover_rate          │
           │  - pass_rate_neutral          │
           │  - pace_of_play               │
           │  - time_of_possession_avg     │
           │  - created_at                 │
           │                               │
           │  ~2,080 records total         │
           └───────────────────────────────┘
```

## Rolling Window Illustration

```
Example: Calculate features for KC, Week 10, 2024

┌─────────────────────────────────────────────────────────────┐
│                     Season 2024                              │
└─────────────────────────────────────────────────────────────┘

Week:    1    2    3    4    5    6    7    8    9   [10]
        ───  ───  ───  ───  ───  ───  ───  ───  ───  ═════
                                       ▲──────▲──────▲
                                       │      │      │
                                    Week 7  Week 8  Week 9
                                       │      │      │
                                       └──────┴──────┘
                                             │
                                    Rolling 3-game window
                                    Used for Week 10 features

Data included:
✓ Week 7: KC vs LAC (offensive EPA, defensive EPA, etc.)
✓ Week 8: KC vs LV  (aggregate all plays)
✓ Week 9: KC vs TB  (rolling calculations)
✗ Week 10: EXCLUDED (can't use future/current week data)

Output: One record in team_rolling_features
  - team: 'KC'
  - season: 2024
  - week: 10
  - off_epa_per_play_last3: AVG of weeks 7-9
  - def_epa_per_play_last3: AVG of weeks 7-9
  - ... (all other metrics)
```

## Validation Flow

```
┌─────────────────────────────────────────────────────────────┐
│              _validate_team_aggregates()                     │
└────────────────────────────┬────────────────────────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
        ▼                    ▼                    ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│  EPA Range    │   │ Success Rate  │   │  NULL Value   │
│  Check        │   │  Check        │   │  Check        │
│               │   │               │   │               │
│  EPA should   │   │  Rate should  │   │  No NULLs in  │
│  be in range  │   │  be in [0,1]  │   │  EPA fields   │
│  [-3, 3]      │   │               │   │               │
│               │   │  Count        │   │  Count        │
│  Count        │   │  violations   │   │  violations   │
│  violations   │   │               │   │               │
│               │   │  Log warning  │   │  Log warning  │
│  Log warning  │   │  if > 0       │   │  if > 0       │
│  if > 0       │   │               │   │               │
└───────┬───────┘   └───────┬───────┘   └───────┬───────┘
        │                   │                   │
        └───────────────────┼───────────────────┘
                            │
                    ┌───────▼────────┐
                    │  Log Summary   │
                    │  - Total rows  │
                    │  - Issues found│
                    │  - Pass/Fail   │
                    └────────────────┘
```

## Performance Optimization Strategy

```
┌─────────────────────────────────────────────────────────────┐
│                   Query Optimization                         │
└─────────────────────────────────────────────────────────────┘

1. Use Existing Indexes
   ┌────────────────────────────────────────┐
   │  idx_pbp_season_week                   │
   │  idx_pbp_posteam_season                │
   │  idx_pbp_defteam_season                │
   └────────────────────────────────────────┘
         │
         ▼
   Fast filtering by team/season/week

2. Batch Processing
   ┌────────────────────────────────────────┐
   │  Process all teams for season together │
   │  Avoid N+1 query patterns              │
   └────────────────────────────────────────┘
         │
         ▼
   Single pass through data

3. CTE Usage
   ┌────────────────────────────────────────┐
   │  WITH offensive_plays AS (...)         │
   │  WITH red_zone_drives AS (...)         │
   └────────────────────────────────────────┘
         │
         ▼
   Readable, maintainable, optimized

4. Bulk Insert
   ┌────────────────────────────────────────┐
   │  Collect all records in DataFrame      │
   │  Single INSERT statement               │
   └────────────────────────────────────────┘
         │
         ▼
   Faster than row-by-row inserts

Expected Performance:
- Single season (2024): ~30 seconds
- All seasons (2022-2025): ~2 minutes
```

## Error Handling Strategy

```
┌─────────────────────────────────────────────────────────────┐
│                  Error Handling Points                       │
└─────────────────────────────────────────────────────────────┘

1. Missing PBP Data (pre-2022)
   ┌──────────────────────┐
   │ Check season >= 2022 │
   │ If not, use fallback │
   └──────────────────────┘

2. Insufficient Rolling Window (Week 1-3)
   ┌──────────────────────┐
   │ Skip weeks <= 3      │
   │ Log info message     │
   └──────────────────────┘

3. Division by Zero
   ┌──────────────────────┐
   │ Use NULLIF(denom, 0) │
   │ Return 0.0 as default│
   └──────────────────────┘

4. NULL EPA Values
   ┌──────────────────────┐
   │ Filter: epa IS NOT   │
   │ NULL before AVG()    │
   └──────────────────────┘

5. Database Connection Issues
   ┌──────────────────────┐
   │ Try-except blocks    │
   │ Log error, re-raise  │
   └──────────────────────┘
```

---

## Team Aggregate Features Usage Downstream

```
┌─────────────────────────────────────────────────────────────┐
│              team_rolling_features                           │
│              (Output of Stage 3c)                            │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
             ┌───────────────────────────────┐
             │   Stage 3d: Matchup Features  │
             │   (combines team aggregates   │
             │   with opponent aggregates)   │
             └───────────────┬───────────────┘
                             │
                             ▼
             ┌───────────────────────────────┐
             │   Stage 4: ML Dataset         │
             │   (final training features)   │
             └───────────────┬───────────────┘
                             │
                             ▼
             ┌───────────────────────────────┐
             │   Model Training              │
             │   - Player performance        │
             │   - Team total points         │
             │   - Win probability           │
             └───────────────────────────────┘
```
