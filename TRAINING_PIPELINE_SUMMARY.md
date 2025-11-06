# Training Pipeline Implementation Summary

## Files Created

### 1. `/home/user/nfl/src/training/__init__.py` (7 lines)
- Basic module exports
- Exports `NFLTrainer` class

### 2. `/home/user/nfl/src/training/trainer.py` (809 lines)
Main orchestration class with complete implementation:

**Key Classes:**
- `NFLTrainer`: Main trainer orchestration class

**Core Methods:**
1. **Data Loading:**
   - `load_training_data()`: Loads ml_training_features from DuckDB with filtering
   - Supports position filtering, season ranges, quality score thresholds
   - Returns Polars DataFrame

2. **Data Splitting:**
   - `split_data_temporal()`: Temporal train/val/test splits (70/20/10)
   - Prevents data leakage by respecting time ordering
   - Returns three DataFrames for train/val/test

3. **Feature Extraction:**
   - `prepare_features_and_targets()`: Extracts numerical_features array and targets
   - Parses actual_outcomes JSON for position-specific targets
   - Returns (X, y, feature_names) tuples

4. **Training Orchestration:**
   - `train_position_models()`: Trains all targets for a specific position
   - `train_all_positions()`: Trains all positions and all their targets
   - Integration with XGBoostPredictor for actual model training

5. **Evaluation:**
   - `evaluate_model()`: Evaluates model on test set
   - Calculates RMSE, MAE, R² metrics

6. **Reporting:**
   - `generate_training_report()`: Creates markdown report with results
   - Includes metrics tables, model paths, error summaries

### 3. `/home/user/nfl/train_models.py` (306 lines)
CLI entry point with comprehensive argument handling:

**Command-Line Arguments:**
- `--position`: Position to train (QB, RB, WR, TE, K)
- `--target`: Specific target (optional, for single-target training)
- `--all-positions`: Train all positions
- `--tune`: Enable hyperparameter tuning
- `--n-trials`: Number of Optuna trials
- `--db-path`: Database path
- `--season-start/--season-end`: Data range
- `--model-dir`: Model output directory
- `--report`: Generate markdown report
- `--no-mlflow`: Disable MLflow tracking
- `--random-seed`: Random seed

**Usage Examples:**
```bash
# Train all QB models
python train_models.py --position QB

# Train all QB models with tuning
python train_models.py --position QB --tune --n-trials 100

# Train all positions
python train_models.py --all-positions

# Train specific target
python train_models.py --position RB --target rushing_yards --tune

# Generate report
python train_models.py --all-positions --report training_report.md
```

## Key Implementation Details

### 1. Position-Target Mapping
```python
POSITION_TARGETS = {
    "QB": ["passing_yards", "passing_tds", "passing_interceptions", 
           "completions", "attempts", "rushing_yards", "rushing_tds", 
           "fantasy_points_ppr"],
    "RB": ["rushing_yards", "rushing_tds", "carries", "receptions", 
           "receiving_yards", "receiving_tds", "fantasy_points_ppr"],
    "WR": ["receiving_yards", "receiving_tds", "receptions", "targets", 
           "catch_rate", "rushing_yards", "fantasy_points_ppr"],
    "TE": ["receiving_yards", "receiving_tds", "receptions", "targets", 
           "catch_rate", "fantasy_points_ppr"],
    "K": ["fg_made", "fg_att", "fg_pct", "pat_made", "pat_att", 
          "fantasy_points_standard"],
}
```

### 2. DuckDB Integration
```python
# Query ml_training_features with position filtering
query = """
    SELECT
        feature_id, entity_id, season, week, game_date,
        numerical_features, feature_names,
        categorical_features, actual_outcomes,
        data_quality_score, player_experience_level
    FROM ml_training_features
    WHERE season BETWEEN ? AND ?
        AND actual_outcomes IS NOT NULL
        AND json_extract(categorical_features, '$.position.value') = ?
    ORDER BY season, week
"""
df = conn.execute(query, params).pl()
```

### 3. Temporal Data Splitting
```python
# Temporal split (70/20/10) - prevents data leakage
n_train = int(n_total * 0.7)
n_val = int(n_total * 0.2)
n_test = n_total - n_train - n_val

train_df = df[:n_train]
val_df = df[n_train:n_train + n_val]
test_df = df[n_train + n_val:]
```

### 4. Feature and Target Extraction
```python
# Extract numerical_features array (47 features)
X = np.array([row[0] for row in df.select("numerical_features").to_numpy()])

# Extract target from actual_outcomes JSON
outcomes = json.loads(row["actual_outcomes"])
y = outcomes.get(target, 0.0)
```

### 5. XGBoostPredictor Integration
```python
# Train using XGBoostPredictor
result = self.predictor.train(
    position="QB",
    target="passing_yards",
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val,
    feature_names=feature_names,
    hyperparameter_tune=True,
    n_trials=50
)
```

### 6. Error Handling
- Validates ml_training_features table exists
- Checks for sufficient data before training
- Handles missing targets gracefully
- Provides detailed error messages
- Continues training on errors (stores error in results)

### 7. Progress Logging
- Detailed logging at each stage
- Summary statistics (position distribution, season ranges)
- Training metrics (RMSE, MAE, R²)
- Final summary reports

## Integration Points

### 1. XGBoostPredictor (Stage 7)
- Uses `train()` method for model training
- Uses `predict()` for evaluation
- Leverages MLflow integration
- Accesses model registry

### 2. DuckDB (Stage 4)
- Reads from `ml_training_features` table
- Extracts `numerical_features` (FLOAT[])
- Parses `categorical_features` (JSON)
- Parses `actual_outcomes` (JSON)

### 3. MLflow (Optional)
- Experiment tracking via XGBoostPredictor
- Can be disabled with `--no-mlflow`

### 4. Config (src/config.py)
- Uses default database path
- Integrates with existing configuration

## Workflow

```
1. Load Data
   └─> Query ml_training_features from DuckDB
       └─> Filter by position, season, quality score
           └─> Sort by (season, week)

2. Split Data
   └─> Temporal split (70/20/10)
       └─> Train: earliest data
       └─> Val: middle data
       └─> Test: latest data

3. For Each Position:
   └─> For Each Target:
       └─> Extract Features (X)
       └─> Extract Target (y)
       └─> Train Model (XGBoostPredictor)
       └─> Evaluate on Test Set
       └─> Store Results

4. Generate Report
   └─> Create DataFrame with all results
   └─> Generate markdown report
   └─> Save to file
```

## Output

### Training Results DataFrame
```
| position | target         | status  | train_rmse | val_rmse | test_rmse | test_r2 | n_train |
|----------|----------------|---------|------------|----------|-----------|---------|---------|
| QB       | passing_yards  | success | 45.23      | 48.12    | 47.89     | 0.756   | 12,450  |
| QB       | passing_tds    | success | 0.89       | 0.94     | 0.92      | 0.623   | 12,450  |
| ...      | ...            | ...     | ...        | ...      | ...       | ...     | ...     |
```

### Markdown Report
- Overall summary (total/success/failed/skipped)
- Results by position (metrics tables)
- Failed/skipped models with reasons
- Model artifact paths

## File Structure

```
nfl/
├── src/
│   ├── training/
│   │   ├── __init__.py          # Module exports
│   │   └── trainer.py            # NFLTrainer class (809 lines)
│   └── ...
├── train_models.py               # CLI entry point (306 lines)
└── models/                       # Model output directory
    └── xgboost/                  # XGBoost models
        ├── QB_passing_yards_model.json
        ├── QB_passing_yards_model.features.json
        ├── QB_passing_yards_metadata.json
        └── ...
```

## Testing

The implementation includes:
- Input validation (position, target, database existence)
- Data validation (sufficient samples, table existence)
- Error handling and recovery
- Detailed logging for debugging
- Graceful degradation (skips targets with insufficient data)

## Next Steps

To use the training pipeline:

1. Ensure data pipeline (Stages 1-4) has run and populated `ml_training_features`
2. Install dependencies: `uv sync`
3. Run training:
   ```bash
   # Simple example
   python train_models.py --position QB
   
   # Full training with tuning
   python train_models.py --all-positions --tune --n-trials 100 --report results.md
   ```

## Notes

- All file paths in implementation are absolute (not relative)
- Temporal splitting ensures no data leakage
- Supports both single-target and multi-target training
- Integrates seamlessly with existing XGBoostPredictor and data pipeline
- MLflow tracking is optional and can be disabled
- Training results are stored and can be accessed later
