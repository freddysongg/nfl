# XGBoost Model Architecture

## System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         XGBoostPredictor                            │
│                         (BaseModel)                                 │
└────────────┬────────────────────────────────────────────────────────┘
             │
             ├─── Model Registry
             │    └─── {position: {target: XGBModel}}
             │         ├─── QB: {passing_yards, passing_tds, ...}
             │         ├─── RB: {rushing_yards, rushing_tds, ...}
             │         ├─── WR: {receiving_yards, receptions, ...}
             │         ├─── TE: {receiving_yards, receptions, ...}
             │         ├─── K: {fg_made, fg_pct, ...}
             │         └─── DEF: {sacks, interceptions, ...}
             │
             ├─── Feature Management
             │    ├─── Load from ml_training_features table
             │    ├─── Preprocess (imputation, scaling)
             │    └─── Feature selection (top N important)
             │
             ├─── Training Pipeline
             │    ├─── Data Loading (by position/target)
             │    ├─── Train/Val Split (temporal)
             │    ├─── Hyperparameter Tuning (Optuna)
             │    ├─── Model Training (early stopping)
             │    ├─── Evaluation (metrics calculation)
             │    └─── Model Persistence
             │
             ├─── Prediction Pipeline
             │    ├─── Feature validation
             │    ├─── Model selection (position/target)
             │    └─── Prediction + confidence intervals
             │
             ├─── Interpretability
             │    ├─── Feature Importance (gain/weight/cover)
             │    ├─── SHAP Values (TreeExplainer)
             │    └─── Visualizations (summary, dependence plots)
             │
             └─── Experiment Tracking (MLflow)
                  ├─── Parameters logging
                  ├─── Metrics logging
                  ├─── Artifact storage
                  └─── Model registry
```

---

## Data Flow

```
┌──────────────────┐
│  ml_training_    │
│  features table  │
└────────┬─────────┘
         │
         │ 1. Query by position/target/season
         │
         ▼
┌──────────────────────────────┐
│ Feature Preparation          │
│ • Extract numerical_features │
│ • Extract actual_outcomes    │
│ • Filter by quality score    │
└────────┬─────────────────────┘
         │
         │ 2. Temporal split
         │
         ▼
┌──────────────────┬───────────────────┐
│   X_train        │     X_val         │
│   y_train        │     y_val         │
└────────┬─────────┴──────────┬────────┘
         │                    │
         │ 3. Optuna tuning   │
         │    (optional)      │
         │                    │
         ▼                    ▼
┌────────────────────────────────────┐
│     XGBoost Training               │
│   • Early stopping on X_val        │
│   • Best iteration selection       │
└────────┬───────────────────────────┘
         │
         │ 4. Evaluation
         │
         ▼
┌────────────────────────────────────┐
│  Metrics Calculation               │
│  • RMSE, MAE, R² (regression)      │
│  • Accuracy, AUC (classification)  │
└────────┬───────────────────────────┘
         │
         │ 5. Interpretability
         │
         ▼
┌────────────────────────────────────┐
│  SHAP Analysis                     │
│  • TreeExplainer                   │
│  • Summary plots                   │
│  • Dependence plots                │
└────────┬───────────────────────────┘
         │
         │ 6. Persistence
         │
         ▼
┌─────────────────┬──────────────────┬──────────────────┐
│  model.json     │  metadata.json   │  features.json   │
└─────────────────┴──────────────────┴──────────────────┘
```

---

## Model Training Workflow

```python
# Pseudo-code workflow

# 1. Initialize
predictor = XGBoostPredictor(db, config)

# 2. Load data for specific position/target
X_train, y_train, X_val, y_val, feature_names = load_data(
    position='QB',
    target='passing_yards',
    season_range=(2021, 2024)
)

# 3. Hyperparameter tuning (optional)
if tune:
    best_params = optuna_optimize(X_train, y_train, X_val, y_val)
else:
    best_params = default_params

# 4. Train final model
model = XGBRegressor(**best_params)
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=20
)

# 5. Evaluate
metrics = {
    'val_rmse': calculate_rmse(model, X_val, y_val),
    'val_r2': calculate_r2(model, X_val, y_val)
}

# 6. SHAP analysis
shap_values = calculate_shap(model, X_val[:1000])
create_visualizations(shap_values)

# 7. Save
model.save_model('models/QB_passing_yards_model.json')
save_metadata('models/QB_passing_yards_metadata.json')

# 8. Log to MLflow
mlflow.log_params(best_params)
mlflow.log_metrics(metrics)
mlflow.log_artifacts(['model.json', 'shap_plots/'])
```

---

## Position-Specific Model Matrix

| Position | Primary Targets | Feature Count | Task Type | Priority |
|----------|----------------|---------------|-----------|----------|
| **QB** | passing_yards, passing_tds, interceptions, rushing_yards, fantasy_points | ~30-40 | Regression | High |
| **RB** | rushing_yards, rushing_tds, receptions, receiving_yards, fantasy_points | ~25-35 | Regression | High |
| **WR** | receiving_yards, receptions, targets, receiving_tds, fantasy_points | ~25-35 | Regression | High |
| **TE** | receiving_yards, receptions, targets, receiving_tds, fantasy_points | ~20-30 | Regression | Medium |
| **K** | fg_made, fg_pct, fg_made_40_49, fg_made_50_plus, fantasy_points | ~15-25 | Regression/Class | Medium |
| **DEF** | sacks, interceptions, fumbles_forced, points_allowed, fantasy_points | ~20-30 | Regression | Low |

**Total Models to Train**: ~35-40 (6 positions × 5-8 targets each)

---

## Feature Engineering Pipeline Integration

```
┌─────────────────────────────────────────────────────────────────┐
│                   Stage 4: ML Dataset Creation                  │
│                 (build_ml_dataset from pipeline)                │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            │ Creates ml_training_features table
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│              ml_training_features Table Schema                  │
│                                                                 │
│  • feature_id (PK)                                              │
│  • entity_type (player/team)                                    │
│  • entity_id                                                    │
│  • prediction_target (passing_yards, rushing_tds, etc.)         │
│  • season, week, game_date                                      │
│  • numerical_features (FLOAT[])                                 │
│  • feature_names (VARCHAR[])                                    │
│  • categorical_features (JSON)                                  │
│  • actual_outcomes (JSON)                                       │
│  • data_quality_score                                           │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            │ XGBoostPredictor queries this table
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                     XGBoostPredictor                            │
│                                                                 │
│  SELECT numerical_features, actual_outcomes, feature_names      │
│  FROM ml_training_features                                      │
│  WHERE entity_type = 'player'                                   │
│    AND prediction_target = ?                                    │
│    AND season BETWEEN ? AND ?                                   │
│    AND data_quality_score >= 0.7                                │
│  ORDER BY season, week                                          │
└─────────────────────────────────────────────────────────────────┘
```

---

## Hyperparameter Search Space

### Regression Models

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| max_depth | int | 3-10 | 6 | Maximum tree depth |
| learning_rate | float | 0.01-0.3 (log) | 0.05 | Step size shrinkage |
| n_estimators | int | 100-500 | 200 | Number of boosting rounds |
| subsample | float | 0.6-1.0 | 0.8 | Subsample ratio of training instances |
| colsample_bytree | float | 0.6-1.0 | 0.8 | Subsample ratio of columns |
| min_child_weight | int | 1-10 | 3 | Minimum sum of instance weight |
| gamma | float | 0.0-1.0 | 0.1 | Minimum loss reduction |
| reg_alpha | float | 0.0-1.0 | 0.1 | L1 regularization |
| reg_lambda | float | 0.0-2.0 | 1.0 | L2 regularization |

### Classification Models

Same as regression, plus:

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| scale_pos_weight | float | 0.5-2.0 | 1.0 | Balance positive/negative classes |

---

## Evaluation Metrics by Task Type

### Regression Metrics

```python
{
    'train_rmse': float,      # Lower is better
    'train_mae': float,       # Lower is better
    'train_r2': float,        # Higher is better (max 1.0)
    'val_rmse': float,        # Primary metric
    'val_mae': float,
    'val_r2': float,
    'mape': float,            # Mean absolute percentage error
    'max_error': float        # Worst prediction
}
```

**Performance Benchmarks:**

| Position | Target | Excellent RMSE | Good RMSE | Acceptable RMSE |
|----------|--------|----------------|-----------|-----------------|
| QB | passing_yards | < 50 | < 75 | < 100 |
| QB | passing_tds | < 0.5 | < 0.8 | < 1.2 |
| RB | rushing_yards | < 20 | < 35 | < 50 |
| WR | receiving_yards | < 15 | < 25 | < 40 |

### Classification Metrics

```python
{
    'train_accuracy': float,
    'val_accuracy': float,    # Primary metric
    'val_precision': float,
    'val_recall': float,
    'val_f1': float,
    'val_roc_auc': float,     # Secondary metric
    'val_log_loss': float
}
```

---

## SHAP Visualization Types

### 1. Summary Plot (Bar)
- Shows average absolute SHAP value per feature
- Identifies most important features globally
- Use case: Feature selection, model debugging

### 2. Summary Plot (Beeswarm)
- Shows distribution of SHAP values across samples
- Color: Feature value (red=high, blue=low)
- Position: Impact on prediction
- Use case: Understanding feature effects

### 3. Dependence Plot
- Shows relationship between feature value and SHAP value
- Includes interaction effects (colored by other feature)
- Use case: Understanding non-linear relationships

### 4. Force Plot
- Individual prediction explanation
- Shows how each feature pushes prediction up/down
- Use case: Explaining specific predictions

### 5. Waterfall Plot
- Individual prediction breakdown
- Shows cumulative effect of features
- Use case: Debugging predictions

---

## MLflow Experiment Structure

```
mlruns/
├── 0/  (Default experiment)
└── 1/  (xgboost_nfl_predictor)
    ├── meta.yaml
    ├── runs/
    │   ├── run_id_1/  (QB_passing_yards)
    │   │   ├── params/
    │   │   │   ├── max_depth
    │   │   │   ├── learning_rate
    │   │   │   └── ...
    │   │   ├── metrics/
    │   │   │   ├── val_rmse
    │   │   │   ├── val_r2
    │   │   │   └── ...
    │   │   ├── artifacts/
    │   │   │   ├── model/
    │   │   │   │   └── model.json
    │   │   │   ├── shap_visualizations/
    │   │   │   │   ├── summary_bar.png
    │   │   │   │   ├── summary_beeswarm.png
    │   │   │   │   └── dependence_plots/
    │   │   │   └── metadata.json
    │   │   └── tags/
    │   │       ├── position: QB
    │   │       ├── target: passing_yards
    │   │       └── model_type: xgboost
    │   ├── run_id_2/  (QB_passing_tds)
    │   └── ...
    └── models/
        └── QB_passing_yards_xgboost/
            ├── version_1/
            │   ├── model/
            │   └── stage: Production
            └── version_2/
                ├── model/
                └── stage: Staging
```

---

## Error Handling Strategy

```python
class XGBoostPredictorError(Exception):
    """Base exception for XGBoostPredictor."""
    pass

class ModelNotFoundError(XGBoostPredictorError):
    """Model not found in registry."""
    pass

class InsufficientDataError(XGBoostPredictorError):
    """Not enough data for training."""
    pass

class FeatureMismatchError(XGBoostPredictorError):
    """Feature dimensions don't match."""
    pass

# Usage in methods
def predict(self, position, target, X):
    if position not in self.model_registry:
        raise ModelNotFoundError(
            f"No models trained for position: {position}"
        )

    expected_features = len(self.feature_info[position][target])
    if X.shape[1] != expected_features:
        raise FeatureMismatchError(
            f"Expected {expected_features} features, got {X.shape[1]}"
        )

    return self.model_registry[position][target].predict(X)
```

---

## Quick Reference: Key Methods

| Method | Purpose | Input | Output |
|--------|---------|-------|--------|
| `train()` | Train single model | position, target, X_train, y_train, X_val, y_val | Training results dict |
| `predict()` | Make predictions | position, target, X | Predictions array |
| `save_model()` | Save model to disk | position, target, filepath | Model path |
| `load_model()` | Load model from disk | position, target, filepath | None (updates registry) |
| `get_feature_importance()` | Get feature importance | position, target | DataFrame |
| `calculate_shap_values()` | Calculate SHAP | position, target, X_sample | SHAP Explanation |
| `train_all_position_models()` | Train all models for position | position, season_range | Results dict |
| `get_position_targets()` | Get available targets | position | List of targets |
| `interpret_shap_results()` | Generate interpretation | position, target | Interpretation text |

---

## Configuration Example

```python
# config.py additions

XGBOOST_CONFIG = {
    'model_dir': 'models/xgboost',
    'experiment_name': 'xgboost_nfl_predictor',
    'mlflow_tracking_uri': 'sqlite:///mlruns.db',

    'hyperparameter_tuning': {
        'enabled': True,
        'n_trials': 50,
        'timeout_seconds': 3600,  # 1 hour max per model
        'n_jobs': -1,  # Use all cores
    },

    'shap_config': {
        'max_samples': 1000,  # Limit for performance
        'visualizations': ['summary_bar', 'summary_beeswarm', 'dependence'],
        'save_values': True,
    },

    'training_config': {
        'validation_split': 0.2,
        'min_samples': 100,  # Minimum samples to train model
        'quality_threshold': 0.7,  # Minimum data quality score
        'early_stopping_rounds': 20,
    },

    'prediction_config': {
        'confidence_intervals': True,
        'quantiles': [0.1, 0.9],  # 80% prediction interval
    }
}
```

---

## Performance Optimization Tips

1. **Data Loading**
   - Use DuckDB's parallel query execution
   - Pre-filter by data_quality_score
   - Index on (entity_type, prediction_target, season)

2. **Training**
   - Use `tree_method='hist'` for speed
   - Set `n_jobs=-1` to use all cores
   - Use early stopping to avoid overfitting
   - Cache preprocessed data

3. **SHAP Calculation**
   - Limit sample size to 1000 rows
   - Use TreeExplainer (fast for XGBoost)
   - Calculate only when needed (not every training run)

4. **Hyperparameter Tuning**
   - Use TPE sampler (more efficient than grid search)
   - Enable pruning to stop unpromising trials
   - Start with small n_trials (10-20) for testing
   - Use distributed optimization for production

5. **Model Storage**
   - Use JSON format (human-readable, version control friendly)
   - Compress large models with gzip
   - Store metadata separately for quick lookup

---

## Monitoring & Logging

```python
# Training logs
[2024-11-06 10:23:45] INFO: Starting training for QB passing_yards
[2024-11-06 10:23:45] INFO: Loaded data: Train=2547, Val=637
[2024-11-06 10:23:45] INFO: Features: 35
[2024-11-06 10:23:45] INFO: Starting hyperparameter tuning...
[2024-11-06 10:28:32] INFO: Optuna trials: 50/50
[2024-11-06 10:28:32] INFO: Best trial: 42, Best value: 48.23
[2024-11-06 10:28:32] INFO: Training final model...
[2024-11-06 10:29:15] INFO: Training complete. Val RMSE: 47.89
[2024-11-06 10:29:15] INFO: Calculating SHAP values...
[2024-11-06 10:29:47] INFO: SHAP visualizations saved
[2024-11-06 10:29:47] INFO: Model saved: models/QB_passing_yards_model.json
[2024-11-06 10:29:47] INFO: MLflow run: run_id_abc123

# Prediction logs
[2024-11-06 11:15:23] INFO: Predicting for QB passing_yards
[2024-11-06 11:15:23] INFO: Input shape: (1, 35)
[2024-11-06 11:15:23] INFO: Prediction: 284.5 yards
[2024-11-06 11:15:23] INFO: Confidence interval: [235.2, 333.8]
```

---

## Next Steps

After implementation:

1. **Validation**: Compare predictions against actual 2024 week-by-week results
2. **Calibration**: Ensure prediction intervals are well-calibrated
3. **Feature Analysis**: Review SHAP values to identify key predictive features
4. **Model Updates**: Retrain weekly as new data becomes available
5. **API Integration**: Expose predictions via REST API
6. **Dashboard**: Create visualization dashboard for predictions and explanations

---

## References

- XGBoost Documentation: https://xgboost.readthedocs.io/
- SHAP Documentation: https://shap.readthedocs.io/
- Optuna Documentation: https://optuna.readthedocs.io/
- MLflow Documentation: https://mlflow.org/docs/latest/index.html
- NFL Data Sources: nflverse (via nflreadpy)
