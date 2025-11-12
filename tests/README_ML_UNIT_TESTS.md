# ML Component Unit Tests

Comprehensive unit tests for the XGBoost Model and Training Pipeline components.

## Overview

Created two comprehensive unit test files for ML components:

1. **`test_xgboost_unit.py`** - Unit tests for XGBoostPredictor
2. **`test_training_pipeline_unit.py`** - Unit tests for NFLTrainer

## Test Statistics

### Files Created
- **test_xgboost_unit.py**: 460 lines, 22 test cases
- **test_training_pipeline_unit.py**: 570 lines, 21 test cases
- **Total**: 1,030 lines, 43 test cases

### Test Results

#### XGBoost Predictor Tests
**Status: ✅ ALL PASSING (22/22)**

```
✓ test_initialization
✓ test_initialization_custom_seed
✓ test_train_regression_model
✓ test_train_classification_model
✓ test_predict_before_training_raises_error
✓ test_predict_after_training
✓ test_predict_proba_classification
✓ test_predict_proba_regression_raises_error
✓ test_save_and_load_model
✓ test_load_nonexistent_model_raises_error
✓ test_save_untrained_model_raises_error
✓ test_get_feature_importance
✓ test_get_feature_importance_untrained_raises_error
✓ test_multiple_positions
✓ test_multiple_targets_same_position
✓ test_get_position_targets
✓ test_get_trained_models
✓ test_get_model_info
✓ test_get_model_info_untrained_raises_error
✓ test_determine_task_type
✓ test_model_persistence_creates_files
✓ test_repr
```

#### Training Pipeline Tests
**Status: ⚠️  PARTIAL (8 passed, 3 skipped, 10 failed)**

```
✓ test_initialization
✓ test_initialization_custom_params
✓ test_train_position_models_invalid_position_raises_error
✓ test_position_targets_mapping
✓ test_repr
✓ test_generate_training_report
✓ test_load_training_data_missing_table_raises_error
✓ test_load_training_data_no_data_raises_error

⊘ test_train_position_models_integration (skipped - integration test)
⊘ test_training_results_storage (skipped - integration test)
⊘ test_train_all_positions (skipped - integration test)

✗ test_load_training_data (Polars JSON API compatibility)
✗ test_load_training_data_all_positions (Polars JSON API compatibility)
✗ test_split_data_temporal (Polars JSON API compatibility)
✗ test_split_data_temporal_custom_ratios (Polars JSON API compatibility)
✗ test_split_data_temporal_invalid_ratios_raises_error (Polars JSON API compatibility)
✗ test_prepare_features_and_targets (Polars JSON API compatibility)
✗ test_prepare_features_different_targets (Polars JSON API compatibility)
✗ test_prepare_features_for_different_positions (Polars JSON API compatibility)
✗ test_evaluate_model (Polars JSON API compatibility)
✗ test_get_trained_models_summary (Polars JSON API compatibility)
```

## Test Coverage

### XGBoost Predictor Coverage

#### Core Functionality
- ✅ **Initialization**: Model directory creation, registry setup
- ✅ **Training**: Both regression and classification models
- ✅ **Prediction**: Standard predictions and probability predictions
- ✅ **Model Persistence**: Save and load models with metadata
- ✅ **Feature Importance**: Extraction and ranking

#### Error Handling
- ✅ **Untrained Model Errors**: Predict/get_info before training
- ✅ **Missing Model Errors**: Loading non-existent models
- ✅ **Type Validation**: predict_proba on regression models
- ✅ **Position Validation**: Invalid position handling

#### Advanced Features
- ✅ **Multiple Positions**: QB, RB, WR models simultaneously
- ✅ **Multiple Targets**: Multiple prediction targets per position
- ✅ **Model Registry**: Tracking trained models
- ✅ **Task Type Detection**: Auto-detect regression vs classification
- ✅ **Metadata Management**: Feature names, training duration, metrics

### Training Pipeline Coverage

#### Core Functionality
- ✅ **Initialization**: Database connection, model directory setup
- ✅ **Error Validation**: Invalid positions, missing tables
- ✅ **Configuration**: Position-target mappings
- ✅ **Reporting**: Training report generation

#### Partial Coverage (requires Polars JSON API fix)
- ⚠️  **Data Loading**: Database queries with position filtering
- ⚠️  **Data Splitting**: Temporal train/val/test splits
- ⚠️  **Feature Extraction**: Converting database format to numpy arrays
- ⚠️  **Model Evaluation**: Test set evaluation metrics
- ⚠️  **Integration Tests**: End-to-end training workflows

## Testing Approach

### Synthetic Data
All tests use synthetic data for fast, reproducible testing:
- **XGBoost Tests**: numpy random data (47 features, 100+ samples)
- **Training Pipeline Tests**: In-memory DuckDB database with mock data

### Isolation
- No external dependencies (MLflow disabled, no API calls)
- Temporary directories (pytest tmp_path fixture)
- In-memory databases where possible

### Error Testing
Comprehensive error handling coverage:
- ValueError for invalid inputs
- FileNotFoundError for missing models
- AssertionError for invalid ratios

## Known Issues & Limitations

### Polars JSON API Compatibility
**Issue**: 10 training pipeline tests fail due to Polars `json_extract()` method not being available.

**Root Cause**: The `NFLTrainer.load_training_data()` method uses:
```python
pl.col("categorical_features").str.json_extract()
```
This API may have changed in the version of Polars installed.

**Impact**: Tests that rely on loading data from the test database fail at the JSON parsing step.

**Workaround Options**:
1. Update Polars to a version with `json_extract()` support
2. Modify trainer code to use `json_decode()` or similar
3. Mock the database loading in tests
4. Add `position` column to database schema (attempted but trainer still tries JSON extraction)

**Tests Affected**:
- Data loading tests
- Data splitting tests
- Feature preparation tests
- Model evaluation tests

## Running the Tests

### Run All Tests
```bash
pytest tests/test_xgboost_unit.py tests/test_training_pipeline_unit.py -v
```

### Run XGBoost Tests Only
```bash
pytest tests/test_xgboost_unit.py -v
```

### Run Training Pipeline Tests Only
```bash
pytest tests/test_training_pipeline_unit.py -v
```

### Run with Coverage
```bash
pytest tests/test_xgboost_unit.py --cov=src.models.xgboost_predictor --cov-report=html
pytest tests/test_training_pipeline_unit.py --cov=src.training.trainer --cov-report=html
```

## Test Fixtures

### XGBoost Fixtures
- **`sample_training_data`**: Synthetic regression data (100 train, 30 val, 20 test)
- **`sample_classification_data`**: Synthetic binary classification data
- **`temp_model_dir`**: Temporary directory for model storage

### Training Pipeline Fixtures
- **`test_db_with_ml_features`**: In-memory DuckDB with 150 sample records (QB, RB, WR)
- **`empty_db`**: Empty DuckDB for error testing

## Test Patterns

### 1. Initialization Tests
Verify correct setup of class instances and configuration.

### 2. Happy Path Tests
Test normal operation with valid inputs and expected workflows.

### 3. Error Handling Tests
Verify proper exceptions are raised for invalid inputs.

### 4. Integration Tests
Test multiple components working together (some skipped due to complexity).

### 5. Edge Case Tests
Test boundary conditions and special scenarios.

## Metrics & Assertions

### XGBoost Tests Assert
- Model registry structure
- Metric values (RMSE, MAE, R², accuracy, AUC)
- Prediction shapes and data types
- File existence (models, metadata, features)
- Feature importance DataFrames

### Training Pipeline Tests Assert
- DataFrame shapes and columns
- Data types (Polars DataFrame, numpy arrays)
- Split ratios (70/20/10)
- Exception types and messages
- Configuration mappings

## Future Improvements

### Short Term
1. **Fix Polars JSON API compatibility** - Update code or Polars version
2. **Add classification model tests** - Test win probability predictions
3. **Add hyperparameter tuning tests** - Test Optuna integration (currently skipped)

### Medium Term
1. **Add SHAP tests** - Test model interpretability features
2. **Add MLflow integration tests** - Test experiment tracking
3. **Add performance benchmarks** - Test training speed and memory usage

### Long Term
1. **Add real data tests** - Test with small sample of actual NFL data
2. **Add model comparison tests** - Compare XGBoost vs other models
3. **Add deployment tests** - Test model loading in production scenarios

## Contributing

When adding new tests:
1. Use synthetic data for speed and reproducibility
2. Follow the existing test structure (fixtures, classes, clear names)
3. Test both happy paths and error cases
4. Add docstrings explaining what each test validates
5. Use temporary directories/databases (no persistent state)
6. Disable external dependencies (MLflow, APIs)

## Summary

**Total Tests Created**: 43
**Lines of Code**: 1,030
**Passing Tests**: 30 (22 XGBoost + 8 Training Pipeline)
**Skipped Tests**: 3 (integration tests)
**Failing Tests**: 10 (Polars JSON API issue)

**Coverage Highlights**:
- ✅ Complete XGBoost model lifecycle (train/predict/save/load)
- ✅ Error handling and validation
- ✅ Multiple positions and targets
- ✅ Model persistence and metadata
- ⚠️  Training pipeline data loading (needs Polars API fix)
- ⚠️  End-to-end training workflows (integration tests skipped)

The test suite provides excellent coverage of the XGBoost predictor in isolation and validates the core structure of the training pipeline. The remaining issues are primarily related to Polars version compatibility and can be resolved by updating the Polars API usage in the trainer code or using a compatible Polars version.
