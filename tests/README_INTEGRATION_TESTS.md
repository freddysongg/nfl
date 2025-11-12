# Integration Tests - Quick Reference

## Test Files

1. **test_data_pipeline_integration.py** - Tests data pipeline (Stages 2-4)
2. **test_ml_pipeline_integration.py** - Tests ML training and prediction
3. **test_end_to_end.py** - Tests complete system workflow

## Quick Start

### Run All Integration Tests
```bash
pytest tests/test_data_pipeline_integration.py tests/test_ml_pipeline_integration.py tests/test_end_to_end.py -v
```

### Run Individual Files
```bash
# Data pipeline tests (~10-15 seconds)
pytest tests/test_data_pipeline_integration.py -v

# ML pipeline tests (~15-20 seconds)
pytest tests/test_ml_pipeline_integration.py -v

# End-to-end tests (~30-60 seconds)
pytest tests/test_end_to_end.py -v -s
```

### Run Specific Test
```bash
pytest tests/test_data_pipeline_integration.py::TestDataPipelineIntegration::test_stage2_to_stage3a_flow -v
```

### Run with Output
```bash
pytest tests/test_end_to_end.py::test_full_system_end_to_end -v -s
```

## Test Summary

| File | Tests | Runtime | Purpose |
|------|-------|---------|---------|
| test_data_pipeline_integration.py | 13 | ~10-15s | Data flow through Stages 2-4 |
| test_ml_pipeline_integration.py | 12 | ~15-20s | ML training and prediction |
| test_end_to_end.py | 2 | ~30-60s | Complete system workflow |

**Total:** 27 tests, ~60-90 seconds

## Test Coverage

### Data Pipeline (13 tests)
- ✅ Stage 2 → 3a flow
- ✅ Stage 3 → 4 flow
- ✅ Full pipeline execution
- ✅ Data quality filtering
- ✅ Temporal consistency
- ✅ Rolling statistics
- ✅ Position-specific features
- ✅ Multi-season handling
- ✅ Feature engineering
- ✅ Matchup features
- ✅ Team aggregates
- ✅ Data completeness
- ✅ Outlier detection

### ML Pipeline (12 tests)
- ✅ Data loading from database
- ✅ Temporal train/val/test splits
- ✅ Feature/target preparation
- ✅ XGBoost training
- ✅ XGBoost prediction
- ✅ Train → predict → evaluate workflow
- ✅ Multi-target training
- ✅ Model save/load
- ✅ Model registry basics
- ✅ Feature extraction (47 features)
- ✅ Categorical features
- ✅ Target extraction

### End-to-End (2 tests)
- ✅ Complete system workflow (raw data → predictions)
- ✅ Multi-position workflow (QB, WR, RB)

## Common Use Cases

### 1. Verify Data Pipeline Works
```bash
pytest tests/test_data_pipeline_integration.py -v
```

### 2. Verify ML Training Works
```bash
pytest tests/test_ml_pipeline_integration.py -v
```

### 3. Verify Complete System
```bash
pytest tests/test_end_to_end.py::test_full_system_end_to_end -v -s
```

### 4. Quick Smoke Test (Fast Tests Only)
```bash
pytest tests/test_data_pipeline_integration.py tests/test_ml_pipeline_integration.py -v
```

### 5. Full Integration Test Suite
```bash
pytest tests/test_*_integration.py tests/test_end_to_end.py -v
```

## Debugging Failed Tests

### Enable Verbose Output
```bash
pytest tests/test_end_to_end.py -v -s --tb=long
```

### Run Single Test with Debug Info
```bash
pytest tests/test_ml_pipeline_integration.py::TestMLPipelineIntegration::test_xgboost_training_and_prediction -v -s --pdb
```

### Check Test Data
Tests use temporary databases in `/tmp`. To inspect:
```bash
# Run test with --pdb to pause at failure
pytest tests/test_data_pipeline_integration.py --pdb
```

## Test Data Characteristics

### Data Pipeline Tests
- 3 players (QB, WR, RB)
- 2 seasons × 10 weeks = 60 samples
- Realistic NFL statistics

### ML Pipeline Tests
- 6 players (2 QB, 2 WR, 2 RB)
- 2 seasons × 17 weeks = ~200 samples
- 47 numerical features per sample

### End-to-End Tests
- 5 players (2 QB, 2 WR, 1 RB)
- 2 seasons × 15 weeks = ~150 samples
- Complete workflow validation

## Expected Output

### Successful Run
```
tests/test_data_pipeline_integration.py::TestDataPipelineIntegration::test_stage2_to_stage3a_flow PASSED
tests/test_data_pipeline_integration.py::TestDataPipelineIntegration::test_stage3_to_stage4_flow PASSED
...
========================= 27 passed in 75.23s =========================
```

### Failed Test
```
tests/test_ml_pipeline_integration.py::TestMLPipelineIntegration::test_xgboost_training_and_prediction FAILED

AssertionError: Should predict for all test samples
```

## Troubleshooting

### Issue: "Table does not exist"
**Cause:** Test fixture didn't create tables properly
**Solution:** Check fixture setup, ensure all `create_*_table()` calls succeed

### Issue: "RMSE too high"
**Cause:** Model didn't train properly or test data is unusual
**Solution:** Check training data size, verify features are populated

### Issue: "Temporal consistency failed"
**Cause:** Data ordering issue in test data
**Solution:** Verify seasons/weeks are in ascending order

### Issue: "Feature count mismatch"
**Cause:** Expected 47 features but got different count
**Solution:** Check feature generation code, ensure all 47 features created

## Performance Notes

- Tests use small datasets for speed (60-200 samples)
- XGBoost trained with n_estimators=10-20 for speed
- Real production models use much larger datasets
- Temporary databases cleaned up automatically

## CI/CD Integration

### GitHub Actions Example
```yaml
- name: Run Integration Tests
  run: |
    pytest tests/test_data_pipeline_integration.py tests/test_ml_pipeline_integration.py -v
    pytest tests/test_end_to_end.py -v -s --tb=short
```

### Skip Slow Tests
```bash
pytest tests/ -v -m "not slow"
```

## Next Steps

After running integration tests:
1. ✅ Verify all tests pass
2. ✅ Check coverage report
3. ✅ Review any warnings
4. ✅ Run on real data subset
5. ✅ Deploy to staging environment

## Additional Resources

- **Full Documentation:** `INTEGRATION_TESTS_SUMMARY.md`
- **Unit Tests:** `tests/test_stage*_unit.py`
- **Pipeline Guide:** `CLAUDE.md`
- **Implementation Plans:** `STAGE*_IMPLEMENTATION_PLAN.md`

---

**Questions?** Check `INTEGRATION_TESTS_SUMMARY.md` for detailed information.
