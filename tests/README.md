# Unit Tests - Quick Reference

## Overview

This directory contains comprehensive unit tests for the NFL Data Pipeline Stages 2-4.

## Test Files

```
tests/
├── __init__.py                 # Test module initialization
├── test_stage2_unit.py         # Stage 2: Player Lifecycle (14 tests)
├── test_stage3a_unit.py        # Stage 3a: Rolling Statistics (16 tests)
├── test_stage3b_unit.py        # Stage 3b: Matchup Features (13 tests)
├── test_stage3c_unit.py        # Stage 3c: Team Aggregates (18 tests)
├── test_stage4_unit.py         # Stage 4: ML Dataset Assembly (17 tests)
└── README.md                   # This file
```

**Total: 78 test cases**

## Running Tests

### All Tests
```bash
# Using UV
uv run pytest tests/ -v

# Or with Python directly (if pytest installed)
pytest tests/ -v
```

### Specific Stage
```bash
# Stage 2 - Player Lifecycle
pytest tests/test_stage2_unit.py -v

# Stage 3a - Rolling Statistics
pytest tests/test_stage3a_unit.py -v

# Stage 3b - Matchup Features
pytest tests/test_stage3b_unit.py -v

# Stage 3c - Team Aggregates
pytest tests/test_stage3c_unit.py -v

# Stage 4 - ML Dataset Assembly
pytest tests/test_stage4_unit.py -v
```

### Specific Test
```bash
# Run single test
pytest tests/test_stage2_unit.py::TestStage2PlayerLifecycle::test_build_player_lifecycle_table_creates_records -v

# Run test class
pytest tests/test_stage3a_unit.py::TestStage3aRollingStatistics -v
```

### With Coverage
```bash
# Coverage report
pytest tests/ --cov=src --cov-report=html

# View coverage
open htmlcov/index.html
```

## Test Structure

### Fixtures
Each test file uses consistent pytest fixtures:

- **`test_db_path`**: Temporary database path (cleaned up automatically)
- **`test_db`**: Initialized test database with schema
- **`sample_*`**: Sample data specific to each stage

### Example Test
```python
def test_build_player_lifecycle_table_creates_records(self, test_db, sample_player_stats):
    """Test player lifecycle table creation."""
    pipeline = NFLDataPipeline(test_db)

    # Run the method
    rows_created = pipeline.build_player_lifecycle_table()

    # Verify records were created
    assert rows_created == 3
```

## What's Tested

### Stage 2: Player Lifecycle
- ✅ Player lifecycle table creation
- ✅ First/last season calculation
- ✅ Career statistics tracking
- ✅ Roster snapshots with JSON structure
- ✅ Experience classification (rookie/developing/veteran)
- ✅ Confidence scores (0.6, 0.8, 1.0)

### Stage 3a: Rolling Statistics
- ✅ Rolling windows (3, 5, 10 games)
- ✅ Position-specific stat selection (QB vs RB vs WR)
- ✅ Trend calculation (REGR_SLOPE)
- ✅ JSON structure validation
- ✅ Temporal consistency (no data leakage)

### Stage 3b: Matchup Features
- ✅ Rest days calculation from schedules
- ✅ Divisional game detection
- ✅ Opponent history aggregation
- ✅ Home/away splits
- ✅ Matchup context features

### Stage 3c: Team Aggregates
- ✅ Team aggregates from play-by-play
- ✅ Offensive/Defensive EPA metrics
- ✅ Success rate calculation
- ✅ Fallback to team_stats (no PBP)
- ✅ Red zone and third down efficiency

### Stage 4: ML Dataset Assembly
- ✅ Feature combination across tables
- ✅ Numerical features array (47 elements)
- ✅ Data quality scoring
- ✅ Completeness and outlier detection
- ✅ Target creation from actual stats
- ✅ Temporal consistency validation
- ✅ Quality filtering (< 0.5 removed)

## Edge Cases Covered

- Empty database (no data)
- Single game players
- Missing/NULL values
- Insufficient game history
- Zero stats
- Extreme outliers
- First game of season
- Short week games
- Future data leakage prevention

## Test Isolation

Each test:
1. Creates its own temporary database
2. Inserts minimal sample data
3. Runs the test
4. Automatically cleans up

**No cross-test dependencies** - tests can run in any order.

## Performance

Tests are designed for speed:
- Use minimal sample data (2-3 players, 2-10 weeks)
- Temporary in-memory databases where possible
- Fast execution (~seconds, not minutes)

## Troubleshooting

### pytest not found
```bash
# Install pytest
uv add pytest --dev
```

### Import errors
```bash
# Run from project root
cd /home/user/nfl
pytest tests/ -v
```

### Database locked
```bash
# Clean up test databases
rm -f tests/test_*.duckdb
```

## Adding New Tests

1. Choose appropriate test file based on stage
2. Add test method to existing test class or create new class
3. Use existing fixtures or create new ones
4. Follow naming convention: `test_<method>_<behavior>`
5. Add docstring describing what's tested

Example:
```python
def test_new_method_behavior(self, test_db, sample_data):
    """Test new method does X correctly."""
    pipeline = NFLDataPipeline(test_db)

    result = pipeline.new_method()

    assert result == expected_value
```

## Documentation

See `UNIT_TESTS_SUMMARY.md` in project root for:
- Detailed test descriptions
- Full coverage report
- Testing limitations
- Maintenance guide

## Quick Stats

- **Total Tests**: 78
- **Total Lines**: 2,766
- **Coverage**: All major Stage 2-4 methods
- **Edge Cases**: Comprehensive
- **Execution Time**: Fast (< 1 minute for all tests)

---

**Need Help?** Check the main documentation or run `pytest tests/ -v` to see all available tests.
