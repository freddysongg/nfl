# Troubleshooting Guide

This document covers common issues and their solutions when working with the NFL Predictor project.

## Data Pipeline Issues

### 403 Forbidden Error When Downloading Data

**Symptoms:**
```
Failed to download https://github.com/nflverse/nflverse-data/releases/download/...
403 Client Error: Forbidden for url: https://release-assets.githubusercontent.com/...
```

**Root Cause:**
The nflverse data is hosted as GitHub release assets, which are served from Azure Blob Storage (`release-assets.githubusercontent.com`). Some environments, particularly:
- Sandboxed/containerized environments
- Corporate networks with restrictive firewalls
- VPNs or proxies that filter traffic
- Some cloud compute instances

may block access to Azure Blob Storage, resulting in 403 Forbidden errors when trying to download the parquet files.

**Workarounds:**

1. **Run from a different environment** (Recommended)
   - Run the data pipeline from your local machine or a non-restricted environment
   - The database file (`nfl_predictions.duckdb`) can be copied to the restricted environment after data is loaded

2. **Manual data download**
   - Download the parquet files manually from GitHub: https://github.com/nflverse/nflverse-data/releases
   - Import them into the database using a custom script

3. **Network configuration**
   - If behind a corporate firewall, request that `release-assets.githubusercontent.com` and `*.blob.core.windows.net` be whitelisted
   - Try using a different network or VPN

4. **Use cached data**
   - If running in a shared environment, check if another user has already populated a database you can copy

**Verification:**
Test if you can access GitHub release assets:
```bash
curl -I "https://github.com/nflverse/nflverse-data/releases/download/stats_player/stats_player_week_2025.parquet"
```

If this returns a `302 Found` redirect, but subsequent Python requests fail with 403, this confirms the Azure Blob Storage access issue.

### Database Exists But Is Empty

**Symptoms:**
```
Error: Database exists but contains no training data
```

**Cause:**
The database schema was created (`setup_database.py` ran successfully) but the data pipeline failed to load data, typically due to the 403 error above.

**Solution:**
Follow the workarounds for the 403 Forbidden error above.

### nflreadpy Version Issues

**Symptoms:**
- Import errors
- Missing functions
- Unexpected API behavior

**Solution:**
Ensure you're using the correct version of nflreadpy:
```bash
uv sync  # Reinstall all dependencies
uv pip list | grep nflreadpy  # Check version (should be 0.1.2+)
```

## Training Issues

### No Data for Position

**Symptoms:**
```
ERROR [src.training.trainer] Error loading training data: No data found for position=QB, seasons=2021-2025
```

**Cause:**
The `ml_training_features` table is empty because the data pipeline never completed successfully (see 403 error above).

**Solution:**
Load data into the database using one of the workarounds above.

### Model Training Crashes

**Symptoms:**
- Out of memory errors
- Training hangs
- DuckDB connection errors

**Solutions:**
1. Reduce the number of seasons being trained on
2. Train one position at a time instead of `--all-positions`
3. Reduce the number of Optuna trials (`--n-trials 20` instead of 50+)
4. Check available memory and close other applications

## MLflow Issues

### MLflow UI Not Accessible

**Symptom:**
Can't access MLflow tracking UI

**Solution:**
Start the MLflow UI:
```bash
mlflow ui --backend-store-uri sqlite:///mlruns.db
```

Access at: http://localhost:5000

### Permission Denied Errors with MLflow

**Solution:**
Check permissions on the `mlruns` directory and `mlruns.db` file:
```bash
chmod -R 755 mlruns/
chmod 644 mlruns.db
```

## Development Issues

### Import Errors

**Symptom:**
```
ModuleNotFoundError: No module named 'src'
```

**Solution:**
Ensure you're running Python from the project root:
```bash
uv run python train_models.py  # Correct
cd src && python ../train_models.py  # Wrong
```

### Database Lock Errors

**Symptom:**
```
database is locked
```

**Solution:**
- Close all database connections
- Check for running processes that may have the database open
- Restart your environment if necessary

## Getting Help

If you continue to experience issues:

1. Check the logs for detailed error messages
2. Verify your environment meets the requirements (Python 3.10+, `uv` installed)
3. Try running with verbose logging:
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```
4. Open an issue on GitHub with:
   - Full error traceback
   - Environment details (OS, Python version, network type)
   - Steps to reproduce

## Known Limitations

- **Data pipeline requires external network access**: Cannot run fully offline
- **Large memory footprint**: Training all positions requires significant RAM (8GB+ recommended)
- **Long training times**: Hyperparameter tuning with 50+ trials can take hours
- **Platform-specific issues**: Some DuckDB extensions (like Polars) may not be available on all platforms
