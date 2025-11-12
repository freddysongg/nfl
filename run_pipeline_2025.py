"""
Quick script to run the full NFL data pipeline for 2025 season only
This reduces download volume and may avoid rate limiting issues
"""

from src.data_pipeline import NFLDataPipeline

if __name__ == "__main__":
    print("Starting NFL data pipeline for 2025 season only...")
    print("This will help avoid GitHub rate limiting issues")

    pipeline = NFLDataPipeline()
    # Try with just 2025 season to minimize downloads
    pipeline.run_full_pipeline(seasons=[2025])

    print("\n✅ Pipeline complete! Database is ready for training.")
