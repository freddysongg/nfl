"""
Quick script to run the full NFL data pipeline
Loads data for the default seasons (2021-2025) and executes all 4 stages
"""

from src.data_pipeline import NFLDataPipeline

if __name__ == "__main__":
    print("Starting NFL data pipeline...")
    print("This will load data for seasons 2021-2025 and execute all 4 stages")
    print("This may take a while...")

    pipeline = NFLDataPipeline()
    pipeline.run_full_pipeline()

    print("\n✅ Pipeline complete! Database is ready for training.")
