"""
Core training and backtest runner for Copper Brain v2.
"""
import os
from datetime import datetime
from typing import Tuple

import pandas as pd
import yfinance as yf

from features import engineer_all_features
from model_utils import walk_forward_backtest


DATA_DIR = "data"
OUTPUTS_DIR = "outputs"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)


def download_ticker(ticker: str, start: str, end: str) -> pd.DataFrame:
    print(f"Downloading {ticker} from yfinance...")
    df = yf.download(ticker, start=start, end=end, interval="1d", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index)
    df.index = df.index.tz_localize(None) if df.index.tz is not None else df.index
    price_col = "Adj Close" if "Adj Close" in df.columns else "Close"
    df = df.rename(columns={price_col: "close"})
    df = df[["close", "Open", "High", "Low", "Volume"]].dropna()
    return df


def fetch_all_data(start: str, end: str):
    # Copper
    copper = download_ticker("HG=F", start, end)

    # DXY: try DX-Y.NYB then fallback to ^DXY
    try:
        dxy = download_ticker("DX-Y.NYB", start, end)
    except Exception:
        dxy = download_ticker("^DXY", start, end)

    # Gold
    gold = download_ticker("GC=F", start, end)

    # Save raw data
    copper.to_csv(os.path.join(DATA_DIR, "copper_raw.csv"))
    dxy.to_csv(os.path.join(DATA_DIR, "dxy_raw.csv"))
    gold.to_csv(os.path.join(DATA_DIR, "gold_raw.csv"))

    return copper, dxy, gold


def main():
    start = "1990-01-01"
    end = datetime.today().strftime("%Y-%m-%d")

    copper, dxy, gold = fetch_all_data(start, end)

    print("Engineering features...")
    df, feature_cols = engineer_all_features(copper, dxy, gold, horizon_days=21)

    target_col = "target_21d"

    print(f"Features shape: {df.shape}, features count: {len(feature_cols)}")

    print("Running walk-forward backtest...")
    backtest_df, summary = walk_forward_backtest(
        df=df,
        feature_cols=feature_cols,
        target_col=target_col,
        train_years=5,
        test_step_days=21,
        model_params={
            "n_estimators": 500,
            "max_depth": 6,
            "learning_rate": 0.03,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
        },
        threshold=0.45,
        outputs_path=OUTPUTS_DIR,
    )

    print("Backtest complete. Summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    print(f"Saved backtest CSV to {os.path.join(OUTPUTS_DIR, 'backtest_results.csv')}")


if __name__ == "__main__":
    main()
"""
Copper Brain v2 - Production-Grade Copper Direction Predictor
=============================================================
Walk-forward validated XGBoost model for predicting 21-day copper price direction.

Features:
- Binary classification (direction prediction)
- Walk-forward rolling validation
- Multi-asset alpha features (DXY, Gold)
- Comprehensive feature engineering
- Production-ready backtesting

Usage:
    python copper_brain_v2.py
"""

import os
import sys
from datetime import datetime
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

from features import engineer_all_features
from model_utils import (
    walk_forward_validation,
    calculate_overall_metrics,
    save_backtest_results,
    train_final_model
)


# =========================
#  CONFIGURATION
# =========================

# Data parameters
START_DATE = "1990-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")

# Asset tickers (free data only)
COPPER_TICKER = "HG=F"      # COMEX Copper Futures
DXY_TICKER = "DX-Y.NYB"     # US Dollar Index
GOLD_TICKER = "GC=F"        # COMEX Gold Futures

# Forecasting parameters
FORECAST_HORIZON_DAYS = 21  # Predict 21-day direction

# Walk-forward validation parameters
TRAIN_WINDOW_YEARS = 5      # 5-year training window
TEST_STEP_DAYS = 21         # 21-day test steps
PROBABILITY_THRESHOLD = 0.45  # Classification threshold

# Directories
DATA_DIR = "data"
OUTPUT_DIR = "outputs"


# =========================
#  DATA FETCHING
# =========================

def fetch_asset_data(ticker: str, start: str, end: str, name: str) -> pd.DataFrame:
    """
    Fetch daily data for a single asset from yfinance.
    """
    print(f"Downloading {name} data ({ticker}) from yfinance...")

    try:
        df = yf.download(ticker, start=start, end=end, interval="1d", auto_adjust=False, progress=False)

        if df.empty:
            raise RuntimeError(f"No data downloaded for {ticker}")

        # Handle MultiIndex columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Ensure DatetimeIndex
        df.index = pd.to_datetime(df.index)
        df.index = df.index.tz_localize(None) if df.index.tz is not None else df.index

        # Use Adjusted Close if available, otherwise Close
        price_col = "Adj Close" if "Adj Close" in df.columns else "Close"
        df = df.rename(columns={price_col: "close"})

        # Keep required columns
        required_cols = ["close", "Open", "High", "Low", "Volume"]
        df = df[required_cols]
        df = df.dropna()

        print(f"  Downloaded {len(df)} days of {name} data")

        return df

    except Exception as e:
        print(f"  Error downloading {name} data: {str(e)}")
        return pd.DataFrame()


def fetch_all_data(start_date: str, end_date: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Fetch data for all required assets.
    """
    print("Fetching data for all assets...")

    # Fetch copper data
    copper_df = fetch_asset_data(COPPER_TICKER, start_date, end_date, "Copper")

    # Fetch DXY data
    dxy_df = fetch_asset_data(DXY_TICKER, start_date, end_date, "DXY")
    if dxy_df.empty:
        # Fallback to ^DXY
        print("  Trying fallback ticker ^DXY...")
        dxy_df = fetch_asset_data("^DXY", start_date, end_date, "DXY")

    # Fetch gold data
    gold_df = fetch_asset_data(GOLD_TICKER, start_date, end_date, "Gold")

    # Validate data availability
    if copper_df.empty:
        raise RuntimeError("Copper data is required but could not be fetched")

    if dxy_df.empty:
        print("  Warning: DXY data not available - some alpha features will be missing")

    if gold_df.empty:
        print("  Warning: Gold data not available - some alpha features will be missing")

    return copper_df, dxy_df, gold_df


def save_raw_data(copper_df: pd.DataFrame, dxy_df: pd.DataFrame, gold_df: pd.DataFrame):
    """
    Save raw data to disk for caching.
    """
    os.makedirs(DATA_DIR, exist_ok=True)

    copper_df.to_csv(os.path.join(DATA_DIR, "copper_raw.csv"))
    if not dxy_df.empty:
        dxy_df.to_csv(os.path.join(DATA_DIR, "dxy_raw.csv"))
    if not gold_df.empty:
        gold_df.to_csv(os.path.join(DATA_DIR, "gold_raw.csv"))

    print(f"Saved raw data to {DATA_DIR}/")


def load_cached_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load cached data if available.
    """
    copper_path = os.path.join(DATA_DIR, "copper_raw.csv")
    dxy_path = os.path.join(DATA_DIR, "dxy_raw.csv")
    gold_path = os.path.join(DATA_DIR, "gold_raw.csv")

    copper_df = pd.read_csv(copper_path, index_col=0, parse_dates=True) if os.path.exists(copper_path) else pd.DataFrame()
    dxy_df = pd.read_csv(dxy_path, index_col=0, parse_dates=True) if os.path.exists(dxy_path) else pd.DataFrame()
    gold_df = pd.read_csv(gold_path, index_col=0, parse_dates=True) if os.path.exists(gold_path) else pd.DataFrame()

    return copper_df, dxy_df, gold_df


# =========================
#  MAIN PIPELINE
# =========================

def main():
    """
    Main execution pipeline.
    """
    print("=" * 60)
    print("COPPER BRAIN v2 - Production Direction Predictor")
    print("=" * 60)
    print(f"Forecast Horizon: {FORECAST_HORIZON_DAYS} days")
    print(f"Training Window: {TRAIN_WINDOW_YEARS} years")
    print(f"Test Step: {TEST_STEP_DAYS} days")
    print(f"Probability Threshold: {PROBABILITY_THRESHOLD}")
    print()

    # Create output directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    # Step 1: Fetch or load data
    print("Step 1: Loading data...")

    # Try to load cached data first
    try:
        copper_df, dxy_df, gold_df = load_cached_data()
        if not copper_df.empty:
            print("  Loaded cached data")
        else:
            raise FileNotFoundError
    except (FileNotFoundError, pd.errors.EmptyDataError):
        print("  No cached data found, fetching from yfinance...")
        copper_df, dxy_df, gold_df = fetch_all_data(START_DATE, END_DATE)
        save_raw_data(copper_df, dxy_df, gold_df)

    print(f"  Copper data: {len(copper_df)} days")
    print(f"  DXY data: {len(dxy_df)} days")
    print(f"  Gold data: {len(gold_df)} days")
    print()

    # Step 2: Feature engineering
    print("Step 2: Engineering features...")

    processed_df, feature_cols = engineer_all_features(
        copper_df, dxy_df, gold_df, FORECAST_HORIZON_DAYS
    )

    print(f"  Processed dataset: {len(processed_df)} samples")
    print(f"  Number of features: {len(feature_cols)}")
    print(f"  Target distribution: {processed_df[f'target_{FORECAST_HORIZON_DAYS}d'].value_counts().to_dict()}")
    print()

    # Step 3: Walk-forward validation
    print("Step 3: Running walk-forward validation...")

    backtest_df = walk_forward_validation(
        processed_df,
        feature_cols,
        f"target_{FORECAST_HORIZON_DAYS}d",
        TRAIN_WINDOW_YEARS,
        TEST_STEP_DAYS,
        PROBABILITY_THRESHOLD
    )

    print(f"  Completed {len(backtest_df)} prediction steps")
    print()

    # Step 4: Calculate overall metrics
    print("Step 4: Calculating performance metrics...")

    metrics = calculate_overall_metrics(backtest_df)

    print("  OVERALL PERFORMANCE METRICS:")
    print(f"    Directional Accuracy: {metrics.get('overall_accuracy', 0):.3f}")
    print(f"    Precision: {metrics.get('overall_precision', 0):.3f}")
    print(f"    Recall: {metrics.get('overall_recall', 0):.3f}")
    print(f"    F1 Score: {metrics.get('overall_f1', 0):.3f}")
    print(f"    ROC-AUC: {metrics.get('overall_roc_auc', 0):.3f}")
    print()
    print("  CONFUSION MATRIX:")
    print(f"    True Positives: {metrics.get('true_positives', 0)}")
    print(f"    True Negatives: {metrics.get('true_negatives', 0)}")
    print(f"    False Positives: {metrics.get('false_positives', 0)}")
    print(f"    False Negatives: {metrics.get('false_negatives', 0)}")
    print(f"    Total Predictions: {metrics.get('total_predictions', 0)}")
    print()

    # Step 5: Save results
    print("Step 5: Saving results...")

    save_backtest_results(backtest_df, OUTPUT_DIR)

    # Save metrics summary
    metrics_path = os.path.join(OUTPUT_DIR, "performance_metrics.txt")
    with open(metrics_path, 'w') as f:
        f.write("COPPER BRAIN v2 - PERFORMANCE METRICS\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Training completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("CONFIGURATION:\n")
        f.write(f"  Forecast Horizon: {FORECAST_HORIZON_DAYS} days\n")
        f.write(f"  Training Window: {TRAIN_WINDOW_YEARS} years\n")
        f.write(f"  Test Step: {TEST_STEP_DAYS} days\n")
        f.write(f"  Probability Threshold: {PROBABILITY_THRESHOLD}\n\n")
        f.write("OVERALL METRICS:\n")
        for key, value in metrics.items():
            if isinstance(value, float):
                f.write(f"  {key}: {value:.4f}\n")
            else:
                f.write(f"  {key}: {value}\n")

    print(f"  Saved performance metrics to {metrics_path}")

    # Step 6: Train final model for production
    print("Step 6: Training final production model...")

    final_model, final_threshold = train_final_model(
        processed_df,
        feature_cols,
        f"target_{FORECAST_HORIZON_DAYS}d",
        PROBABILITY_THRESHOLD
    )

    # Save final model (using joblib for XGBoost compatibility)
    import joblib
    model_path = os.path.join(OUTPUT_DIR, "final_model.pkl")
    joblib.dump(final_model, model_path)
    print(f"  Saved final model to {model_path}")

    # Save feature columns for future use
    features_path = os.path.join(OUTPUT_DIR, "feature_columns.pkl")
    joblib.dump(feature_cols, features_path)
    print(f"  Saved feature columns to {features_path}")

    print()
    print("=" * 60)
    print("✅ COPPER BRAIN v2 TRAINING COMPLETE")
    print("=" * 60)
    print()
    print("Next steps:")
    print("1. Review backtest results in outputs/backtest_results.csv")
    print("2. Check performance plots in outputs/backtest_summary_plots.png")
    print("3. Launch dashboard: streamlit run app.py")
    print()
    print("🎯 Target KPI Achieved!" if metrics.get('overall_accuracy', 0) >= 0.60 else "⚠️  Target KPI Not Met - Consider parameter tuning")


if __name__ == "__main__":
    main()