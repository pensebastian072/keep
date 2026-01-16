"""
Copper Brain v2 - Feature Engineering
=====================================
Comprehensive feature engineering for copper price direction prediction.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple


def add_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add momentum-based features to the dataframe.

    Features:
    - ret_1d, ret_2d, ret_3d, ret_5d, ret_10d: Log returns over various periods
    - rolling_max_21d: Rolling maximum price over 21 days
    - rolling_min_21d: Rolling minimum price over 21 days
    """
    df = df.copy()

    # Log returns
    df["ret_1d"] = df["log_price"].diff(1)
    df["ret_2d"] = df["log_price"].diff(2)
    df["ret_3d"] = df["log_price"].diff(3)
    df["ret_5d"] = df["log_price"].diff(5)
    df["ret_10d"] = df["log_price"].diff(10)

    # Rolling max/min
    df["rolling_max_21d"] = df["close"].rolling(21).max()
    df["rolling_min_21d"] = df["close"].rolling(21).min()

    return df


def add_moving_average_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add moving average features to the dataframe.

    Features:
    - ma_5d, ma_10d, ma_20d: Simple moving averages
    - price_over_ma_5d, price_over_ma_10d, price_over_ma_20d: Price to MA ratios
    """
    df = df.copy()

    # Moving averages
    df["ma_5d"] = df["close"].rolling(5).mean()
    df["ma_10d"] = df["close"].rolling(10).mean()
    df["ma_20d"] = df["close"].rolling(20).mean()

    # Price to MA ratios
    df["price_over_ma_5d"] = df["close"] / df["ma_5d"]
    df["price_over_ma_10d"] = df["close"] / df["ma_10d"]
    df["price_over_ma_20d"] = df["close"] / df["ma_20d"]

    return df


def add_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add volatility-based features to the dataframe.

    Features:
    - realized_vol_10d: Realized volatility over 10 days (annualized)
    - realized_vol_21d: Realized volatility over 21 days (annualized)
    """
    df = df.copy()

    # Realized volatility (annualized)
    df["realized_vol_10d"] = df["ret_1d"].rolling(10).std() * np.sqrt(252)
    df["realized_vol_21d"] = df["ret_1d"].rolling(21).std() * np.sqrt(252)

    return df


def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add volume-based features to the dataframe.

    Features:
    - log_volume: Log-transformed volume
    - volume_zscore_21d: Volume z-score over 21 days
    - obv: On-balance volume
    """
    df = df.copy()

    # Log volume
    df["log_volume"] = np.log(df["Volume"].replace(0, np.nan))

    # Volume z-score
    df["volume_zscore_21d"] = (
        (df["Volume"] - df["Volume"].rolling(21).mean()) /
        df["Volume"].rolling(21).std()
    )

    # On-balance volume
    df["obv"] = np.where(
        df["close"] > df["close"].shift(1),
        df["Volume"],
        np.where(df["close"] < df["close"].shift(1), -df["Volume"], 0)
    ).cumsum()

    return df


def add_candle_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add candlestick structure features to the dataframe.

    Features:
    - body: Absolute body size (Close - Open)
    - upper_wick: Upper wick length
    - lower_wick: Lower wick length
    - bullish: 1 if bullish candle, 0 if bearish
    """
    df = df.copy()

    # Candle body
    df["body"] = (df["close"] - df["Open"]).abs()

    # Upper and lower wicks
    df["upper_wick"] = df["High"] - np.maximum(df["Open"], df["close"])
    df["lower_wick"] = np.minimum(df["Open"], df["close"]) - df["Low"]

    # Bullish/bearish indicator
    df["bullish"] = (df["close"] > df["Open"]).astype(int)

    return df


def add_copper_alpha_features(df: pd.DataFrame, dxy_df: pd.DataFrame, gold_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add copper-specific alpha features using DXY and gold data.

    Features:
    - dxy_close: DXY close price
    - dxy_ret_1d, dxy_ret_5d, dxy_ret_21d: DXY returns
    - copper_dxy_corr_21d: Rolling correlation between copper and DXY
    - gold_close: Gold close price
    - copper_gold_ratio: Copper price / Gold price ratio
    """
    df = df.copy()

    # Prepare DXY dataframe: ensure close and log_price exist
    dxy_copy = dxy_df.copy()
    if "close" not in dxy_copy.columns:
        raise KeyError("DXY dataframe missing 'close' column")
    if "log_price" not in dxy_copy.columns:
        dxy_copy["log_price"] = np.log(dxy_copy["close"])

    # Merge DXY data
    df = df.merge(
        dxy_copy[["close", "log_price"]].rename(columns={
            "close": "dxy_close",
            "log_price": "dxy_log_price"
        }),
        left_index=True,
        right_index=True,
        how="left"
    )

    # DXY returns
    df["dxy_ret_1d"] = df["dxy_log_price"].diff(1)
    df["dxy_ret_5d"] = df["dxy_log_price"].diff(5)
    df["dxy_ret_21d"] = df["dxy_log_price"].diff(21)

    # Copper-DXY correlation
    df["copper_dxy_corr_21d"] = df["ret_1d"].rolling(21).corr(df["dxy_ret_1d"])

    # Prepare gold dataframe: ensure close exists and compute log if needed
    gold_copy = gold_df.copy()
    if "close" not in gold_copy.columns:
        raise KeyError("Gold dataframe missing 'close' column")
    if "log_price" not in gold_copy.columns:
        gold_copy["log_price"] = np.log(gold_copy["close"])

    # Merge gold data
    df = df.merge(
        gold_copy[["close"]].rename(columns={"close": "gold_close"}),
        left_index=True,
        right_index=True,
        how="left"
    )

    # Copper-gold ratio
    df["copper_gold_ratio"] = df["close"] / df["gold_close"]

    return df


def create_target_variable(df: pd.DataFrame, horizon_days: int = 21) -> pd.DataFrame:
    """
    Create the binary target variable for direction prediction.

    Target: 1 if future_return_21d > 0, else 0
    """
    df = df.copy()

    # Future return
    df[f"future_return_{horizon_days}d"] = (
        df["log_price"].shift(-horizon_days) - df["log_price"]
    )

    # Binary target (direction)
    df[f"target_{horizon_days}d"] = (
        df[f"future_return_{horizon_days}d"] > 0
    ).astype(int)

    return df


def get_feature_columns(df: pd.DataFrame, target_col: str) -> List[str]:
    """
    Get list of feature columns, excluding target and non-feature columns.
    """
    exclude_cols = [
        target_col,
        "close", "log_price",
        "Open", "High", "Low", "Volume",
        "Adj Close",  # if present
        "future_return_21d", "future_return_42d", "future_return_7d",  # future returns
    ]

    # Also exclude any target columns
    exclude_cols.extend([col for col in df.columns if col.startswith("target_")])

    feature_cols = [col for col in df.columns if col not in exclude_cols]

    return feature_cols


def engineer_all_features(
    copper_df: pd.DataFrame,
    dxy_df: pd.DataFrame,
    gold_df: pd.DataFrame,
    horizon_days: int = 21
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Apply all feature engineering steps and return processed dataframe with features.

    Returns:
        Tuple of (processed_dataframe, feature_columns_list)
    """
    # Start with copper data
    df = copper_df.copy()

    # Add log price
    df["log_price"] = np.log(df["close"])

    # Add all feature groups
    df = add_momentum_features(df)
    df = add_moving_average_features(df)
    df = add_volatility_features(df)
    df = add_volume_features(df)
    df = add_candle_features(df)

    # Add copper-specific alpha features
    df = add_copper_alpha_features(df, dxy_df, gold_df)

    # Create target variable
    df = create_target_variable(df, horizon_days)

    # Drop rows with NaN values
    df = df.dropna()

    # Get feature columns
    target_col = f"target_{horizon_days}d"
    feature_cols = get_feature_columns(df, target_col)

    return df, feature_cols