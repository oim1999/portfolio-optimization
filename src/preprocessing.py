"""
Data cleaning, type-checking, and feature engineering utilities.

These functions are intentionally pure (DataFrame in, DataFrame/Series out)
so they are easy to unit test and reuse across the notebook, scripts, and
future modeling stages.
"""

import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

EXPECTED_NUMERIC_COLUMNS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]


def check_dtypes(df: pd.DataFrame) -> pd.Series:
    """Return the dtype of each column for a quick sanity check."""
    return df.dtypes


def enforce_numeric_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Coerce OHLCV columns to numeric, turning any unparsable values into NaN
    (which are then handled explicitly by `handle_missing_values`, rather
    than silently propagating bad data).
    """
    df = df.copy()
    for col in EXPECTED_NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def summarize_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Return count and percentage of missing values per column."""
    missing_count = df.isna().sum()
    missing_pct = (missing_count / len(df) * 100).round(3)
    return pd.DataFrame({"missing_count": missing_count, "missing_pct": missing_pct})


def handle_missing_values(df: pd.DataFrame, method: str = "ffill") -> pd.DataFrame:
    """
    Handle missing values in a price series.

    Parameters
    ----------
    method : {"ffill", "interpolate", "drop"}
        - "ffill": forward-fill (standard for financial time series - carries
          the last known price forward across non-trading days / gaps).
        - "interpolate": linear interpolation between known points.
        - "drop": drop rows containing any missing values.

    Notes
    -----
    Forward-fill is the conventional default for daily price data because it
    reflects the fact that a security's value doesn't reset to zero or
    become undefined between observations - the last traded price is a
    reasonable stand-in. It is applied only after the DataFrame has been
    reindexed to a continuous business-day calendar; on the raw yfinance
    output there should be very few, if any, gaps.
    """
    df = df.copy()

    if method == "ffill":
        df = df.ffill().bfill()  # bfill only catches a possible leading NaN
    elif method == "interpolate":
        df = df.interpolate(method="linear", limit_direction="both")
    elif method == "drop":
        df = df.dropna()
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'ffill', 'interpolate', or 'drop'.")

    return df


def basic_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Return descriptive statistics (mean, std, min, max, quartiles) for numeric columns."""
    return df.describe()


def calculate_daily_returns(df: pd.DataFrame, price_col: str = "Adj Close") -> pd.Series:
    """Calculate simple daily percentage returns from a price series."""
    return df[price_col].pct_change().rename("daily_return")


def calculate_rolling_volatility(returns: pd.Series, window: int = 21) -> pd.Series:
    """
    Calculate rolling standard deviation of returns as a volatility proxy.
    Default window of 21 trading days approximates one calendar month.
    """
    return returns.rolling(window=window).std().rename(f"rolling_vol_{window}d")


def calculate_rolling_mean(series: pd.Series, window: int = 21) -> pd.Series:
    """Calculate rolling mean, e.g. for smoothing price or return series."""
    return series.rolling(window=window).mean().rename(f"rolling_mean_{window}d")


def detect_outliers_zscore(returns: pd.Series, z_threshold: float = 3.0) -> pd.DataFrame:
    """
    Flag days with unusually high or low returns using a z-score threshold.

    Returns
    -------
    pd.DataFrame
        Subset of the input series (as a DataFrame) where |z-score| exceeds
        the threshold, alongside the computed z-score, sorted by date.
    """
    mean = returns.mean()
    std = returns.std()
    z_scores = (returns - mean) / std

    outliers = pd.DataFrame({"daily_return": returns, "z_score": z_scores})
    outliers = outliers[outliers["z_score"].abs() > z_threshold]
    return outliers.sort_index()


def clean_pipeline(
    raw_df: pd.DataFrame,
    missing_method: str = "ffill",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convenience wrapper running the full cleaning pipeline on a single
    ticker's raw DataFrame.

    Returns
    -------
    (cleaned_df, missing_value_report)
    """
    df = enforce_numeric_types(raw_df)
    missing_report = summarize_missing_values(df)
    df = handle_missing_values(df, method=missing_method)
    return df, missing_report
