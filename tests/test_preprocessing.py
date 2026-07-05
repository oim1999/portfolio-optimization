import numpy as np
import pandas as pd
import pytest

from src.preprocessing import (
    calculate_daily_returns,
    calculate_rolling_volatility,
    detect_outliers_zscore,
    enforce_numeric_types,
    handle_missing_values,
    summarize_missing_values,
)


@pytest.fixture
def sample_df():
    dates = pd.date_range("2024-01-01", periods=10, freq="B")
    df = pd.DataFrame(
        {
            "Open": [10, 11, 12, np.nan, 14, 15, 16, 17, 18, 19],
            "High": range(10, 20),
            "Low": range(10, 20),
            "Close": [10, 11, 12, 13, 14, 15, 16, 17, 18, 100],  # last value is an outlier
            "Adj Close": [10, 11, 12, 13, 14, 15, 16, 17, 18, 100],
            "Volume": [1000] * 10,
        },
        index=dates,
    )
    return df


def test_enforce_numeric_types(sample_df):
    df = sample_df.copy()
    df["Close"] = df["Close"].astype(str)
    result = enforce_numeric_types(df)
    assert pd.api.types.is_numeric_dtype(result["Close"])


def test_summarize_missing_values(sample_df):
    report = summarize_missing_values(sample_df)
    assert report.loc["Open", "missing_count"] == 1


def test_handle_missing_values_ffill(sample_df):
    cleaned = handle_missing_values(sample_df, method="ffill")
    assert cleaned["Open"].isna().sum() == 0
    # Forward fill: NaN at index 3 should equal value at index 2 (12)
    assert cleaned["Open"].iloc[3] == 12


def test_handle_missing_values_invalid_method(sample_df):
    with pytest.raises(ValueError):
        handle_missing_values(sample_df, method="not_a_method")


def test_calculate_daily_returns(sample_df):
    returns = calculate_daily_returns(sample_df, price_col="Adj Close")
    assert returns.iloc[0] != returns.iloc[0]  # first value should be NaN
    assert round(returns.iloc[1], 4) == round((11 - 10) / 10, 4)


def test_calculate_rolling_volatility(sample_df):
    returns = calculate_daily_returns(sample_df)
    vol = calculate_rolling_volatility(returns, window=3)
    assert vol.name == "rolling_vol_3d"
    assert len(vol) == len(returns)


def test_detect_outliers_zscore(sample_df):
    returns = calculate_daily_returns(sample_df)
    outliers = detect_outliers_zscore(returns, z_threshold=1.5)
    assert not outliers.empty
    # the huge jump to 100 should be flagged
    assert outliers["daily_return"].max() > 1.0
