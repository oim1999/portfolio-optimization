"""
Unit tests for src/data_loader.py.

These tests mock out yfinance entirely so CI can run them with no network
access and no dependency on live market data availability.
"""

from unittest.mock import patch

import pandas as pd
import pytest

from src.data_loader import (
    DataFetchError,
    combine_to_long_format,
    fetch_all_tickers,
    fetch_ticker_data,
)


def _make_fake_price_df(n_rows: int = 10) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=n_rows, freq="B")
    dates.name = "Date"
    return pd.DataFrame(
        {
            "Open": range(n_rows),
            "High": range(n_rows),
            "Low": range(n_rows),
            "Close": range(n_rows),
            "Adj Close": range(n_rows),
            "Volume": [1000] * n_rows,
        },
        index=dates,
    )


@patch("src.data_loader.yf")
def test_fetch_ticker_data_success(mock_yf):
    mock_yf.download.return_value = _make_fake_price_df()

    df = fetch_ticker_data("TSLA", "2024-01-01", "2024-01-31")

    assert not df.empty
    assert list(df.columns) == ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    assert df.index.name == "Date"


@patch("src.data_loader.yf")
def test_fetch_ticker_data_empty_result_raises(mock_yf):
    mock_yf.download.return_value = pd.DataFrame()

    with pytest.raises(DataFetchError):
        fetch_ticker_data("BADTICKER", "2024-01-01", "2024-01-31", max_retries=1, retry_delay_seconds=0)


@patch("src.data_loader.yf")
def test_fetch_ticker_data_retries_then_succeeds(mock_yf):
    # First call raises, second call succeeds.
    mock_yf.download.side_effect = [Exception("network blip"), _make_fake_price_df()]

    df = fetch_ticker_data("TSLA", "2024-01-01", "2024-01-31", max_retries=2, retry_delay_seconds=0)
    assert not df.empty
    assert mock_yf.download.call_count == 2


@patch("src.data_loader.yf")
def test_fetch_all_tickers_partial_failure(mock_yf):
    def side_effect(ticker, **kwargs):
        if ticker == "BAD":
            return pd.DataFrame()
        return _make_fake_price_df()

    mock_yf.download.side_effect = side_effect

    results = fetch_all_tickers(["TSLA", "BAD", "SPY"], "2024-01-01", "2024-01-31")

    assert "TSLA" in results
    assert "SPY" in results
    assert "BAD" not in results


@patch("src.data_loader.yf")
def test_fetch_all_tickers_all_fail_raises(mock_yf):
    mock_yf.download.return_value = pd.DataFrame()

    with pytest.raises(DataFetchError):
        fetch_all_tickers(["BAD1", "BAD2"], "2024-01-01", "2024-01-31")


def test_combine_to_long_format():
    data = {
        "TSLA": _make_fake_price_df(5),
        "BND": _make_fake_price_df(5),
    }
    combined = combine_to_long_format(data)

    assert "asset" in combined.columns
    assert set(combined["asset"].unique()) == {"TSLA", "BND"}
    assert len(combined) == 10
