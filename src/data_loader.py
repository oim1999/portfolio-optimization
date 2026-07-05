"""
Data extraction utilities for pulling historical price data via yfinance.

Design notes:
- Each ticker is fetched independently so a single API failure (rate limit,
  network blip, delisting, etc.) does not abort the entire batch.
- Data is returned with a MultiIndex-free, tidy structure: one DataFrame per
  ticker, plus a helper to combine them into a single long-format frame
  tagged with an `asset` column for downstream analysis.
"""

import logging
import time
from typing import Dict, List, Optional

import pandas as pd

try:
    import yfinance as yf
except ImportError:  # pragma: no cover - import guard for environments
    yf = None  # without yfinance installed (e.g. this authoring sandbox)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class DataFetchError(Exception):
    """Raised when a ticker cannot be fetched after all retries are exhausted."""


def fetch_ticker_data(
    ticker: str,
    start: str,
    end: str,
    max_retries: int = 3,
    retry_delay_seconds: float = 2.0,
) -> pd.DataFrame:
    """
    Fetch historical OHLCV data for a single ticker from Yahoo Finance.

    Parameters
    ----------
    ticker : str
        Ticker symbol, e.g. "TSLA".
    start, end : str
        Date strings in "YYYY-MM-DD" format.
    max_retries : int
        Number of attempts before giving up.
    retry_delay_seconds : float
        Delay between retries (simple linear backoff).

    Returns
    -------
    pd.DataFrame
        Indexed by Date, with columns [Open, High, Low, Close, Adj Close, Volume].

    Raises
    ------
    DataFetchError
        If the ticker cannot be fetched after `max_retries` attempts, or if
        the API returns an empty result (e.g. invalid ticker / no data in range).
    """
    if yf is None:
        raise DataFetchError(
            "yfinance is not installed in this environment. "
            "Run `pip install -r requirements.txt` before executing this function."
        )

    last_exception: Optional[Exception] = None

    for attempt in range(1, max_retries + 1):
        try:
            logger.info("Fetching %s (attempt %d/%d)...", ticker, attempt, max_retries)
            df = yf.download(
                ticker,
                start=start,
                end=end,
                auto_adjust=False,  # keep raw Close AND Adj Close as separate columns
                progress=False,
            )

            if df is None or df.empty:
                raise DataFetchError(f"No data returned for ticker '{ticker}' in range {start}..{end}.")

            # yfinance sometimes returns a MultiIndex column structure even
            # for a single ticker depending on version; normalize it.
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df.index.name = "Date"
            logger.info("Successfully fetched %d rows for %s.", len(df), ticker)
            return df

        except Exception as exc:  # noqa: BLE001 - we want to catch and retry broadly here
            last_exception = exc
            logger.warning("Attempt %d for %s failed: %s", attempt, ticker, exc)
            if attempt < max_retries:
                time.sleep(retry_delay_seconds * attempt)

    raise DataFetchError(
        f"Failed to fetch data for '{ticker}' after {max_retries} attempts."
    ) from last_exception


def fetch_all_tickers(
    tickers: List[str],
    start: str,
    end: str,
) -> Dict[str, pd.DataFrame]:
    """
    Fetch historical data for multiple tickers, continuing past individual failures.

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping of ticker -> DataFrame for every ticker that was fetched
        successfully. Tickers that failed are logged and skipped, not silently
        dropped.

    Raises
    ------
    DataFetchError
        If ALL tickers fail to fetch (nothing usable was retrieved).
    """
    results: Dict[str, pd.DataFrame] = {}
    failures: Dict[str, str] = {}

    for ticker in tickers:
        try:
            results[ticker] = fetch_ticker_data(ticker, start, end)
        except DataFetchError as exc:
            failures[ticker] = str(exc)
            logger.error("Skipping '%s' due to fetch failure: %s", ticker, exc)

    if not results:
        raise DataFetchError(f"All ticker fetches failed: {failures}")

    if failures:
        logger.warning(
            "Completed fetch with partial failures. Missing tickers: %s",
            list(failures.keys()),
        )

    return results


def combine_to_long_format(ticker_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Combine a dict of per-ticker DataFrames into one long-format DataFrame
    with an `asset` identifier column, suitable for groupby-based analysis.
    """
    frames = []
    for ticker, df in ticker_data.items():
        tagged = df.copy()
        tagged["asset"] = ticker
        frames.append(tagged)

    combined = pd.concat(frames, axis=0)
    combined = combined.reset_index().sort_values(["asset", "Date"]).reset_index(drop=True)
    return combined


def save_processed_data(ticker_data: Dict[str, pd.DataFrame], output_dir: str = "data/processed") -> None:
    """Persist each ticker's cleaned data to CSV for reproducibility."""
    import os

    os.makedirs(output_dir, exist_ok=True)
    for ticker, df in ticker_data.items():
        path = os.path.join(output_dir, f"{ticker}.csv")
        df.to_csv(path)
        logger.info("Saved %s to %s", ticker, path)
