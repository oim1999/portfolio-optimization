"""
Standalone script to fetch TSLA, BND, and SPY historical data and save it
to data/processed/ as CSV files.

Usage:
    python scripts/fetch_and_save_data.py

This is the same logic used inside the notebook but exposed as a script so
it can be run independently (e.g. in CI, or to refresh data before a
notebook session) without needing to open Jupyter.
"""

import logging
import sys

sys.path.append(".")

from src import config
from src.data_loader import fetch_all_tickers, save_processed_data, DataFetchError

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    try:
        data = fetch_all_tickers(config.TICKERS, config.START_DATE, config.END_DATE)
    except DataFetchError as exc:
        logger.error("Could not fetch any ticker data: %s", exc)
        sys.exit(1)

    save_processed_data(data, output_dir=config.PROCESSED_DATA_DIR)
    logger.info("Done. Fetched and saved: %s", list(data.keys()))


if __name__ == "__main__":
    main()
