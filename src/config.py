"""
Central configuration for the GMF Investments portfolio optimization project.

Keeping these values in one place means every notebook/script pulls from the
same source of truth for tickers, date ranges, and the train/test cutoff.
"""

from datetime import date

# --- Assets ------------------------------------------------------------
TICKERS = ["TSLA", "BND", "SPY"]

TICKER_METADATA = {
    "TSLA": {
        "name": "Tesla Inc.",
        "sector": "Consumer Discretionary (Automobile Manufacturing)",
        "risk_profile": "High risk, high potential return",
    },
    "BND": {
        "name": "Vanguard Total Bond Market ETF",
        "sector": "Fixed Income (U.S. investment-grade bonds)",
        "risk_profile": "Low risk, stability and income",
    },
    "SPY": {
        "name": "SPDR S&P 500 ETF Trust",
        "sector": "Broad Market Index",
        "risk_profile": "Moderate risk, broad market exposure",
    },
}

# --- Date ranges ---------------------------------------------------------
START_DATE = "2015-01-01"
END_DATE = "2026-06-30"

# Chronological train/test split for Task 2 modeling.
# Rationale documented in README / report: train on the bulk of history,
# test on the most recent ~1.5 years so evaluation reflects genuinely
# unseen, out-of-sample market conditions.
TRAIN_START = "2015-01-01"
TRAIN_END = "2024-12-31"
TEST_START = "2025-01-01"
TEST_END = "2026-06-30"

# Backtesting window (Task 5) - held out from all model training.
BACKTEST_START = "2025-01-01"
BACKTEST_END = "2026-01-31"

# --- Risk metric parameters ---------------------------------------------
RISK_FREE_RATE_ANNUAL = 0.02  # Approximate short-term T-bill rate assumption
TRADING_DAYS_PER_YEAR = 252
VAR_CONFIDENCE_LEVEL = 0.95

# --- Paths -----------------------------------------------------------
RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"

TODAY = date.today().isoformat()
