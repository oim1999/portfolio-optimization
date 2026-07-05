# Time Series Forecasting for Portfolio Management Optimization

Project for **GMF Investments** — a personalized portfolio advisory firm —
applying time series forecasting to historical financial data to inform
portfolio management decisions across three assets with distinct risk
profiles: **TSLA** (high-growth equity), **BND** (investment-grade bonds),
and **SPY** (broad market index).

> **Note on modeling philosophy:** Per the Efficient Market Hypothesis,
> predicting exact future prices from historical price data alone is
> extremely difficult. The forecasting models in this project are built to
> characterize trend and volatility as **one input among several** into
> GMF's broader portfolio decision-making process — not as a standalone
> price-prediction guarantee.

## Project Status

This repository currently contains the **interim submission**:
- ✅ Task 1 complete: data extraction, cleaning, EDA, stationarity testing, risk metrics
- 🔄 Task 2 in progress: chronological train/test split + initial ARIMA model implemented
- ⏳ Tasks 3–5 (future forecasting, MPT/Efficient Frontier, backtesting): planned for final submission

## Data Sources

All price data is pulled live via the [`yfinance`](https://pypi.org/project/yfinance/)
Python library (Yahoo Finance), covering **January 1, 2015 – June 30, 2026**:

| Asset | Ticker | Description | Risk Profile |
|---|---|---|---|
| Tesla | `TSLA` | High-growth consumer discretionary stock | High risk, high potential return |
| Vanguard Total Bond Market ETF | `BND` | U.S. investment-grade bonds | Low risk, stability and income |
| S&P 500 ETF | `SPY` | Broad U.S. equity market index | Moderate risk, broad market exposure |

Fields used: `Open`, `High`, `Low`, `Close`, `Adj Close`, `Volume`.

## Project Structure

```
portfolio-optimization/
├── .github/workflows/unittests.yml   # CI: lint + pytest on push/PR
├── .vscode/settings.json
├── .gitignore
├── requirements.txt
├── README.md
├── data/                   # generated CSVs (git-ignored)
├── notebooks/
│   ├── 01_task1_eda_and_task2_initial.ipynb
│   └── README.md
├── src/                              # reusable, unit-tested logic
│   ├── config.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── eda.py
│   ├── stationarity.py
│   ├── risk_metrics.py
│   └── modeling.py
├── scripts/
│   └── fetch_and_save_data.py
└── tests/                            # pytest unit tests (mocked, no network needed)
    ├── test_data_loader.py
    ├── test_preprocessing.py
    ├── test_risk_metrics.py
    └── test_stationarity.py
```

## Setup Instructions

1. **Clone the repository and create a virtual environment:**
   ```bash
   git clone https://github.com/oim1999/portfolio-optimization.git
   cd portfolio-optimization
   py -3.11 -m venv .venv
   .venv\Scripts\activate
   python.exe -m pip install --upgrade pip
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the unit test suite** (no network access required — all external
   calls are mocked):
   ```bash
   pytest tests/ -v
   ```

4. **Fetch data and run the notebook:**
   ```bash
   python scripts/fetch_and_save_data.py     # optional: pre-fetch and cache CSVs
   ```
   Run all cells top to bottom. This requires internet access to Yahoo Finance.

## Methodology Summary (Task 1)

- **Cleaning:** OHLCV columns coerced to numeric types; missing values
  handled via forward-fill (standard for daily price series — carries the
  last traded price forward across gaps) with a backward-fill safety net
  for any leading NaNs.
- **EDA:** Closing price trends, daily percentage returns, rolling
  mean/volatility (21-trading-day window), return distributions, and
  z-score-based outlier detection (|z| > 3).
- **Stationarity:** Augmented Dickey-Fuller (ADF) test applied to both
  price levels (expected non-stationary) and daily returns (expected
  stationary), confirming the need for differencing before ARIMA modeling.
- **Risk metrics:** 95% historical and parametric Value at Risk (VaR), and
  annualized Sharpe Ratio, computed per asset.

## Methodology Summary (Task 2 — Initial Progress)

- **Train/test split:** Chronological (not random) — trains on
  2015-01-01–2024-12-31, tests on 2025-01-01–2026-06-30 — to preserve
  temporal order and avoid look-ahead bias.
- **Initial model:** ARIMA via `pmdarima.auto_arima`, fit on the training
  set and evaluated against the held-out test set using MAE, RMSE, and
  MAPE.

## Roadmap

- **Task 2 (complete):** Add and tune an LSTM model; formally compare
  against ARIMA/SARIMA.
- **Task 3:** Generate 6–12 month forward forecasts with confidence
  intervals; assess how interval width evolves over the horizon.
- **Task 4:** Build the Efficient Frontier using Modern Portfolio Theory
  (`PyPortfolioOpt`); identify the Maximum Sharpe Ratio and Minimum
  Volatility portfolios; recommend final weights.
- **Task 5:** Backtest the recommended portfolio against a static 60%
  SPY / 40% BND benchmark over the most recent ~1 year of data.

## Testing & CI

Unit tests live in `tests/` and mock all external API calls (`yfinance`),
so they run deterministically with no network dependency. GitHub Actions
(`.github/workflows/unittests.yml`) runs linting and the full test suite on
every push/PR to `main`/`develop`.
