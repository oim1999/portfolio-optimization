"""
Strategy backtesting utilities for Task 5: simulate a fixed-weight portfolio
over a held-out window, with either a static hold or monthly rebalancing,
and compare it against a benchmark.
"""

import logging
from typing import Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def simulate_buy_and_hold(
    returns_df: pd.DataFrame,
    weights: Dict[str, float],
    initial_value: float = 1.0,
) -> pd.Series:
    """
    Simulate a portfolio that is bought once at the target weights and then
    held with NO rebalancing - individual asset weights drift as prices move.

    Parameters
    ----------
    returns_df : pd.DataFrame
        Daily returns, columns = tickers matching `weights` keys.
    weights : dict[str, float]
        Target weights at t=0. Should sum to ~1.0.

    Returns
    -------
    pd.Series
        Cumulative portfolio value over time, starting at `initial_value`.
    """
    tickers = list(weights.keys())
    w = np.array([weights[t] for t in tickers])

    asset_values = initial_value * w  # dollar value allocated to each asset at t=0
    portfolio_values = [initial_value]

    for _, day_returns in returns_df[tickers].iterrows():
        asset_values = asset_values * (1 + day_returns.values)
        portfolio_values.append(asset_values.sum())

    return pd.Series(portfolio_values[1:], index=returns_df.index, name="portfolio_value")


def simulate_monthly_rebalance(
    returns_df: pd.DataFrame,
    weights: Dict[str, float],
    initial_value: float = 1.0,
) -> pd.Series:
    """
    Simulate a portfolio that is rebalanced back to target weights at the
    start of every calendar month.
    """
    tickers = list(weights.keys())
    w_target = np.array([weights[t] for t in tickers])

    portfolio_value = initial_value
    asset_values = initial_value * w_target
    portfolio_values = []

    current_month = None
    for date, day_returns in returns_df[tickers].iterrows():
        month_key = (date.year, date.month)
        if month_key != current_month:
            # Rebalance to target weights at the start of a new month
            asset_values = portfolio_value * w_target
            current_month = month_key

        asset_values = asset_values * (1 + day_returns.values)
        portfolio_value = asset_values.sum()
        portfolio_values.append(portfolio_value)

    return pd.Series(portfolio_values, index=returns_df.index, name="portfolio_value")


def compute_backtest_metrics(
    cumulative_value: pd.Series,
    risk_free_rate_annual: float = 0.02,
    trading_days_per_year: int = 252,
) -> Dict[str, float]:
    """
    Compute total return, annualized return, Sharpe Ratio, and max drawdown
    from a cumulative portfolio value series.
    """
    daily_returns = cumulative_value.pct_change().dropna()

    total_return = cumulative_value.iloc[-1] / cumulative_value.iloc[0] - 1
    n_days = len(cumulative_value)
    annualized_return = (1 + total_return) ** (trading_days_per_year / n_days) - 1

    daily_rf = risk_free_rate_annual / trading_days_per_year
    excess = daily_returns - daily_rf
    sharpe = (
        (excess.mean() / excess.std()) * np.sqrt(trading_days_per_year)
        if excess.std() > 0
        else np.nan
    )

    running_max = cumulative_value.cummax()
    drawdown = (cumulative_value - running_max) / running_max
    max_drawdown = drawdown.min()

    return {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown,
    }


def run_backtest_comparison(
    returns_df: pd.DataFrame,
    strategy_weights: Dict[str, float],
    benchmark_weights: Dict[str, float],
    rebalance: str = "none",
    risk_free_rate_annual: float = 0.02,
) -> Dict:
    """
    Convenience wrapper: simulate both the strategy and benchmark portfolios
    over the same window and return cumulative series + metrics for both.

    Parameters
    ----------
    rebalance : {"none", "monthly"}
        Rebalancing policy applied to BOTH strategy and benchmark, so the
        comparison isolates the effect of asset weights rather than
        rebalancing frequency.
    """
    sim_fn = simulate_monthly_rebalance if rebalance == "monthly" else simulate_buy_and_hold

    strategy_curve = sim_fn(returns_df, strategy_weights)
    benchmark_curve = sim_fn(returns_df, benchmark_weights)

    strategy_metrics = compute_backtest_metrics(strategy_curve, risk_free_rate_annual)
    benchmark_metrics = compute_backtest_metrics(benchmark_curve, risk_free_rate_annual)

    return {
        "strategy_curve": strategy_curve,
        "benchmark_curve": benchmark_curve,
        "strategy_metrics": strategy_metrics,
        "benchmark_metrics": benchmark_metrics,
    }
