"""
Modern Portfolio Theory utilities for Task 4: expected returns preparation,
covariance matrix computation, and Efficient Frontier generation via
PyPortfolioOpt.
"""

import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def prepare_expected_returns(
    historical_returns: Dict[str, pd.Series],
    forecast_annual_returns: Dict[str, float],
    trading_days_per_year: int = 252,
) -> pd.Series:
    """
    Build the expected-annual-return vector used as MPT input.

    Parameters
    ----------
    historical_returns : dict[str, pd.Series]
        Daily return series per asset (used as the fallback / historical basis).
    forecast_annual_returns : dict[str, float]
        Explicit annualized expected return overrides per asset - typically
        {"TSLA": <derived from the Task 2/3 model forecast>}. Any asset NOT
        present here falls back to its historical mean annualized return.
    trading_days_per_year : int
        Trading days used to annualize historical daily returns.

    Returns
    -------
    pd.Series
        Expected annual return per asset, indexed by ticker.

    Notes
    -----
    This mirrors a common analyst workflow: use a specific forward-looking
    "view" on one asset (here, TSLA, informed by the forecasting model) while
    relying on historical averages for the other assets.
    """
    expected = {}
    for ticker, returns in historical_returns.items():
        if ticker in forecast_annual_returns:
            expected[ticker] = forecast_annual_returns[ticker]
            logger.info("%s: using forecast-based expected return = %.4f", ticker, expected[ticker])
        else:
            hist_annual = returns.mean() * trading_days_per_year
            expected[ticker] = hist_annual
            logger.info("%s: using historical mean annualized return = %.4f", ticker, hist_annual)

    return pd.Series(expected)


def compute_covariance_matrix(
    returns_df: pd.DataFrame,
    trading_days_per_year: int = 252,
) -> pd.DataFrame:
    """
    Compute the annualized covariance matrix from a DataFrame of daily
    returns (columns = tickers, rows = dates).
    """
    daily_cov = returns_df.cov()
    annualized_cov = daily_cov * trading_days_per_year
    return annualized_cov


def portfolio_performance(
    weights: np.ndarray,
    mu: pd.Series,
    cov: pd.DataFrame,
    risk_free_rate: float = 0.02,
) -> Tuple[float, float, float]:
    """
    Compute (expected_return, volatility, sharpe_ratio) for a given weight vector.
    """
    weights = np.array(weights)
    expected_return = float(np.dot(weights, mu.values))
    volatility = float(np.sqrt(weights.T @ cov.values @ weights))
    sharpe = (expected_return - risk_free_rate) / volatility if volatility > 0 else np.nan
    return expected_return, volatility, sharpe


def run_efficient_frontier(
    mu: pd.Series,
    cov: pd.DataFrame,
    risk_free_rate: float = 0.02,
    n_frontier_points: int = 50,
):
    """
    Build the Efficient Frontier using PyPortfolioOpt, plus identify the
    Maximum Sharpe Ratio (tangency) and Minimum Volatility portfolios.

    Returns
    -------
    dict with keys:
        "frontier_returns", "frontier_volatility" : np.ndarray - the frontier curve
        "max_sharpe": {"weights": pd.Series, "return": float, "volatility": float, "sharpe": float}
        "min_vol":    {"weights": pd.Series, "return": float, "volatility": float, "sharpe": float}
    """
    from pypfopt.efficient_frontier import EfficientFrontier

    tickers = list(mu.index)

    # --- Max Sharpe (tangency) portfolio ---
    ef_sharpe = EfficientFrontier(mu, cov, weight_bounds=(0, 1))
    ef_sharpe.max_sharpe(risk_free_rate=risk_free_rate)
    sharpe_weights = pd.Series(ef_sharpe.clean_weights())
    sharpe_ret, sharpe_vol, sharpe_sharpe = ef_sharpe.portfolio_performance(risk_free_rate=risk_free_rate)

    # --- Min Volatility portfolio ---
    ef_minvol = EfficientFrontier(mu, cov, weight_bounds=(0, 1))
    ef_minvol.min_volatility()
    minvol_weights = pd.Series(ef_minvol.clean_weights())
    minvol_ret, minvol_vol, minvol_sharpe = ef_minvol.portfolio_performance(risk_free_rate=risk_free_rate)

    # --- Frontier curve: sweep target returns between min-vol and max-return asset ---
    min_ret = minvol_ret
    max_ret = mu.max() * 0.999  # slightly inside the max achievable return to keep solver feasible
    target_returns = np.linspace(min_ret, max_ret, n_frontier_points)

    frontier_vols = []
    frontier_rets = []
    for target in target_returns:
        try:
            ef = EfficientFrontier(mu, cov, weight_bounds=(0, 1))
            ef.efficient_return(target_return=target)
            ret, vol, _ = ef.portfolio_performance(risk_free_rate=risk_free_rate)
            frontier_rets.append(ret)
            frontier_vols.append(vol)
        except Exception as exc:  # noqa: BLE001 - some targets may be infeasible
            logger.debug("Skipping infeasible frontier target %.4f: %s", target, exc)
            continue

    return {
        "frontier_returns": np.array(frontier_rets),
        "frontier_volatility": np.array(frontier_vols),
        "max_sharpe": {
            "weights": sharpe_weights,
            "return": sharpe_ret,
            "volatility": sharpe_vol,
            "sharpe": sharpe_sharpe,
        },
        "min_vol": {
            "weights": minvol_weights,
            "return": minvol_ret,
            "volatility": minvol_vol,
            "sharpe": minvol_sharpe,
        },
        "tickers": tickers,
    }
