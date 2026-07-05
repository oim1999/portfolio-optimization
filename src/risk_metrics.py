"""
Foundational risk metric calculations: Value at Risk (VaR) and Sharpe Ratio.
"""

import numpy as np
import pandas as pd


def calculate_historical_var(
    returns: pd.Series,
    confidence_level: float = 0.95,
) -> float:
    """
    Calculate historical (non-parametric) daily Value at Risk.

    VaR answers: "Over a given day, what is the maximum loss we would expect
    NOT to exceed, at the given confidence level, based on historical
    return behavior?"

    Returns
    -------
    float
        The VaR expressed as a positive number representing a loss
        magnitude (e.g., 0.032 means a 3.2% potential daily loss).
    """
    clean_returns = returns.dropna()
    var_percentile = 1 - confidence_level
    var_value = -np.percentile(clean_returns, var_percentile * 100)
    return var_value


def calculate_parametric_var(
    returns: pd.Series,
    confidence_level: float = 0.95,
) -> float:
    """
    Calculate parametric (variance-covariance) VaR assuming normally
    distributed returns. Provided alongside historical VaR since real
    return distributions are typically fat-tailed - comparing the two
    highlights how much the normality assumption understates tail risk.
    """
    from scipy.stats import norm

    clean_returns = returns.dropna()
    mean = clean_returns.mean()
    std = clean_returns.std()
    z_score = norm.ppf(1 - confidence_level)
    var_value = -(mean + z_score * std)
    return var_value


def calculate_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate_annual: float = 0.02,
    trading_days_per_year: int = 252,
) -> float:
    """
    Calculate the annualized Sharpe Ratio from a daily return series.

    Sharpe Ratio = (mean annualized excess return) / (annualized volatility)

    A higher Sharpe Ratio indicates better risk-adjusted returns - more
    return generated per unit of volatility taken on.
    """
    clean_returns = returns.dropna()
    daily_rf = risk_free_rate_annual / trading_days_per_year

    excess_returns = clean_returns - daily_rf
    mean_excess = excess_returns.mean()
    std_excess = excess_returns.std()

    if np.isclose(std_excess, 0.0, atol=1e-12):
        return np.nan

    sharpe_daily = mean_excess / std_excess
    sharpe_annualized = sharpe_daily * np.sqrt(trading_days_per_year)
    return sharpe_annualized


def calculate_max_drawdown(cumulative_returns: pd.Series) -> float:
    """
    Calculate maximum drawdown from a cumulative return series (e.g. (1+r).cumprod()).
    Returns a negative number representing the largest peak-to-trough decline.
    """
    running_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns - running_max) / running_max
    return drawdown.min()


def risk_summary(
    returns: pd.Series,
    risk_free_rate_annual: float = 0.02,
    confidence_level: float = 0.95,
    trading_days_per_year: int = 252,
) -> dict:
    """Convenience wrapper bundling all headline risk metrics for one asset."""
    cumulative = (1 + returns.dropna()).cumprod()
    return {
        "historical_var_95": calculate_historical_var(returns, confidence_level),
        "parametric_var_95": calculate_parametric_var(returns, confidence_level),
        "sharpe_ratio_annualized": calculate_sharpe_ratio(
            returns, risk_free_rate_annual, trading_days_per_year
        ),
        "max_drawdown": calculate_max_drawdown(cumulative),
        "annualized_volatility": returns.std() * np.sqrt(trading_days_per_year),
        "annualized_return": returns.mean() * trading_days_per_year,
    }
