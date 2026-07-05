import numpy as np
import pandas as pd
import pytest

from src.risk_metrics import (
    calculate_historical_var,
    calculate_max_drawdown,
    calculate_parametric_var,
    calculate_sharpe_ratio,
    risk_summary,
)


@pytest.fixture
def sample_returns():
    np.random.seed(42)
    # Simulated daily returns: mean slightly positive, realistic daily vol
    return pd.Series(np.random.normal(loc=0.0005, scale=0.02, size=500))


def test_historical_var_is_positive_number(sample_returns):
    var = calculate_historical_var(sample_returns, confidence_level=0.95)
    assert var > 0  # VaR expressed as a positive loss magnitude


def test_parametric_var_close_to_historical(sample_returns):
    hist_var = calculate_historical_var(sample_returns, 0.95)
    param_var = calculate_parametric_var(sample_returns, 0.95)
    # For roughly normal synthetic data, both should be in the same ballpark
    assert abs(hist_var - param_var) < 0.02


def test_sharpe_ratio_reasonable_range(sample_returns):
    sharpe = calculate_sharpe_ratio(sample_returns, risk_free_rate_annual=0.02)
    assert isinstance(sharpe, float)
    assert -5 < sharpe < 5


def test_sharpe_ratio_zero_std_returns_nan():
    constant_returns = pd.Series([0.001] * 100)
    sharpe = calculate_sharpe_ratio(constant_returns)
    assert np.isnan(sharpe)


def test_max_drawdown_is_non_positive(sample_returns):
    cumulative = (1 + sample_returns).cumprod()
    mdd = calculate_max_drawdown(cumulative)
    assert mdd <= 0


def test_risk_summary_keys(sample_returns):
    summary = risk_summary(sample_returns)
    expected_keys = {
        "historical_var_95",
        "parametric_var_95",
        "sharpe_ratio_annualized",
        "max_drawdown",
        "annualized_volatility",
        "annualized_return",
    }
    assert expected_keys.issubset(summary.keys())
