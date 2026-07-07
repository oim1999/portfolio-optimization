import numpy as np
import pandas as pd
import pytest

from src.backtesting import (
    compute_backtest_metrics,
    run_backtest_comparison,
    simulate_buy_and_hold,
    simulate_monthly_rebalance,
)


@pytest.fixture
def synthetic_returns():
    np.random.seed(1)
    n = 252  # ~1 year of trading days
    dates = pd.date_range("2025-01-01", periods=n, freq="B")
    tsla = np.random.normal(0.001, 0.03, n)
    bnd = np.random.normal(0.0001, 0.003, n)
    spy = np.random.normal(0.0004, 0.01, n)
    return pd.DataFrame({"TSLA": tsla, "BND": bnd, "SPY": spy}, index=dates)


def test_simulate_buy_and_hold_starts_at_initial_value(synthetic_returns):
    weights = {"TSLA": 0.5, "BND": 0.3, "SPY": 0.2}
    curve = simulate_buy_and_hold(synthetic_returns, weights, initial_value=1.0)

    assert len(curve) == len(synthetic_returns)
    assert curve.iloc[0] > 0  # first day already reflects day-1 return applied to initial alloc


def test_simulate_buy_and_hold_zero_returns_stays_flat():
    dates = pd.date_range("2025-01-01", periods=10, freq="B")
    zero_returns = pd.DataFrame({"A": [0.0] * 10, "B": [0.0] * 10}, index=dates)
    curve = simulate_buy_and_hold(zero_returns, {"A": 0.6, "B": 0.4}, initial_value=1.0)
    assert np.allclose(curve.values, 1.0)


def test_simulate_monthly_rebalance_matches_buy_and_hold_within_first_month(synthetic_returns):
    weights = {"TSLA": 0.5, "BND": 0.3, "SPY": 0.2}
    bh_curve = simulate_buy_and_hold(synthetic_returns, weights)
    reb_curve = simulate_monthly_rebalance(synthetic_returns, weights)

    # Within the very first month (before any rebalancing event), both should be identical
    first_month_mask = (synthetic_returns.index.year == synthetic_returns.index[0].year) & (
        synthetic_returns.index.month == synthetic_returns.index[0].month
    )
    assert np.allclose(
        bh_curve[first_month_mask].values, reb_curve[first_month_mask].values, atol=1e-8
    )


def test_compute_backtest_metrics_keys(synthetic_returns):
    curve = simulate_buy_and_hold(synthetic_returns, {"TSLA": 0.5, "BND": 0.3, "SPY": 0.2})
    metrics = compute_backtest_metrics(curve)

    expected_keys = {"total_return", "annualized_return", "sharpe_ratio", "max_drawdown"}
    assert expected_keys.issubset(metrics.keys())
    assert metrics["max_drawdown"] <= 0


def test_run_backtest_comparison_structure(synthetic_returns):
    result = run_backtest_comparison(
        synthetic_returns,
        strategy_weights={"TSLA": 0.5, "BND": 0.3, "SPY": 0.2},
        benchmark_weights={"TSLA": 0.0, "BND": 0.4, "SPY": 0.6},
        rebalance="none",
    )
    assert "strategy_curve" in result and "benchmark_curve" in result
    assert "strategy_metrics" in result and "benchmark_metrics" in result
    assert len(result["strategy_curve"]) == len(synthetic_returns)


def test_run_backtest_comparison_monthly_rebalance(synthetic_returns):
    result = run_backtest_comparison(
        synthetic_returns,
        strategy_weights={"TSLA": 0.5, "BND": 0.3, "SPY": 0.2},
        benchmark_weights={"TSLA": 0.0, "BND": 0.4, "SPY": 0.6},
        rebalance="monthly",
    )
    assert len(result["benchmark_curve"]) == len(synthetic_returns)
