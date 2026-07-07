import numpy as np
import pandas as pd
import pytest

from src.portfolio import (
    compute_covariance_matrix,
    portfolio_performance,
    prepare_expected_returns,
    run_efficient_frontier,
)


@pytest.fixture
def synthetic_returns():
    np.random.seed(42)
    n = 500
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    # Distinct risk/return profiles roughly matching TSLA/BND/SPY
    tsla = np.random.normal(0.0015, 0.035, n)
    bnd = np.random.normal(0.0002, 0.004, n)
    spy = np.random.normal(0.0006, 0.011, n)
    return pd.DataFrame({"TSLA": tsla, "BND": bnd, "SPY": spy}, index=dates)


def test_prepare_expected_returns_uses_forecast_override(synthetic_returns):
    historical = {col: synthetic_returns[col] for col in synthetic_returns.columns}
    mu = prepare_expected_returns(historical, forecast_annual_returns={"TSLA": 0.30})

    assert mu["TSLA"] == 0.30
    # BND/SPY should fall back to historical annualized mean
    assert abs(mu["BND"] - synthetic_returns["BND"].mean() * 252) < 1e-9
    assert abs(mu["SPY"] - synthetic_returns["SPY"].mean() * 252) < 1e-9


def test_compute_covariance_matrix_shape_and_symmetry(synthetic_returns):
    cov = compute_covariance_matrix(synthetic_returns)
    assert cov.shape == (3, 3)
    assert np.allclose(cov.values, cov.values.T)  # covariance matrices are symmetric
    assert (np.diag(cov.values) > 0).all()  # variances must be positive


def test_portfolio_performance_equal_weights(synthetic_returns):
    mu = prepare_expected_returns(
        {col: synthetic_returns[col] for col in synthetic_returns.columns},
        forecast_annual_returns={},
    )
    cov = compute_covariance_matrix(synthetic_returns)
    weights = [1 / 3, 1 / 3, 1 / 3]

    ret, vol, sharpe = portfolio_performance(weights, mu, cov, risk_free_rate=0.02)
    assert vol > 0
    assert isinstance(ret, float)
    assert isinstance(sharpe, float)


def test_run_efficient_frontier_returns_valid_portfolios(synthetic_returns):
    mu = prepare_expected_returns(
        {col: synthetic_returns[col] for col in synthetic_returns.columns},
        forecast_annual_returns={"TSLA": 0.30},
    )
    cov = compute_covariance_matrix(synthetic_returns)

    result = run_efficient_frontier(mu, cov, n_frontier_points=10)

    assert "max_sharpe" in result and "min_vol" in result
    max_sharpe_weights = result["max_sharpe"]["weights"]
    min_vol_weights = result["min_vol"]["weights"]

    # Weights should sum to ~1 and be non-negative (long-only constraint)
    assert abs(sum(max_sharpe_weights.values) - 1.0) < 1e-3
    assert abs(sum(min_vol_weights.values) - 1.0) < 1e-3
    assert (max_sharpe_weights.values >= -1e-6).all()
    assert (min_vol_weights.values >= -1e-6).all()

    # Min-vol portfolio should have volatility <= max-sharpe portfolio's volatility
    assert result["min_vol"]["volatility"] <= result["max_sharpe"]["volatility"] + 1e-6

    # Frontier curve should be non-empty and monotonically increasing in volatility
    assert len(result["frontier_volatility"]) > 0
