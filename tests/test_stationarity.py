import numpy as np
import pandas as pd

from src.stationarity import run_adf_test, summarize_adf_results


def test_adf_on_stationary_white_noise():
    np.random.seed(0)
    stationary_series = pd.Series(np.random.normal(0, 1, 500))
    result = run_adf_test(stationary_series, "white_noise")

    assert result["is_stationary"] is True
    assert result["p_value"] < 0.05
    assert "stationary" in result["interpretation"]


def test_adf_on_trending_nonstationary_series():
    trend = pd.Series(np.arange(500) + np.random.normal(0, 1, 500))
    result = run_adf_test(trend, "trend_series")

    assert result["is_stationary"] is False
    assert result["p_value"] >= 0.05


def test_summarize_adf_results():
    np.random.seed(1)
    series_a = pd.Series(np.random.normal(0, 1, 300))
    series_b = pd.Series(np.arange(300).astype(float))

    results = {
        "a": run_adf_test(series_a, "a"),
        "b": run_adf_test(series_b, "b"),
    }
    summary = summarize_adf_results(results)

    assert list(summary["series"]) == ["a", "b"]
    assert "is_stationary" in summary.columns
