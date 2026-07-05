"""
Stationarity testing utilities.

Why this matters: ARIMA assumes the series being modeled is stationary
(constant mean/variance/autocorrelation structure over time). Raw price
series are almost always non-stationary (they trend), while daily returns
are typically much closer to stationary. The Augmented Dickey-Fuller (ADF)
test formally checks this: the null hypothesis is that the series HAS a
unit root (i.e., is non-stationary).
"""

from typing import Dict

import pandas as pd
from statsmodels.tsa.stattools import adfuller


def run_adf_test(series: pd.Series, series_name: str = "series") -> Dict:
    """
    Run the Augmented Dickey-Fuller test on a time series.

    Returns
    -------
    dict
        {
            "series_name": str,
            "adf_statistic": float,
            "p_value": float,
            "n_lags_used": int,
            "n_observations": int,
            "critical_values": dict,
            "is_stationary": bool,  # True if p_value < 0.05
            "interpretation": str,
        }
    """
    clean_series = series.dropna()
    result = adfuller(clean_series, autolag="AIC")

    adf_statistic, p_value, n_lags, n_obs, critical_values, _ = result
    is_stationary = bool(p_value < 0.05)

    interpretation = (
        f"The ADF statistic for {series_name} is {adf_statistic:.4f} with a "
        f"p-value of {p_value:.4f}. "
        + (
            "Since the p-value is below the 0.05 significance threshold, we "
            "reject the null hypothesis of a unit root: the series is "
            "statistically stationary."
            if is_stationary
            else "Since the p-value is above the 0.05 significance threshold, "
            "we fail to reject the null hypothesis of a unit root: the "
            "series is non-stationary and would need differencing "
            "(the 'd' parameter in ARIMA) before it can be modeled."
        )
    )

    return {
        "series_name": series_name,
        "adf_statistic": adf_statistic,
        "p_value": p_value,
        "n_lags_used": n_lags,
        "n_observations": n_obs,
        "critical_values": critical_values,
        "is_stationary": is_stationary,
        "interpretation": interpretation,
    }


def summarize_adf_results(results: Dict[str, Dict]) -> pd.DataFrame:
    """Convert a dict of {name: run_adf_test(...) output} into a summary table."""
    rows = []
    for name, res in results.items():
        rows.append(
            {
                "series": name,
                "adf_statistic": res["adf_statistic"],
                "p_value": res["p_value"],
                "n_lags_used": res["n_lags_used"],
                "is_stationary": res["is_stationary"],
            }
        )
    return pd.DataFrame(rows)
