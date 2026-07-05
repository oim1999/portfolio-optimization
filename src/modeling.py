"""
Task 2 modeling utilities: chronological train/test splitting, ARIMA/SARIMA
fitting via pmdarima, and LSTM sequence preparation + model construction.

Only ONE of ARIMA or LSTM is strictly required for the interim submission
per the rubric ("At Least One Model Implemented"), but both are provided
here as reusable, independently testable functions so Phase 2 (full Task 2)
can build directly on top of this module without rework.
"""

import logging
from typing import Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def chronological_train_test_split(
    series: pd.Series,
    train_end: str,
    test_start: str,
    test_end: str = None,
) -> Tuple[pd.Series, pd.Series]:
    """
    Split a time-indexed series into train/test sets by DATE, not by random
    sampling. This is critical for time series: shuffling would leak future
    information into training and produce an unrealistically optimistic
    evaluation.

    Parameters
    ----------
    series : pd.Series
        Must have a DatetimeIndex.
    train_end : str
        Last date (inclusive) to include in the training set, e.g. "2024-12-31".
    test_start : str
        First date (inclusive) of the test set, e.g. "2025-01-01".
    test_end : str, optional
        Last date (inclusive) of the test set. If None, uses all remaining data.

    Returns
    -------
    (train_series, test_series)
    """
    if not isinstance(series.index, pd.DatetimeIndex):
        raise TypeError("Series must have a DatetimeIndex for chronological splitting.")

    train = series.loc[:train_end]
    test = series.loc[test_start:test_end] if test_end else series.loc[test_start:]

    logger.info(
        "Chronological split -> train: %s to %s (%d obs), test: %s to %s (%d obs)",
        train.index.min(), train.index.max(), len(train),
        test.index.min(), test.index.max(), len(test),
    )
    return train, test


# ---------------------------------------------------------------------
# ARIMA / SARIMA
# ---------------------------------------------------------------------

def fit_auto_arima(train_series: pd.Series, seasonal: bool = False, m: int = 1):
    """
    Fit an ARIMA (or SARIMA if seasonal=True) model using pmdarima's
    auto_arima, which performs a stepwise search over (p,d,q) [and (P,D,Q,m)
    if seasonal] to minimize AIC.

    Returns the fitted pmdarima model object. Access `model.order` and
    `model.seasonal_order` to document the chosen parameters.
    """
    import pmdarima as pm

    model = pm.auto_arima(
        train_series,
        seasonal=seasonal,
        m=m if seasonal else 1,
        stepwise=True,
        suppress_warnings=True,
        error_action="ignore",
        trace=True,
    )
    logger.info("Best ARIMA order: %s, seasonal_order: %s", model.order,
                getattr(model, "seasonal_order", None))
    return model


def forecast_arima(model, n_periods: int, return_conf_int: bool = True, alpha: float = 0.05):
    """
    Generate forecasts (and optionally confidence intervals) from a fitted
    pmdarima model for `n_periods` steps ahead.
    """
    if return_conf_int:
        forecast, conf_int = model.predict(n_periods=n_periods, return_conf_int=True, alpha=alpha)
        return forecast, conf_int
    forecast = model.predict(n_periods=n_periods)
    return forecast


# ---------------------------------------------------------------------
# LSTM
# ---------------------------------------------------------------------

def create_sequences(scaled_series: np.ndarray, window_size: int = 60) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a 1D scaled array into overlapping (X, y) sequences for
    supervised LSTM training: use the previous `window_size` days to
    predict the next single day.

    Returns
    -------
    (X, y) : np.ndarray
        X shape -> (n_samples, window_size, 1)
        y shape -> (n_samples,)
    """
    X, y = [], []
    for i in range(window_size, len(scaled_series)):
        X.append(scaled_series[i - window_size:i])
        y.append(scaled_series[i])
    X = np.array(X).reshape(-1, window_size, 1)
    y = np.array(y)
    return X, y


def build_lstm_model(window_size: int = 60, units: int = 50, dropout: float = 0.2):
    """
    Build a simple stacked LSTM architecture:
        Input(window_size, 1) -> LSTM -> Dropout -> LSTM -> Dropout -> Dense(1)

    Kept intentionally modest (2 LSTM layers) as a strong, fast-training
    baseline; Phase 2 can expand this (more units, layers, or a
    Bidirectional/GRU variant) during hyperparameter experimentation.
    """
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input

    model = Sequential([
        Input(shape=(window_size, 1)),
        LSTM(units, return_sequences=True),
        Dropout(dropout),
        LSTM(units, return_sequences=False),
        Dropout(dropout),
        Dense(25, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


def evaluate_forecast(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute MAE, RMSE, and MAPE between true and predicted values."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((np.array(y_true) - np.array(y_pred)) / np.array(y_true))) * 100

    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}
