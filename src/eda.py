"""
Exploratory data analysis plotting utilities.

Each function returns the matplotlib Figure/Axes so callers (notebooks,
scripts, or tests) can further customize, save, or embed the plot rather
than being forced into a single global side-effect (plt.show()).
"""

import matplotlib.pyplot as plt
import pandas as pd


def plot_closing_price(price_series: pd.Series, ticker: str, ax=None):
    """Plot the closing (or adjusted close) price over time."""
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 5))
    ax.plot(price_series.index, price_series.values, linewidth=1.2)
    ax.set_title(f"{ticker} - Adjusted Close Price Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.grid(alpha=0.3)
    return ax


def plot_daily_returns(returns: pd.Series, ticker: str, ax=None):
    """Plot the daily percentage change to visualize volatility clustering."""
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 5))
    ax.plot(returns.index, returns.values, linewidth=0.6, alpha=0.8)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(f"{ticker} - Daily Percentage Returns")
    ax.set_xlabel("Date")
    ax.set_ylabel("Daily Return")
    ax.grid(alpha=0.3)
    return ax


def plot_rolling_stats(price_series: pd.Series, window: int, ticker: str, ax=None):
    """Plot the rolling mean and rolling standard deviation over a price series."""
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 5))
    rolling_mean = price_series.rolling(window).mean()
    rolling_std = price_series.rolling(window).std()

    ax.plot(price_series.index, price_series.values, label="Price", linewidth=1.0, alpha=0.6)
    ax.plot(rolling_mean.index, rolling_mean.values, label=f"{window}D Rolling Mean", linewidth=1.4)
    ax.plot(rolling_std.index, rolling_std.values, label=f"{window}D Rolling Std", linewidth=1.4)
    ax.set_title(f"{ticker} - Rolling Mean & Std Dev ({window}D window)")
    ax.set_xlabel("Date")
    ax.legend()
    ax.grid(alpha=0.3)
    return ax


def plot_return_distribution(returns: pd.Series, ticker: str, ax=None):
    """Plot a histogram of daily returns to visualize the return distribution shape."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    ax.hist(returns.dropna(), bins=100, alpha=0.75)
    ax.set_title(f"{ticker} - Distribution of Daily Returns")
    ax.set_xlabel("Daily Return")
    ax.set_ylabel("Frequency")
    ax.grid(alpha=0.3)
    return ax
