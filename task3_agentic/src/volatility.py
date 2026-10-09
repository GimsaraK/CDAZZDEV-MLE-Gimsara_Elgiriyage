"""Historical volatility and the price ranges a hedge is sized from. Pure functions, no I/O.

Volatility is the close-to-close estimator: the sample standard deviation (ddof=1) of daily
log returns, scaled by sqrt(252) to an annual figure. Over a horizon of t trading days,
one standard deviation of the log price move is sigma * sqrt(t / 252), and the price band
is spot * exp(+/- n * that move) (a log-normal band, so it never goes below zero).
"""
# AI-ASSISTED: Claude Code (claude-opus-5-5), Prompt: 'Implement the Task 3A plan (the plan approved in Entry 13)', Date: 2026-10-09 (see CITATIONS.md Entry 14)

import math
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from . import config


def validate_window(window: int) -> int:
    """Reject windows that are not ints in [MIN_VOL_WINDOW, MAX_VOL_WINDOW]."""
    if isinstance(window, bool) or not isinstance(window, (int, np.integer)):
        raise ValueError(f"window must be an integer, got {window!r}")
    if not config.MIN_VOL_WINDOW <= int(window) <= config.MAX_VOL_WINDOW:
        raise ValueError(
            f"window must be between {config.MIN_VOL_WINDOW} and {config.MAX_VOL_WINDOW} trading days, got {window}"
        )
    return int(window)


def log_returns(prices: pd.Series) -> pd.Series:
    """Daily log returns. Non-positive or missing prices are dropped first (log is undefined there)."""
    clean = pd.to_numeric(prices, errors="coerce")
    clean = clean[clean > 0].dropna()
    return np.log(clean / clean.shift(1)).dropna()


def annualised_vol(returns: pd.Series, window: int) -> Optional[float]:
    """Annualised volatility (fraction) over the last `window` returns, or None if there are too few."""
    tail = returns.tail(window)
    # The sample std needs at least two points, and the window must be complete.
    if len(tail) < max(window, 2):
        return None
    return float(tail.std(ddof=1) * math.sqrt(config.TRADING_DAYS_PER_YEAR))


def rolling_annualised_vol(returns: pd.Series, window: int) -> pd.Series:
    return returns.rolling(window).std(ddof=1).dropna() * math.sqrt(config.TRADING_DAYS_PER_YEAR)


def vol_percentile(rolling: pd.Series, current: float, lookback: int = config.VOL_PERCENTILE_LOOKBACK) -> Optional[float]:
    """Share (0-100) of the last `lookback` rolling readings at or below the current one."""
    history = rolling.tail(lookback)
    if history.empty:
        return None
    return float((history <= current).mean() * 100)


def vol_regime(percentile: Optional[float]) -> Optional[str]:
    if percentile is None:
        return None
    if percentile < config.VOL_REGIME_LOW_PERCENTILE:
        return config.VOL_REGIME_LOW
    if percentile > config.VOL_REGIME_HIGH_PERCENTILE:
        return config.VOL_REGIME_ELEVATED
    return config.VOL_REGIME_NORMAL


def horizon_trading_days(calendar_days: int = config.HEDGE_HORIZON_CALENDAR_DAYS) -> int:
    """90 calendar days is about 62 trading sessions (252 sessions per 365.25 days)."""
    return max(1, round(calendar_days * config.TRADING_DAYS_PER_YEAR / config.CALENDAR_DAYS_PER_YEAR))


def expected_move(annual_vol: float, trading_days: int) -> float:
    """One-sigma log move (fraction) over `trading_days`."""
    return annual_vol * math.sqrt(trading_days / config.TRADING_DAYS_PER_YEAR)


def price_band(spot: float, move: float, n_sigma: float = 1.0) -> Tuple[float, float]:
    """(low, high) price for +/- n_sigma of a log move around spot."""
    return spot * math.exp(-n_sigma * move), spot * math.exp(n_sigma * move)


def max_drawdown(prices: pd.Series, lookback: int = config.DRAWDOWN_LOOKBACK) -> Optional[float]:
    """Largest peak-to-trough fall (negative fraction) over the last `lookback` sessions."""
    tail = pd.to_numeric(prices, errors="coerce").dropna().tail(lookback)
    if len(tail) < 2:
        return None
    running_peak = tail.cummax()
    return float((tail / running_peak - 1).min())
