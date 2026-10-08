"""Technical indicators implemented from first principles (no TA-Lib).

All functions take a price ``pd.Series`` and return a Series/DataFrame aligned
to the same index. Values are NaN until enough observations exist (warm-up),
so callers never receive misleading early values.

References
----------
- SMA / Bollinger Bands: J. Bollinger, "Bollinger on Bollinger Bands" (2001).
- RSI and Wilder smoothing: J. Welles Wilder, "New Concepts in Technical Trading Systems" (1978).
- MACD: G. Appel, "Technical Analysis: Power Tools for Active Investors" (2005).
"""
# AI-ASSISTED: Cursor Agent (claude-sonnet-5.5), Prompt: 'yes implement the plan @TASK1A_PLAN.md', Date: 2026-10-08 (see CITATIONS.md Entry 4)

from typing import Optional

import numpy as np
import pandas as pd

from . import config

# Output column names, derived from config so they stay in sync with the parameters.
COL_SMA_SHORT = f"SMA_{config.SMA_SHORT_WINDOW}"
COL_SMA_LONG = f"SMA_{config.SMA_LONG_WINDOW}"
COL_RSI = f"RSI_{config.RSI_PERIOD}"
COL_MACD = "MACD"
COL_MACD_SIGNAL = "MACD_Signal"
COL_MACD_HIST = "MACD_Hist"
COL_BB_MIDDLE = "BB_Middle"
COL_BB_UPPER = "BB_Upper"
COL_BB_LOWER = "BB_Lower"
COL_BB_PCT_B = "BB_PctB"
COL_BB_BANDWIDTH = "BB_Bandwidth"

# Every column added by add_indicators(), in display order (used by the notebook and tests).
INDICATOR_COLUMNS = (
    COL_SMA_SHORT,
    COL_SMA_LONG,
    COL_RSI,
    COL_MACD,
    COL_MACD_SIGNAL,
    COL_MACD_HIST,
    COL_BB_MIDDLE,
    COL_BB_UPPER,
    COL_BB_LOWER,
    COL_BB_PCT_B,
    COL_BB_BANDWIDTH,
)


def _validate_window(window: int, name: str) -> None:
    """Reject zero, negative or non-integer window sizes early with a clear message."""
    if not isinstance(window, (int, np.integer)) or window < 1:
        raise ValueError(f"{name} must be a positive integer, got {window!r}")


def _as_float_series(series: pd.Series) -> pd.Series:
    """Convert any input (list, int Series, strings) to a float Series; unparseable values become NaN."""
    return pd.to_numeric(pd.Series(series), errors="coerce").astype(float)


def sma(series: pd.Series, window: int) -> pd.Series:
    """Simple moving average: SMA_t = mean(x_{t-window+1} .. x_t).

    The first ``window - 1`` values are NaN.
    """
    _validate_window(window, "window")
    # min_periods=window: emit NaN until a full window is available, instead of a partial average.
    return _as_float_series(series).rolling(window=window, min_periods=window).mean()


def ema(series: pd.Series, span: int, min_periods: Optional[int] = None) -> pd.Series:
    """Exponential moving average via the explicit recursion.

    EMA_t = alpha * x_t + (1 - alpha) * EMA_{t-1}, with alpha = 2 / (span + 1),
    seeded with the first non-NaN observation. Output is NaN until ``min_periods``
    (default ``span``) valid observations have been seen. NaN inputs are skipped:
    the previous EMA is carried forward and NaN is emitted at that position.

    Numerically identical to ``pandas.Series.ewm(span=span, adjust=False)`` on
    series without gaps (verified in tests).
    """
    _validate_window(span, "span")
    min_periods = span if min_periods is None else min_periods
    # Smoothing factor: a larger span gives a smaller alpha, i.e. a slower-moving average.
    alpha = 2.0 / (span + 1.0)

    # Work on a NumPy array (faster than row-by-row pandas access); start with all NaN.
    values = _as_float_series(series).to_numpy()
    out = np.full(values.shape, np.nan)
    prev = np.nan  # EMA value at the previous valid observation
    seen = 0  # number of valid (non-NaN) observations processed so far
    for i, x in enumerate(values):
        if np.isnan(x):
            # Missing price: skip it and keep the previous EMA (out[i] stays NaN).
            continue
        # First valid value seeds the EMA; after that apply the recursion.
        prev = x if seen == 0 else alpha * x + (1.0 - alpha) * prev
        seen += 1
        # Only publish the EMA once enough observations have been seen (warm-up).
        if seen >= min_periods:
            out[i] = prev
    return pd.Series(out, index=series.index)


def wilder_rma(series: pd.Series, period: int) -> pd.Series:
    """Wilder's smoothing (running moving average), used by RSI.

    Seed: simple average of the first ``period`` valid values.
    Then: RMA_t = (RMA_{t-1} * (period - 1) + x_t) / period, i.e. an EMA with alpha = 1 / period.
    """
    _validate_window(period, "period")
    values = _as_float_series(series).to_numpy()
    out = np.full(values.shape, np.nan)
    seed_window = []  # collects the first `period` valid values for the seed average
    prev = np.nan  # previous RMA value; NaN until the seed has been computed
    for i, x in enumerate(values):
        if np.isnan(x):
            continue
        if np.isnan(prev):
            # Warm-up phase: Wilder seeds the series with a plain average of the first `period` values.
            seed_window.append(x)
            if len(seed_window) == period:
                prev = sum(seed_window) / period
                out[i] = prev
        else:
            # Smoothing phase: the new value gets weight 1/period, the history gets (period-1)/period.
            prev = (prev * (period - 1) + x) / period
            out[i] = prev
    return pd.Series(out, index=series.index)


def rsi(close: pd.Series, period: int = config.RSI_PERIOD) -> pd.Series:
    """Relative Strength Index with Wilder smoothing.

    gain_t = max(close_t - close_{t-1}, 0), loss_t = max(close_{t-1} - close_t, 0)
    RS = RMA(gain) / RMA(loss);  RSI = 100 - 100 / (1 + RS)

    Edge cases: no losses -> 100; no gains and no losses (flat prices) -> 50.
    The first valid value appears at position ``period`` (one change is lost to diff()).
    """
    _validate_window(period, "period")
    # Day-over-day price change, split into upward moves (gains) and downward moves (losses).
    # clip(lower=0) zeroes the opposite direction, so both series are non-negative.
    delta = _as_float_series(close).diff()
    gains = delta.clip(lower=0.0)
    losses = (-delta).clip(lower=0.0)

    # Smooth gains and losses with Wilder's method (this is what distinguishes RSI from a plain ratio).
    avg_gain = wilder_rma(gains, period)
    avg_loss = wilder_rma(losses, period)

    # Division by zero (no losses) is handled explicitly below, so silence NumPy's warnings here.
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = avg_gain / avg_loss
        values = config.RSI_MAX_VALUE - config.RSI_MAX_VALUE / (1.0 + rs)

    # Edge case 1: only gains in the window -> RSI is 100 by definition.
    values = values.where(avg_loss != 0, config.RSI_MAX_VALUE)
    # Edge case 2: price did not move at all -> neither overbought nor oversold, so 50.
    flat = (avg_gain == 0) & (avg_loss == 0)
    values = values.where(~flat, config.RSI_NEUTRAL_VALUE)
    # Keep warm-up positions NaN (the where() calls above must not fill them).
    return values.where(avg_gain.notna() & avg_loss.notna())


def macd(
    close: pd.Series,
    fast: int = config.MACD_FAST,
    slow: int = config.MACD_SLOW,
    signal: int = config.MACD_SIGNAL,
) -> pd.DataFrame:
    """Moving Average Convergence Divergence.

    MACD = EMA_fast(close) - EMA_slow(close)
    Signal = EMA_signal(MACD)
    Histogram = MACD - Signal
    """
    for value, name in ((fast, "fast"), (slow, "slow"), (signal, "signal")):
        _validate_window(value, name)
    # MACD measures fast-minus-slow trend, so the "fast" EMA must actually be faster.
    if fast >= slow:
        raise ValueError(f"fast span ({fast}) must be shorter than slow span ({slow})")

    # Positive MACD = short-term trend above long-term trend (upward momentum).
    macd_line = ema(close, fast) - ema(close, slow)
    # The signal line is a smoothed MACD; crossovers between the two are the classic trade signal.
    signal_line = ema(macd_line, signal)
    # Histogram > 0 means MACD is above its signal line (bullish), < 0 means below (bearish).
    return pd.DataFrame(
        {
            COL_MACD: macd_line,
            COL_MACD_SIGNAL: signal_line,
            COL_MACD_HIST: macd_line - signal_line,
        },
        index=close.index,
    )


def bollinger_bands(
    close: pd.Series,
    window: int = config.BB_WINDOW,
    num_std: float = config.BB_NUM_STD,
    ddof: int = config.BB_STD_DDOF,
) -> pd.DataFrame:
    """Bollinger Bands.

    Middle = SMA_window(close); sigma = rolling std (population, ddof=0 by default)
    Upper/Lower = Middle +/- num_std * sigma
    %B = (close - Lower) / (Upper - Lower)   (NaN when the bands have zero width)
    Bandwidth = (Upper - Lower) / Middle
    """
    _validate_window(window, "window")
    prices = _as_float_series(close)
    # Middle band is the moving average; sigma is the rolling volatility over the same window.
    # ddof=0 (population std) follows Bollinger's own definition; pandas defaults to ddof=1.
    middle = sma(prices, window)
    sigma = prices.rolling(window=window, min_periods=window).std(ddof=ddof)
    upper = middle + num_std * sigma
    lower = middle - num_std * sigma

    # %B locates the price inside the bands: 0 = on the lower band, 1 = on the upper band,
    # > 1 or < 0 = outside the bands. Zero-width bands (flat prices) become NaN to avoid 0/0.
    width = upper - lower
    safe_width = width.where(width > 0)
    pct_b = (prices - lower) / safe_width
    # Bandwidth = band width relative to price level; low values mean a volatility "squeeze".
    bandwidth = width / middle.where(middle != 0)

    return pd.DataFrame(
        {
            COL_BB_MIDDLE: middle,
            COL_BB_UPPER: upper,
            COL_BB_LOWER: lower,
            COL_BB_PCT_B: pct_b,
            COL_BB_BANDWIDTH: bandwidth,
        },
        index=close.index,
    )


def add_indicators(df: pd.DataFrame, price_col: str = config.PRICE_COL_ADJ) -> pd.DataFrame:
    """Return a copy of ``df`` with all indicator columns computed from ``price_col``."""
    if price_col not in df.columns:
        raise KeyError(f"price column {price_col!r} not found; available: {list(df.columns)}")

    # Copy so the caller's DataFrame is never modified in place.
    out = df.copy()
    close = out[price_col]
    out[COL_SMA_SHORT] = sma(close, config.SMA_SHORT_WINDOW)
    out[COL_SMA_LONG] = sma(close, config.SMA_LONG_WINDOW)
    out[COL_RSI] = rsi(close, config.RSI_PERIOD)
    # MACD and Bollinger return several columns each; join() aligns them on the date index.
    out = out.join(macd(close))
    out = out.join(bollinger_bands(close))
    return out
