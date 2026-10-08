"""Unit tests for indicators.py using hand-computed and published reference values."""
# AI-ASSISTED: Cursor Agent (claude-sonnet-5.5), Prompt: 'yes implement the plan @TASK1A_PLAN.md', Date: 2026-10-08 (see CITATIONS.md Entry 4)

import math

import numpy as np
import pandas as pd
import pytest

from task1_financial.src import indicators as ind

# SOURCE: Adapted from https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/relative-strength-index-rsi, file: RSI worked-example spreadsheet (cs-rsi.xls), Lines: closing prices and RSI(14) column
STOCKCHARTS_CLOSES = [
    44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42, 45.84, 46.08,
    45.89, 46.03, 45.61, 46.28, 46.28, 46.00, 46.03, 46.41, 46.22, 45.64,
    46.21, 46.25, 45.71, 46.45, 45.78, 45.35, 44.03, 44.18, 44.22, 44.57,
    43.42, 42.66, 43.13,
]
STOCKCHARTS_RSI_FIRST_VALUES = [70.53, 66.32, 66.55, 69.41, 66.36, 57.97, 62.93, 63.26, 56.06, 62.38]
# The published table displays closes rounded to 2 decimals but computes RSI from
# unrounded prices, so values reproduced from the rounded closes differ by < 0.1.
STOCKCHARTS_TOLERANCE = 0.1

# Hand calculation from the first 15 rounded closes:
# gains sum = 3.34, losses sum = 1.40 -> avg gain = 3.34/14, avg loss = 0.10
HAND_FIRST_RSI = 100 - 100 / (1 + (3.34 / 14) / (1.40 / 14))


def _series(values):
    return pd.Series(values, index=pd.RangeIndex(len(values)), dtype=float)


# --------------------------------------------------------------------------- SMA
def test_sma_hand_computed():
    result = ind.sma(_series(range(1, 11)), 3)
    assert result.iloc[:2].isna().all()
    expected = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    assert np.allclose(result.iloc[2:].to_numpy(), expected)


def test_sma_rejects_bad_window():
    with pytest.raises(ValueError):
        ind.sma(_series([1, 2, 3]), 0)


# --------------------------------------------------------------------------- EMA
def test_ema_explicit_recursion_by_hand():
    # span=3 -> alpha=0.5; seed=10; then 0.5*11+0.5*10=10.5; 0.5*12+0.5*10.5=11.25; ...
    result = ind.ema(_series([10, 11, 12, 13, 14]), span=3, min_periods=1)
    assert np.allclose(result.to_numpy(), [10.0, 10.5, 11.25, 12.125, 13.0625])


def test_ema_min_periods_masks_warmup():
    result = ind.ema(_series([10, 11, 12, 13, 14]), span=3)
    assert result.iloc[:2].isna().all()
    assert result.iloc[2] == pytest.approx(11.25)


def test_ema_matches_pandas_ewm(ohlcv_3y):
    close = ohlcv_3y["Adj Close"]
    ours = ind.ema(close, 12)
    reference = close.ewm(span=12, adjust=False).mean()
    valid = ours.notna()
    assert np.allclose(ours[valid], reference[valid])


# --------------------------------------------------------------------------- Wilder RMA
def test_wilder_rma_seed_and_recursion():
    result = ind.wilder_rma(_series([1, 2, 3, 4, 5]), period=3)
    assert result.iloc[:2].isna().all()
    assert result.iloc[2] == pytest.approx(2.0)  # seed = mean(1, 2, 3)
    assert result.iloc[3] == pytest.approx((2.0 * 2 + 4) / 3)
    assert result.iloc[4] == pytest.approx((((2.0 * 2 + 4) / 3) * 2 + 5) / 3)


def test_wilder_rma_skips_leading_nan():
    result = ind.wilder_rma(_series([np.nan, 1, 2, 3]), period=3)
    assert result.iloc[:3].isna().all()
    assert result.iloc[3] == pytest.approx(2.0)


# --------------------------------------------------------------------------- RSI
def test_rsi_first_value_matches_hand_calculation():
    result = ind.rsi(_series(STOCKCHARTS_CLOSES), period=14)
    assert result.iloc[:14].isna().all()
    assert result.iloc[14] == pytest.approx(HAND_FIRST_RSI, abs=1e-9)


def test_rsi_matches_stockcharts_worked_example():
    result = ind.rsi(_series(STOCKCHARTS_CLOSES), period=14)
    first_values = result.iloc[14 : 14 + len(STOCKCHARTS_RSI_FIRST_VALUES)].to_list()
    assert first_values == pytest.approx(STOCKCHARTS_RSI_FIRST_VALUES, abs=STOCKCHARTS_TOLERANCE)


def test_rsi_strictly_increasing_is_100():
    result = ind.rsi(_series(range(1, 40)), period=14)
    assert (result.dropna() == 100.0).all()


def test_rsi_strictly_decreasing_is_0():
    result = ind.rsi(_series(range(40, 1, -1)), period=14)
    assert np.allclose(result.dropna(), 0.0)


def test_rsi_flat_prices_is_neutral():
    result = ind.rsi(_series([50.0] * 30), period=14)
    assert (result.dropna() == 50.0).all()


def test_rsi_bounded(ohlcv_3y):
    result = ind.rsi(ohlcv_3y["Adj Close"]).dropna()
    assert ((result >= 0) & (result <= 100)).all()


# --------------------------------------------------------------------------- MACD
def test_macd_histogram_identity(ohlcv_3y):
    result = ind.macd(ohlcv_3y["Adj Close"])
    valid = result.dropna()
    assert np.allclose(valid["MACD_Hist"], valid["MACD"] - valid["MACD_Signal"])


def test_macd_constant_series_is_zero():
    result = ind.macd(_series([100.0] * 60)).dropna()
    assert not result.empty
    assert np.allclose(result.to_numpy(), 0.0)


def test_macd_warmup_lengths():
    result = ind.macd(_series(np.linspace(1, 2, 60)), fast=12, slow=26, signal=9)
    assert result["MACD"].first_valid_index() == 25
    assert result["MACD_Signal"].first_valid_index() == 25 + 8


def test_macd_rejects_fast_not_less_than_slow():
    with pytest.raises(ValueError):
        ind.macd(_series(range(60)), fast=26, slow=12)


# --------------------------------------------------------------------------- Bollinger
def test_bollinger_hand_computed_population_std():
    result = ind.bollinger_bands(_series([1, 2, 3, 4, 5]), window=5, num_std=2, ddof=0)
    last = result.iloc[-1]
    pop_std = math.sqrt(2.0)  # population std of 1..5
    assert last["BB_Middle"] == pytest.approx(3.0)
    assert last["BB_Upper"] == pytest.approx(3.0 + 2 * pop_std)
    assert last["BB_Lower"] == pytest.approx(3.0 - 2 * pop_std)
    assert last["BB_PctB"] == pytest.approx((5 - (3.0 - 2 * pop_std)) / (4 * pop_std))


def test_bollinger_constant_series_bands_collapse_and_pct_b_nan():
    result = ind.bollinger_bands(_series([10.0] * 25), window=20).dropna(subset=["BB_Middle"])
    assert np.allclose(result["BB_Upper"], result["BB_Middle"])
    assert np.allclose(result["BB_Lower"], result["BB_Middle"])
    assert result["BB_PctB"].isna().all()


# --------------------------------------------------------------------------- add_indicators
def test_add_indicators_adds_all_columns_without_mutating(ohlcv_3y):
    original_columns = list(ohlcv_3y.columns)
    result = ind.add_indicators(ohlcv_3y)
    assert list(ohlcv_3y.columns) == original_columns
    for column in ind.INDICATOR_COLUMNS:
        assert column in result.columns
    assert result.iloc[-1][list(ind.INDICATOR_COLUMNS)].notna().all()


def test_add_indicators_missing_price_column():
    with pytest.raises(KeyError):
        ind.add_indicators(pd.DataFrame({"Close": [1.0, 2.0]}))
