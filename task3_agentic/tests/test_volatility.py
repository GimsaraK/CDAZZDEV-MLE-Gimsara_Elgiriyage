"""Volatility maths against hand-computed values."""
# AI-ASSISTED: Claude Code (claude-sonnet-5.5), Prompt: 'Implement the Task 3A plan (the plan approved in Entry 13)', Date: 2026-10-09 (see CITATIONS.md Entry 14)

import math

import numpy as np
import pandas as pd
import pytest

from task3_agentic.src import config
from task3_agentic.src import volatility as vol


def test_log_returns_match_definition_and_drop_bad_prices():
    prices = pd.Series([100.0, 110.0, 0.0, 99.0, np.nan, 121.0])
    returns = vol.log_returns(prices)
    # 0 and NaN are removed first, so the pairs are 100->110, 110->99, 99->121.
    expected = [math.log(110 / 100), math.log(99 / 110), math.log(121 / 99)]
    assert returns.tolist() == pytest.approx(expected)


def test_annualised_vol_is_sample_std_times_sqrt_252():
    returns = pd.Series([0.01, -0.02, 0.015, -0.005, 0.0, 0.012])
    window = 5
    tail = returns.tail(window).to_numpy()
    mean = tail.mean()
    sample_std = math.sqrt(((tail - mean) ** 2).sum() / (window - 1))
    assert vol.annualised_vol(returns, window) == pytest.approx(sample_std * math.sqrt(252))


def test_constant_prices_have_zero_volatility():
    returns = vol.log_returns(pd.Series([50.0] * 30))
    assert vol.annualised_vol(returns, 20) == pytest.approx(0.0)


def test_annualised_vol_needs_a_full_window():
    assert vol.annualised_vol(pd.Series([0.01, 0.02]), 5) is None


@pytest.mark.parametrize("bad", [config.MIN_VOL_WINDOW - 1, config.MAX_VOL_WINDOW + 1, 2.5, "20", True, None])
def test_validate_window_rejects_bad_values(bad):
    with pytest.raises(ValueError):
        vol.validate_window(bad)


def test_validate_window_accepts_bounds():
    assert vol.validate_window(config.MIN_VOL_WINDOW) == config.MIN_VOL_WINDOW
    assert vol.validate_window(np.int64(config.MAX_VOL_WINDOW)) == config.MAX_VOL_WINDOW


def test_horizon_and_expected_move():
    # 90 calendar days * 252 / 365.25 = 62.1 -> 62 sessions.
    assert vol.horizon_trading_days(90) == 62
    # A full year of trading days moves by exactly the annual volatility.
    assert vol.expected_move(0.2, 252) == pytest.approx(0.2)
    assert vol.expected_move(0.2, 63) == pytest.approx(0.1)


def test_price_band_is_symmetric_in_log_space():
    low, high = vol.price_band(100.0, 0.1, 1)
    assert low == pytest.approx(100 * math.exp(-0.1))
    assert high == pytest.approx(100 * math.exp(0.1))
    assert math.log(100 / low) == pytest.approx(math.log(high / 100))


def test_percentile_and_regime():
    rolling = pd.Series(np.linspace(0.1, 0.5, 100))
    assert vol.vol_percentile(rolling, 0.5) == pytest.approx(100.0)
    assert vol.vol_regime(10.0) == config.VOL_REGIME_LOW
    assert vol.vol_regime(50.0) == config.VOL_REGIME_NORMAL
    assert vol.vol_regime(90.0) == config.VOL_REGIME_ELEVATED
    assert vol.vol_regime(None) is None


def test_max_drawdown():
    # Peak 120, trough 90 -> -25%.
    assert vol.max_drawdown(pd.Series([100.0, 120.0, 90.0, 110.0])) == pytest.approx(-0.25)
    assert vol.max_drawdown(pd.Series([100.0])) is None
