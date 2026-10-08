"""Offline tests for summary.py: momentum votes, 52-week range, YTD, P/E chain, full summary."""
# AI-ASSISTED: Cursor Agent (claude-sonnet-5.5), Prompt: 'yes implement the plan @TASK1A_PLAN.md', Date: 2026-10-08 (see CITATIONS.md Entry 4)

import json

import numpy as np
import pandas as pd
import pytest

from task1_financial.src import config
from task1_financial.src import indicators as ind
from task1_financial.src import summary as sm
from task1_financial.src.data import DataQualityReport, MarketData, validate_and_clean_ohlcv
from task1_financial.src.news import NewsItem, NewsResult

from .conftest import make_ohlcv

N_ROWS = 40


def _indicator_frame(**last_values) -> pd.DataFrame:
    """Flat indicator frame; ``last_values`` override the final row."""
    index = pd.bdate_range("2026-01-02", periods=N_ROWS)
    base = {
        "Adj Close": 110.0,
        ind.COL_SMA_SHORT: 105.0,
        ind.COL_SMA_LONG: 100.0,
        ind.COL_MACD: 1.0,
        ind.COL_MACD_SIGNAL: 0.5,
        ind.COL_MACD_HIST: np.linspace(0.1, 0.5, N_ROWS),
        ind.COL_RSI: 60.0,
        ind.COL_BB_PCT_B: 0.7,
    }
    df = pd.DataFrame({k: (v if isinstance(v, np.ndarray) else [v] * N_ROWS) for k, v in base.items()}, index=index)
    for column, value in last_values.items():
        df.iloc[-1, df.columns.get_loc(column)] = value
    return df


# --------------------------------------------------------------------------- momentum
def test_all_bullish_components_give_strong_bullish():
    """All seven rules bullish -> score 1.0 and the Strong Bullish label."""
    signal = sm.compute_momentum(_indicator_frame())
    assert all(c.vote == 1 for c in signal.components)
    assert signal.score == 1.0
    assert signal.label == config.MOMENTUM_LABEL_STRONG_BULLISH
    assert signal.unavailable == []


def test_all_bearish_components_give_strong_bearish():
    """All seven rules bearish -> score -1.0 and the Strong Bearish label."""
    df = _indicator_frame(**{"Adj Close": 90.0, ind.COL_SMA_SHORT: 95.0, ind.COL_MACD: -1.0, ind.COL_RSI: 40.0,
                             ind.COL_BB_PCT_B: 0.2})
    df[ind.COL_MACD_HIST] = np.linspace(0.5, -0.5, N_ROWS)
    df[ind.COL_SMA_SHORT] = 95.0
    signal = sm.compute_momentum(df)
    assert signal.score == -1.0
    assert signal.label == config.MOMENTUM_LABEL_STRONG_BEARISH


def test_overbought_rsi_and_upper_band_do_not_vote_but_are_flagged():
    """Stretched RSI and %B abstain (vote 0) and add flags, so the score is 5/7."""
    signal = sm.compute_momentum(_indicator_frame(**{ind.COL_RSI: 78.0, ind.COL_BB_PCT_B: 1.2}))
    votes = {c.name: c.vote for c in signal.components}
    assert votes["rsi_zone"] == 0 and votes["bollinger_pct_b"] == 0
    assert any("overbought" in f for f in signal.flags)
    assert any("upper Bollinger" in f for f in signal.flags)
    assert signal.score == pytest.approx(5 / 7, abs=1e-4)


def test_nan_components_are_skipped_and_reported():
    """Rules that need a missing SMA-200 are listed as unavailable and left out of the average."""
    signal = sm.compute_momentum(_indicator_frame(**{ind.COL_SMA_LONG: np.nan}))
    assert {"price_vs_sma_200", "trend_regime"} <= set(signal.unavailable)
    assert signal.score == 1.0  # the remaining components are all bullish


def test_no_data_gives_unavailable():
    """An empty frame gives the Unavailable label instead of an error."""
    assert sm.compute_momentum(pd.DataFrame()).label == config.MOMENTUM_LABEL_UNAVAILABLE


@pytest.mark.parametrize(
    "score,label",
    [
        (0.6, config.MOMENTUM_LABEL_STRONG_BULLISH),
        (0.2, config.MOMENTUM_LABEL_BULLISH),
        (0.19, config.MOMENTUM_LABEL_NEUTRAL),
        (-0.19, config.MOMENTUM_LABEL_NEUTRAL),
        (-0.2, config.MOMENTUM_LABEL_BEARISH),
        (-0.6, config.MOMENTUM_LABEL_STRONG_BEARISH),
        (None, config.MOMENTUM_LABEL_UNAVAILABLE),
    ],
)
def test_momentum_label_thresholds_are_symmetric(score, label):
    """Label boundaries are symmetric around 0 (+/-0.2 mild, +/-0.6 strong)."""
    assert sm.momentum_label(score) == label


def test_golden_cross_flag():
    """SMA-50 crossing above SMA-200 inside the lookback window raises a golden-cross flag."""
    df = _indicator_frame()
    df[ind.COL_SMA_SHORT] = [99.0] * (N_ROWS - 3) + [101.0] * 3  # crossed above SMA-200 3 sessions ago
    signal = sm.compute_momentum(df)
    assert any(f.startswith("Golden cross") and "2 sessions ago" in f for f in signal.flags)


# --------------------------------------------------------------------------- 52-week range
def test_week_52_uses_calendar_window_and_raw_prices():
    """A spike ~60 weeks ago is outside the window; spikes inside it set the high and low."""
    df = make_ohlcv(400)
    df.index = df.index.tz_localize(None)
    df.loc[df.index[-300], "High"] = 999.0  # ~60 weeks ago: outside the window
    df.loc[df.index[-50], "High"] = 500.0  # ~10 weeks ago: inside
    df.loc[df.index[-20], "Low"] = 1.0
    high, high_date, low, low_date = sm.compute_week_52_range(df)
    assert high == 500.0 and high_date == df.index[-50].date().isoformat()
    assert low == 1.0 and low_date == df.index[-20].date().isoformat()


def test_week_52_mismatch_warning():
    """Only gaps above the tolerance vs Yahoo's values produce a warning (the high here, not the low)."""
    warnings = sm.check_week_52_against_source(100.0, 50.0, {"fiftyTwoWeekHigh": 110.0, "fiftyTwoWeekLow": 50.2})
    assert len(warnings) == 1 and "high" in warnings[0]


# --------------------------------------------------------------------------- YTD
def test_ytd_uses_last_close_of_previous_year():
    """YTD is measured from the previous year's last close, not this year's first close."""
    index = pd.to_datetime(["2025-12-30", "2025-12-31", "2026-01-02", "2026-03-02"])
    df = pd.DataFrame({"Adj Close": [95.0, 100.0, 101.0, 110.0]}, index=index)
    ytd, base_date, base_price, reason = sm.compute_ytd_return(df)
    assert ytd == pytest.approx(0.10)
    assert base_date == "2025-12-31" and base_price == 100.0 and reason is None


def test_ytd_unavailable_without_prior_year():
    """Without data from the previous year, YTD is None with a reason."""
    df = pd.DataFrame({"Adj Close": [100.0, 101.0]}, index=pd.to_datetime(["2026-01-02", "2026-01-05"]))
    ytd, _, _, reason = sm.compute_ytd_return(df)
    assert ytd is None and "before the start" in reason


# --------------------------------------------------------------------------- P/E
@pytest.mark.parametrize(
    "info,price,expected_pe,expected_source,has_reason",
    [
        ({"trailingPE": 30.0, "trailingEps": 5.0}, 150.0, 30.0, "trailingPE", False),
        ({"trailingEps": 5.0}, 150.0, 30.0, "computed (price / trailingEps)", False),
        ({"trailingPE": "Infinity", "trailingEps": 5.0}, 150.0, 30.0, "computed (price / trailingEps)", False),
        ({"trailingEps": -2.0}, 150.0, None, None, True),
        ({}, 150.0, None, None, True),
        ({"trailingPE": float("nan")}, None, None, None, True),
    ],
)
def test_pe_fallback_chain(info, price, expected_pe, expected_source, has_reason):
    """P/E: trailingPE first, then price / EPS, otherwise None with a reason (bad values treated as missing)."""
    pe, source, reason, _ = sm.compute_pe(info, price)
    assert pe == (pytest.approx(expected_pe) if expected_pe else None)
    assert source == expected_source
    assert (reason is not None) == has_reason


def test_forward_pe_reported_separately():
    """Forward P/E is returned separately and does not replace the trailing P/E."""
    _, _, _, forward = sm.compute_pe({"trailingPE": 30.0, "forwardPE": 25.0}, 150.0)
    assert forward == 25.0


# --------------------------------------------------------------------------- full summary
def _market(df, info):
    """Clean a synthetic frame and wrap it in MarketData, as load_market_data would."""
    cleaned, quality = validate_and_clean_ohlcv(df)
    return MarketData(ticker="AAPL", ohlcv=cleaned, info=info, quality=quality), cleaned


def test_build_summary_has_all_required_fields_and_is_json_serialisable():
    """Every spec field is filled, values are consistent, and the summary serialises to JSON."""
    market, cleaned = _market(make_ohlcv(756), {"trailingPE": 30.0, "longName": "Apple Inc.", "currency": "USD"})
    news = NewsResult(ticker="AAPL", items=[NewsItem(title=f"h{i}", source="google_rss") for i in range(12)],
                      sources_used=["google_rss"])
    summary = sm.build_summary(market, ind.add_indicators(cleaned), news)

    core = summary.core_dict()
    for key in ("current_price", "week_52_high", "week_52_low", "pe_ratio", "ytd_return", "momentum_signal"):
        assert core[key] is not None, key
    assert summary.week_52_low <= summary.current_price <= summary.week_52_high
    assert summary.current_price == round(cleaned["Close"].iloc[-1], 2)  # raw, not adjusted
    assert summary.news_count == 12
    json.dumps(summary.model_dump(mode="json"))


def test_build_summary_degrades_gracefully_with_empty_info():
    """Missing company info leaves P/E and the name empty but the price fields still work."""
    market, cleaned = _market(make_ohlcv(756), {})
    summary = sm.build_summary(market, ind.add_indicators(cleaned))
    assert summary.pe_ratio is None and summary.pe_unavailable_reason
    assert summary.company_name is None
    assert summary.current_price is not None


def test_build_summary_with_short_history_marks_components_unavailable():
    """With 120 sessions, SMA-200 rules are unavailable but a score is still produced."""
    market, cleaned = _market(make_ohlcv(120), {"trailingPE": 30.0})
    summary = sm.build_summary(market, ind.add_indicators(cleaned))
    assert "price_vs_sma_200" in summary.momentum.unavailable
    assert summary.momentum.score is not None
    assert any("below the" in w for w in summary.warnings)
