"""Offline tests for data.py: validation, cleaning, retries, and snapshot fallback."""
# AI-ASSISTED: Cursor Agent (claude-sonnet-5.5), Prompt: 'yes implement the plan @TASK1A_PLAN.md', Date: 2026-10-08 (see CITATIONS.md Entry 4)

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from task1_financial.src import config, data
from task1_financial.src.errors import DataUnavailableError

from .conftest import make_ohlcv

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
ISO_DATE_PATTERN = re.compile(r"\b(19|20)\d{2}-\d{2}-\d{2}\b")


# --------------------------------------------------------------------------- validate & clean
def test_clean_frame_passes_and_index_is_tz_naive(ohlcv_3y):
    cleaned, report = data.validate_and_clean_ohlcv(ohlcv_3y)
    assert cleaned.index.tz is None
    assert cleaned.index.is_monotonic_increasing
    assert report.rows == len(ohlcv_3y)
    assert report.meets_min_history
    assert report.span_years >= config.MIN_HISTORY_YEARS
    assert report.warnings == []


def test_duplicates_unsorted_and_missing_prices_are_cleaned(ohlcv_3y):
    messy = pd.concat([ohlcv_3y.iloc[::-1], ohlcv_3y.iloc[:3]])  # reversed + 3 duplicate dates
    messy.iloc[10, messy.columns.get_loc("Close")] = np.nan
    messy.iloc[20, messy.columns.get_loc("Adj Close")] = -1.0
    cleaned, report = data.validate_and_clean_ohlcv(messy)
    assert cleaned.index.is_monotonic_increasing
    assert not cleaned.index.duplicated().any()
    assert report.duplicate_rows_removed == 3
    assert report.non_positive_prices_removed == 1
    assert report.rows_dropped_missing_price == 2
    assert cleaned["Close"].notna().all() and cleaned["Adj Close"].notna().all()


def test_short_history_is_flagged_not_fatal():
    cleaned, report = data.validate_and_clean_ohlcv(make_ohlcv(100))
    assert not report.meets_min_history
    assert any("below the" in w for w in report.warnings)
    assert len(cleaned) == 100


def test_missing_adj_close_falls_back_to_close(ohlcv_3y):
    cleaned, report = data.validate_and_clean_ohlcv(ohlcv_3y.drop(columns=["Adj Close"]))
    assert (cleaned["Adj Close"] == cleaned["Close"]).all()
    assert any("Adj Close" in w for w in report.warnings)


def test_missing_volume_is_filled_with_nan(ohlcv_3y):
    cleaned, report = data.validate_and_clean_ohlcv(ohlcv_3y.drop(columns=["Volume"]))
    assert cleaned["Volume"].isna().all()
    assert report.remaining_nan_counts["Volume"] == len(cleaned)


@pytest.mark.parametrize("frame", [None, pd.DataFrame()])
def test_empty_frame_raises_data_unavailable(frame):
    with pytest.raises(DataUnavailableError):
        data.validate_and_clean_ohlcv(frame)


def test_missing_critical_column_raises(ohlcv_3y):
    with pytest.raises(DataUnavailableError):
        data.validate_and_clean_ohlcv(ohlcv_3y.drop(columns=["High"]))


# --------------------------------------------------------------------------- fetch
class _FakeTicker:
    def __init__(self, history=None, info=None, info_error=None):
        self._history = history
        self._info = info
        self._info_error = info_error

    def history(self, **_kwargs):
        return self._history

    @property
    def info(self):
        if self._info_error:
            raise self._info_error
        return self._info


def test_unknown_ticker_raises_without_retrying(monkeypatch):
    calls = []

    def fake_ticker(symbol):
        calls.append(symbol)
        return _FakeTicker(history=pd.DataFrame())

    monkeypatch.setattr(data.yf, "Ticker", fake_ticker)
    with pytest.raises(DataUnavailableError):
        data.fetch_ohlcv("NOTREAL")
    assert len(calls) == 1


def test_transient_error_is_retried(monkeypatch, ohlcv_3y):
    attempts = {"n": 0}

    def fake_ticker(_symbol):
        attempts["n"] += 1
        if attempts["n"] < config.RETRY_MAX_ATTEMPTS:
            raise ConnectionError("temporary network failure")
        return _FakeTicker(history=ohlcv_3y)

    monkeypatch.setattr(data.yf, "Ticker", fake_ticker)
    result = data.fetch_ohlcv("AAPL")
    assert attempts["n"] == config.RETRY_MAX_ATTEMPTS
    assert len(result) == len(ohlcv_3y)


def test_ticker_info_failure_returns_empty_dict(monkeypatch):
    monkeypatch.setattr(data.yf, "Ticker", lambda _s: _FakeTicker(info_error=ConnectionError("down")))
    assert data.fetch_ticker_info("AAPL") == {}


def test_ticker_info_keeps_only_configured_keys(monkeypatch):
    payload = {"trailingPE": 30.5, "longName": "Apple Inc.", "irrelevantKey": 1, "forwardPE": None}
    monkeypatch.setattr(data.yf, "Ticker", lambda _s: _FakeTicker(info=payload))
    assert data.fetch_ticker_info("AAPL") == {"trailingPE": 30.5, "longName": "Apple Inc."}


@pytest.mark.parametrize("bad", ["", "   ", None])
def test_invalid_ticker_string_rejected(bad):
    with pytest.raises(DataUnavailableError):
        data.normalise_ticker(bad)


# --------------------------------------------------------------------------- snapshot fallback
def test_snapshot_round_trip(tmp_path, ohlcv_3y):
    cleaned, _ = data.validate_and_clean_ohlcv(ohlcv_3y)
    data.save_market_snapshot("AAPL", cleaned, {"trailingPE": 30.0}, tmp_path)
    loaded, info, fetched_at = data.load_market_snapshot("AAPL", tmp_path)
    assert len(loaded) == len(cleaned)
    assert np.allclose(loaded["Adj Close"], cleaned["Adj Close"])
    assert info == {"trailingPE": 30.0}
    assert fetched_at is not None


def test_live_failure_falls_back_to_snapshot(monkeypatch, tmp_path, ohlcv_3y):
    cleaned, _ = data.validate_and_clean_ohlcv(ohlcv_3y)
    data.save_market_snapshot("AAPL", cleaned, {"trailingPE": 30.0}, tmp_path)

    def failing_fetch(*_args, **_kwargs):
        raise ConnectionError("rate limited")

    monkeypatch.setattr(data, "fetch_ohlcv", failing_fetch)
    market = data.load_market_data("AAPL", data_dir=tmp_path)
    assert market.quality.source == data.SOURCE_SNAPSHOT
    assert "Live fetch failed" in market.quality.warnings[0]
    assert market.info == {"trailingPE": 30.0}


def test_live_failure_without_snapshot_raises_data_unavailable(monkeypatch, tmp_path):
    monkeypatch.setattr(data, "fetch_ohlcv", lambda *a, **k: (_ for _ in ()).throw(ConnectionError("down")))
    with pytest.raises(DataUnavailableError):
        data.load_market_data("AAPL", data_dir=tmp_path)


def test_live_success_saves_snapshot(monkeypatch, tmp_path, ohlcv_3y):
    monkeypatch.setattr(data, "fetch_ohlcv", lambda *a, **k: ohlcv_3y)
    monkeypatch.setattr(data, "fetch_ticker_info", lambda _t: {"trailingPE": 31.0})
    market = data.load_market_data("AAPL", data_dir=tmp_path)
    assert market.quality.source == data.SOURCE_LIVE
    assert (tmp_path / "AAPL_ohlcv.csv").exists()
    assert (tmp_path / "AAPL_info.json").exists()


# --------------------------------------------------------------------------- spec rule
def test_no_hardcoded_date_strings_in_source():
    # Comment lines are excluded: citation headers legitimately carry a date.
    offenders = [
        f"{path.name}:{lineno}"
        for path in SRC_DIR.glob("*.py")
        for lineno, line in enumerate(path.read_text().splitlines(), start=1)
        if ISO_DATE_PATTERN.search(line) and not line.lstrip().startswith("#")
    ]
    assert offenders == [], f"Hardcoded date strings found: {offenders}"
