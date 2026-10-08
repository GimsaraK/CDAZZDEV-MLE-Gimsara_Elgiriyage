"""Market data ingestion: OHLCV + company info from yfinance, validation, and snapshot fallback.

Flow (``load_market_data``):
    live yfinance fetch (with retries) -> validate & clean -> save snapshot
    on any live failure -> load the last committed snapshot (logged warning)
    if neither works   -> DataUnavailableError (caught by the pipeline)
"""
# AI-ASSISTED: Cursor Agent (claude-sonnet-5.5), Prompt: 'yes implement the plan @TASK1A_PLAN.md', Date: 2026-10-08 (see CITATIONS.md Entry 4)

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yfinance as yf
from pydantic import BaseModel, Field

from . import config
from .errors import DataUnavailableError
from .logging_utils import get_logger
from .retry import call_with_retry

logger = get_logger("data")

# All price columns (checked for non-positive values during cleaning).
PRICE_COLUMNS = ("Open", config.HIGH_COL, config.LOW_COL, config.PRICE_COL_RAW, config.PRICE_COL_ADJ)
# Columns the summary cannot be built without (current price and 52-week range need them).
CRITICAL_COLUMNS = (config.HIGH_COL, config.LOW_COL, config.PRICE_COL_RAW)
# Values for DataQualityReport.source, so the summary shows whether data is live or from the snapshot.
SOURCE_LIVE = "live"
SOURCE_SNAPSHOT = "snapshot"


class DataQualityReport(BaseModel):
    """What was fetched and what had to be cleaned - surfaced in the summary for transparency."""

    source: str
    fetched_at: Optional[str] = None
    rows: int = 0
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    span_years: float = 0.0
    meets_min_history: bool = False
    duplicate_rows_removed: int = 0
    non_positive_prices_removed: int = 0
    rows_dropped_missing_price: int = 0
    remaining_nan_counts: Dict[str, int] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)


@dataclass
class MarketData:
    """Cleaned OHLCV history plus the subset of company info the pipeline needs."""

    ticker: str
    ohlcv: pd.DataFrame
    info: Dict[str, Any] = field(default_factory=dict)
    quality: DataQualityReport = field(default_factory=lambda: DataQualityReport(source=SOURCE_LIVE))


def _utc_now_iso() -> str:
    """Current UTC time as an ISO-8601 string (used to timestamp fetches and snapshots)."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def display_path(path: Path) -> str:
    """Repository-relative path for logs (avoids leaking local absolute paths)."""
    try:
        return path.resolve().relative_to(config.TASK_DIR.parent).as_posix()
    except ValueError:
        # Path is outside the repository (e.g. a temp dir in tests): show only the file name.
        return path.name


def normalise_ticker(ticker: str) -> str:
    """Strip whitespace and upper-case the symbol (' aapl ' -> 'AAPL'); reject empty or non-string input."""
    if not isinstance(ticker, str) or not ticker.strip():
        raise DataUnavailableError(f"Invalid ticker: {ticker!r}")
    return ticker.strip().upper()


# --------------------------------------------------------------------------- fetch
def _download_history(ticker: str, period: str, interval: str) -> pd.DataFrame:
    """Single yfinance history call (wrapped by call_with_retry in fetch_ohlcv)."""
    # auto_adjust=False returns both raw 'Close' and split/dividend-adjusted 'Adj Close'.
    history = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=False)
    if history is None or history.empty:
        # yfinance returns an empty frame (not an error) for unknown/delisted symbols.
        raise DataUnavailableError(f"No price history returned for {ticker!r} (unknown or delisted symbol?)")
    return history


def fetch_ohlcv(
    ticker: str,
    period: str = config.HISTORY_PERIOD,
    interval: str = config.HISTORY_INTERVAL,
) -> pd.DataFrame:
    """Fetch daily OHLCV for a relative ``period`` (e.g. '3y') - never hardcoded dates."""
    ticker = normalise_ticker(ticker)
    logger.info("Fetching %s OHLCV history for %s (interval=%s)", period, ticker, interval)
    # Network errors are retried with backoff; DataUnavailableError (unknown symbol) is not retried.
    return call_with_retry(_download_history, f"OHLCV fetch for {ticker}", ticker, period, interval)


def _download_info(ticker: str) -> Dict[str, Any]:
    """Single yfinance .info call (company name, P/E, EPS, 52-week values, ...)."""
    info = yf.Ticker(ticker).info
    # Guard against yfinance schema changes: anything other than a dict is unusable.
    if not isinstance(info, dict):
        raise TypeError(f"Unexpected info payload type: {type(info).__name__}")
    return info


def fetch_ticker_info(ticker: str) -> Dict[str, Any]:
    """Return the needed subset of yfinance ``.info``; never raises (returns {} on failure)."""
    try:
        ticker = normalise_ticker(ticker)
        raw = call_with_retry(_download_info, f"info fetch for {ticker}", ticker)
    except Exception as exc:  # noqa: BLE001 - info is optional; any failure degrades gracefully
        logger.warning("Company info unavailable for %r: %s", ticker, exc)
        return {}
    # .info has ~150 keys; keep only the ones listed in config.INFO_KEYS and drop empty values.
    return {key: raw.get(key) for key in config.INFO_KEYS if raw.get(key) is not None}


# --------------------------------------------------------------------------- validate & clean
def validate_and_clean_ohlcv(
    df: Optional[pd.DataFrame],
    source: str = SOURCE_LIVE,
    fetched_at: Optional[str] = None,
) -> Tuple[pd.DataFrame, DataQualityReport]:
    """Validate structure, remove bad rows, and report every change made.

    - Raises DataUnavailableError only if the frame is empty or critical columns are missing.
    - Missing 'Adj Close' falls back to 'Close' (with a warning); missing Open/Volume become NaN.
    - Non-positive prices are treated as missing; rows without a close price are dropped.
    """
    report = DataQualityReport(source=source, fetched_at=fetched_at)
    if df is None or df.empty:
        raise DataUnavailableError("Price history is empty")

    # Step 1 - structural checks. Work on a copy so the caller's frame is untouched.
    out = df.copy()
    missing_critical = [col for col in CRITICAL_COLUMNS if col not in out.columns]
    if missing_critical:
        raise DataUnavailableError(f"Price history is missing critical columns: {missing_critical}")

    # Step 2 - fill in non-critical columns so later code can rely on them existing.
    if config.PRICE_COL_ADJ not in out.columns:
        out[config.PRICE_COL_ADJ] = out[config.PRICE_COL_RAW]
        report.warnings.append("'Adj Close' missing - using raw 'Close' for indicators")
    for optional_col in ("Open", config.VOLUME_COL):
        if optional_col not in out.columns:
            out[optional_col] = float("nan")
            report.warnings.append(f"'{optional_col}' missing - filled with NaN")

    # Step 3 - index: tz-naive exchange dates, ascending, unique.
    # yfinance returns timezone-aware timestamps (America/New_York); dropping the timezone and the
    # time-of-day makes dates comparable with plain dates (e.g. year boundaries for YTD).
    out.index = pd.to_datetime(out.index)
    if out.index.tz is not None:
        out.index = out.index.tz_localize(None)
    out.index = out.index.normalize()
    out.index.name = "Date"
    out = out.sort_index()
    # If a date appears more than once, keep a single row for it (the last occurrence).
    duplicated = out.index.duplicated(keep="last")
    report.duplicate_rows_removed = int(duplicated.sum())
    out = out[~duplicated]

    # Step 4 - force numeric types; strings or garbage values become NaN instead of crashing later.
    for col in (*PRICE_COLUMNS, config.VOLUME_COL):
        out[col] = pd.to_numeric(out[col], errors="coerce")

    # Step 5 - a price of zero or below is impossible, so treat it as missing (mask() sets it to NaN).
    non_positive = out[list(PRICE_COLUMNS)] <= 0
    report.non_positive_prices_removed = int(non_positive.to_numpy().sum())
    out[list(PRICE_COLUMNS)] = out[list(PRICE_COLUMNS)].mask(non_positive)

    # Step 6 - drop rows with no close price: without it the row is useless for indicators.
    missing_price = out[config.PRICE_COL_RAW].isna() | out[config.PRICE_COL_ADJ].isna()
    report.rows_dropped_missing_price = int(missing_price.sum())
    out = out[~missing_price]

    if out.empty:
        raise DataUnavailableError("No usable rows remain after removing missing prices")

    # Step 7 - record any NaNs that remain in other columns (reported, not dropped).
    nan_counts = out[list(config.REQUIRED_OHLCV_COLUMNS)].isna().sum()
    report.remaining_nan_counts = {col: int(n) for col, n in nan_counts.items() if n > 0}

    # Step 8 - summary statistics and the minimum-history check (2 years required by the spec).
    report.rows = len(out)
    report.start_date = out.index[0].date().isoformat()
    report.end_date = out.index[-1].date().isoformat()
    report.span_years = round((out.index[-1] - out.index[0]).days / config.DAYS_PER_YEAR, 2)
    report.meets_min_history = report.span_years >= config.MIN_HISTORY_YEARS
    if not report.meets_min_history:
        report.warnings.append(
            f"History spans {report.span_years} years, below the {config.MIN_HISTORY_YEARS}-year minimum"
        )

    # Turn each non-zero cleaning count into a readable warning (zero counts produce "" and are skipped).
    for message in (
        f"{report.duplicate_rows_removed} duplicate dates removed" if report.duplicate_rows_removed else "",
        f"{report.non_positive_prices_removed} non-positive prices treated as missing"
        if report.non_positive_prices_removed
        else "",
        f"{report.rows_dropped_missing_price} rows dropped (missing close)" if report.rows_dropped_missing_price else "",
    ):
        if message:
            report.warnings.append(message)

    for warning in report.warnings:
        logger.warning("Data quality: %s", warning)
    return out, report


# --------------------------------------------------------------------------- snapshots
def write_json_snapshot(path: Path, payload: Any) -> None:
    """Write ``{"fetched_at": ..., "data": payload}``; failures are logged, never raised."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"fetched_at": _utc_now_iso(), "data": payload}, indent=2, default=str))
    except OSError as exc:
        logger.warning("Could not write snapshot %s: %s", display_path(path), exc)


def read_json_snapshot(path: Path) -> Tuple[Any, Optional[str]]:
    """Return (data, fetched_at); raises DataUnavailableError if missing or unreadable."""
    try:
        content = json.loads(path.read_text())
        return content["data"], content.get("fetched_at")
    except (OSError, ValueError, KeyError) as exc:
        raise DataUnavailableError(f"Snapshot {path.name} unavailable: {exc}") from exc


def _snapshot_paths(ticker: str, data_dir: Path) -> Dict[str, Path]:
    """File locations of the three snapshot files for a ticker (names come from config templates)."""
    return {
        "ohlcv": data_dir / config.SNAPSHOT_OHLCV_TEMPLATE.format(ticker=ticker),
        "ohlcv_meta": data_dir / config.SNAPSHOT_OHLCV_META_TEMPLATE.format(ticker=ticker),
        "info": data_dir / config.SNAPSHOT_INFO_TEMPLATE.format(ticker=ticker),
    }


def save_market_snapshot(ticker: str, ohlcv: pd.DataFrame, info: Dict[str, Any], data_dir: Path) -> None:
    """Save the latest good data so a later run can fall back to it. Never raises (a failed save is only logged)."""
    paths = _snapshot_paths(ticker, data_dir)
    # OHLCV goes to CSV (readable diff in git); the fetch timestamp goes to a small JSON sidecar.
    try:
        data_dir.mkdir(parents=True, exist_ok=True)
        ohlcv.to_csv(paths["ohlcv"])
    except OSError as exc:
        logger.warning("Could not write OHLCV snapshot: %s", exc)
        return
    write_json_snapshot(paths["ohlcv_meta"], {"ticker": ticker, "rows": len(ohlcv)})
    # Do not overwrite a good info snapshot with an empty dict when live info failed.
    if info:
        write_json_snapshot(paths["info"], info)
    logger.info("Saved market data snapshot for %s to %s", ticker, display_path(data_dir))


def load_market_snapshot(ticker: str, data_dir: Path) -> Tuple[pd.DataFrame, Dict[str, Any], Optional[str]]:
    """Read a saved snapshot: returns (ohlcv, info, fetched_at). Only the OHLCV CSV is mandatory."""
    paths = _snapshot_paths(ticker, data_dir)
    if not paths["ohlcv"].exists():
        raise DataUnavailableError(f"No snapshot found for {ticker} in {display_path(data_dir)}")
    # First CSV column is the Date index; parse_dates turns it back into a DatetimeIndex.
    ohlcv = pd.read_csv(paths["ohlcv"], index_col=0, parse_dates=True)
    # The timestamp and company info are optional extras; missing files just mean less context.
    try:
        _, fetched_at = read_json_snapshot(paths["ohlcv_meta"])
    except DataUnavailableError:
        fetched_at = None
    try:
        info, _ = read_json_snapshot(paths["info"])
    except DataUnavailableError:
        info = {}
    return ohlcv, info or {}, fetched_at


# --------------------------------------------------------------------------- orchestration
def load_market_data(
    ticker: str = config.TICKER,
    period: str = config.HISTORY_PERIOD,
    data_dir: Path = config.DATA_DIR,
    use_snapshot_fallback: bool = True,
    save_snapshot: bool = True,
) -> MarketData:
    """Live fetch with snapshot fallback. Raises DataUnavailableError only if both fail."""
    ticker = normalise_ticker(ticker)
    try:
        # Happy path: live prices (with retries) -> clean -> live company info.
        raw = fetch_ohlcv(ticker, period)
        ohlcv, quality = validate_and_clean_ohlcv(raw, source=SOURCE_LIVE, fetched_at=_utc_now_iso())
        info = fetch_ticker_info(ticker)
        # Prices can succeed while .info fails (separate Yahoo endpoint), so info has its own fallback.
        if not info:
            info = _snapshot_info_fallback(ticker, data_dir, quality) if use_snapshot_fallback else {}
        # Refresh the snapshot so the fallback is as recent as possible.
        if save_snapshot:
            save_market_snapshot(ticker, ohlcv, info, data_dir)
        logger.info("Loaded %d live rows for %s (%s to %s)", quality.rows, ticker, quality.start_date, quality.end_date)
        return MarketData(ticker=ticker, ohlcv=ohlcv, info=info, quality=quality)
    except Exception as live_exc:  # noqa: BLE001 - any live failure triggers the fallback path
        if not use_snapshot_fallback:
            raise DataUnavailableError(f"Live fetch failed for {ticker}: {live_exc}") from live_exc
        # Fallback path: load the last saved snapshot and run it through the same cleaning.
        logger.warning("Live fetch failed for %s (%s) - trying snapshot", ticker, live_exc)
        try:
            snap_ohlcv, snap_info, fetched_at = load_market_snapshot(ticker, data_dir)
        except DataUnavailableError as snap_exc:
            # Both sources failed: report both reasons so the cause is obvious.
            raise DataUnavailableError(
                f"Live fetch failed ({live_exc}) and no usable snapshot exists ({snap_exc})"
            ) from live_exc
        ohlcv, quality = validate_and_clean_ohlcv(snap_ohlcv, source=SOURCE_SNAPSHOT, fetched_at=fetched_at)
        # Put the fallback warning first so the summary makes clear the data is not live.
        quality.warnings.insert(0, f"Live fetch failed ({live_exc}); using snapshot fetched at {fetched_at}")
        return MarketData(ticker=ticker, ohlcv=ohlcv, info=snap_info, quality=quality)


def _snapshot_info_fallback(ticker: str, data_dir: Path, quality: DataQualityReport) -> Dict[str, Any]:
    """Company info from the snapshot when only the live .info call failed; records a warning either way."""
    try:
        info, fetched_at = read_json_snapshot(_snapshot_paths(ticker, data_dir)["info"])
    except DataUnavailableError:
        quality.warnings.append("Company info unavailable (live and snapshot)")
        return {}
    quality.warnings.append(f"Company info from snapshot fetched at {fetched_at} (live info unavailable)")
    return info or {}
