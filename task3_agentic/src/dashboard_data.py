"""Data functions behind the Streamlit trace dashboard (dashboard/app.py). Pure pandas, tested offline."""
# AI-ASSISTED: Claude Code (claude-opus-5-5), Prompt: 'Implement the Task 3C plan', Date: 2026-10-09 (see CITATIONS.md Entry 18)

import json
from pathlib import Path
from typing import Dict, Iterable, Optional

import pandas as pd

from . import config
from .tracing import TraceLogger

TRACE_COLUMNS = ["timestamp", "session_id", "agent", "tool", "status", "duration_ms", "args", "output", "error"]
EVENT_COLUMNS = ["timestamp", "session_id", "event", "detail"]
# Status names in the trace, plus "denied" for calls the tool-access guard refused.
DENIED = "denied"


def load_trace(path: Path = config.TRACE_FILE) -> pd.DataFrame:
    """agent_trace.jsonl as a DataFrame, oldest first. Missing file -> empty frame with the right columns."""
    frame = pd.DataFrame(TraceLogger.read(Path(path)))
    if frame.empty:
        return pd.DataFrame(columns=TRACE_COLUMNS)
    for column in TRACE_COLUMNS:
        if column not in frame:
            frame[column] = None
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame["duration_ms"] = pd.to_numeric(frame["duration_ms"], errors="coerce")
    frame["args"] = frame["args"].apply(lambda value: json.dumps(value, ensure_ascii=False) if isinstance(value, dict) else value)
    # The guard's refusals are logged as errors; give them their own status so they stand out.
    denied = frame["error"].fillna("").str.startswith("ToolAccessError")
    frame.loc[denied, "status"] = DENIED
    return frame.sort_values("timestamp").reset_index(drop=True)[TRACE_COLUMNS]


def load_events(path: Path = config.EVENTS_FILE) -> pd.DataFrame:
    frame = pd.DataFrame(TraceLogger.read(Path(path)))
    if frame.empty:
        return pd.DataFrame(columns=EVENT_COLUMNS)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame["detail"] = frame["detail"].apply(lambda value: json.dumps(value, ensure_ascii=False))
    return frame.sort_values("timestamp").reset_index(drop=True)[EVENT_COLUMNS]


def filter_trace(
    frame: pd.DataFrame,
    sessions: Optional[Iterable[str]] = None,
    agents: Optional[Iterable[str]] = None,
    tools: Optional[Iterable[str]] = None,
    statuses: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Keep rows matching every non-empty filter."""
    mask = pd.Series(True, index=frame.index)
    for column, allowed in (("session_id", sessions), ("agent", agents), ("tool", tools), ("status", statuses)):
        allowed = list(allowed or [])
        if allowed:
            mask &= frame[column].isin(allowed)
    return frame[mask]


def summarise(frame: pd.DataFrame) -> Dict[str, float]:
    """Headline numbers for the KPI row."""
    durations = frame["duration_ms"].dropna()
    return {
        "calls": int(len(frame)),
        "sessions": int(frame["session_id"].nunique()) if len(frame) else 0,
        "errors": int((frame["status"] == config.STATUS_ERROR).sum()),
        "empty": int((frame["status"] == config.STATUS_EMPTY).sum()),
        "denied": int((frame["status"] == DENIED).sum()),
        "total_s": round(float(durations.sum()) / 1000, 2) if len(durations) else 0.0,
        "p50_ms": round(float(durations.median()), 1) if len(durations) else 0.0,
        "p95_ms": round(float(durations.quantile(0.95)), 1) if len(durations) else 0.0,
    }


def per_session(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per session: when it started, which agents ran, calls, failures, and total tool time."""
    if frame.empty:
        return pd.DataFrame(columns=["session_id", "started", "agents", "calls", "not_ok", "tool_time_s"])
    grouped = frame.groupby("session_id", sort=False)
    table = grouped.agg(
        started=("timestamp", "min"),
        agents=("agent", lambda values: ", ".join(sorted(set(values)))),
        calls=("tool", "size"),
        not_ok=("status", lambda values: int((values != config.STATUS_OK).sum())),
        tool_time_s=("duration_ms", lambda values: round(values.sum() / 1000, 2)),
    )
    return table.reset_index().sort_values("started")


def timeline(frame: pd.DataFrame) -> pd.DataFrame:
    """Each call's start offset (seconds since its session's first call) and duration, for a Gantt-style chart.

    The trace stamps a call when it finishes, so start = finish - duration.
    """
    if frame.empty:
        return frame.assign(start_s=[], end_s=[], call=[])
    finished = frame["timestamp"]
    started = finished - pd.to_timedelta(frame["duration_ms"].fillna(0), unit="ms")
    session_start = started.groupby(frame["session_id"]).transform("min")
    out = frame.copy()
    out["start_s"] = (started - session_start).dt.total_seconds().round(3)
    out["end_s"] = (finished - session_start).dt.total_seconds().round(3)
    out["call"] = out.groupby("session_id").cumcount() + 1
    out["label"] = out["call"].astype(str) + ". " + out["tool"]
    return out
