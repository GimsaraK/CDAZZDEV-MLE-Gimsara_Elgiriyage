"""3C persistent memory: the final research brief cached as JSON, keyed by ticker and date.

    cache/<TICKER>_<YYYY-MM-DD>_<pipeline>.json     date = the market's calendar date (US/Eastern)

cached_research() looks for today's file first. If it is there and valid, the brief is
returned with no tool call at all. Otherwise the pipeline runs, and its report (validated, or
accepted with warnings) is written atomically for the next run. A failed run is never cached.
A corrupt or out-of-date file counts as a miss, never as an error. The single-agent (3A) and
multi-agent (3B) pipelines share this layer, but a lookup only matches its own pipeline.
"""
# AI-ASSISTED: Claude Code (claude-sonnet-5.5), Prompt: 'Implement the Task 3C plan', Date: 2026-10-09 (see CITATIONS.md Entry 18)

import json
import os
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional

import pandas as pd
from pydantic import BaseModel, Field, ValidationError

from . import config
from .logging_utils import get_logger
from .react import models_used
from .schemas import FinalReport
from .session import SessionContext
from .tracing import EventLogger

logger = get_logger("cache")

Pipeline = Literal["single_agent", "multi_agent"]


class CachedBrief(BaseModel):
    """What is stored. `final` is the same FinalReport the pipeline returned."""

    schema_version: int
    ticker: str
    cache_date: str
    pipeline: Pipeline
    created_at: str
    session_id: str
    query: str
    status: str
    final: FinalReport
    handoffs: Optional[List[Dict[str, Any]]] = None
    tool_calls: int = Field(description="Tool calls the original run made; a cache hit makes none.")
    models: Dict[str, int] = Field(default_factory=dict)


@dataclass
class CacheOutcome:
    hit: bool
    brief: Optional[CachedBrief]
    path: Path
    run: Any
    tool_calls_this_call: int
    seconds: float
    note: str


def market_date() -> str:
    """Today's date on the market's calendar. pandas carries its own time-zone data (zoneinfo needs tzdata on Windows)."""
    return pd.Timestamp.now(tz=config.CACHE_TIMEZONE).strftime("%Y-%m-%d")


def cache_path(ticker: str, pipeline: str, date: Optional[str] = None, cache_dir: Optional[Path] = None) -> Path:
    if pipeline not in config.PIPELINES:
        raise ValueError(f"Unknown pipeline {pipeline!r}; expected one of {config.PIPELINES}")
    name = config.CACHE_FILE_TEMPLATE.format(ticker=ticker.strip().upper(), date=date or market_date(), pipeline=pipeline)
    return Path(cache_dir or config.CACHE_DIR) / name


def _log(events: Optional[EventLogger], session_id: str, event: str, **detail: Any) -> None:
    if events is not None:
        events.log(session_id, event, **detail)


def load_cached(
    ticker: str,
    pipeline: str,
    date: Optional[str] = None,
    cache_dir: Optional[Path] = None,
    events: Optional[EventLogger] = None,
    session_id: str = "",
) -> Optional[CachedBrief]:
    """The cached brief for this key, or None. A missing, corrupt, old-schema or mismatched file is a miss."""
    path = cache_path(ticker, pipeline, date, cache_dir)
    if not path.exists():
        return None
    try:
        brief = CachedBrief.model_validate(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError, ValidationError) as exc:
        logger.warning("Ignoring unreadable cache file %s: %s", path.name, str(exc).splitlines()[0])
        _log(events, session_id, "cache_corrupt", path=path.name, reason=type(exc).__name__)
        return None
    # The file name already encodes ticker, date and pipeline, but the contents are checked too: a renamed
    # or hand-copied file, or one written by an older schema version, must not be served as today's brief.
    expected = (config.CACHE_SCHEMA_VERSION, ticker.strip().upper(), pipeline, date or market_date())
    found = (brief.schema_version, brief.ticker, brief.pipeline, brief.cache_date)
    if found != expected:
        logger.warning("Ignoring cache file %s: key %s does not match %s", path.name, found, expected)
        _log(events, session_id, "cache_corrupt", path=path.name, reason="key mismatch")
        return None
    return brief


def save_cached(brief: CachedBrief, path: Path) -> Path:
    """Write atomically: a temp file in the same folder, then os.replace, so a reader never sees half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # mkstemp creates a uniquely named temp file in the *same folder* (os.replace is only atomic within one
    # filesystem). It returns an open OS-level handle plus the file name.
    handle, temp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.stem}.", suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as temp:
            temp.write(brief.model_dump_json(indent=2))
        # One atomic step: readers see either the old file or the complete new one, never a partial write.
        os.replace(temp_name, path)
    finally:
        # Only true if something failed before the replace; never leave a stray .tmp file behind.
        if os.path.exists(temp_name):
            os.remove(temp_name)
    return path


def _messages_of(run: Any) -> List[Any]:
    if hasattr(run, "analyst_messages"):
        return list(run.analyst_messages) + list(run.writer_messages)
    return list(getattr(run, "messages", []))


def cached_research(
    ctx: SessionContext,
    pipeline: str,
    runner: Callable[[], Any],
    force_refresh: bool = False,
    events: Optional[EventLogger] = None,
    cache_dir: Optional[Path] = None,
    date: Optional[str] = None,
) -> CacheOutcome:
    """Load today's brief for ctx.ticker if it exists; otherwise run `runner()` and cache its report.

    `runner` returns an AgentRun (3A) or a MultiAgentRun (3B); both carry `.final`.
    """
    started = time.perf_counter()
    date = date or market_date()
    path = cache_path(ctx.ticker, pipeline, date, cache_dir)
    # Tool calls are counted as "history length after minus before", so a cache hit provably made zero.
    history_before = ctx.history_length()

    # Step 1: look for today's file (unless the caller forces a fresh run).
    if force_refresh:
        _log(events, ctx.session_id, "cache_bypass", path=path.name, reason="force_refresh")
    else:
        cached = load_cached(ctx.ticker, pipeline, date, cache_dir, events, ctx.session_id)
        if cached is not None:
            _log(events, ctx.session_id, "cache_hit", path=path.name, cached_session=cached.session_id, status=cached.status)
            return CacheOutcome(
                hit=True,
                brief=cached,
                path=path,
                run=None,
                tool_calls_this_call=ctx.history_length() - history_before,
                seconds=round(time.perf_counter() - started, 3),
                note=f"Loaded {path.name} (written by session {cached.session_id}); no tools were run",
            )
        _log(events, ctx.session_id, "cache_miss", path=path.name)

    # Step 2: no usable cache, so run the pipeline. `runner` is a zero-argument function that runs the 3A
    # agent or the 3B pipeline; the cache does not need to know which.
    run = runner()
    calls = ctx.history_length() - history_before
    final: FinalReport = run.final
    # Step 3: cache only a run that produced a report. A failed run must not be served tomorrow as a "hit".
    if final.report is None:
        _log(events, ctx.session_id, "cache_skip", path=path.name, reason=f"run status {final.status}")
        return CacheOutcome(False, None, path, run, calls, round(time.perf_counter() - started, 3), "Run failed; nothing cached")

    handoffs = [h.model_dump(mode="json") for h in getattr(run, "handoffs", [])] or None
    brief = CachedBrief(
        schema_version=config.CACHE_SCHEMA_VERSION,
        ticker=ctx.ticker,
        cache_date=date,
        pipeline=pipeline,
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        session_id=ctx.session_id,
        query=final.query,
        status=final.status,
        final=final,
        handoffs=handoffs,
        tool_calls=calls,
        models=models_used(_messages_of(run)),
    )
    save_cached(brief, path)
    _log(events, ctx.session_id, "cache_write", path=path.name, status=final.status, tool_calls=calls)
    return CacheOutcome(False, brief, path, run, calls, round(time.perf_counter() - started, 3), f"Ran the pipeline and wrote {path.name}")


def research_with_cache(ctx: SessionContext, force_refresh: bool = False, events: Optional[EventLogger] = None, **run_kwargs: Any) -> CacheOutcome:
    """3A single agent behind the cache."""
    from .agent import run_research

    return cached_research(ctx, config.PIPELINE_SINGLE_AGENT, lambda: run_research(ctx, **run_kwargs), force_refresh, events)


def multi_agent_with_cache(ctx: SessionContext, force_refresh: bool = False, events: Optional[EventLogger] = None, **run_kwargs: Any) -> CacheOutcome:
    """3B two-agent pipeline behind the cache."""
    from .multi_agent import run_multi_agent

    return cached_research(ctx, config.PIPELINE_MULTI_AGENT, lambda: run_multi_agent(ctx, **run_kwargs), force_refresh, events)
