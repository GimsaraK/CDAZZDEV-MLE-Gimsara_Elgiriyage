"""3C persistent cache: keyed by ticker, market date and pipeline; a hit makes no tool call."""
# AI-ASSISTED: Claude Code (claude-opus-5-5), Prompt: 'Implement the Task 3C plan', Date: 2026-10-09 (see CITATIONS.md Entry 18)

import json
from types import SimpleNamespace

import pandas as pd
import pytest
from langchain_core.runnables import RunnableLambda

from task3_agentic.src import cache, config
from task3_agentic.src.cache import cache_path, cached_research, load_cached, market_date, research_with_cache
from task3_agentic.src.schemas import FinalReport
from task3_agentic.src.tools import ResearchTools
from task3_agentic.src.tracing import EventLogger

from .conftest import valid_report
from .test_memory import RESEARCH, Script

MULTI = config.PIPELINE_MULTI_AGENT
SINGLE = config.PIPELINE_SINGLE_AGENT


@pytest.fixture(autouse=True)
def cache_dir(tmp_path, monkeypatch):
    folder = tmp_path / "cache"
    monkeypatch.setattr(config, "CACHE_DIR", folder)
    return folder


def make_runner(ctx, with_report=True):
    """A pipeline stand-in that makes two real (offline) tool calls and returns a FinalReport."""
    calls = []

    def runner():
        calls.append(1)
        toolkit = ResearchTools(ctx)
        toolkit.get_price_data("AAPL", "1y")
        toolkit.calculate_volatility("AAPL", 20)
        final = FinalReport(
            ticker=ctx.ticker,
            query="q",
            session_id=ctx.session_id,
            generated_at="2026-10-09T10:00:00+00:00",
            status=config.REPORT_STATUS_VALIDATED if with_report else config.REPORT_STATUS_FAILED,
            report=valid_report(ctx) if with_report else None,
        )
        return SimpleNamespace(final=final, messages=[])

    return runner, calls


def test_key_uses_ticker_market_date_and_pipeline(cache_dir):
    assert market_date() == pd.Timestamp.now(tz="America/New_York").strftime("%Y-%m-%d")
    path = cache_path(" aapl ", MULTI, "2026-10-09")
    assert path == cache_dir / "AAPL_2026-10-09_multi_agent.json"
    with pytest.raises(ValueError):
        cache_path("AAPL", "three_agents")


def test_miss_runs_and_writes_then_hit_loads_without_tools(ctx, offline_sources, cache_dir):
    events = EventLogger(None)
    runner, calls = make_runner(ctx)
    first = cached_research(ctx, MULTI, runner, events=events)
    assert not first.hit and first.tool_calls_this_call == 2 and first.path.exists()
    second = cached_research(ctx, MULTI, runner, events=events)
    assert second.hit and second.tool_calls_this_call == 0 and len(calls) == 1
    assert second.brief.final.report == first.brief.final.report
    assert second.brief.tool_calls == 2
    assert [e["event"] for e in events.records] == ["cache_miss", "cache_write", "cache_hit"]
    # No temp file is left next to the cache file.
    assert [p.name for p in cache_dir.iterdir()] == [first.path.name]


def test_a_different_date_or_pipeline_is_a_miss(ctx, offline_sources):
    runner, calls = make_runner(ctx)
    cached_research(ctx, MULTI, runner, date="2026-10-08")
    assert load_cached("AAPL", MULTI, "2026-10-09") is None
    assert load_cached("AAPL", SINGLE, "2026-10-08") is None
    assert load_cached("MSFT", MULTI, "2026-10-08") is None
    assert load_cached("AAPL", MULTI, "2026-10-08") is not None


def test_failed_runs_are_not_cached(ctx, offline_sources):
    events = EventLogger(None)
    runner, _ = make_runner(ctx, with_report=False)
    outcome = cached_research(ctx, MULTI, runner, events=events)
    assert outcome.brief is None and not outcome.path.exists()
    assert events.records[-1]["event"] == "cache_skip"


@pytest.mark.parametrize("content", ["{not json", json.dumps({"ticker": "AAPL"})])
def test_corrupt_file_is_a_miss_not_a_crash(ctx, offline_sources, content):
    events = EventLogger(None)
    path = cache_path("AAPL", MULTI)
    path.parent.mkdir(parents=True)
    path.write_text(content, encoding="utf-8")
    runner, calls = make_runner(ctx)
    outcome = cached_research(ctx, MULTI, runner, events=events)
    assert not outcome.hit and len(calls) == 1
    assert "cache_corrupt" in [e["event"] for e in events.records]
    assert load_cached("AAPL", MULTI) is not None  # rewritten with a valid brief


def test_old_schema_version_is_a_miss(ctx, offline_sources, monkeypatch):
    runner, _ = make_runner(ctx)
    cached_research(ctx, MULTI, runner)
    monkeypatch.setattr(config, "CACHE_SCHEMA_VERSION", config.CACHE_SCHEMA_VERSION + 1)
    assert load_cached("AAPL", MULTI) is None


def test_force_refresh_reruns_and_overwrites(ctx, offline_sources):
    events = EventLogger(None)
    runner, calls = make_runner(ctx)
    cached_research(ctx, MULTI, runner)
    outcome = cached_research(ctx, MULTI, runner, force_refresh=True, events=events)
    assert not outcome.hit and len(calls) == 2
    assert events.records[0]["event"] == "cache_bypass"


def test_research_with_cache_wraps_the_single_agent(ctx, offline_sources):
    def write(_messages):
        return valid_report(ctx, risk_tools=("get_price_data", "calculate_volatility", "llm_sentiment"))

    first = research_with_cache(ctx, agent_model=RunnableLambda(Script(RESEARCH)), report_model=RunnableLambda(write))
    assert not first.hit and first.brief.pipeline == SINGLE and first.brief.tool_calls == 4
    assert first.brief.models == {"scripted-model": 1, "unknown": 4}
    second = research_with_cache(ctx)
    assert second.hit and second.tool_calls_this_call == 0
    assert second.brief.final.report.ticker == "AAPL"
