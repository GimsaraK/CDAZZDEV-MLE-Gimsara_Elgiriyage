"""3C observability: the trace audit, the session-events log, and the dashboard's data functions."""
# AI-ASSISTED: Claude Code (claude-sonnet-5.5), Prompt: 'Implement the Task 3C plan', Date: 2026-10-09 (see CITATIONS.md Entry 18)

import json

import pytest

from task3_agentic.src import config
from task3_agentic.src.dashboard_data import DENIED, filter_trace, load_events, load_trace, per_session, summarise, timeline
from task3_agentic.src.session import SessionContext
from task3_agentic.src.tools import ResearchTools
from task3_agentic.src.tracing import EventLogger, TraceLogger, audit_trace, trace_line_count


@pytest.fixture
def trace_file(tmp_path, offline_sources):
    """A real trace from two sessions: an analyst with one denied call, and a writer."""
    path = tmp_path / "agent_trace.jsonl"
    trace = TraceLogger(path)
    first = SessionContext("AAPL", trace=trace)
    first.llm_clients = {"fake": object()}
    analyst = ResearchTools(first, agent_name=config.AGENT_A_NAME, allowed=config.AGENT_A_TOOLS)
    analyst.get_price_data("AAPL", "1y")
    analyst.calculate_volatility("AAPL", 1000)  # error
    analyst.web_search("Apple")  # denied
    second = SessionContext("AAPL", trace=trace)
    ResearchTools(second, agent_name=config.AGENT_B_NAME, allowed=config.AGENT_B_TOOLS).get_news("AAPL", 4)
    return path


def test_audit_passes_a_real_trace(trace_file):
    audit = audit_trace(TraceLogger.read(trace_file))
    assert audit["lines"] == 4 and audit["problems"] == 0
    assert audit["by_agent"] == {config.AGENT_A_NAME: 3, config.AGENT_B_NAME: 1}
    assert audit["by_status"] == {"ok": 2, "error": 2}
    assert len(audit["by_session"]) == 2


def test_audit_flags_each_kind_of_problem():
    good = {name: "x" for name in config.TRACE_REQUIRED_FIELDS} | {"duration_ms": 1.0}
    records = [
        good,
        {k: v for k, v in good.items() if k != "args"},
        good | {"output": "y" * (config.TRACE_OUTPUT_MAX_CHARS + 1)},
        good | {"duration_ms": -3},
        good | {"duration_ms": None},
    ]
    audit = audit_trace(records)
    assert audit["missing_fields"] == [(2, ["args"])]
    assert audit["output_over_limit"] == [3]
    assert audit["bad_duration"] == [4, 5]
    assert audit["problems"] == 4


def test_event_logger_and_line_count(tmp_path):
    path = tmp_path / "logs" / "session_events.jsonl"
    events = EventLogger(path)
    events.log("s1", "cache_hit", path="AAPL.json")
    events.log("s1", "followup_answered", tool_calls=[])
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [line["event"] for line in lines] == ["cache_hit", "followup_answered"]
    assert lines[0]["detail"] == {"path": "AAPL.json"}
    assert trace_line_count(path) == 2 and trace_line_count(tmp_path / "missing.jsonl") == 0
    frame = load_events(path)
    assert list(frame["event"]) == ["cache_hit", "followup_answered"]


def test_dashboard_data(trace_file):
    frame = load_trace(trace_file)
    assert list(frame["status"]) == ["ok", "error", DENIED, "ok"]
    kpi = summarise(frame)
    assert (kpi["calls"], kpi["sessions"], kpi["errors"], kpi["denied"]) == (4, 2, 1, 1)
    assert kpi["p95_ms"] >= kpi["p50_ms"] >= 0
    assert len(filter_trace(frame, agents=[config.AGENT_B_NAME])) == 1
    assert len(filter_trace(frame, statuses=["ok"], tools=["get_news"])) == 1
    sessions = per_session(frame)
    assert list(sessions["calls"]) == [3, 1] and list(sessions["not_ok"]) == [2, 0]
    chart = timeline(frame)
    assert (chart["start_s"] >= 0).all() and (chart["end_s"] >= chart["start_s"]).all()
    assert list(chart["label"])[:2] == ["1. get_price_data", "2. calculate_volatility"]


def test_dashboard_handles_missing_files(tmp_path):
    assert load_trace(tmp_path / "none.jsonl").empty and load_events(tmp_path / "none.jsonl").empty
    assert summarise(load_trace(tmp_path / "none.jsonl"))["calls"] == 0
