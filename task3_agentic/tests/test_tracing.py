"""agent_trace.jsonl: one complete record per tool call, output truncated to 200 characters."""
# AI-ASSISTED: Claude Code (claude-sonnet-5.5), Prompt: 'Implement the Task 3A plan (the plan approved in Entry 13)', Date: 2026-10-09 (see CITATIONS.md Entry 14)

import json

from task3_agentic.src import config
from task3_agentic.src.schemas import ToolResult
from task3_agentic.src.session import SessionContext
from task3_agentic.src.tracing import TraceLogger, run_traced, safe_error

REQUIRED_FIELDS = {"timestamp", "session_id", "agent", "tool", "args", "status", "output", "duration_ms"}


def test_every_call_is_written_with_required_fields(tmp_path):
    path = tmp_path / "logs" / "agent_trace.jsonl"
    ctx = SessionContext("AAPL", trace=TraceLogger(path))
    long_payload = "x" * 1000

    def tool(query):
        return ToolResult(status="ok", tool="web_search", warnings=[long_payload])

    run_traced(ctx, "web_search", tool, {"query": "apple"})
    run_traced(ctx, "web_search", tool, {"query": "apple risk"})

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    record = json.loads(lines[0])
    assert REQUIRED_FIELDS <= set(record)
    assert record["tool"] == "web_search" and record["args"] == {"query": "apple"}
    assert record["session_id"] == ctx.session_id and record["agent"] == config.DEFAULT_AGENT_NAME
    assert len(record["output"]) == config.TRACE_OUTPUT_MAX_CHARS
    assert record["duration_ms"] >= 0
    assert TraceLogger.read(path) == ctx.trace.records


def test_exceptions_are_logged_as_errors_and_not_raised():
    ctx = SessionContext("AAPL", trace=TraceLogger(None))

    def broken(ticker):
        raise ConnectionError("socket closed")

    result = run_traced(ctx, "get_news", broken, {"ticker": "AAPL"}, error_hint="try web_search")
    assert result.status == "error" and result.hint == "try web_search"
    record = ctx.trace.records[-1]
    assert record["status"] == "error" and "socket closed" in record["error"]
    assert ctx.history[-1].result is result


def test_safe_error_hides_provider_account_details():
    class FakeRateLimit(Exception):
        status_code = 429

    exc = FakeRateLimit(
        "Rate limit reached for model `openai/gpt-oss-120b` in organization `org_01abcDEF` "
        "on tokens per day (TPD): Limit 200000, Used 199848"
    )
    text = safe_error(exc)
    assert text == "FakeRateLimit: HTTP 429, daily limit reached"
    plain = safe_error(RuntimeError("failed for org_01abcDEF with key gsk_abcdefghijklmnop"))
    assert "org_01abcDEF" not in plain and "gsk_abcdefghijklmnop" not in plain


def test_read_skips_corrupt_lines(tmp_path):
    path = tmp_path / "trace.jsonl"
    path.write_text('{"tool": "a"}\nnot json\n\n{"tool": "b"}\n', encoding="utf-8")
    assert [r["tool"] for r in TraceLogger.read(path)] == ["a", "b"]
    assert TraceLogger.read(tmp_path / "missing.jsonl") == []
