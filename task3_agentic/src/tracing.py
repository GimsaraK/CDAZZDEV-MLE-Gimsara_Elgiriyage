"""agent_trace.jsonl: one JSON line per tool call, plus the wrapper every tool runs through.

Each line holds the tool name, its input arguments, the output truncated to 200
characters, and the wall-clock duration, as the specification asks. It also records the
timestamp, session id, agent name and status, so 3B and 3C can filter by agent or run.
"""
# AI-ASSISTED: Claude Code (claude-opus-5-5), Prompt: 'Implement the Task 3A plan (the plan approved in Entry 13)', Date: 2026-10-09 (see CITATIONS.md Entry 14)
# AI-ASSISTED: Claude Code (claude-opus-5-5), Prompt: 'Implement the Task 3B plan', Date: 2026-10-09 (see CITATIONS.md Entry 16): agent attribution and the tool-access guard

import json
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, FrozenSet, List, Optional, Tuple

from . import config
from .logging_utils import get_logger
from .schemas import ToolResult

if TYPE_CHECKING:
    from .session import SessionContext

logger = get_logger("tracing")

# Provider error bodies can contain organisation ids; key-shaped strings must never reach a log either.
_ORG_ID = re.compile(r"org_[A-Za-z0-9]+")
_KEY_LIKE = re.compile(r"\b(gsk_|sk-or-|sk-|hf_)[A-Za-z0-9_\-]{8,}")
SAFE_ERROR_MAX_CHARS = 300
# Daily quotas (Groq tokens per day, OpenRouter free requests per day) do not reset within a run.
DAILY_LIMIT_REASON = "daily limit reached"


def safe_error(exc: BaseException, limit: int = SAFE_ERROR_MAX_CHARS) -> str:
    """Exception text that is safe to log, show, and commit.

    HTTP errors from LLM providers are reduced to the status code and a short reason,
    because their bodies include account details. Anything else is redacted and cut.
    """
    status = getattr(exc, "status_code", None)
    text = str(exc)
    if isinstance(status, int):
        lowered = text.lower()
        if "per day" in lowered or "tpd" in lowered or "per-day" in lowered:
            reason = DAILY_LIMIT_REASON
        elif status == 429:
            reason = "rate limited"
        else:
            reason = "request rejected"
        text = f"HTTP {status}, {reason}"
    text = _KEY_LIKE.sub("[redacted]", _ORG_ID.sub("org_[redacted]", text))
    return f"{type(exc).__name__}: {text}"[:limit]


def truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit]


class TraceLogger:
    """Appends tool-call records to a JSONL file and keeps them in memory for display."""

    def __init__(self, path: Optional[Path] = config.TRACE_FILE) -> None:
        # path=None keeps records in memory only (used by tests).
        self.path = Path(path) if path is not None else None
        self.records: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(
        self,
        *,
        session_id: str,
        agent: str,
        tool: str,
        args: Dict[str, Any],
        status: str,
        output: str,
        duration_ms: float,
        error: Optional[str] = None,
    ) -> Dict[str, Any]:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "session_id": session_id,
            "agent": agent,
            "tool": tool,
            "args": json.loads(json.dumps(args, default=str)),
            "status": status,
            "output": truncate(output, config.TRACE_OUTPUT_MAX_CHARS),
            "duration_ms": round(duration_ms, 1),
        }
        if error:
            record["error"] = truncate(error, config.TRACE_OUTPUT_MAX_CHARS)
        line = json.dumps(record, ensure_ascii=False)
        with self._lock:
            self.records.append(record)
            if self.path is not None:
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(line + "\n")
        return record

    @staticmethod
    def read(path: Path = config.TRACE_FILE) -> List[Dict[str, Any]]:
        """All records in a trace file. Unreadable lines are skipped, not fatal."""
        path = Path(path)
        if not path.exists():
            return []
        records = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("Skipping unreadable trace line: %.80s", line)
        return records


def error_result(tool: str, exc: BaseException, hint: Optional[str] = None) -> ToolResult:
    return ToolResult(status=config.STATUS_ERROR, tool=tool, error=safe_error(exc), hint=hint)


class ToolAccessError(PermissionError):
    """An agent called a tool outside its allowed set (3B tool restriction, enforced in code)."""


ACCESS_DENIED_HINT = "This tool belongs to the other agent. Ask that agent for the data instead."


def run_traced(
    ctx: "SessionContext",
    tool: str,
    func: Callable[..., ToolResult],
    args: Dict[str, Any],
    error_hint: Optional[str] = None,
    agent: Optional[str] = None,
    allowed: Optional[FrozenSet[str]] = None,
) -> ToolResult:
    """Run one tool call: check access, inject faults, time it, turn any exception into an error envelope, log it.

    This is the single place that guarantees a tool never raises into the agent loop. `agent`
    attributes the call in the trace. `allowed`, when given, is that agent's tool list; a call
    outside it is refused here, before the tool body runs, and the attempt is still logged.
    """
    agent_name = agent or ctx.agent_name
    start = time.perf_counter()
    denied = allowed is not None and tool not in allowed
    if denied:
        exc = ToolAccessError(f"{agent_name} may not call {tool} (allowed: {', '.join(sorted(allowed))})")
        logger.warning("Blocked: %s", exc)
        result = error_result(tool, exc, ACCESS_DENIED_HINT)
    else:
        try:
            ctx.faults.check(tool)
            result = func(**args)
        except Exception as exc:  # noqa: BLE001 - any failure becomes an error envelope the agent can read
            logger.warning("Tool %s failed: %s", tool, safe_error(exc))
            result = error_result(tool, exc, error_hint)
    duration_ms = (time.perf_counter() - start) * 1000
    ctx.record(tool, args, result, duration_ms, agent=agent_name, denied=denied)
    ctx.trace.log(
        session_id=ctx.session_id,
        agent=agent_name,
        tool=tool,
        args=args,
        status=result.status,
        output=result.to_llm_json(),
        duration_ms=duration_ms,
        error=result.error,
    )
    return result


# --------------------------------------------------------------------------- 3C: session events and the trace audit
# AI-ASSISTED: Claude Code (claude-opus-5-5), Prompt: 'Implement the Task 3C plan', Date: 2026-10-09 (see CITATIONS.md Entry 18)
class EventLogger:
    """Non-tool events (cache hit/miss/write, follow-up answered) in logs/session_events.jsonl.

    Kept apart from agent_trace.jsonl, so that file stays one line per tool call and an
    unchanged line count is clean proof that no tool ran.
    """

    def __init__(self, path: Optional[Path] = config.EVENTS_FILE) -> None:
        self.path = Path(path) if path is not None else None
        self.records: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, session_id: str, event: str, **detail: Any) -> Dict[str, Any]:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "session_id": session_id,
            "event": event,
            "detail": json.loads(json.dumps(detail, default=str)),
        }
        with self._lock:
            self.records.append(record)
            if self.path is not None:
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record


def trace_line_count(path: Optional[Path] = None) -> int:
    """Lines in a trace file (0 if missing). Used before/after a step to prove no tool ran."""
    path = Path(path or config.TRACE_FILE)
    if not path.exists():
        return 0
    with path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def audit_trace(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Check agent_trace.jsonl records against the specification and summarise them.

    Problems: a missing required field, an output longer than TRACE_OUTPUT_MAX_CHARS, or a
    missing or negative duration. Counts are by session, agent, tool and status.
    """
    missing: List[Tuple[int, List[str]]] = []
    too_long: List[int] = []
    bad_duration: List[int] = []
    counts: Dict[str, Dict[str, int]] = {"session_id": {}, "agent": {}, "tool": {}, "status": {}}
    for number, record in enumerate(records, start=1):
        absent = [name for name in config.TRACE_REQUIRED_FIELDS if name not in record]
        if absent:
            missing.append((number, absent))
        if len(str(record.get("output", ""))) > config.TRACE_OUTPUT_MAX_CHARS:
            too_long.append(number)
        duration = record.get("duration_ms")
        if not isinstance(duration, (int, float)) or duration < 0:
            bad_duration.append(number)
        for key, bucket in counts.items():
            value = str(record.get(key))
            bucket[value] = bucket.get(value, 0) + 1
    return {
        "lines": len(records),
        "missing_fields": missing,
        "output_over_limit": too_long,
        "bad_duration": bad_duration,
        "problems": len(missing) + len(too_long) + len(bad_duration),
        "by_session": counts["session_id"],
        "by_agent": counts["agent"],
        "by_tool": counts["tool"],
        "by_status": counts["status"],
    }
