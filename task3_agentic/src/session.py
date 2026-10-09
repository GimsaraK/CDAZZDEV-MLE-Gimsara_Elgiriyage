"""Per-session state shared by the tools: results history, cached prices, fault injection.

The agent only sees compact tool JSON. The session keeps the full objects (for example
the 3-year OHLCV frame) so later tools and the report check can use them without
another network call.
"""
# AI-ASSISTED: Claude Code (claude-sonnet-5.5), Prompt: 'Implement the Task 3A plan (the plan approved in Entry 13)', Date: 2026-10-09 (see CITATIONS.md Entry 14)
# AI-ASSISTED: Claude Code (claude-sonnet-5.5), Prompt: 'Implement the Task 3B plan', Date: 2026-10-09 (see CITATIONS.md Entry 16): per-agent records

import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Union

import pandas as pd

from . import config
from .schemas import ToolResult
from .tracing import TraceLogger

FAULT_ALWAYS = "always"
FAULT_FIRST = "first"

# Messages used when a fault is injected. They imitate the real failures each tool can hit.
DEFAULT_FAULT_MESSAGES = {
    "web_search": "Simulated DuckDuckGo failure: RatelimitException (202 Ratelimit)",
    "get_price_data": "Simulated Yahoo Finance failure: read timeout after 30s",
    "get_news": "Simulated news failure: all sources unreachable",
    "calculate_volatility": "Simulated volatility failure: price history unavailable",
    "llm_sentiment": "Simulated LLM failure: HTTP 503 from every provider",
}


class InjectedFault(RuntimeError):
    """Raised inside a tool call by FaultConfig to demonstrate recovery. Never raised in a normal run."""


@dataclass
class FaultConfig:
    """Which tools should fail on purpose, for the error-handling demonstration.

    `modes` maps a tool name to "always", "first" (fail the first call only), or an int n
    (fail the first n calls). An empty config injects nothing.
    """

    modes: Dict[str, Union[str, int]] = field(default_factory=dict)
    messages: Dict[str, str] = field(default_factory=dict)
    _calls: Dict[str, int] = field(default_factory=dict, init=False, repr=False)

    def check(self, tool: str) -> None:
        """Count this call and raise InjectedFault if the tool is configured to fail now."""
        mode = self.modes.get(tool)
        # Count every call to this tool (1st, 2nd, ...), so "first" and "n" modes know which call this is.
        self._calls[tool] = self._calls.get(tool, 0) + 1
        if mode is None:
            return
        call_number = self._calls[tool]
        if mode == FAULT_ALWAYS:
            failing = True
        elif mode == FAULT_FIRST:
            failing = call_number == 1
        else:
            failing = call_number <= int(mode)
        if failing:
            message = self.messages.get(tool) or DEFAULT_FAULT_MESSAGES.get(tool, f"Simulated failure in {tool}")
            raise InjectedFault(message)


@dataclass
class ToolCallRecord:
    """One completed tool call, in the order it happened."""

    tool: str
    args: Dict[str, Any]
    result: ToolResult
    duration_ms: float
    timestamp: str
    agent: str
    # True when the tool-access guard refused the call; it says nothing about the tool itself.
    denied: bool = False


class SessionContext:
    """Everything one research session shares. Tools receive it through a closure in build_tools()."""

    def __init__(
        self,
        ticker: str = config.TICKER,
        trace: Optional[TraceLogger] = None,
        faults: Optional[FaultConfig] = None,
        session_id: Optional[str] = None,
        agent_name: str = config.DEFAULT_AGENT_NAME,
    ) -> None:
        self.ticker = ticker.strip().upper()
        self.session_id = session_id or uuid.uuid4().hex[:12]
        self.agent_name = agent_name
        self.trace = trace if trace is not None else TraceLogger()
        self.faults = faults or FaultConfig()
        self.history: List[ToolCallRecord] = []
        # Full OHLCV + indicator frames by ticker, filled by get_price_data / calculate_volatility.
        self.frames: Dict[str, pd.DataFrame] = {}
        self.company_names: Dict[str, str] = {}
        # Task 1 LLM clients for llm_sentiment, built on first use.
        self.llm_clients: Optional[Dict[str, Any]] = None
        # ToolNode may run several tool calls of one turn in parallel threads.
        self._lock = threading.Lock()

    def record(
        self,
        tool: str,
        args: Dict[str, Any],
        result: ToolResult,
        duration_ms: float,
        agent: Optional[str] = None,
        denied: bool = False,
    ) -> ToolCallRecord:
        entry = ToolCallRecord(
            tool=tool,
            args=args,
            result=result,
            duration_ms=duration_ms,
            timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            agent=agent or self.agent_name,
            denied=denied,
        )
        with self._lock:
            self.history.append(entry)
        return entry

    def results_for(
        self, tool: str, ok_only: bool = True, agent: Optional[str] = None, since: int = 0
    ) -> List[ToolCallRecord]:
        """Records for one tool, newest first. `agent` filters by caller; `since` skips the first n history entries."""
        # Filters, in order: this tool only; never a call the access guard refused (it returned no data);
        # only successes unless ok_only=False; only the given agent's calls when one is named.
        # The list is reversed at the end so index [0] is always the most recent call.
        with self._lock:
            records = [
                r
                for r in self.history[since:]
                if r.tool == tool
                and not r.denied
                and (r.result.ok or not ok_only)
                and (agent is None or r.agent == agent)
            ]
        return list(reversed(records))

    def succeeded_tools(self, agent: Optional[str] = None) -> Set[str]:
        with self._lock:
            return {r.tool for r in self.history if r.result.ok and (agent is None or r.agent == agent)}

    def failed_tools(self) -> Set[str]:
        """Tools that never returned ok this session (only errors or empty results). Denied calls don't count."""
        with self._lock:
            attempted = {r.tool for r in self.history if not r.denied}
        # A tool that failed once but later succeeded is not a failure; only "tried, never worked" remains.
        # (succeeded_tools takes the lock itself, so it is called after the `with` block is released.)
        return attempted - self.succeeded_tools()

    def tool_sequence(self, agent: Optional[str] = None) -> List[str]:
        with self._lock:
            return [r.tool for r in self.history if agent is None or r.agent == agent]

    def history_length(self) -> int:
        with self._lock:
            return len(self.history)

    def known_headlines(self) -> List[str]:
        """Every headline or result title a news or search tool actually returned this session.

        llm_sentiment only scores titles from this list, so a model cannot score headlines it made up.
        """
        titles: List[str] = []
        with self._lock:
            for record in self.history:
                if not record.result.ok or record.tool not in ("get_news", "web_search"):
                    continue
                data = record.result.data
                # get_news results have `headlines` and web_search results have `hits`; getattr with a []
                # default lets one loop read whichever field the record has.
                titles += [h.title for h in getattr(data, "headlines", [])]
                titles += [h.title for h in getattr(data, "hits", [])]
        return titles
