"""Readable trace of an agent run for the notebook: DECIDE -> CALL -> OBSERVE -> DECIDE ...

TracePrinter is passed to run_research(on_message=...) and prints each message as it
arrives. A DECIDE that follows tool results is labelled with the cycle number and the
tools it observed, which makes the observe-and-replan loop visible.
"""
# AI-ASSISTED: Claude Code (claude-sonnet-5.5), Prompt: 'Implement the Task 3A plan (the plan approved in Entry 13)', Date: 2026-10-09 (see CITATIONS.md Entry 14)

import json
from typing import Any, Dict, List, Optional

import pandas as pd
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from . import config, prompts
from .schemas import ToolResult
from .session import SessionContext

_RULE = "-" * 100


def _clip(text: str, limit: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def decision_text(message: AIMessage) -> str:
    """The model's own note, or an excerpt of its reasoning when the note is empty (gpt-oss tool turns)."""
    content = str(message.content or "").strip()
    if content:
        return _clip(content, config.DISPLAY_DECISION_MAX_CHARS)
    reasoning = message.additional_kwargs.get("reasoning_content") or message.additional_kwargs.get("reasoning")
    if reasoning:
        return "(reasoning) " + _clip(reasoning, config.DISPLAY_DECISION_MAX_CHARS)
    return "(no note)"


def summarize_result(result: ToolResult) -> str:
    """One line per tool result, with the numbers that matter for the next decision."""
    if not result.ok:
        return f"{result.status.upper()}: {result.error} | hint: {result.hint}"
    data = result.data
    tool = result.tool
    if tool == "get_price_data":
        f = data.fundamentals
        return (
            f"{data.company_name or data.ticker}: price {data.current_price}, {data.period} return {data.period_return_pct}%, "
            f"RSI {data.indicators.get('RSI_14')}, momentum {data.momentum_label}, fwd P/E {f.forward_pe}, "
            f"margin {f.profit_margin_pct}% [source={result.source}]"
        )
    if tool == "get_news":
        first = data.headlines[0].title if data.headlines else ""
        return f"{data.count} headlines from {data.sources_used}; newest: '{first}'"
    if tool == "calculate_volatility":
        return (
            f"{data.window}d vol {data.annualised_vol_pct}% ({data.regime}, {data.percentile_1y} pctile 1y); "
            f"90d 1-sigma move +/-{data.expected_move_pct}% -> band {data.band_1sigma_low}-{data.band_1sigma_high}"
        )
    if tool == "llm_sentiment":
        return (
            f"score {data.score} ({data.label}) from {data.n_scored} headlines: "
            f"{data.positive} positive / {data.negative} negative / {data.neutral} neutral"
        )
    if tool == "web_search":
        top = data.hits[0].title if data.hits else ""
        return f"{len(data.hits)} hits via {data.backend} for '{data.effective_query}'; top: '{top}'"
    return result.to_llm_json(config.DISPLAY_OBSERVATION_MAX_CHARS)


def _observation(message: ToolMessage) -> str:
    artifact = getattr(message, "artifact", None)
    if isinstance(artifact, ToolResult):
        return summarize_result(artifact)
    # Schema errors handled by ToolNode arrive as plain text with status="error".
    return _clip(message.content, config.DISPLAY_OBSERVATION_MAX_CHARS)


class TracePrinter:
    """Prints messages as they arrive. Use as run_research(on_message=TracePrinter()).

    `prefix` labels every line with the agent in 3B; `finish_text` says what happens when the agent stops.
    """

    def __init__(self, prefix: str = "", finish_text: str = "write the report from the gathered evidence") -> None:
        # The printer keeps a little state between messages, because a "cycle" spans several of them:
        # tool results arrive (OBSERVE), then the agent's next turn decides (DECIDE) based on them.
        self.step = 0  # agent turns printed so far
        self.cycle = 0  # observe -> decide cycles completed
        self.prefix = prefix
        self.finish_text = finish_text
        self._observed: List[str] = []  # tools observed since the agent's last decision

    def __call__(self, message: BaseMessage) -> None:
        for block in self.format(message):
            for line in block.splitlines():
                print(f"{self.prefix}{line}")

    def format(self, message: BaseMessage) -> List[str]:
        if isinstance(message, HumanMessage):
            text = str(message.content)
            if text.startswith(prompts.REPORT_CHECK_MARKER):
                self._observed = []
                return [_RULE, "REPLAN   the report check sent the agent back:", *("         " + l for l in text.splitlines()[1:] if l.strip()), _RULE]
            return [_RULE, f"QUERY    {_clip(text, 400)}", _RULE]

        if isinstance(message, AIMessage):
            self.step += 1
            lines = []
            model = message.response_metadata.get("model_name")
            label = f"DECIDE   [step {self.step}{' | ' + model if model else ''}]"
            # A decision that follows tool results closes one observe -> decide cycle: number it and name
            # the tools it was based on, which is the visible evidence of replanning from observations.
            if self._observed:
                self.cycle += 1
                label += f" cycle {self.cycle}: after observing {', '.join(self._observed)}"
                self._observed = []
            lines.append(f"{label}\n         {decision_text(message)}")
            if message.tool_calls:
                for call in message.tool_calls:
                    args = json.dumps(call.get("args", {}), default=str)
                    lines.append(f"CALL     {call['name']}({_clip(args, 200)})")
            else:
                lines.append(f"FINISH   no tool call -> {self.finish_text}")
            return lines

        if isinstance(message, ToolMessage):
            name = message.name or "tool"
            self._observed.append(name)
            return [f"OBSERVE  {name} -> {_observation(message)}"]
        return [f"{type(message).__name__}: {_clip(message.content, 200)}"]


def print_messages(messages: List[BaseMessage]) -> None:
    """Re-print a finished run's trace (same format as the live printer)."""
    printer = TracePrinter()
    for message in messages:
        printer(message)


def tool_call_table(ctx: SessionContext) -> pd.DataFrame:
    """One row per tool call in the order the agent made them."""
    rows: List[Dict[str, Any]] = []
    for number, record in enumerate(ctx.history, start=1):
        rows.append(
            {
                "#": number,
                "agent": record.agent,
                "tool": record.tool,
                "args": json.dumps(record.args, default=str)[:80],
                "status": "denied" if record.denied else record.result.status,
                "source": record.result.source,
                "duration_ms": round(record.duration_ms),
            }
        )
    return pd.DataFrame(rows)


def trace_table(records: List[Dict[str, Any]], session_id: Optional[str] = None) -> pd.DataFrame:
    """agent_trace.jsonl records as a table, optionally for one session."""
    chosen = [r for r in records if session_id is None or r.get("session_id") == session_id]
    frame = pd.DataFrame(chosen)
    columns = [c for c in ("timestamp", "session_id", "agent", "tool", "args", "status", "duration_ms", "output") if c in frame]
    return frame[columns] if not frame.empty else frame


# --------------------------------------------------------------------------- 3B: two agents
# AI-ASSISTED: Claude Code (claude-sonnet-5.5), Prompt: 'Implement the Task 3B plan', Date: 2026-10-09 (see CITATIONS.md Entry 16)
_HANDOFF_RULE = "=" * 100


class MultiAgentPrinter:
    """Live 3B trace: each agent's DECIDE / CALL / OBSERVE lines with its name, and a block per handoff."""

    def __init__(self) -> None:
        self.printers = {
            config.AGENT_A_NAME: TracePrinter(prefix="[A data_analyst]    ", finish_text="hand off to the next step"),
            config.AGENT_B_NAME: TracePrinter(prefix="[B research_writer] ", finish_text="hand off to the next step"),
        }
        self.count = 0

    def on_message(self, agent: str, message: BaseMessage) -> None:
        printer = self.printers.get(agent) or TracePrinter(prefix=f"[{agent}] ")
        printer(message)

    def on_handoff(self, message: Any) -> None:
        self.count += 1
        flag = "  (fallback)" if message.fallback else ""
        print(_HANDOFF_RULE)
        print(f"HANDOFF #{self.count}  {message.sender} -> {message.recipient}  [{message.kind}]{flag}")
        print(f"  summary: {message.summary}")
        print(f"  payload: {_clip(json.dumps(message.payload, ensure_ascii=False, default=str), config.DISPLAY_HANDOFF_MAX_CHARS)}")
        print(_HANDOFF_RULE)

    def on_event(self, text: str) -> None:
        print(f"[pipeline] {text}")


def handoff_table(handoffs: List[Any]) -> pd.DataFrame:
    """One row per inter-agent message, in order."""
    return pd.DataFrame(
        [
            {
                "#": number,
                "sender": h.sender,
                "recipient": h.recipient,
                "kind": h.kind,
                "fallback": h.fallback,
                "payload fields": ", ".join(h.payload)[:90],
                "summary": h.summary[:120],
            }
            for number, h in enumerate(handoffs, start=1)
        ]
    )


def schema_table(model: Any) -> pd.DataFrame:
    """A Pydantic model's fields: name, type, required, description (for the handoff schema section)."""
    rows = []
    for name, info in model.model_fields.items():
        annotation = getattr(info.annotation, "__name__", None) or str(info.annotation).replace("typing.", "")
        rows.append(
            {
                "field": name,
                "type": annotation,
                "required": info.is_required(),
                "description": (info.description or "")[:110],
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- 3C: memory and cache evidence
# AI-ASSISTED: Claude Code (claude-sonnet-5.5), Prompt: 'Implement the Task 3C plan', Date: 2026-10-09 (see CITATIONS.md Entry 18)
def followup_table(answer: Any) -> pd.DataFrame:
    """The proof for a follow-up: tools run during it, trace lines before/after, and which remembered values it cites."""
    rows = [
        ("Question", answer.question),
        ("Answered by", answer.model or "-"),
        ("Tool calls during the follow-up", ", ".join(answer.tool_calls_made) or "none"),
        ("agent_trace.jsonl lines before -> after", f"{answer.trace_lines_before} -> {answer.trace_lines_after}"),
        ("Answered from memory", answer.answered_from_memory),
        ("Error", answer.error or "none"),
    ]
    rows += [
        (f"Cites remembered {name} = {value}", "yes" if cited else "no")
        for name, (value, cited) in answer.cited_values.items()
    ]
    return pd.DataFrame(rows, columns=["Check", "Result"])


def cache_table(outcome: Any) -> pd.DataFrame:
    """One cache lookup: hit or miss, file, tool calls made by this call, and time taken."""
    brief = outcome.brief
    rows = [
        ("Result", "HIT" if outcome.hit else "MISS (pipeline ran)"),
        ("Cache file", outcome.path.name),
        ("Tool calls made by this call", outcome.tool_calls_this_call),
        ("Seconds", outcome.seconds),
        ("Note", outcome.note),
    ]
    if brief is not None:
        rows += [
            ("Cached by session", brief.session_id),
            ("Cache date (US/Eastern)", brief.cache_date),
            ("Report status", brief.status),
            ("Tool calls of the original run", brief.tool_calls),
        ]
    return pd.DataFrame(rows, columns=["Item", "Value"])
