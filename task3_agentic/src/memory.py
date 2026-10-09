"""3C short-term memory: a follow-up question answered from the conversation the agent already has.

The 3A research graph is compiled with a LangGraph checkpointer (InMemorySaver) and run on
thread_id = session id. ask_followup() sends one more message on that thread in "followup"
mode. The agent sees every earlier tool result in its own history and answers. With no tool
call, the graph ends there, so no new report is written. The tools stay bound, so not calling
one is the agent's own decision; the result reports honestly whether any tool ran.
"""
# AI-ASSISTED: Claude Code (claude-sonnet-5.5), Prompt: 'Implement the Task 3C plan', Date: 2026-10-09 (see CITATIONS.md Entry 18)

import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from . import config, prompts
from .agent import AgentRun
from .logging_utils import get_logger
from .react import count_tool_rounds
from .tracing import EventLogger, safe_error, trace_line_count

logger = get_logger("memory")

_NUMBER = re.compile(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?")
_MINUS_SIGNS = ("−", "‑", "–")


@dataclass
class FollowUpAnswer:
    question: str
    answer: str
    model: Optional[str]
    tool_calls_made: List[str]
    trace_lines_before: int
    trace_lines_after: int
    # field -> (remembered value, whether the answer cites it)
    cited_values: Dict[str, tuple] = field(default_factory=dict)
    messages: List[BaseMessage] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def answered_from_memory(self) -> bool:
        """True when the agent answered without any tool running during the follow-up."""
        return self.error is None and not self.tool_calls_made and self.trace_lines_after == self.trace_lines_before


def numbers_in(text: str) -> List[float]:
    for sign in _MINUS_SIGNS:
        text = text.replace(sign, "-")
    values = []
    for match in _NUMBER.findall(text):
        try:
            values.append(float(match.replace(",", "")))
        except ValueError:
            continue
    return values


def cites(text: str, value: float) -> bool:
    """Does `text` contain `value`, allowing for rounding (e.g. 313.46 written as 313.5)?"""
    # The larger of an absolute and a relative tolerance: small values (8.33%) need an absolute margin,
    # large ones (a $313.46 price) need a relative one so "313.5" still counts.
    tolerance = max(config.FOLLOWUP_MATCH_TOLERANCE, abs(value) * config.FOLLOWUP_RELATIVE_TOLERANCE)
    return any(abs(number - value) <= tolerance for number in numbers_in(text))


def volatility_memory(run: AgentRun) -> Dict[str, float]:
    """The values of the latest calculate_volatility result in the run: what a volatility follow-up should cite."""
    records = run.ctx.results_for("calculate_volatility")
    if not records:
        return {}
    data = records[0].result.data
    return {
        "window": float(data.window),
        "annualised_vol_pct": data.annualised_vol_pct,
        "expected_move_pct": data.expected_move_pct,
        "band_1sigma_low": data.band_1sigma_low,
        "band_1sigma_high": data.band_1sigma_high,
    }


def _trace_count(run: AgentRun) -> int:
    trace = run.ctx.trace
    return trace_line_count(trace.path) if trace.path is not None else len(trace.records)


def ask_followup(
    run: AgentRun,
    question: str,
    remembered: Optional[Dict[str, float]] = None,
    on_message: Optional[Callable[[BaseMessage], None]] = None,
    events: Optional[EventLogger] = None,
) -> FollowUpAnswer:
    """Ask a follow-up on the research run's own thread. Never raises."""
    ctx = run.ctx
    # Snapshot both counters before the question; comparing them afterwards is the proof that no tool ran.
    history_before = ctx.history_length()
    trace_before = _trace_count(run)

    # One place that builds the result for every exit path (success, model down, budget used), so the
    # before/after evidence and the event-log line are always recorded the same way.
    def result(answer: str = "", model: Optional[str] = None, new: Optional[List[BaseMessage]] = None, error: Optional[str] = None) -> FollowUpAnswer:
        made = [r.tool for r in ctx.history[history_before:]]
        values = remembered or {}
        outcome = FollowUpAnswer(
            question=question,
            answer=answer,
            model=model,
            tool_calls_made=made,
            trace_lines_before=trace_before,
            trace_lines_after=_trace_count(run),
            cited_values={name: (value, cites(answer, value)) for name, value in values.items()},
            messages=new or [],
            error=error,
        )
        if events is not None:
            events.log(
                ctx.session_id,
                "followup_answered" if error is None else "followup_failed",
                question=question,
                tool_calls=made,
                answered_from_memory=outcome.answered_from_memory,
                error=error,
            )
        return outcome

    if run.graph is None or run.thread_id is None:
        return result(error="The research run has no checkpointed conversation to continue")
    thread = {"configurable": {"thread_id": run.thread_id}, "recursion_limit": config.RECURSION_LIMIT}
    try:
        # get_state reads the conversation the checkpointer saved at the end of the research run.
        prior = run.graph.get_state(thread).values.get("messages", [])
        question_message = HumanMessage(content=prompts.FOLLOWUP.format(question=question))
        state = {}
        seen = len(prior)
        # Streaming on the same thread_id *continues* that conversation: the checkpointer restores the saved
        # state and the new message is appended to it (add_messages), so the agent sees its old tool results.
        # The flags are reset because the research run may have ended with budget_exhausted=True.
        for state in run.graph.stream(
            {
                "messages": [question_message],
                "mode": config.MODE_FOLLOWUP,
                "followup_base_rounds": count_tool_rounds(prior),
                "budget_exhausted": False,
                "llm_error": None,
            },
            config=thread,
            stream_mode="values",
        ):
            messages = state.get("messages", [])
            if on_message is not None:
                for message in messages[seen:]:
                    on_message(message)
            seen = len(messages)
    except Exception as exc:  # noqa: BLE001 - a follow-up must never crash the notebook
        return result(error=safe_error(exc))

    # Only the messages added by this follow-up; the answer is the agent's last reply without a tool call.
    new = list(state.get("messages", []))[len(prior):]
    replies = [m for m in new if isinstance(m, AIMessage) and not m.tool_calls]
    if state.get("llm_error") and not replies:
        return result(new=new, error=state["llm_error"])
    if not replies:
        return result(new=new, error="The agent used its follow-up budget without answering")
    last = replies[-1]
    return result(answer=str(last.content).strip(), model=last.response_metadata.get("model_name"), new=new)
