"""3C short-term memory: a follow-up on the research run's checkpointed thread, answered without tools."""
# AI-ASSISTED: Claude Code (claude-sonnet-5.5), Prompt: 'Implement the Task 3C plan', Date: 2026-10-09 (see CITATIONS.md Entry 18)

import itertools
from typing import List

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableLambda

from task3_agentic.src import config
from task3_agentic.src.agent import run_research
from task3_agentic.src.memory import ask_followup, cites, numbers_in, volatility_memory
from task3_agentic.src.tracing import EventLogger

from .conftest import FAKE_HEADLINES, valid_report

RESEARCH = [
    ("get_price_data", {"ticker": "AAPL", "period": "1y"}),
    ("calculate_volatility", {"ticker": "AAPL", "window": 20}),
    ("get_news", {"ticker": "AAPL", "n": 4}),
    ("llm_sentiment", {"headlines": FAKE_HEADLINES}),
    "Enough evidence: summary.",
]
QUESTION = "Which volatility window did you compute, and what 90-day 1-sigma price band did it give?"


class Script:
    """Scripted model: a step is (tool, args), an answer string, or a function of the messages (returns either)."""

    def __init__(self, steps):
        self.steps = list(steps)
        self.calls: List[List[BaseMessage]] = []
        self._ids = itertools.count(1)

    def __call__(self, messages):
        self.calls.append(list(messages))
        step = self.steps.pop(0) if self.steps else "done"
        if callable(step):
            step = step(messages)
        if isinstance(step, tuple):
            name, args = step
            return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"c{next(self._ids)}", "type": "tool_call"}])
        return AIMessage(content=step, response_metadata={"model_name": "scripted-model"})


def researched(ctx, followup_steps):
    model = Script(RESEARCH + list(followup_steps))
    report_calls = []

    def write(_messages):
        report_calls.append(1)
        return valid_report(ctx, risk_tools=("get_price_data", "calculate_volatility", "llm_sentiment"))

    run = run_research(ctx, agent_model=RunnableLambda(model), report_model=RunnableLambda(write))
    return run, model, report_calls


def answer_from_history(ctx):
    """Answers from the volatility result it can see in its own history (fails the test if it cannot)."""

    def step(messages):
        seen = [m for m in messages if isinstance(m, ToolMessage) and m.name == "calculate_volatility"]
        assert seen, "the earlier tool result is not in the follow-up's context"
        data = ctx.results_for("calculate_volatility")[0].result.data
        return (
            f"From the calculate_volatility result: window {data.window} days, annualised {data.annualised_vol_pct}%, "
            f"90-day 1-sigma move {data.expected_move_pct}%, band {data.band_1sigma_low} to {data.band_1sigma_high}."
        )

    return step


def test_followup_is_answered_from_memory(ctx, offline_sources):
    run, model, report_calls = researched(ctx, [answer_from_history(ctx)])
    assert run.final.status == config.REPORT_STATUS_VALIDATED and run.thread_id == ctx.session_id
    answer = ask_followup(run, QUESTION, remembered=volatility_memory(run))
    assert answer.error is None and answer.answered_from_memory
    assert answer.tool_calls_made == [] and answer.trace_lines_before == answer.trace_lines_after
    assert all(cited for _, cited in answer.cited_values.values())
    assert answer.model == "scripted-model"
    # The follow-up ended the turn: no second report was written.
    assert len(report_calls) == 1


def test_followup_sees_the_whole_thread(ctx, offline_sources):
    run, model, _ = researched(ctx, ["The window was 20 days."])
    ask_followup(run, QUESTION)
    context = model.calls[-1]
    assert context[0].type == "system"
    assert isinstance(context[-1], HumanMessage) and QUESTION in context[-1].content
    assert {m.name for m in context if isinstance(m, ToolMessage)} == {"get_price_data", "calculate_volatility", "get_news", "llm_sentiment"}


def test_a_followup_that_calls_a_tool_is_reported_honestly(ctx, offline_sources):
    run, _, _ = researched(ctx, [("calculate_volatility", {"ticker": "AAPL", "window": 20}), "Recomputed: 20 days."])
    answer = ask_followup(run, QUESTION)
    assert answer.error is None and not answer.answered_from_memory
    assert answer.tool_calls_made == ["calculate_volatility"]
    assert answer.trace_lines_after == answer.trace_lines_before + 1


def test_followup_budget_stops_a_tool_loop(ctx, offline_sources):
    looping = [("get_news", {"ticker": "AAPL", "n": 3})] * 5
    run, _, _ = researched(ctx, looping)
    answer = ask_followup(run, QUESTION)
    assert len(answer.tool_calls_made) == config.FOLLOWUP_TOOL_ROUNDS
    assert answer.error and "budget" in answer.error


def test_followup_with_a_dead_model_returns_an_error(ctx, offline_sources):
    def down(_messages):
        raise ConnectionError("all providers unreachable")

    run, _, _ = researched(ctx, [down])
    answer = ask_followup(run, QUESTION)
    assert answer.error and "unreachable" in answer.error and not answer.answered_from_memory


def test_followup_events_and_missing_thread(ctx, offline_sources):
    events = EventLogger(None)
    run, _, _ = researched(ctx, ["20 days."])
    ask_followup(run, QUESTION, events=events)
    assert events.records[-1]["event"] == "followup_answered"
    assert events.records[-1]["detail"]["answered_from_memory"] is True
    run.graph = None
    assert "no checkpointed conversation" in ask_followup(run, QUESTION).error


def test_cites_allows_rounding_and_unicode_minus():
    assert numbers_in("band 313.46 to 1,234.5 and −0.25") == [313.46, 1234.5, -0.25]
    assert cites("band from 313.5 upward", 313.46)
    assert not cites("band from 320 upward", 313.46)
    assert cites("score −0.25", -0.25)
