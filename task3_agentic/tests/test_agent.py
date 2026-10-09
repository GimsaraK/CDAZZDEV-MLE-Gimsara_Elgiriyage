"""The LangGraph loop with scripted models: the model decides tool order, observations feed the next
decision, the report check sends the agent back, and limits end the run without exceptions."""
# AI-ASSISTED: Claude Code (claude-opus-5-5), Prompt: 'Implement the Task 3A plan (the plan approved in Entry 13)', Date: 2026-10-09 (see CITATIONS.md Entry 14)

import itertools
from typing import Callable, List, Optional, Sequence, Tuple, Union

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableLambda

from task3_agentic.src import config, prompts
from task3_agentic.src.agent import count_tool_rounds, run_research
from task3_agentic.src.display import TracePrinter
from task3_agentic.src.session import FaultConfig

from .conftest import FAKE_HEADLINES, valid_report

Step = Union[Tuple[str, dict], Callable[[List[BaseMessage]], Optional[Tuple[str, dict]]], None]


class ScriptedAgent:
    """Plays a list of steps. A step is (tool, args), a function of the messages, or None to finish."""

    def __init__(self, steps: Sequence[Step]) -> None:
        self.steps = list(steps)
        self.calls: List[List[BaseMessage]] = []
        self._ids = itertools.count(1)

    def __call__(self, messages: List[BaseMessage]) -> AIMessage:
        self.calls.append(list(messages))
        step = self.steps.pop(0) if self.steps else None
        if callable(step):
            step = step(messages)
        if step is None:
            return AIMessage(content="Enough evidence; summary follows.")
        name, args = step
        return AIMessage(
            content=f"Observed: so far -> Next: {name}",
            tool_calls=[{"name": name, "args": args, "id": f"call_{next(self._ids)}", "type": "tool_call"}],
        )


def report_model_for(ctx, drafts: Optional[List[Callable]] = None):
    """Report writer stand-in. By default it always returns a report consistent with the session."""
    queue = list(drafts or [])

    def write(_messages):
        draft = queue.pop(0) if queue else (lambda: valid_report(ctx, risk_tools=("get_price_data", "calculate_volatility", "llm_sentiment")))
        return draft()

    return RunnableLambda(write)


FULL_RESEARCH = [
    ("get_price_data", {"ticker": "AAPL", "period": "1y"}),
    ("calculate_volatility", {"ticker": "AAPL", "window": 20}),
    ("get_news", {"ticker": "AAPL", "n": 4}),
    ("llm_sentiment", {"headlines": FAKE_HEADLINES}),
]


@pytest.mark.parametrize("order", [FULL_RESEARCH, list(reversed(FULL_RESEARCH[:2])) + FULL_RESEARCH[2:]])
def test_tool_order_comes_from_the_model(ctx, offline_sources, order):
    agent = ScriptedAgent(order + [None])
    run = run_research(ctx, agent_model=RunnableLambda(agent), report_model=report_model_for(ctx))
    assert run.final.status == config.REPORT_STATUS_VALIDATED
    assert run.tool_sequence == [name for name, _ in order]
    assert sum(isinstance(m, ToolMessage) for m in run.messages) == len(order)
    assert run.final.report.ticker == "AAPL"


def test_observation_drives_the_next_decision(ctx, offline_sources):
    """After an error envelope the scripted model retries; it only can because it saw the error."""
    ctx.faults = FaultConfig({"get_price_data": "first"})

    def retry_if_error(messages):
        last = messages[-1]
        assert isinstance(last, ToolMessage)
        if '"status":"error"' in last.content:
            return "get_price_data", {"ticker": "AAPL", "period": "1y"}
        return "calculate_volatility", {"ticker": "AAPL", "window": 20}

    agent = ScriptedAgent([FULL_RESEARCH[0], retry_if_error, retry_if_error, FULL_RESEARCH[2], FULL_RESEARCH[3], None])
    run = run_research(ctx, agent_model=RunnableLambda(agent), report_model=report_model_for(ctx))
    assert run.tool_sequence[:3] == ["get_price_data", "get_price_data", "calculate_volatility"]
    assert run.final.status == config.REPORT_STATUS_VALIDATED
    # The system prompt is sent on every turn, followed by the whole conversation.
    assert all(call[0].type == "system" for call in agent.calls)


def test_report_check_sends_agent_back_then_passes(ctx, offline_sources):
    bad = lambda: valid_report(ctx, vol_shift=10.0, risk_tools=("get_price_data", "calculate_volatility", "llm_sentiment"))
    agent = ScriptedAgent(FULL_RESEARCH + [None, ("web_search", {"query": "Apple analyst"}), None])
    run = run_research(ctx, agent_model=RunnableLambda(agent), report_model=report_model_for(ctx, [bad]))
    assert run.final.status == config.REPORT_STATUS_VALIDATED
    assert run.final.revisions == 1
    feedback = [m for m in run.messages if isinstance(m, HumanMessage) and m.content.startswith(prompts.REPORT_CHECK_MARKER)]
    assert len(feedback) == 1 and "calculate_volatility" in feedback[0].content
    # The agent acted after the feedback: web_search came after the first report attempt.
    assert run.tool_sequence[-1] == "web_search"


def test_unfixable_report_is_accepted_with_warnings(ctx, offline_sources, monkeypatch):
    monkeypatch.setattr(config, "MAX_REPORT_REVISIONS", 1)
    always_bad = lambda: valid_report(ctx, vol_shift=10.0)
    agent = ScriptedAgent(FULL_RESEARCH + [None, None])
    run = run_research(ctx, agent_model=RunnableLambda(agent), report_model=report_model_for(ctx, [always_bad] * 5))
    assert run.final.status == config.REPORT_STATUS_UNVALIDATED
    assert run.final.validation_warnings and run.final.report is not None


def test_tool_budget_ends_the_loop(ctx, offline_sources, monkeypatch):
    monkeypatch.setattr(config, "MAX_TOOL_ROUNDS", 2)
    monkeypatch.setattr(config, "MAX_REPORT_REVISIONS", 0)
    agent = ScriptedAgent([("get_news", {"ticker": "AAPL", "n": 3})] * 10)
    run = run_research(ctx, agent_model=RunnableLambda(agent), report_model=report_model_for(ctx))
    assert count_tool_rounds(run.messages) == 2
    assert any("budget" in event for event in run.events)
    assert run.final.status in (config.REPORT_STATUS_UNVALIDATED, config.REPORT_STATUS_VALIDATED)


def test_llm_failure_degrades_to_a_failed_report(ctx, offline_sources):
    def down(_messages):
        raise ConnectionError("all providers unreachable")

    run = run_research(ctx, agent_model=RunnableLambda(down), report_model=RunnableLambda(down))
    assert run.final.status == config.REPORT_STATUS_FAILED
    assert run.final.report is None and "unreachable" in run.final.error


def test_unknown_tool_and_bad_arguments_do_not_crash(ctx, offline_sources):
    agent = ScriptedAgent([("delete_files", {}), ("calculate_volatility", {"ticker": "AAPL", "window": {"bad": 1}})] + FULL_RESEARCH + [None])
    run = run_research(ctx, agent_model=RunnableLambda(agent), report_model=report_model_for(ctx))
    errors = [m for m in run.messages if isinstance(m, ToolMessage) and m.status == "error"]
    assert len(errors) == 2
    assert run.final.status == config.REPORT_STATUS_VALIDATED


def test_report_writer_output_with_raw_message(ctx, offline_sources):
    """with_structured_output(include_raw=True) output is unwrapped and the writer's model recorded."""

    def write(_messages):
        raw = AIMessage(content="", response_metadata={"model_name": "openai/gpt-oss-20b"})
        report = valid_report(ctx, risk_tools=("get_price_data", "calculate_volatility", "llm_sentiment"))
        return {"raw": raw, "parsed": report, "parsing_error": None}

    agent = ScriptedAgent(FULL_RESEARCH + [None])
    run = run_research(ctx, agent_model=RunnableLambda(agent), report_model=RunnableLambda(write))
    assert run.final.status == config.REPORT_STATUS_VALIDATED
    assert run.final.report_model == "openai/gpt-oss-20b"
    assert sum(run.final.agent_models.values()) == len(FULL_RESEARCH) + 1


def test_parsing_error_triggers_repair(ctx, offline_sources):
    attempts = []

    def write(_messages):
        attempts.append(1)
        if len(attempts) == 1:
            return {"raw": AIMessage(content=""), "parsed": None, "parsing_error": ValueError("bad json")}
        return valid_report(ctx, risk_tools=("get_price_data", "calculate_volatility", "llm_sentiment"))

    agent = ScriptedAgent(FULL_RESEARCH + [None])
    run = run_research(ctx, agent_model=RunnableLambda(agent), report_model=RunnableLambda(write))
    assert len(attempts) == 2 and run.final.status == config.REPORT_STATUS_VALIDATED


def test_trace_printer_labels_cycles(ctx, offline_sources, capsys):
    agent = ScriptedAgent(FULL_RESEARCH + [None])
    run_research(ctx, agent_model=RunnableLambda(agent), report_model=report_model_for(ctx), on_message=TracePrinter())
    out = capsys.readouterr().out
    assert "CALL     get_price_data" in out and "OBSERVE  get_price_data -> " in out
    assert "cycle 1: after observing get_price_data" in out
    assert "FINISH" in out
