"""The 3B pipeline with scripted models: order of handoffs, the critique round trip, tool restriction
in all three layers, per-agent trace attribution, and the degraded paths."""
# AI-ASSISTED: Claude Code (claude-opus-5-5), Prompt: 'Implement the Task 3B plan', Date: 2026-10-09 (see CITATIONS.md Entry 16)

import json
from typing import List

import pytest
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableLambda

from task3_agentic.src import config
from task3_agentic.src.display import MultiAgentPrinter, handoff_table, schema_table, tool_call_table
from task3_agentic.src.handoff import AnalystInterpretation, ClarificationRequest, DataBrief
from task3_agentic.src.multi_agent import PipelineModels, build_agent_toolkits, run_multi_agent, save_handoffs
from task3_agentic.src.react import build_react_loop, run_react_loop

from .conftest import FAKE_HEADLINES, valid_report
from .test_agent import ScriptedAgent

A, B = config.AGENT_A_NAME, config.AGENT_B_NAME
HANDOFF_ORDER = ["task", "data_brief", "clarification_request", "clarification_response", "final_report"]


def request_in(messages: List[BaseMessage]) -> dict:
    """The clarification JSON inside A's latest instruction."""
    text = next(m.content for m in reversed(messages) if isinstance(m, HumanMessage))
    return json.loads(text.split("(JSON):\n", 1)[1].split("\n\nAnswer it", 1)[0])


def answer_request(messages):
    request = request_in(messages)
    assert request["request_kind"] == "score_headlines"
    return "llm_sentiment", {"headlines": request["headlines"]}


def interpretation(_messages):
    return AnalystInterpretation(key_findings=["Uptrend with a rich multiple.", "Volatility is low."], confidence_notes="No sentiment yet.")


def headline_request(_messages):
    return ClarificationRequest(
        request_kind="score_headlines",
        question="What is the sentiment score of these headlines?",
        why_needed="Market sentiment section.",
        headlines=FAKE_HEADLINES,
    )


def report_for(ctx, with_clarification=True):
    def write(_messages):
        report = valid_report(ctx, risk_tools=("get_price_data", "calculate_volatility", "get_news"))
        records = ctx.results_for("llm_sentiment", agent=A)
        if with_clarification and records:
            report.clarification_used = f"Asked A to score B's headlines: score {records[0].result.data.score}."
        return report

    return write


def models_for(ctx, analyst_steps=None, writer_steps=None, request=headline_request, report=None, brief=interpretation):
    analyst_steps = analyst_steps if analyst_steps is not None else [
        ("get_price_data", {"ticker": "AAPL", "period": "1y"}),
        ("calculate_volatility", {"ticker": "AAPL", "window": 20}),
        None,
        answer_request,
        None,
    ]
    writer_steps = writer_steps if writer_steps is not None else [
        ("get_news", {"ticker": "AAPL", "n": 4}),
        ("web_search", {"query": "Apple analyst outlook"}),
        None,
    ]
    return PipelineModels(
        analyst_agent=RunnableLambda(ScriptedAgent(analyst_steps)),
        writer_agent=RunnableLambda(ScriptedAgent(writer_steps)),
        analyst_brief=RunnableLambda(brief),
        writer_request=RunnableLambda(request),
        writer_report=RunnableLambda(report or report_for(ctx)),
    )


# --------------------------------------------------------------------------- happy path
def test_pipeline_runs_end_to_end(ctx, offline_sources):
    run = run_multi_agent(ctx, models=models_for(ctx))
    assert run.final.status == config.REPORT_STATUS_VALIDATED
    assert [h.kind for h in run.handoffs] == HANDOFF_ORDER
    assert [(h.sender, h.recipient) for h in run.handoffs] == [
        (config.ORCHESTRATOR_NAME, A), (A, B), (B, A), (A, B), (B, config.ORCHESTRATOR_NAME)
    ]
    assert run.critique_rounds == 1
    # Each agent used only its own tools, in the order its model chose.
    assert run.tool_sequence(A) == ["get_price_data", "calculate_volatility", "llm_sentiment"]
    assert run.tool_sequence(B) == ["get_news", "web_search"]
    assert run.final.agent_tools == {A: ["calculate_volatility", "get_price_data", "llm_sentiment"], B: ["get_news", "web_search"]}


def test_headlines_travel_from_b_to_a_through_the_critique(ctx, offline_sources):
    run = run_multi_agent(ctx, models=models_for(ctx))
    assert run.brief.sentiment is None and "no news access" in run.brief.sentiment_unavailable_reason
    assert run.request.headlines == FAKE_HEADLINES
    scored = ctx.results_for("llm_sentiment", agent=A)[0]
    assert scored.args["headlines"] == FAKE_HEADLINES
    assert run.response.status == "ok" and run.response.fulfilled_by == "agent"
    assert run.response.sentiment.score == scored.result.data.score
    assert str(run.response.sentiment.score) in run.final.report.clarification_used


def test_agents_keep_separate_conversations(ctx, offline_sources):
    run = run_multi_agent(ctx, models=models_for(ctx))
    analyst_tools = {m.name for m in run.analyst_messages if isinstance(m, ToolMessage)}
    writer_tools = {m.name for m in run.writer_messages if isinstance(m, ToolMessage)}
    assert analyst_tools == set(config.AGENT_A_TOOLS) and writer_tools == set(config.AGENT_B_TOOLS)
    # A's second phase continues its own history: its first tool call is still there.
    assert isinstance(run.analyst_messages[0], HumanMessage) and "orchestrator" in run.analyst_messages[0].content


def test_trace_lines_are_attributed_per_agent(ctx, offline_sources):
    run_multi_agent(ctx, models=models_for(ctx))
    by_agent = {}
    for record in ctx.trace.records:
        by_agent.setdefault(record["agent"], set()).add(record["tool"])
    assert by_agent == {A: set(config.AGENT_A_TOOLS), B: set(config.AGENT_B_TOOLS)}


def test_handoffs_are_saved_and_tabulated(ctx, offline_sources, tmp_path):
    run = run_multi_agent(ctx, models=models_for(ctx))
    path = save_handoffs(run, tmp_path)
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert [m["kind"] for m in saved] == HANDOFF_ORDER
    assert DataBrief.model_validate(saved[1]["payload"]).ticker == "AAPL"
    assert list(handoff_table(run.handoffs)["kind"]) == HANDOFF_ORDER
    assert "agent" in tool_call_table(ctx).columns
    assert "request_kind" in set(schema_table(ClarificationRequest)["field"])


def test_live_printer_labels_agents_and_handoffs(ctx, offline_sources, capsys):
    run_multi_agent(ctx, models=models_for(ctx), observer=MultiAgentPrinter())
    out = capsys.readouterr().out
    assert "[A data_analyst]    CALL     get_price_data" in out
    assert "[B research_writer] CALL     get_news" in out
    assert "HANDOFF #3  research_writer -> data_analyst  [clarification_request]" in out
    assert "HANDOFF #4  data_analyst -> research_writer  [clarification_response]" in out


# --------------------------------------------------------------------------- tool restriction, three layers
def test_layer1_each_agent_is_bound_only_to_its_tools(ctx, offline_sources):
    run = run_multi_agent(ctx, models=models_for(ctx))
    assert run.bound_tools == {A: list(config.AGENT_A_TOOLS), B: list(config.AGENT_B_TOOLS)}


def test_layer2_tool_node_rejects_a_forged_call(ctx, offline_sources):
    """A model that names a tool it was not given (as a jailbroken or confused model might) gets an error back."""
    _, tools_a = build_agent_toolkits(ctx)[A]
    forger = ScriptedAgent([("web_search", {"query": "Apple"}), None])
    loop = build_react_loop(RunnableLambda(forger), tools_a, "You are Agent A.", max_rounds=3)
    state = run_react_loop(loop, [HumanMessage(content="go")])
    message = next(m for m in state["messages"] if isinstance(m, ToolMessage))
    assert message.status == "error" and "web_search" in message.content
    assert ctx.history == []  # the tool body never ran, not even the guard
    # The model saw the rejection before its next turn.
    assert any(isinstance(m, ToolMessage) for m in forger.calls[-1])


def test_layer3_toolkit_guard_refuses_and_logs(ctx, offline_sources):
    toolkits = build_agent_toolkits(ctx)
    toolkit_a, toolkit_b = toolkits[A][0], toolkits[B][0]
    for result in (toolkit_a.web_search("Apple"), toolkit_a.get_news("AAPL", 5), toolkit_b.get_price_data("AAPL", "1y")):
        assert result.status == "error" and "ToolAccessError" in result.error
    assert [r.denied for r in ctx.history] == [True, True, True]
    assert {r["error"].split(":")[0] for r in ctx.trace.records} == {"ToolAccessError"}
    # A refusal is not a tool failure: no tool counts as failed for the report check.
    assert ctx.failed_tools() == set()
    assert offline_sources == []  # DuckDuckGo was never called


def test_forged_call_inside_a_run_is_contained(ctx, offline_sources):
    writer_steps = [("get_price_data", {"ticker": "AAPL", "period": "1y"}), ("get_news", {"ticker": "AAPL", "n": 4}), None]
    run = run_multi_agent(ctx, models=models_for(ctx, writer_steps=writer_steps))
    errors = [m for m in run.writer_messages if isinstance(m, ToolMessage) and m.status == "error"]
    assert len(errors) == 1 and "get_price_data" in errors[0].content
    assert "get_price_data" not in run.tool_sequence(B)
    assert run.final.status == config.REPORT_STATUS_VALIDATED


# --------------------------------------------------------------------------- degraded paths
def test_invalid_request_falls_back_to_scoring_bs_headlines(ctx, offline_sources):
    def broken(_messages):
        raise ValueError("model returned junk")

    run = run_multi_agent(ctx, models=models_for(ctx, request=broken))
    request_message = run.handoffs[2]
    assert request_message.kind == "clarification_request" and request_message.fallback
    assert run.request.request_kind == "score_headlines" and run.request.headlines == FAKE_HEADLINES
    assert run.response.status == "ok"


def test_analyst_that_ignores_the_request_is_covered_by_its_own_toolkit(ctx, offline_sources):
    analyst_steps = [("get_price_data", {"ticker": "AAPL", "period": "1y"}), ("calculate_volatility", {"ticker": "AAPL", "window": 20}), None, None]
    run = run_multi_agent(ctx, models=models_for(ctx, analyst_steps=analyst_steps))
    assert run.response.status == "ok" and run.response.fulfilled_by == "fallback"
    # Still Agent A's tool, called under A's name and access rules.
    assert ctx.results_for("llm_sentiment", agent=A)
    assert run.final.status == config.REPORT_STATUS_VALIDATED


def test_report_that_ignores_the_answer_gets_one_repair(ctx, offline_sources):
    drafts = [report_for(ctx, with_clarification=False), report_for(ctx)]

    def write(messages):
        return drafts.pop(0)(messages)

    run = run_multi_agent(ctx, models=models_for(ctx, report=write))
    assert run.final.status == config.REPORT_STATUS_VALIDATED and run.final.revisions == 1
    assert any("repair pass" in e for e in run.events)


def test_report_that_never_uses_the_answer_is_flagged(ctx, offline_sources):
    run = run_multi_agent(ctx, models=models_for(ctx, report=report_for(ctx, with_clarification=False)))
    assert run.final.status == config.REPORT_STATUS_UNVALIDATED
    assert any("clarification_used" in w for w in run.final.validation_warnings)


def test_analyst_cannot_score_headlines_it_made_up(ctx, offline_sources):
    """Seen in a live run: A, with no news access, invented headlines. The tool refuses them."""
    analyst_steps = [
        ("get_price_data", {"ticker": "AAPL", "period": "1y"}),
        ("llm_sentiment", {"headlines": ["Apple reports Q3 earnings beat expectations"]}),
        ("calculate_volatility", {"ticker": "AAPL", "window": 20}),
        None,
        answer_request,
        None,
    ]
    run = run_multi_agent(ctx, models=models_for(ctx, analyst_steps=analyst_steps))
    invented = ctx.results_for("llm_sentiment", ok_only=False, agent=A)[-1]
    assert invented.result.status == "error" and "get_news or web_search" in invented.result.error
    # The brief carries no sentiment: the only score in the run is on B's real headlines.
    assert run.brief.sentiment is None
    assert run.response.sentiment is not None and ctx.results_for("llm_sentiment", agent=A)[0].args["headlines"] == FAKE_HEADLINES
    assert run.final.status == config.REPORT_STATUS_VALIDATED


def test_fallback_request_without_headlines_asks_for_a_new_window(ctx, offline_sources):
    def broken(_messages):
        raise ValueError("model returned junk")

    run = run_multi_agent(ctx, models=models_for(ctx, writer_steps=[None], request=broken))
    assert run.request.request_kind == "volatility_window"
    assert run.request.window != run.brief.volatility.window


def test_every_model_down_still_returns_a_run(ctx, offline_sources):
    def down(_messages):
        raise ConnectionError("all providers unreachable")

    models = PipelineModels(*(RunnableLambda(down) for _ in range(5)))
    run = run_multi_agent(ctx, models=models)
    assert run.final.status == config.MULTI_AGENT_STATUS_FAILED and run.final.report is None
    # The brief (numbers only), the fallback request and A's answer still flowed between the agents.
    assert [h.kind for h in run.handoffs] == HANDOFF_ORDER[:4]
    assert run.handoffs[2].fallback and run.response.fulfilled_by == "fallback"
