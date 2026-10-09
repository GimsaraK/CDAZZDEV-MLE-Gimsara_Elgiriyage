"""3B handoff schemas: validation, code-filled numbers, and the incorporation check."""
# AI-ASSISTED: Claude Code (claude-opus-5-5), Prompt: 'Implement the Task 3B plan', Date: 2026-10-09 (see CITATIONS.md Entry 16)

import pytest
from pydantic import ValidationError

from task3_agentic.src import config
from task3_agentic.src.handoff import (
    AgentMessage,
    AnalystInterpretation,
    ClarificationRequest,
    ClarificationResponse,
    DataBrief,
    RiskFlag,
    build_brief_numbers,
    build_response,
    check_incorporation,
    lookup_metric,
    merge_interpretation,
    report_numbers,
)
from task3_agentic.src.tools import ResearchTools

from .conftest import FAKE_HEADLINES, valid_report

A = config.AGENT_A_NAME


@pytest.fixture
def analyst(ctx, offline_sources):
    # The writer fetches the headlines first: A can only score titles a news tool returned.
    ResearchTools(ctx, agent_name=config.AGENT_B_NAME, allowed=config.AGENT_B_TOOLS).get_news("AAPL", len(FAKE_HEADLINES))
    return ResearchTools(ctx, agent_name=A, allowed=config.AGENT_A_TOOLS)


# --------------------------------------------------------------------------- ClarificationRequest
def test_score_headlines_needs_headlines_and_dedupes():
    with pytest.raises(ValidationError):
        ClarificationRequest(request_kind="score_headlines", question="Score these headlines please", why_needed="x")
    many = [f"Headline {i}" for i in range(15)] + ["headline 0"]
    request = ClarificationRequest(request_kind="score_headlines", question="Score these headlines please", why_needed="x", headlines=many)
    assert len(request.headlines) == config.MAX_SENTIMENT_HEADLINES and request.headlines[0] == "Headline 0"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"request_kind": "volatility_window", "window": 2},
        {"request_kind": "volatility_window"},
        {"request_kind": "price_period", "period": "7y"},
        {"request_kind": "metric_check", "metric": "  "},
        {"request_kind": "unknown_kind"},
    ],
)
def test_request_rejects_missing_or_bad_parameters(kwargs):
    with pytest.raises(ValidationError):
        ClarificationRequest(question="A specific question here", why_needed="x", **kwargs)


def test_request_accepts_each_kind():
    base = {"question": "A specific question here", "why_needed": "hedge sizing"}
    assert ClarificationRequest(request_kind="volatility_window", window=60, **base).window == 60
    assert ClarificationRequest(request_kind="price_period", period=" YTD ", **base).period == "ytd"
    assert ClarificationRequest(request_kind="metric_check", metric="forward_pe", **base).metric == "forward_pe"


def test_question_must_be_specific():
    with pytest.raises(ValidationError):
        ClarificationRequest(request_kind="volatility_window", window=20, question="vol?", why_needed="x")


# --------------------------------------------------------------------------- AgentMessage
def test_agent_message_round_trip_and_receiver_validation():
    request = ClarificationRequest(request_kind="volatility_window", window=60, question="Volatility over a quarter?", why_needed="hedge")
    message = AgentMessage.wrap("research_writer", "data_analyst", "clarification_request", request, summary="vol 60d")
    assert message.open(ClarificationRequest) == request
    # A payload that does not match the schema the receiver expects is rejected, not silently used.
    with pytest.raises(ValidationError):
        message.open(DataBrief)
    with pytest.raises(ValidationError):
        AgentMessage.wrap("a", "b", "gossip", request, summary="x")


# --------------------------------------------------------------------------- the data brief
def test_brief_numbers_come_from_the_analysts_tool_results(ctx, analyst):
    price = analyst.get_price_data("AAPL", "1y").data
    vol = analyst.calculate_volatility("AAPL", 20).data
    brief = build_brief_numbers(ctx, A)
    assert brief.price.current_price == price.current_price
    assert brief.price.indicators == price.indicators
    assert brief.price.fundamentals == price.fundamentals
    assert brief.volatility == vol
    assert "recent_bars" not in brief.model_dump()["price"]
    assert brief.sentiment is None and "no news access" in brief.sentiment_unavailable_reason
    assert brief.tool_status == {"get_price_data": "ok", "calculate_volatility": "ok", "llm_sentiment": "not_called"}
    assert brief.data_gaps == []


def test_brief_records_failed_tools_as_gaps(ctx, analyst):
    analyst.calculate_volatility("AAPL", 1000)
    brief = build_brief_numbers(ctx, A)
    assert brief.volatility is None and brief.tool_status["calculate_volatility"] == "error"
    assert any("calculate_volatility" in gap for gap in brief.data_gaps)


def test_brief_ignores_other_agents_results(ctx, offline_sources):
    ResearchTools(ctx, agent_name="someone_else").get_price_data("AAPL", "1y")
    assert build_brief_numbers(ctx, A).price is None


def test_merge_interpretation(ctx, analyst):
    analyst.get_price_data("AAPL", "1y")
    numbers = build_brief_numbers(ctx, A)
    interpretation = AnalystInterpretation(
        key_findings=[f"finding {i}" for i in range(8)],
        quant_risk_flags=[RiskFlag(metric="forward_pe", value="28.2", concern="rich valuation")],
        confidence_notes="No sentiment yet.",
    )
    brief = merge_interpretation(numbers, interpretation, "openai/gpt-oss-20b")
    assert len(brief.key_findings) == config.BRIEF_MAX_FINDINGS and brief.interpretation_model == "openai/gpt-oss-20b"
    assert brief.price == numbers.price
    numbers_only = merge_interpretation(numbers, None, None)
    assert numbers_only.key_findings == [] and any("interpretation" in gap for gap in numbers_only.data_gaps)


# --------------------------------------------------------------------------- the clarification response
def test_response_uses_only_results_after_the_request(ctx, analyst):
    analyst.llm_sentiment(FAKE_HEADLINES[:1])  # scored before the request: must not count
    since = ctx.history_length()
    request = ClarificationRequest(request_kind="score_headlines", question="Score B's headlines please", why_needed="x", headlines=FAKE_HEADLINES)
    assert build_response(ctx, request, "nothing yet", since).status == "failed"
    analyst.llm_sentiment(request.headlines)
    response = build_response(ctx, request, "Neutral overall.", since)
    assert response.status == "ok" and response.sentiment.n_scored == len(FAKE_HEADLINES)
    assert response.tools_used == ["llm_sentiment"] and response.fulfilled_by == "agent"


def test_response_prefers_the_requested_window_and_period(ctx, analyst):
    since = ctx.history_length()
    analyst.calculate_volatility("AAPL", 60)
    analyst.calculate_volatility("AAPL", 20)
    request = ClarificationRequest(request_kind="volatility_window", window=60, question="Quarterly volatility?", why_needed="x")
    assert build_response(ctx, request, "", since).volatility.window == 60
    analyst.get_price_data("AAPL", "ytd")
    request = ClarificationRequest(request_kind="price_period", period="ytd", question="Year-to-date return?", why_needed="x")
    response = build_response(ctx, request, "", since)
    assert response.period_return.period == "ytd" and response.period_return.period_return_pct is not None


def test_metric_check_lookup(ctx, analyst):
    price = analyst.get_price_data("AAPL", "1y").data
    request = ClarificationRequest(request_kind="metric_check", metric="Forward P/E", question="Confirm forward P/E", why_needed="x")
    response = build_response(ctx, request, "", ctx.history_length())
    assert response.metric_value == pytest.approx(price.fundamentals.forward_pe)
    assert lookup_metric("RSI_14", price, None) == price.indicators["RSI_14"]
    assert lookup_metric("no such metric", price, None) is None


# --------------------------------------------------------------------------- incorporation
def _sentiment_response(ctx, analyst) -> ClarificationResponse:
    since = ctx.history_length()
    analyst.llm_sentiment(FAKE_HEADLINES)
    request = ClarificationRequest(request_kind="score_headlines", question="Score B's headlines please", why_needed="x", headlines=FAKE_HEADLINES)
    return build_response(ctx, request, "Mildly negative.", since)


def test_incorporation_passes_when_the_value_is_cited(ctx, analyst):
    analyst.get_price_data("AAPL", "1y")
    analyst.calculate_volatility("AAPL", 20)
    response = _sentiment_response(ctx, analyst)
    report = valid_report(ctx)
    report.clarification_used = f"Asked A to score 4 headlines; score {response.sentiment.score} used in market sentiment."
    assert check_incorporation(report, response) == []


def test_incorporation_fails_when_value_or_note_missing(ctx, analyst):
    analyst.get_price_data("AAPL", "1y")
    analyst.calculate_volatility("AAPL", 20)
    response = _sentiment_response(ctx, analyst)
    report = valid_report(ctx)
    issues = check_incorporation(report, response)
    assert any("clarification_used is empty" in i for i in issues)
    assert any("sentiment score" in i for i in issues)


def test_failed_response_only_needs_the_note(ctx, analyst):
    response = ClarificationResponse(request_kind="score_headlines", status="failed", answer="", error="no data")
    report = valid_report(ctx)
    assert check_incorporation(report, response) == ["clarification_used is empty: say what you asked the Data Analyst, what it returned, and where the report uses it."]
    report.clarification_used = "A could not score the headlines; recorded as a gap."
    assert check_incorporation(report, response) == []


def test_report_numbers_handle_unicode_minus(ctx):
    report = valid_report(ctx)
    report.clarification_used = "Score −0.18 and vol 1,234.5"
    numbers = report_numbers(report)
    assert -0.18 in numbers and 1234.5 in numbers
