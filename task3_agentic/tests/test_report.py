"""ResearchReport schema, the deterministic report check, digest and rendering."""
# AI-ASSISTED: Claude Code (claude-opus-5-5), Prompt: 'Implement the Task 3A plan (the plan approved in Entry 13)', Date: 2026-10-09 (see CITATIONS.md Entry 14)

import json

import pytest
from pydantic import ValidationError

from task3_agentic.src import config
from task3_agentic.src.report import build_evidence_digest, check_report, render_markdown, save_report
from task3_agentic.src.schemas import Evidence, FinalReport, ResearchReport, Risk
from task3_agentic.src.session import FaultConfig
from task3_agentic.src.tools import ResearchTools

from .conftest import FAKE_HEADLINES, valid_report


@pytest.fixture
def researched(ctx, offline_sources):
    """A session in which every tool has succeeded once."""
    toolkit = ResearchTools(ctx)
    toolkit.get_price_data("AAPL", "1y")
    toolkit.calculate_volatility("AAPL", 20)
    toolkit.get_news("AAPL", 4)
    toolkit.llm_sentiment(FAKE_HEADLINES)
    toolkit.web_search("Apple analyst")
    return ctx


def _risk(tool="get_price_data"):
    return Risk(name="r", description="d", evidence=[Evidence(source_tool=tool, detail="x")])


@pytest.mark.parametrize("count", [2, 4])
def test_report_requires_exactly_three_risks(researched, count):
    data = valid_report(researched).model_dump()
    data["risks"] = [_risk().model_dump()] * count
    with pytest.raises(ValidationError):
        ResearchReport.model_validate(data)


def test_risk_requires_evidence():
    with pytest.raises(ValidationError):
        Risk(name="r", description="d", evidence=[])


def test_evidence_must_name_a_real_tool():
    with pytest.raises(ValidationError):
        Evidence(source_tool="memory", detail="x")


def test_data_gaps_drop_fragments(researched):
    data = valid_report(researched).model_dump()
    data["data_gaps"] = [":{", "  ", "web_search failed"]
    assert ResearchReport.model_validate(data).data_gaps == ["web_search failed"]


def test_valid_report_passes(researched):
    assert check_report(valid_report(researched), researched) == []


def test_no_report():
    from task3_agentic.src.session import SessionContext
    from task3_agentic.src.tracing import TraceLogger

    assert check_report(None, SessionContext("AAPL", trace=TraceLogger(None)))


def test_hedge_volatility_must_match_a_computed_value(researched):
    issues = check_report(valid_report(researched, vol_shift=5.0), researched)
    assert any("match no calculate_volatility result" in issue for issue in issues)


def test_tolerance_allows_rounding(researched):
    shift = config.VOL_CHECK_TOLERANCE_PCT_POINTS / 2
    assert check_report(valid_report(researched, vol_shift=shift), researched) == []


def test_risk_citing_a_failed_tool_is_flagged(ctx, offline_sources):
    ctx.faults = FaultConfig({"web_search": "always"})
    toolkit = ResearchTools(ctx)
    toolkit.get_price_data("AAPL", "1y")
    toolkit.calculate_volatility("AAPL", 20)
    toolkit.get_news("AAPL", 4)
    toolkit.llm_sentiment(FAKE_HEADLINES)
    toolkit.web_search("Apple analyst")
    report = valid_report(ctx, risk_tools=("web_search", "get_price_data", "calculate_volatility"))
    issues = check_report(report, ctx)
    assert any("Risk 1" in issue and "no evidence from a tool that succeeded" in issue for issue in issues)


def test_failed_tool_must_be_recorded_in_data_gaps(ctx, offline_sources):
    ctx.faults = FaultConfig({"web_search": "always"})
    toolkit = ResearchTools(ctx)
    toolkit.get_price_data("AAPL", "1y")
    toolkit.calculate_volatility("AAPL", 20)
    toolkit.get_news("AAPL", 4)
    toolkit.llm_sentiment(FAKE_HEADLINES)
    toolkit.web_search("Apple analyst")
    report = valid_report(ctx, risk_tools=("get_price_data", "calculate_volatility", "llm_sentiment"))
    assert check_report(report, ctx) == []
    report.data_gaps = []
    assert any("web_search failed" in issue for issue in check_report(report, ctx))


def test_missing_sentiment_and_volatility_are_flagged(ctx, offline_sources):
    ResearchTools(ctx).get_price_data("AAPL", "1y")
    issues = " ".join(check_report(valid_report(ctx, risk_tools=("get_price_data",) * 3), ctx))
    assert "llm_sentiment" in issues and "calculate_volatility has not succeeded" in issues


def test_wrong_ticker_is_flagged(researched):
    report = valid_report(researched)
    report.ticker = "MSFT"
    assert any("session ticker" in issue for issue in check_report(report, researched))


def test_digest_omits_bars_and_lists_failures(ctx, offline_sources):
    ctx.faults = FaultConfig({"web_search": "always"})
    toolkit = ResearchTools(ctx)
    toolkit.get_price_data("AAPL", "1y")
    toolkit.web_search("Apple")
    digest = json.loads(build_evidence_digest(ctx))
    price = digest["succeeded"]["get_price_data"][0]["result"]["data"]
    assert "recent_bars" not in price and "indicators" in price
    assert digest["failed"]["web_search"]["status"] == "error"


def test_render_and_save(researched, tmp_path):
    final = FinalReport(
        ticker="AAPL",
        query="q",
        session_id=researched.session_id,
        generated_at="2026-10-09T10:00:00+00:00",
        status=config.REPORT_STATUS_VALIDATED,
        report=valid_report(researched),
        tools_used=sorted(researched.succeeded_tools()),
    )
    markdown = render_markdown(final, "Apple Inc.")
    for heading in ("## 1. Financial Health Summary", "## 2. Top Three Risks", "## 3. Hedge Strategy Recommendation"):
        assert heading in markdown
    assert config.RISK_DISCLAIMER in markdown
    md_path, json_path = save_report(final, "Apple Inc.", suffix="_test", output_dir=tmp_path)
    assert md_path.name == "AAPL_2026-10-09_research_report_test.md"
    assert FinalReport.model_validate_json(json_path.read_text(encoding="utf-8")).report.ticker == "AAPL"


def test_render_failed_report():
    final = FinalReport(
        ticker="AAPL", query="q", session_id="s", generated_at="2026-10-09T10:00:00+00:00",
        status=config.REPORT_STATUS_FAILED, error="all providers down",
    )
    assert "all providers down" in render_markdown(final)
