"""Offline tests for the equity research brief: ranking, required sections, disclaimer, HTML chart."""
# AI-ASSISTED: Cursor Agent (claude-sonnet-5.5), Prompt: 'Task 1 bonus research brief', Date: 2026-10-08

from pathlib import Path

from task1_financial.src import config
from task1_financial.src.llm import HeadlineSentiment, SentimentResult, SignalRecommendation
from task1_financial.src.report import (
    build_brief_html,
    build_brief_markdown,
    headline_contribution,
    select_top_headlines,
    write_research_brief,
)
from task1_financial.src.summary import MomentumComponent, MomentumSignal, StockSummary

# Three sentences, so the signal validator accepts it. Decimals are not sentence breaks.
_JUSTIFICATION = (
    "Price is above both moving averages and the 50-day average is above the 200-day average. "
    "MACD is below its signal line, which argues against adding. "
    "News is slightly negative, so the combined view is Hold."
)

# Smallest valid PNG (1x1). Used only to check that the HTML inlines the chart bytes.
_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02"
    b"\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\xff\xff?\x00\x05\xfe\x02\xfe"
    b"\xdc\xccY\xe7\x00\x00\x00\x00IEND\xaeB`\x82"
)


def _headline(title: str, sentiment: str, confidence: float) -> HeadlineSentiment:
    return HeadlineSentiment(
        headline=title,
        sentiment=sentiment,
        confidence=confidence,
        brief_reason="Test reason.",
    )


def _summary() -> StockSummary:
    return StockSummary(
        ticker="AAPL",
        current_price=100.0,
        week_52_high=110.0,
        week_52_low=80.0,
        pe_ratio=20.0,
        ytd_return=0.10,
        momentum=MomentumSignal(
            score=0.5,
            label="Bullish",
            components=[
                MomentumComponent(name="macd_histogram_direction", value=0.2, vote=1, rationale="rising"),
            ],
        ),
        company_name="Apple Inc.",
        currency="USD",
        as_of_date="2026-10-08",
        pe_source="trailingPE",
        forward_pe=18.0,
        ytd_return_pct=10.0,
        latest_indicators={
            "SMA_50": 95.0,
            "SMA_200": 90.0,
            "RSI_14": 55.0,
            "MACD": 1.2,
            "MACD_Signal": 0.4,
            "MACD_Hist": 0.8,
            "BB_PctB": 0.7,
        },
    )


def _sentiment() -> SentimentResult:
    items = [
        _headline("Neutral filing", "neutral", 0.99),
        _headline("Weak positive", "positive", 0.40),
        _headline("Strong negative", "negative", 0.90),
        _headline("Mid positive", "positive", 0.70),
    ]
    return SentimentResult(items=items, score=-0.1, label="Neutral", n_scored=4, weight_sum=2.99)


def _signal(**overrides) -> SignalRecommendation:
    payload = {"signal": "Hold", "justification": _JUSTIFICATION, "provider": "groq"}
    payload.update(overrides)
    return SignalRecommendation(**payload)


def test_contribution_is_confidence_times_absolute_vote():
    assert headline_contribution(_headline("n", "neutral", 0.9)) == 0.0
    assert headline_contribution(_headline("p", "positive", 0.4)) == 0.4
    assert headline_contribution(_headline("d", "negative", 0.9)) == 0.9


def test_top_headlines_rank_by_contribution_then_confidence():
    # The confident neutral loses to every non-neutral headline.
    top = select_top_headlines(_sentiment().items, n=3)
    assert [item.headline for item in top] == ["Strong negative", "Mid positive", "Weak positive"]


def test_markdown_has_required_sections_and_disclaimer():
    text = build_brief_markdown(_summary(), _sentiment(), _signal(), "AAPL_technical_chart.png")
    for heading in (
        "## Company snapshot",
        "## Technical outlook",
        "## News sentiment",
        "## Recommendation",
        "## Risk disclaimer",
    ):
        assert heading in text
    assert config.RISK_DISCLAIMER in text
    assert "Strong negative" in text
    assert "Neutral filing" not in text
    assert "Hold" in text
    assert "Price is above both moving averages" in text
    assert "above the 50-day SMA" in text
    assert "AAPL_technical_chart.png" in text


def test_fallback_signal_is_labelled_as_fallback():
    text = build_brief_markdown(
        _summary(),
        _sentiment(),
        _signal(fallback=True, justification="", unavailable_reason="both providers failed"),
        None,
    )
    assert "pipeline fallback" in text
    assert "both providers failed" in text
    assert config.RISK_DISCLAIMER in text
    assert "Price is above both moving averages" not in text


def test_non_ascii_punctuation_is_stripped_from_model_text():
    text = build_brief_markdown(
        _summary(),
        SentimentResult(
            items=[_headline("Shares rise \u2014 sharply", "positive", 0.8)],
            score=0.8,
            label="Strong Positive",
            n_scored=1,
        ),
        _signal(),
        None,
    )
    assert "\u2014" not in text
    assert "Shares rise - sharply" in text
    assert text.isascii()


def test_html_inlines_the_chart_and_keeps_the_disclaimer(tmp_path: Path):
    chart = tmp_path / "AAPL_technical_chart.png"
    chart.write_bytes(_PNG)
    markdown_path, html_path = write_research_brief(
        _summary(), _sentiment(), _signal(), chart, output_dir=tmp_path
    )
    page = html_path.read_text(encoding="utf-8")
    assert markdown_path.name == "AAPL_research_brief.md"
    assert "data:image/png;base64," in page
    assert config.RISK_DISCLAIMER in page
    assert "<h2>Company snapshot</h2>" in page
    # The Markdown copy keeps a normal relative image, not the data URI.
    assert "data:image/png;base64," not in markdown_path.read_text(encoding="utf-8")


def test_html_still_renders_when_the_chart_is_missing(tmp_path: Path):
    page = build_brief_html(
        build_brief_markdown(_summary(), _sentiment(), _signal(), None),
        "Apple Inc. (AAPL)",
        tmp_path / "missing.png",
    )
    assert "data:image/png;base64," not in page
    assert config.RISK_DISCLAIMER in page
