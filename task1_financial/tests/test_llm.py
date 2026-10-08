"""Offline tests for llm.py. A scripted client stands in for Groq and OpenRouter, so no network is used."""
# AI-ASSISTED: Cursor Agent (claude-sonnet-5.5), Prompt: 'Implement the Task 1B plan', Date: 2026-10-08 (see CITATIONS.md Entry 6)

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from task1_financial.src import config, prompts
from task1_financial.src.errors import LLMUnavailableError
from task1_financial.src.llm import (
    HeadlineSentiment,
    SentimentResult,
    SignalRecommendation,
    aggregate_sentiment,
    count_sentences,
    score_headlines,
)
from task1_financial.src.llm import sentiment_label
from task1_financial.src.news import NewsItem, NewsResult
from task1_financial.src.summary import MomentumComponent, MomentumSignal, StockSummary


class ScriptedClient:
    """Returns queued strings from complete() and records every request."""

    def __init__(self, outputs, provider):
        self.outputs = list(outputs)
        self.provider = provider
        self.model = "fake-model"
        self.calls = []

    def complete(self, messages, max_tokens, response_format):
        self.calls.append({
            "messages": messages,
            "max_tokens": max_tokens,
            "response_format": response_format,
        })
        if not self.outputs:
            raise AssertionError(f"{self.provider} ran out of scripted responses")
        item = self.outputs.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _sentiment_json(headline="Apple rises on demand", sentiment="positive", confidence=0.9, reason="Demand improved."):
    return json.dumps({
        "headline": headline,
        "sentiment": sentiment,
        "confidence": confidence,
        "brief_reason": reason,
    })


def _item(sentiment, confidence, headline="h"):
    return HeadlineSentiment(
        headline=headline, sentiment=sentiment, confidence=confidence, brief_reason="Because."
    )


def _news(*titles):
    return NewsResult(
        ticker="AAPL",
        items=[NewsItem(title=title, source="google_rss") for title in titles],
    )


def _clients(groq_outputs, router_outputs):
    groq = ScriptedClient(groq_outputs, config.LLM_PROVIDER_GROQ)
    router = ScriptedClient(router_outputs, config.LLM_PROVIDER_OPENROUTER)
    return groq, router, {
        config.LLM_PROVIDER_GROQ: groq,
        config.LLM_PROVIDER_OPENROUTER: router,
    }


# --------------------------------------------------------------------------- aggregation
def test_all_positive_scores_strong_positive():
    """Three positive headlines at high confidence average to +1."""
    score, label, n_scored, weight, reason = aggregate_sentiment([
        _item("positive", 0.9), _item("positive", 0.8), _item("positive", 0.7),
    ])
    assert score == 1.0
    assert label == config.SENTIMENT_LABEL_STRONG_POSITIVE
    assert n_scored == 3 and weight == pytest.approx(2.4) and reason is None


def test_mixed_votes_match_hand_weighted_mean():
    """score = (0.8 * +1 + 0.2 * -1) / (0.8 + 0.2) = 0.6, which is the strong-positive boundary."""
    score, label, _, weight, _ = aggregate_sentiment([
        _item("positive", 0.8), _item("negative", 0.2),
    ])
    assert weight == pytest.approx(1.0)
    assert score == pytest.approx(0.6)
    assert label == config.SENTIMENT_LABEL_STRONG_POSITIVE


def test_neutral_votes_pull_the_score_toward_zero():
    """A confident neutral headline dilutes a positive one: (0.4 * +1 + 0.6 * 0) / 1.0 = 0.4."""
    score, label, _, _, _ = aggregate_sentiment([
        _item("positive", 0.4), _item("neutral", 0.6),
    ])
    assert score == pytest.approx(0.4)
    assert label == config.SENTIMENT_LABEL_POSITIVE


def test_zero_confidence_does_not_invent_a_score():
    """A total weight of 0 is "no evidence", not a neutral 0."""
    score, label, n_scored, _, reason = aggregate_sentiment([_item("positive", 0.0), _item("negative", 0.0)])
    assert score is None and label == config.SENTIMENT_LABEL_UNAVAILABLE
    assert n_scored == 2 and "zero confidence" in reason


def test_empty_input_is_unavailable():
    """No headlines means no score, rather than an error."""
    score, label, n_scored, _, reason = aggregate_sentiment([])
    assert score is None and n_scored == 0
    assert label == config.SENTIMENT_LABEL_UNAVAILABLE and reason


@pytest.mark.parametrize(
    "score,label",
    [
        (0.6, config.SENTIMENT_LABEL_STRONG_POSITIVE),
        (0.2, config.SENTIMENT_LABEL_POSITIVE),
        (0.19, config.SENTIMENT_LABEL_NEUTRAL),
        (-0.19, config.SENTIMENT_LABEL_NEUTRAL),
        (-0.2, config.SENTIMENT_LABEL_NEGATIVE),
        (-0.6, config.SENTIMENT_LABEL_STRONG_NEGATIVE),
        (None, config.SENTIMENT_LABEL_UNAVAILABLE),
    ],
)
def test_sentiment_label_thresholds_match_momentum(score, label):
    """The cut-offs are the momentum ones: 0.6 strong, 0.2 mild, symmetric around 0."""
    assert sentiment_label(score) == label


# --------------------------------------------------------------------------- schema
def test_schema_rejects_bad_payloads():
    """Missing field, unknown sentiment, confidence outside [0, 1], and a blank reason are invalid."""
    valid = {"headline": "h", "sentiment": "positive", "confidence": 0.5, "brief_reason": "Demand rose."}
    with pytest.raises(ValidationError):
        HeadlineSentiment.model_validate({k: v for k, v in valid.items() if k != "headline"})
    for bad in (
        {**valid, "sentiment": "good"},
        {**valid, "confidence": 1.5},
        {**valid, "confidence": -0.1},
        {**valid, "brief_reason": "   "},
    ):
        with pytest.raises(ValidationError):
            HeadlineSentiment.model_validate(bad)


def test_justification_sentence_bounds_and_decimals():
    """2 sentences is too few, 6 is too many, 4 is accepted, and 0.43 is not its own sentence."""
    base = "Price is above both moving averages. "
    two = base + "RSI agrees with that trend."
    four = two + " MACD is above its signal line. News sentiment points the same way."
    six = four + " Volume confirms the move. Risk is still elevated."
    decimal = "The momentum score is 0.43 and still positive. RSI sits in the bullish zone. MACD confirms the trend."

    assert count_sentences(decimal) == 3
    SignalRecommendation.model_validate({"signal": "Buy", "justification": four})
    SignalRecommendation.model_validate({"signal": "Hold", "justification": decimal})
    for text in (two, six):
        with pytest.raises(ValidationError):
            SignalRecommendation.model_validate({"signal": "Buy", "justification": text})


# --------------------------------------------------------------------------- repair and fallback
def test_invalid_json_is_repaired_on_the_second_call():
    """The first response is not JSON; the repair call is, so the headline is scored and the repair is logged."""
    title = "Apple rises on demand"
    groq, router, clients = _clients(
        ["{not json", _sentiment_json(title)],
        [],
    )
    result = score_headlines(_news(title), "AAPL", "Apple Inc.", clients=clients)

    assert result.n_scored == 1 and result.unscored == []
    assert result.items[0].provider == config.LLM_PROVIDER_GROQ
    assert result.items[0].headline == title
    assert any("repair" in warning.lower() for warning in result.warnings)
    assert router.calls == []

    first, second = groq.calls
    assert first["response_format"] == {"type": "json_object"}
    assert [message["role"] for message in first["messages"]] == ["system", "user"]
    assert [message["role"] for message in second["messages"]] == ["system", "user", "user"]
    assert "failed validation" in second["messages"][2]["content"]


def test_groq_failure_falls_back_to_openrouter():
    """Two invalid Groq replies, then a valid OpenRouter reply: the score is kept and the provider recorded."""
    title = "Apple rises on demand"
    groq, router, clients = _clients(
        ["{not json", "still not json"],
        [_sentiment_json(title, confidence=0.7)],
    )
    result = score_headlines(_news(title), "AAPL", clients=clients)

    assert result.n_scored == 1
    assert result.items[0].provider == config.LLM_PROVIDER_OPENROUTER
    assert len(groq.calls) == 2 and len(router.calls) == 1


def test_unscored_headline_is_left_out_of_the_average():
    """A headline both providers reject is listed as unscored and does not move the score."""
    good = "Apple beats estimates"
    bad = "Unreadable headline"
    groq, router, clients = _clients(
        [_sentiment_json(good, confidence=1.0), "{not json", "{not json"],
        ["{not json", "nope"],
    )
    result = score_headlines(_news(good, bad), "AAPL", clients=clients)

    assert [item.headline for item in result.items] == [good]
    assert result.unscored == [bad]
    assert result.score == 1.0
    assert len(groq.calls) == 3 and len(router.calls) == 2


def test_signal_falls_back_to_hold_when_both_providers_fail():
    """No valid signal from either provider becomes Hold with fallback=True, not a real thesis."""
    from task1_financial.src.llm import recommend_signal

    groq, router, clients = _clients(["{not json", "{not json"], ["{not json", "{not json"])
    summary = StockSummary(
        ticker="AAPL",
        current_price=100.0,
        momentum=MomentumSignal(
            score=0.4,
            label="Bullish",
            components=[MomentumComponent(name="trend_regime", value=0.05, vote=1, rationale="Uptrend.")],
        ),
    )
    sentiment = SentimentResult(score=0.5, label="Positive", n_scored=1, weight_sum=0.5)
    result = recommend_signal(summary, sentiment, clients=clients)

    assert result.fallback is True
    assert result.signal == config.SIGNAL_FALLBACK
    assert result.justification == ""
    assert result.unavailable_reason
    assert len(groq.calls) == 2 and len(router.calls) == 2


def test_signal_user_message_contains_the_momentum_votes():
    """The signal prompt is given each component vote, not only the momentum label."""
    from task1_financial.src.llm import build_signal_context, recommend_signal

    justification = (
        "Price is above both moving averages while RSI is not stretched. "
        "MACD agrees with that uptrend. News sentiment supports the same direction."
    )
    payload = json.dumps({"signal": "Buy", "justification": justification})
    groq, _, clients = _clients([payload], [])
    summary = StockSummary(
        ticker="AAPL",
        momentum=MomentumSignal(
            score=0.4,
            label="Bullish",
            components=[MomentumComponent(name="rsi_zone", value=60.0, vote=1, rationale="RSI mid-high.")],
            flags=["none"],
        ),
        latest_indicators={"RSI_14": 60.0},
    )
    recommend_signal(summary, SentimentResult(score=0.4, label="Positive"), clients=clients)
    user_text = groq.calls[0]["messages"][1]["content"]
    assert "rsi_zone" in user_text and "RSI_14" in user_text
    context = build_signal_context(summary, SentimentResult(score=0.4, label="Positive"))
    assert context["momentum"]["components"][0]["vote"] == 1


# --------------------------------------------------------------------------- prompts
def test_prompts_module_has_no_client_and_user_template_is_separate():
    """Prompt text lives on its own: no SDK imports, and the user turn is not the system instructions."""
    source = Path(prompts.__file__).read_text(encoding="utf-8")
    for banned in ("OpenAI", "requests", "httpx"):
        assert banned not in source
    rendered = prompts.SENTIMENT_USER.format(
        ticker="AAPL", company_name="Apple Inc.", headline="Apple rises on demand"
    )
    assert "Apple rises on demand" in rendered
    assert "Reply with a single JSON object" not in rendered


def test_chat_client_rejects_a_missing_key():
    """An empty key fails before any network call."""
    from task1_financial.src.llm import ChatClient

    with pytest.raises(LLMUnavailableError):
        ChatClient(config.LLM_PROVIDER_GROQ, config.LLM_MODEL_GROQ, "  ", config.GROQ_BASE_URL)


def test_pipeline_skips_the_llm_unless_asked(monkeypatch):
    """run_llm defaults to False, so a Part 1A run never calls the sentiment function."""
    from task1_financial.src import pipeline
    from task1_financial.src.data import DataQualityReport, MarketData

    from .conftest import make_ohlcv

    market = MarketData(
        ticker="AAPL",
        ohlcv=make_ohlcv(40),
        info={"longName": "Apple Inc."},
        quality=DataQualityReport(source="snapshot", rows=40),
    )
    monkeypatch.setattr(pipeline, "load_market_data", lambda *args, **kwargs: market)
    monkeypatch.setattr(pipeline, "get_headlines", lambda *args, **kwargs: NewsResult(ticker="AAPL"))
    called = []
    monkeypatch.setattr(pipeline, "score_headlines", lambda *args, **kwargs: called.append("called"))

    result = pipeline.run_pipeline("AAPL", save_snapshot=False)
    assert called == []
    assert result.sentiment is None and result.signal is None
    assert result.summary is not None


def test_pipeline_run_llm_records_a_fallback_as_degraded(monkeypatch):
    """A Hold fallback is a degraded run with the reason in errors, not a failed one."""
    from task1_financial.src import pipeline
    from task1_financial.src.data import DataQualityReport, MarketData

    from .conftest import make_ohlcv

    market = MarketData(
        ticker="AAPL",
        ohlcv=make_ohlcv(40),
        info={},
        quality=DataQualityReport(source="snapshot", rows=40),
    )
    monkeypatch.setattr(pipeline, "load_market_data", lambda *args, **kwargs: market)
    monkeypatch.setattr(pipeline, "get_headlines", lambda *args, **kwargs: NewsResult(ticker="AAPL"))
    monkeypatch.setattr(
        pipeline,
        "score_headlines",
        lambda *args, **kwargs: SentimentResult(score=None, unavailable_reason="no score"),
    )
    monkeypatch.setattr(
        pipeline,
        "recommend_signal",
        lambda *args, **kwargs: SignalRecommendation(
            signal="Hold", fallback=True, unavailable_reason="providers down",
        ),
    )

    result = pipeline.run_pipeline("AAPL", save_snapshot=False, run_llm=True)
    assert result.status == "degraded"
    assert result.sentiment is not None and result.signal.fallback is True
    assert any("providers down" in error for error in result.errors)


def test_load_api_keys_reports_presence_without_the_secret(monkeypatch):
    """The helper says whether a key is set. It does not return the key itself."""
    monkeypatch.setenv(config.ENV_GROQ_API_KEY, "secret-value-not-for-logs")
    monkeypatch.delenv(config.ENV_OPENROUTER_API_KEY, raising=False)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *args, **kwargs: False)

    from task1_financial.src import llm

    presence = llm.load_api_keys()
    assert presence[config.ENV_GROQ_API_KEY] is True
    assert presence[config.ENV_OPENROUTER_API_KEY] is False
    assert "secret-value-not-for-logs" not in repr(presence)
