"""The five tools: correct payload types, error envelopes instead of exceptions, fallbacks, fault injection."""
# AI-ASSISTED: Claude Code (claude-sonnet-5.5), Prompt: 'Implement the Task 3A plan (the plan approved in Entry 13)', Date: 2026-10-09 (see CITATIONS.md Entry 14)

import math

import pytest
from ddgs.exceptions import DDGSException, RatelimitException
from langchain_core.messages import ToolMessage

from task1_financial.src import data as t1_data
from task1_financial.src import news as t1_news
from task1_financial.src.errors import DataUnavailableError

from task3_agentic.src import config, tools
from task3_agentic.src import volatility as vol
from task3_agentic.src.schemas import NewsList, PriceData, SearchResults, SentimentScore, ToolResult, VolatilityResult
from task3_agentic.src.session import FaultConfig, InjectedFault
from task3_agentic.src.tools import ResearchTools, build_tools

from .conftest import FAKE_HEADLINES, synthetic_market


# --------------------------------------------------------------------------- get_price_data
def test_get_price_data_returns_typed_snapshot(ctx, offline_sources):
    result = ResearchTools(ctx).get_price_data("aapl", "1y")
    assert result.status == "ok" and isinstance(result.data, PriceData)
    data = result.data
    assert data.ticker == "AAPL" and data.period == "1y"
    assert {"SMA_50", "SMA_200", "RSI_14", "MACD", "BB_Upper"} <= set(data.indicators)
    assert len(data.recent_bars) == config.RECENT_BARS
    assert data.momentum_label and data.signals
    # Yahoo fractions become percent; other fields are copied.
    assert data.fundamentals.profit_margin_pct == pytest.approx(25.0)
    assert data.fundamentals.debt_to_equity_pct == pytest.approx(145.0)
    # The full frame stays in the session for later tools.
    assert "AAPL" in ctx.frames and len(ctx.frames["AAPL"]) == data.rows_fetched


def test_get_price_data_rejects_unknown_period_without_raising(ctx, offline_sources):
    result = ResearchTools(ctx).get_price_data("AAPL", "7y")
    assert result.status == "error"
    assert "1y" in result.hint and "max" in result.hint


def test_get_price_data_rejects_invalid_ticker(ctx, offline_sources):
    result = ResearchTools(ctx).get_price_data("NOT A TICKER!", "1y")
    assert result.status == "error" and "valid ticker" in result.error


def test_get_price_data_reports_unavailable_data(ctx, monkeypatch):
    def fail(ticker, **kwargs):
        raise DataUnavailableError("live and snapshot both failed")

    monkeypatch.setattr(t1_data, "load_market_data", fail)
    result = ResearchTools(ctx).get_price_data("AAPL", "1y")
    assert result.status == "error"
    assert "both failed" in result.error and result.hint == tools.ERROR_HINTS["get_price_data"]


def test_fundamentals_failure_keeps_the_tool_ok(ctx, offline_sources, monkeypatch):
    def broken(ticker):
        raise ConnectionError("Yahoo down")

    monkeypatch.setattr(tools, "_download_info", broken)
    result = ResearchTools(ctx).get_price_data("AAPL", "6mo")
    assert result.status == "ok"
    assert "Yahoo down" in result.data.fundamentals.unavailable_reason


def test_period_start():
    index = synthetic_market().ohlcv.index
    start = tools.period_start("1y", index)
    assert (index[-1] - start).days <= 366
    assert tools.period_start("max", index) == index[0]
    assert tools.period_start("ytd", index).year == index[-1].year


# --------------------------------------------------------------------------- calculate_volatility
def test_volatility_reuses_session_prices(ctx, offline_sources, monkeypatch):
    toolkit = ResearchTools(ctx)
    toolkit.get_price_data("AAPL", "1y")

    def must_not_fetch(*args, **kwargs):
        raise AssertionError("volatility should reuse the session frame")

    monkeypatch.setattr(t1_data, "load_market_data", must_not_fetch)
    result = toolkit.calculate_volatility("AAPL", 20)
    assert result.status == "ok" and result.source == "session"
    assert isinstance(result.data, VolatilityResult)
    prices = ctx.frames["AAPL"]["Adj Close"]
    expected = vol.annualised_vol(vol.log_returns(prices), 20) * 100
    assert result.data.annualised_vol_pct == pytest.approx(expected, abs=0.01)
    data = result.data
    assert data.horizon_trading_days == 62
    assert data.expected_move_pct == pytest.approx(data.annualised_vol_pct * math.sqrt(62 / 252), abs=0.02)
    assert data.band_1sigma_low < data.spot < data.band_1sigma_high
    assert data.band_2sigma_low < data.band_1sigma_low
    assert set(data.comparison_pct) == {"20d", "60d", "252d"}


def test_volatility_fetches_when_session_is_empty(ctx, offline_sources):
    result = ResearchTools(ctx).calculate_volatility("AAPL", "60")
    assert result.status == "ok" and result.data.window == 60 and result.source == "live"


@pytest.mark.parametrize("window", [1, 500, "abc"])
def test_volatility_rejects_bad_window(ctx, offline_sources, window):
    result = ResearchTools(ctx).calculate_volatility("AAPL", window)
    assert result.status == "error" and "between" in result.hint


def test_volatility_with_too_little_history_is_empty(ctx, monkeypatch):
    monkeypatch.setattr(t1_data, "load_market_data", lambda ticker, **kw: synthetic_market(ticker, sessions=30))
    result = ResearchTools(ctx).calculate_volatility("AAPL", 60)
    assert result.status == "empty" and "at most" in result.hint


# --------------------------------------------------------------------------- get_news
def test_get_news_returns_structured_list(ctx, offline_sources):
    result = ResearchTools(ctx).get_news("AAPL", 3)
    assert result.status == "ok" and isinstance(result.data, NewsList)
    assert result.data.count == 3
    assert result.data.headlines[0].title == FAKE_HEADLINES[0]
    assert result.data.headlines[0].url.startswith("https://example.com/")


def test_get_news_clamps_n_and_drops_long_urls(ctx, monkeypatch):
    seen = {}

    def headlines(ticker, company_name=None, target=10, minimum=10, **kwargs):
        seen["target"] = target
        long_link = "https://news.google.com/rss/articles/" + "x" * 300
        return t1_news.NewsResult(ticker=ticker, items=[t1_news.NewsItem(title="A headline", link=long_link, source="google_rss")])

    monkeypatch.setattr(t1_news, "get_headlines", headlines)
    result = ResearchTools(ctx).get_news("AAPL", 100)
    assert seen["target"] == config.MAX_NEWS_COUNT
    assert any("clamped" in w for w in result.warnings)
    assert result.data.headlines[0].url is None


def test_get_news_empty_points_to_web_search(ctx, monkeypatch):
    monkeypatch.setattr(t1_news, "get_headlines", lambda ticker, **kw: t1_news.NewsResult(ticker=ticker))
    result = ResearchTools(ctx).get_news("AAPL", 10)
    assert result.status == "empty" and "web_search" in result.hint


# --------------------------------------------------------------------------- llm_sentiment
@pytest.fixture
def news_seen(ctx, offline_sources):
    """A toolkit whose session has already fetched FAKE_HEADLINES, so they count as real headlines."""
    toolkit = ResearchTools(ctx)
    toolkit.get_news("AAPL", len(FAKE_HEADLINES))
    return toolkit


def test_llm_sentiment_scores_and_ranks(news_seen):
    result = news_seen.llm_sentiment(FAKE_HEADLINES + [FAKE_HEADLINES[0].upper(), "  "])
    assert result.status == "ok" and isinstance(result.data, SentimentScore)
    data = result.data
    # The duplicate (different case) and the blank string are dropped.
    assert data.n_scored == len(FAKE_HEADLINES)
    assert (data.positive, data.negative, data.neutral) == (1, 2, 1)
    assert -1 <= data.score <= 1
    assert data.top_negative and all(h.sentiment == "negative" for h in data.top_negative)


def test_llm_sentiment_empty_input_hints_get_news(ctx, offline_sources):
    result = ResearchTools(ctx).llm_sentiment([])
    assert result.status == "empty" and "get_news" in result.hint


def test_llm_sentiment_without_providers_is_an_error(ctx, news_seen):
    ctx.llm_clients = {}
    result = news_seen.llm_sentiment(FAKE_HEADLINES)
    assert result.status == "error" and "No LLM provider" in result.error


def test_llm_sentiment_caps_headline_count(ctx, offline_sources, monkeypatch):
    many = [f"Apple headline number {i} about the quarter" for i in range(config.MAX_SENTIMENT_HEADLINES + 5)]

    def lots(ticker, company_name=None, target=10, minimum=10, **kwargs):
        return t1_news.NewsResult(ticker=ticker, items=[t1_news.NewsItem(title=t, source="yfinance") for t in many])

    monkeypatch.setattr(t1_news, "get_headlines", lots)
    toolkit = ResearchTools(ctx)
    toolkit.get_news("AAPL", config.MAX_NEWS_COUNT)
    result = toolkit.llm_sentiment(many)
    assert result.data.n_scored == config.MAX_SENTIMENT_HEADLINES
    assert any("first" in w for w in result.warnings)


def test_llm_sentiment_rejects_invented_headlines(ctx, offline_sources):
    """No news fetched: a model that makes headlines up gets an error, and the scorer is never called."""
    result = ResearchTools(ctx).llm_sentiment(["Apple reports Q3 earnings beat expectations"])
    assert result.status == "error" and "get_news or web_search" in result.error
    assert "agent that has it" in result.hint


def test_llm_sentiment_drops_unknown_but_keeps_real_and_trimmed_titles(news_seen):
    trimmed = FAKE_HEADLINES[1][:-5]  # "...App Store" without " fees": still a real headline
    result = news_seen.llm_sentiment([FAKE_HEADLINES[0], trimmed, "A headline nobody published"])
    assert result.status == "ok" and result.data.n_scored == 2
    assert any("Dropped 1 headline" in w for w in result.warnings)


def test_search_result_titles_count_as_known(ctx, search_calls):
    toolkit = ResearchTools(ctx)
    toolkit.web_search("Apple analyst")
    assert tools.verify_headlines(["Analyst keeps Buy rating on Apple"], ctx.known_headlines()) == [
        "Analyst keeps Buy rating on Apple"
    ]
    # Short strings only match exactly; containment needs both sides to be long.
    assert tools.verify_headlines(["Apple"], ctx.known_headlines()) == []


def test_model_chain_client_skips_a_capped_model():
    from task3_agentic.src.llm import ModelChainClient

    class Fake:
        def __init__(self, model, fail):
            self.model, self.fail, self.calls = model, fail, 0

        def complete(self, messages, max_tokens, response_format):
            self.calls += 1
            if self.fail:
                raise RuntimeError("groq rate limited (HTTP 429)")
            return '{"ok": true}'

    big, small = Fake("openai/gpt-oss-120b", True), Fake("openai/gpt-oss-20b", False)
    chain = ModelChainClient("groq", [big, small])
    assert chain.complete([], 10, {}) == '{"ok": true}'
    assert chain.provider == "groq/openai/gpt-oss-20b"
    chain.complete([], 10, {})
    # The capped model was tried once, then skipped for the rest of the session.
    assert big.calls == 1 and small.calls == 2
    small.fail = True
    with pytest.raises(RuntimeError):
        chain.complete([], 10, {})


def test_fallback_chain_reports_every_error_and_skips_daily_limits():
    from langchain_core.runnables import RunnableLambda

    from task3_agentic.src.llm import AllModelsFailed, FallbackChain

    class Daily(Exception):
        status_code = 429

    calls = []

    def capped(_):
        calls.append("big")
        raise Daily("tokens per day (TPD) exceeded for org_123")

    def flaky(_):
        calls.append("small")
        if len(calls) < 3:
            raise ConnectionError("reset")
        return "answer"

    chain = FallbackChain([RunnableLambda(capped), RunnableLambda(flaky)], ["big", "small"])
    with pytest.raises(AllModelsFailed) as info:
        chain.invoke("x")
    assert "big: Daily: HTTP 429, daily limit reached" in str(info.value) and "small: ConnectionError" in str(info.value)
    assert "org_123" not in str(info.value)
    # The capped model is skipped from now on; the other one is tried again.
    assert chain.invoke("x") == "answer" and calls == ["big", "small", "small"]


# --------------------------------------------------------------------------- web_search
def test_web_search_returns_hits(ctx, search_calls):
    result = ResearchTools(ctx).web_search("Apple analyst price target")
    assert result.status == "ok" and isinstance(result.data, SearchResults)
    assert result.data.backend == "text" and result.data.hits[0].url == "https://example.com/a"
    assert search_calls[0]["timelimit"] == config.SEARCH_TIMELIMIT


def test_web_search_falls_back_to_news_after_rate_limit(ctx, monkeypatch):
    calls = []

    def flaky(kind, query, timelimit):
        calls.append(kind)
        if kind == "text":
            raise RatelimitException("202 Ratelimit")
        return [{"title": "News hit", "url": "https://example.com/n", "body": "x", "source": "Wire", "date": "2026-10-01"}]

    monkeypatch.setattr(tools, "_ddgs_search", flaky)
    result = ResearchTools(ctx).web_search("Apple analyst")
    assert result.status == "ok" and result.data.backend == "news"
    # The text backend was retried before falling back.
    assert calls == ["text"] * config.SEARCH_ATTEMPTS_PER_BACKEND + ["news"]
    assert result.warnings


def test_web_search_uses_simplified_query_last(ctx, monkeypatch):
    queries = []

    def picky(kind, query, timelimit):
        queries.append((kind, query))
        if query.startswith('"'):
            raise DDGSException("No results found.")
        return [{"title": "Hit", "href": "https://example.com"}]

    monkeypatch.setattr(tools, "_ddgs_search", picky)
    result = ResearchTools(ctx).web_search('"Apple" site:example.com analyst downgrade risk')
    assert result.status == "ok" and result.data.backend == "text-simplified"
    assert "site:" not in result.data.effective_query and '"' not in result.data.effective_query


def test_web_search_all_rate_limited_is_error_with_alternative(ctx, monkeypatch):
    def limited(kind, query, timelimit):
        raise RatelimitException("202 Ratelimit")

    monkeypatch.setattr(tools, "_ddgs_search", limited)
    result = ResearchTools(ctx).web_search("Apple analyst")
    assert result.status == "error" and "get_news" in result.hint


def test_web_search_no_results_is_empty(ctx, monkeypatch):
    def nothing(kind, query, timelimit):
        raise DDGSException("No results found.")

    monkeypatch.setattr(tools, "_ddgs_search", nothing)
    assert ResearchTools(ctx).web_search("zzzz").status == "empty"


def test_web_search_empty_query(ctx, search_calls):
    assert ResearchTools(ctx).web_search("   ").status == "error"
    assert not search_calls


def test_simplify_query():
    assert tools.simplify_query('"Apple" site:reuters.com risk, outlook! 2026 iPhone demand China') == (
        "Apple risk outlook 2026 iPhone demand"
    )


# --------------------------------------------------------------------------- robustness and fault injection
def test_unexpected_exception_becomes_error_envelope(ctx, monkeypatch):
    def boom(ticker, **kwargs):
        raise RuntimeError("unexpected")

    monkeypatch.setattr(t1_data, "load_market_data", boom)
    result = ResearchTools(ctx).get_price_data("AAPL", "1y")
    assert result.status == "error" and "RuntimeError" in result.error
    assert ctx.trace.records[-1]["status"] == "error"


def test_fault_config_modes():
    faults = FaultConfig({"a": "first", "b": "always", "c": 2})
    with pytest.raises(InjectedFault):
        faults.check("a")
    faults.check("a")
    for _ in range(3):
        with pytest.raises(InjectedFault):
            faults.check("b")
    for _ in range(2):
        with pytest.raises(InjectedFault):
            faults.check("c")
    faults.check("c")
    faults.check("untouched")


def test_injected_fault_then_recovery(ctx, offline_sources):
    ctx.faults = FaultConfig({"get_price_data": "first"})
    toolkit = ResearchTools(ctx)
    first = toolkit.get_price_data("AAPL", "1y")
    second = toolkit.get_price_data("AAPL", "1y")
    assert first.status == "error" and "Simulated" in first.error and first.hint
    assert second.status == "ok"
    assert ctx.tool_sequence() == ["get_price_data", "get_price_data"]
    assert ctx.failed_tools() == set()


# --------------------------------------------------------------------------- LangChain wrappers
def test_build_tools_exposes_exact_names_and_arguments(ctx):
    built = {tool.name: tool for tool in build_tools(ctx)}
    assert set(built) == {"get_price_data", "get_news", "calculate_volatility", "llm_sentiment", "web_search"}
    assert set(built["get_price_data"].args) == {"ticker", "period"}
    assert set(built["get_news"].args) == {"ticker", "n"}
    assert set(built["calculate_volatility"].args) == {"ticker", "window"}
    assert set(built["llm_sentiment"].args) == {"headlines"}
    assert set(built["web_search"].args) == {"query"}


def test_build_tools_can_restrict_the_set(ctx):
    assert [t.name for t in build_tools(ctx, names=["web_search", "get_news"])] == ["web_search", "get_news"]
    with pytest.raises(ValueError):
        build_tools(ctx, names=["delete_everything"])


def test_tool_message_carries_typed_artifact(ctx, offline_sources):
    search = build_tools(ctx, names=["web_search"])[0]
    message = search.invoke({"type": "tool_call", "name": "web_search", "args": {"query": "Apple"}, "id": "call_1"})
    assert isinstance(message, ToolMessage)
    assert isinstance(message.artifact, ToolResult) and message.artifact.ok
    assert '"status":"ok"' in message.content
    assert len(message.content) <= config.TOOL_MESSAGE_MAX_CHARS
