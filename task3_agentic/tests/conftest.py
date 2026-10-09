"""Shared test setup: repo root on sys.path, a session with an in-memory trace, and offline data sources.

Every network seam the tools use (yfinance prices and .info, news, LLM scoring, DuckDuckGo)
is replaced with synthetic data, so the suite runs without keys or a connection.
"""
# AI-ASSISTED: Claude Code (claude-opus-5-5), Prompt: 'Implement the Task 3A plan (the plan approved in Entry 13)', Date: 2026-10-09 (see CITATIONS.md Entry 14)

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import pytest

# tests/ -> task3_agentic/ -> repo root, so `task1_financial` and `task3_agentic` both import.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from task1_financial.src import config as t1_config  # noqa: E402
from task1_financial.src import data as t1_data  # noqa: E402
from task1_financial.src import llm as t1_llm  # noqa: E402
from task1_financial.src import news as t1_news  # noqa: E402
from task1_financial.tests.conftest import make_ohlcv  # noqa: E402

from task3_agentic.src import config, tools  # noqa: E402
from task3_agentic.src.session import SessionContext  # noqa: E402
from task3_agentic.src.tracing import TraceLogger  # noqa: E402

# Synthetic history long enough for SMA-200 warm-up and a 1-year volatility percentile.
SYNTHETIC_SESSIONS = 800

FAKE_INFO = {
    "marketCap": 3.0e12,
    "trailingPE": 31.5,
    "forwardPE": 28.2,
    "profitMargins": 0.25,
    "revenueGrowth": 0.061,
    "debtToEquity": 145.0,
    "currentRatio": 0.95,
    "freeCashflow": 1.0e11,
    "beta": 1.2,
}

FAKE_HEADLINES = [
    "Apple beats earnings expectations on services strength",
    "Regulators open antitrust probe into App Store fees",
    "Apple shares hold steady ahead of product event",
    "Analyst cuts Apple price target on weak China demand",
]

FAKE_SEARCH_HITS = [
    {"title": "Analyst keeps Buy rating on Apple", "href": "https://example.com/a", "body": "Price target $260."},
    {"title": "Apple faces China headwinds", "href": "https://example.com/b", "body": "Shipments fell 8%."},
]


def synthetic_market(ticker: str = "AAPL", sessions: int = SYNTHETIC_SESSIONS) -> t1_data.MarketData:
    """Task 1's cleaned MarketData built from a random-walk frame."""
    ohlcv, quality = t1_data.validate_and_clean_ohlcv(
        make_ohlcv(sessions), source=t1_data.SOURCE_LIVE, fetched_at=datetime.now(timezone.utc).isoformat()
    )
    return t1_data.MarketData(ticker=ticker, ohlcv=ohlcv, info={"longName": "Apple Inc.", "currency": "USD"}, quality=quality)


def fake_score_headlines(news, ticker, company_name=None, clients=None):
    """Deterministic stand-in for Task 1's LLM scorer: 'beats' positive, 'probe'/'cuts' negative."""
    items = []
    for title in news.headlines:
        lowered = title.lower()
        label = "positive" if "beats" in lowered else "negative" if ("probe" in lowered or "cuts" in lowered) else "neutral"
        items.append(t1_llm.HeadlineSentiment(headline=title, sentiment=label, confidence=0.8, brief_reason="test", provider="fake"))
    score, label, n_scored, weight_sum, reason = t1_llm.aggregate_sentiment(items)
    return t1_llm.SentimentResult(
        items=items, score=score, label=label, n_scored=n_scored, weight_sum=weight_sum, unavailable_reason=reason
    )


def fake_headlines(ticker, company_name=None, target=10, minimum=10, **kwargs):
    items = [
        t1_news.NewsItem(title=title, publisher="Example Wire", link=f"https://example.com/{i}", source="yfinance")
        for i, title in enumerate(FAKE_HEADLINES)
    ]
    return t1_news.NewsResult(ticker=ticker, items=items[:target], sources_used=["yfinance"])


def valid_report(ctx: SessionContext, vol_shift: float = 0.0, risk_tools=("get_price_data", "calculate_volatility", "get_news")):
    """A ResearchReport that passes check_report for `ctx` (its volatility numbers come from the session)."""
    from task3_agentic.src.schemas import Evidence, FinancialHealth, HedgeStrategy, Metric, ResearchReport, Risk

    vol_records = ctx.results_for("calculate_volatility")
    vol_data = vol_records[0].result.data if vol_records else None
    return ResearchReport(
        ticker=ctx.ticker,
        financial_health=FinancialHealth(
            summary="Uptrend with a high multiple.",
            market_sentiment="Neutral headlines.",
            key_metrics=[
                Metric(name="Price", value="100", source_tool="get_price_data"),
                Metric(name="RSI", value="55", source_tool="get_price_data"),
                Metric(name="Forward P/E", value="28", source_tool="get_price_data"),
            ],
        ),
        risks=[
            Risk(name=f"Risk {i}", description="Could move the price.", evidence=[Evidence(source_tool=tool, detail="a number")])
            for i, tool in enumerate(risk_tools, start=1)
        ],
        hedge=HedgeStrategy(
            strategy="Protective put",
            instrument="90-day put near the 1-sigma low",
            rationale="Covers the downside risks.",
            sizing="One put per 100 shares.",
            annualised_vol_pct=(vol_data.annualised_vol_pct if vol_data else 25.0) + vol_shift,
            expected_move_90d_pct=vol_data.expected_move_pct if vol_data else 12.0,
            trade_offs="Premium cost.",
        ),
        data_gaps=[f"{tool} failed; used other tools" for tool in sorted(ctx.failed_tools())],
    )


@pytest.fixture(autouse=True)
def fast_retries(monkeypatch):
    """No sleeping between retries in Task 1's retry helper or the search fallback."""
    monkeypatch.setattr(t1_config, "RETRY_BACKOFF_MIN_SECONDS", 0)
    monkeypatch.setattr(t1_config, "RETRY_BACKOFF_MAX_SECONDS", 0)
    monkeypatch.setattr(config, "SEARCH_RETRY_WAIT_SECONDS", 0)


@pytest.fixture
def ctx() -> SessionContext:
    session = SessionContext("AAPL", trace=TraceLogger(None))
    # A non-empty client map stops llm_sentiment from reading real API keys.
    session.llm_clients = {"fake": object()}
    return session


@pytest.fixture
def search_calls(monkeypatch) -> List[Dict]:
    """Patch DuckDuckGo with canned hits; the returned list records each call."""
    calls: List[Dict] = []

    def fake_search(kind, query, timelimit):
        calls.append({"kind": kind, "query": query, "timelimit": timelimit})
        return FAKE_SEARCH_HITS

    monkeypatch.setattr(tools, "_ddgs_search", fake_search)
    return calls


@pytest.fixture
def offline_sources(monkeypatch, search_calls):
    """All five tools work offline with synthetic data."""
    monkeypatch.setattr(t1_data, "load_market_data", lambda ticker, **kwargs: synthetic_market(ticker))
    monkeypatch.setattr(tools, "_download_info", lambda ticker: dict(FAKE_INFO))
    monkeypatch.setattr(t1_news, "get_headlines", fake_headlines)
    monkeypatch.setattr(t1_llm, "score_headlines", fake_score_headlines)
    return search_calls
