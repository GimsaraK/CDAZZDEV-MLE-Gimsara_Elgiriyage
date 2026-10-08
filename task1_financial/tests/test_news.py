"""Offline tests for news.py using recorded fixtures (no network access)."""
# AI-ASSISTED: Cursor Agent (claude-sonnet-5.5), Prompt: 'yes implement the plan @TASK1A_PLAN.md', Date: 2026-10-08 (see CITATIONS.md Entry 4)

import json
from datetime import datetime, timedelta, timezone

import pytest

from task1_financial.src import news


def _item(title, hours_ago=0, source=news.SOURCE_GOOGLE_RSS, publisher=None):
    """News item with a distinct publisher per title unless one is given (keeps the per-publisher cap out of the way)."""
    published = datetime(2026, 10, 7, tzinfo=timezone.utc) - timedelta(hours=hours_ago)
    return news.NewsItem(
        title=title, publisher=publisher or f"Publisher of {title}", link="https://x", published_at=published,
        source=source,
    )


# --------------------------------------------------------------------------- parsing
def test_parse_yfinance_nested_schema(fixtures_dir):
    """The newer nested yfinance schema is parsed, including the clickThroughUrl link fallback."""
    raw = json.loads((fixtures_dir / "yfinance_news_nested.json").read_text())
    items = news.parse_yfinance_news(raw)
    assert [i.title for i in items] == ["Apple unveils new chip lineup", "Analysts weigh iPhone demand"]
    assert items[0].publisher == "Reuters"
    assert items[0].link == "https://example.com/apple-chips"
    assert items[1].link == "https://example.com/iphone-demand"  # clickThroughUrl fallback
    assert items[0].published_at == datetime(2026, 10, 6, 14, 30, tzinfo=timezone.utc)


def test_parse_yfinance_legacy_schema_skips_items_without_title(fixtures_dir):
    """The legacy flat schema is parsed; items without a title are skipped."""
    raw = json.loads((fixtures_dir / "yfinance_news_legacy.json").read_text())
    items = news.parse_yfinance_news(raw)
    assert len(items) == 1
    assert items[0].publisher == "CNBC"
    assert items[0].published_at is not None


def test_parse_yfinance_tolerates_garbage():
    """Non-dict items and None input are ignored instead of crashing."""
    assert news.parse_yfinance_news([None, 42, "text", {}]) == []
    assert news.parse_yfinance_news(None) == []


def test_parse_google_feed_strips_publisher_suffix(fixtures_dir):
    """The ' - Publisher' suffix moves from the title to the publisher field."""
    items = news.parse_google_news_feed((fixtures_dir / "google_news_sample.xml").read_bytes())
    assert len(items) == 3
    assert items[0].title == "Apple stock climbs on services growth"
    assert items[0].publisher == "MarketWatch"
    # No <source> element: publisher recovered from the title suffix.
    assert items[2].title == "Apple supplier outlook"
    assert items[2].publisher == "Some Blog"
    assert all(i.published_at is not None for i in items)


def test_google_url_contains_ticker_company_and_recency():
    """The RSS URL searches ticker and company name, limited by the when: operator."""
    url = news.build_google_news_url("AAPL", "Apple Inc.")
    assert "AAPL" in url and "Apple%20Inc." in url and "when%3A" in url


# --------------------------------------------------------------------------- dedupe / sort
def test_deduplicate_normalises_case_and_punctuation():
    """Titles differing only in case or punctuation count as duplicates; the first one wins."""
    items = [_item("Apple beats estimates!"), _item("apple beats estimates"), _item("Different story")]
    assert [i.title for i in news.deduplicate(items)] == ["Apple beats estimates!", "Different story"]


def test_sort_newest_first_puts_undated_last():
    """Newest items come first; items without a date go last."""
    undated = news.NewsItem(title="No date", source=news.SOURCE_GOOGLE_RSS)
    ordered = news.sort_newest_first([_item("old", 48), undated, _item("new", 1)])
    assert [i.title for i in ordered] == ["new", "old", "No date"]


@pytest.mark.parametrize(
    "title,low_signal",
    [
        ("AAPL 261005 335.00C (AAPL261005C335000) Stock Options Chain | Quotes & News", True),
        ("AAPL Oct 2026 315.000 call (AAPL261007C00315000) Stock Price, News, Quote & History", True),
        ("2,899 Apple Inc. $AAPL Shares Sold by Tudor Financial Inc.", True),
        ("Apple Inc. $AAPL Shares Acquired by Premier Path Wealth Partners LLC", True),
        ("Knollwood Investment Advisory LLC Trims Stock Holdings in Apple Inc. $AAPL", True),
        ("Vanguard Boosts Position in Apple Inc.", True),
        ("Apple (AAPL): Buy, Sell, or Hold Post Q2 Earnings?", False),
        ("Jefferies downgrades Apple over iPhone setback", False),
        ("Apple shares sold off after the keynote", False),
    ],
)
def test_low_signal_titles_detected(title, low_signal):
    """Quote/options pages and holdings filings are low-signal; real news is not."""
    assert news.is_low_signal_title(title) is low_signal


def test_cap_per_publisher_keeps_first_n_per_outlet():
    """At most N items per publisher are kept, in order; items without a publisher are kept."""
    items = [news.NewsItem(title=f"m{i}", publisher="MarketBeat", source="google_rss") for i in range(5)]
    items += [news.NewsItem(title="r1", publisher="Reuters", source="google_rss"),
              news.NewsItem(title="anon", source="google_rss")]
    kept = news.cap_per_publisher(items, max_per_publisher=3)
    assert [i.title for i in kept] == ["m0", "m1", "m2", "r1", "anon"]


def test_get_headlines_drops_low_signal_and_caps_publishers(monkeypatch, tmp_path):
    """End to end: low-signal titles are removed and no publisher exceeds the cap."""
    feed = [news.NewsItem(title=f"story {i}", publisher=f"Outlet {i % 6}", source="google_rss",
                          published_at=datetime(2026, 10, 7, tzinfo=timezone.utc) - timedelta(hours=i))
            for i in range(30)]
    feed.append(_item("AAPL Stock Options Chain | Quotes & News", 0))
    monkeypatch.setattr(news, "fetch_yfinance_news", lambda _t: [])
    monkeypatch.setattr(news, "fetch_google_news_rss", lambda _t, _c: feed)
    result = news.get_headlines("AAPL", data_dir=tmp_path, target=15, minimum=10)
    assert result.count == 15
    assert not any("Options Chain" in t for t in result.headlines)
    publishers = [i.publisher for i in result.items]
    assert max(publishers.count(p) for p in set(publishers)) <= 3


def test_blank_title_rejected():
    """A whitespace-only title fails NewsItem validation."""
    with pytest.raises(ValueError):
        news.NewsItem(title="   ", source=news.SOURCE_YFINANCE)


# --------------------------------------------------------------------------- orchestration
def test_get_headlines_tops_up_from_rss_and_trims_to_target(monkeypatch, tmp_path):
    """Empty yfinance results are topped up from RSS, trimmed to the target, and saved."""
    monkeypatch.setattr(news, "fetch_yfinance_news", lambda _t: [])
    monkeypatch.setattr(news, "fetch_google_news_rss", lambda _t, _c: [_item(f"story {i}", i) for i in range(20)])
    result = news.get_headlines("AAPL", data_dir=tmp_path, target=15, minimum=10)
    assert result.count == 15
    assert result.sources_used == [news.SOURCE_GOOGLE_RSS]
    assert result.items[0].title == "story 0"  # newest first
    assert result.warnings == []
    assert (tmp_path / "AAPL_news.json").exists()


def test_get_headlines_skips_rss_when_yfinance_suffices(monkeypatch, tmp_path):
    """RSS is not called when yfinance already provides enough headlines."""
    rss_calls = []
    monkeypatch.setattr(
        news, "fetch_yfinance_news", lambda _t: [_item(f"yf {i}", i, news.SOURCE_YFINANCE) for i in range(15)]
    )
    monkeypatch.setattr(news, "fetch_google_news_rss", lambda _t, _c: rss_calls.append(1) or [])
    result = news.get_headlines("AAPL", data_dir=tmp_path, target=15, minimum=10)
    assert result.count == 15 and rss_calls == []


def test_all_sources_fail_uses_snapshot(monkeypatch, tmp_path):
    """When every live source fails, headlines come from the snapshot (with a warning)."""
    snapshot = news.NewsResult(ticker="AAPL", items=[_item(f"cached {i}", i) for i in range(12)])
    news.save_news_snapshot(snapshot, tmp_path)

    def boom(*_args):
        raise ConnectionError("offline")

    monkeypatch.setattr(news, "fetch_yfinance_news", boom)
    monkeypatch.setattr(news, "fetch_google_news_rss", boom)
    result = news.get_headlines("AAPL", data_dir=tmp_path)
    assert result.from_snapshot
    assert result.count == 12
    assert any("snapshot" in w for w in result.warnings)


def test_all_sources_fail_without_snapshot_returns_empty_with_warning(monkeypatch, tmp_path):
    """No sources and no snapshot: an empty result with a warning, never an exception."""
    def boom(*_args):
        raise ConnectionError("offline")

    monkeypatch.setattr(news, "fetch_yfinance_news", boom)
    monkeypatch.setattr(news, "fetch_google_news_rss", boom)
    result = news.get_headlines("AAPL", data_dir=tmp_path)
    assert result.count == 0
    assert any("Only 0 headlines" in w for w in result.warnings)


def test_news_to_records_shape():
    """Display records contain exactly the four expected columns."""
    records = news.news_to_records([_item("hello")])
    assert set(records[0]) == {"published_at", "publisher", "source", "title"}
