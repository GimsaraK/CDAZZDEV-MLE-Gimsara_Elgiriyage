"""Recent news headlines from free sources, normalised to one schema.

Sources, in priority order:
    1. yfinance ``Ticker.news`` (parses both the legacy flat and the newer nested schema)
    2. Google News RSS search (no API key) - tops up when yfinance returns too few items
Results are de-duplicated, sorted newest first, and trimmed to the target count.
If live sources cannot reach the minimum, the last committed snapshot is used instead.
"""
# AI-ASSISTED: Cursor Agent (claude-sonnet-5.5), Prompt: 'yes implement the plan @TASK1A_PLAN.md', Date: 2026-10-08 (see CITATIONS.md Entry 4)

import calendar
import re
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import feedparser
import requests
import yfinance as yf
from pydantic import BaseModel, Field, ValidationError, field_validator

from . import config
from .data import read_json_snapshot, write_json_snapshot
from .errors import DataUnavailableError
from .logging_utils import get_logger
from .retry import call_with_retry

logger = get_logger("news")

SOURCE_YFINANCE = "yfinance"
SOURCE_GOOGLE_RSS = "google_rss"
SOURCE_SNAPSHOT = "snapshot"
_NON_ALNUM = re.compile(r"[^a-z0-9 ]+")
_WHITESPACE = re.compile(r"\s+")
_EXCLUDE_PATTERNS = [re.compile(pattern, re.IGNORECASE) for pattern in config.NEWS_EXCLUDE_TITLE_PATTERNS]


class NewsItem(BaseModel):
    title: str
    publisher: Optional[str] = None
    link: Optional[str] = None
    published_at: Optional[datetime] = None
    source: str

    @field_validator("title")
    @classmethod
    def _title_not_blank(cls, value: str) -> str:
        value = _WHITESPACE.sub(" ", value).strip()
        if not value:
            raise ValueError("title is blank")
        return value


class NewsResult(BaseModel):
    ticker: str
    items: List[NewsItem] = Field(default_factory=list)
    sources_used: List[str] = Field(default_factory=list)
    from_snapshot: bool = False
    snapshot_fetched_at: Optional[str] = None
    warnings: List[str] = Field(default_factory=list)

    @property
    def headlines(self) -> List[str]:
        return [item.title for item in self.items]

    @property
    def count(self) -> int:
        return len(self.items)


# --------------------------------------------------------------------------- parsing (pure)
def _parse_iso_datetime(value: Any) -> Optional[datetime]:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _parse_epoch(value: Any) -> Optional[datetime]:
    try:
        return datetime.fromtimestamp(int(value), tz=timezone.utc)
    except (TypeError, ValueError, OverflowError, OSError):
        return None


def _nested(mapping: Any, *keys: str) -> Any:
    for key in keys:
        if not isinstance(mapping, dict):
            return None
        mapping = mapping.get(key)
    return mapping


def parse_yfinance_news(raw_items: Iterable[Any]) -> List[NewsItem]:
    """Parse yfinance news; supports the nested ``content`` schema and the legacy flat schema."""
    items: List[NewsItem] = []
    for raw in raw_items or []:
        if not isinstance(raw, dict):
            continue
        content = raw.get("content")
        if isinstance(content, dict):  # newer schema
            fields = {
                "title": content.get("title"),
                "publisher": _nested(content, "provider", "displayName"),
                "link": _nested(content, "canonicalUrl", "url") or _nested(content, "clickThroughUrl", "url"),
                "published_at": _parse_iso_datetime(content.get("pubDate") or content.get("displayTime")),
            }
        else:  # legacy schema
            fields = {
                "title": raw.get("title"),
                "publisher": raw.get("publisher"),
                "link": raw.get("link"),
                "published_at": _parse_epoch(raw.get("providerPublishTime")),
            }
        try:
            items.append(NewsItem(source=SOURCE_YFINANCE, **fields))
        except ValidationError:
            logger.debug("Skipping malformed yfinance news item: %s", raw)
    return items


def _split_google_title(title: str, publisher: Optional[str]) -> Tuple[str, Optional[str]]:
    """Google News appends ' - Publisher' to titles; move it into the publisher field."""
    separator = config.GOOGLE_NEWS_TITLE_SEPARATOR
    if publisher and title.endswith(f"{separator}{publisher}"):
        return title[: -len(separator + publisher)].strip(), publisher
    if not publisher and separator in title:
        head, tail = title.rsplit(separator, 1)
        return head.strip(), tail.strip() or None
    return title, publisher


def parse_google_news_feed(content: bytes) -> List[NewsItem]:
    feed = feedparser.parse(content)
    items: List[NewsItem] = []
    for entry in feed.entries:
        publisher = _nested(entry, "source", "title") if isinstance(entry.get("source"), dict) else None
        title, publisher = _split_google_title(entry.get("title", ""), publisher)
        published_struct = entry.get("published_parsed")
        published_at = (
            datetime.fromtimestamp(calendar.timegm(published_struct), tz=timezone.utc) if published_struct else None
        )
        try:
            items.append(
                NewsItem(
                    title=title,
                    publisher=publisher,
                    link=entry.get("link"),
                    published_at=published_at,
                    source=SOURCE_GOOGLE_RSS,
                )
            )
        except ValidationError:
            logger.debug("Skipping malformed RSS entry: %s", entry.get("title"))
    return items


def is_low_signal_title(title: str) -> bool:
    """True for quote/options listing pages and holdings-filing boilerplate returned by search feeds."""
    return any(pattern.search(title) for pattern in _EXCLUDE_PATTERNS)


def filter_low_signal(items: Iterable[NewsItem]) -> List[NewsItem]:
    return [item for item in items if not is_low_signal_title(item.title)]


def cap_per_publisher(items: Iterable[NewsItem], max_per_publisher: int = config.NEWS_MAX_PER_PUBLISHER) -> List[NewsItem]:
    """Keep at most ``max_per_publisher`` items per publisher, preserving order (items without one are kept)."""
    counts: Dict[str, int] = {}
    kept: List[NewsItem] = []
    for item in items:
        key = (item.publisher or "").strip().lower()
        if key:
            counts[key] = counts.get(key, 0) + 1
            if counts[key] > max_per_publisher:
                continue
        kept.append(item)
    return kept


def curate(items: Iterable[NewsItem]) -> List[NewsItem]:
    """De-duplicate, sort newest first, then cap per publisher (so the newest items per outlet survive)."""
    return cap_per_publisher(sort_newest_first(deduplicate(items)))


def _dedupe_key(title: str) -> str:
    return _WHITESPACE.sub(" ", _NON_ALNUM.sub(" ", title.lower())).strip()


def deduplicate(items: Iterable[NewsItem]) -> List[NewsItem]:
    """Drop items whose normalised title was already seen (earlier sources take priority)."""
    seen = set()
    unique: List[NewsItem] = []
    for item in items:
        key = _dedupe_key(item.title)
        if key and key not in seen:
            seen.add(key)
            unique.append(item)
    return unique


def sort_newest_first(items: Iterable[NewsItem]) -> List[NewsItem]:
    oldest = datetime.min.replace(tzinfo=timezone.utc)
    return sorted(items, key=lambda item: item.published_at or oldest, reverse=True)


# --------------------------------------------------------------------------- network
def fetch_yfinance_news(ticker: str) -> List[NewsItem]:
    raw = call_with_retry(lambda: yf.Ticker(ticker).news, f"yfinance news for {ticker}")
    return parse_yfinance_news(raw or [])


def build_google_news_url(ticker: str, company_name: Optional[str]) -> str:
    query = config.GOOGLE_NEWS_QUERY_TEMPLATE.format(
        ticker=ticker,
        company=company_name or ticker,
        recency=config.GOOGLE_NEWS_RECENCY_WINDOW,
    )
    return config.GOOGLE_NEWS_RSS_URL_TEMPLATE.format(query=urllib.parse.quote(query), **config.GOOGLE_NEWS_LOCALE)


def _download(url: str) -> bytes:
    response = requests.get(
        url,
        timeout=config.HTTP_TIMEOUT_SECONDS,
        headers={"User-Agent": config.HTTP_USER_AGENT},
    )
    response.raise_for_status()
    return response.content


def fetch_google_news_rss(ticker: str, company_name: Optional[str] = None) -> List[NewsItem]:
    url = build_google_news_url(ticker, company_name)
    content = call_with_retry(_download, f"Google News RSS for {ticker}", url)
    return parse_google_news_feed(content)


# --------------------------------------------------------------------------- snapshots
def _snapshot_path(ticker: str, data_dir: Path) -> Path:
    return data_dir / config.SNAPSHOT_NEWS_TEMPLATE.format(ticker=ticker)


def save_news_snapshot(result: NewsResult, data_dir: Path) -> None:
    payload = [item.model_dump(mode="json") for item in result.items]
    write_json_snapshot(_snapshot_path(result.ticker, data_dir), payload)


def load_news_snapshot(ticker: str, data_dir: Path) -> Tuple[List[NewsItem], Optional[str]]:
    raw, fetched_at = read_json_snapshot(_snapshot_path(ticker, data_dir))
    items = []
    for entry in raw or []:
        try:
            items.append(NewsItem(**entry))
        except (ValidationError, TypeError):
            logger.debug("Skipping malformed snapshot news item: %s", entry)
    return items, fetched_at


# --------------------------------------------------------------------------- orchestration
def get_headlines(
    ticker: str = config.TICKER,
    company_name: Optional[str] = None,
    target: int = config.NEWS_TARGET_COUNT,
    minimum: int = config.NEWS_MIN_COUNT,
    data_dir: Path = config.DATA_DIR,
    use_snapshot_fallback: bool = True,
    save_snapshot: bool = True,
) -> NewsResult:
    """Collect ``target`` recent headlines (at least ``minimum``). Never raises."""
    ticker = ticker.strip().upper()
    result = NewsResult(ticker=ticker)
    sources: List[Tuple[str, Callable[[], List[NewsItem]]]] = [
        (SOURCE_YFINANCE, lambda: fetch_yfinance_news(ticker)),
        (SOURCE_GOOGLE_RSS, lambda: fetch_google_news_rss(ticker, company_name)),
    ]

    collected: List[NewsItem] = []
    for name, fetcher in sources:
        if len(curate(collected)) >= target:
            break
        try:
            fetched = fetcher()
        except Exception as exc:  # noqa: BLE001 - one failing source must not stop the others
            result.warnings.append(f"{name} failed: {exc}")
            logger.warning("News source %s failed: %s", name, exc)
            continue
        news_only = filter_low_signal(fetched)
        logger.info(
            "News source %s returned %d items (%d listing/filing items removed)",
            name,
            len(fetched),
            len(fetched) - len(news_only),
        )
        if news_only:
            result.sources_used.append(name)
            collected.extend(news_only)

    curated = curate(collected)
    capped_out = len(sort_newest_first(deduplicate(collected))) - len(curated)
    if capped_out:
        logger.info("Per-publisher cap (%d) removed %d items", config.NEWS_MAX_PER_PUBLISHER, capped_out)
    result.items = curated[:target]

    if result.count < minimum and use_snapshot_fallback:
        _apply_snapshot_fallback(result, data_dir)
    elif result.count >= minimum and save_snapshot:
        save_news_snapshot(result, data_dir)

    if result.count < minimum:
        message = f"Only {result.count} headlines found (minimum {minimum})"
        result.warnings.append(message)
        logger.warning(message)
    return result


def _apply_snapshot_fallback(result: NewsResult, data_dir: Path) -> None:
    try:
        snapshot_items, fetched_at = load_news_snapshot(result.ticker, data_dir)
    except DataUnavailableError as exc:
        result.warnings.append(f"No news snapshot available: {exc}")
        return
    if len(snapshot_items) > result.count:
        logger.warning("Using news snapshot fetched at %s (%d items)", fetched_at, len(snapshot_items))
        result.items = snapshot_items
        result.sources_used = [SOURCE_SNAPSHOT]
        result.from_snapshot = True
        result.snapshot_fetched_at = fetched_at
        result.warnings.append(f"Live sources returned too few headlines; using snapshot fetched at {fetched_at}")


def news_to_records(items: Iterable[NewsItem]) -> List[Dict[str, Any]]:
    """Flat records for display (e.g. a pandas DataFrame in the notebook)."""
    return [
        {
            "published_at": item.published_at.strftime("%Y-%m-%d %H:%M UTC") if item.published_at else None,
            "publisher": item.publisher,
            "source": item.source,
            "title": item.title,
        }
        for item in items
    ]
