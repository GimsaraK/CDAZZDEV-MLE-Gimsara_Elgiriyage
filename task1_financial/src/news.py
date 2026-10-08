"""Recent news headlines from free sources, normalised to one schema.

Sources, in priority order:
    1. yfinance ``Ticker.news`` (parses both the legacy flat and the newer nested schema)
    2. Google News RSS search (no API key) - tops up when yfinance returns too few items
Low-signal items (quote/options pages, holdings filings) are filtered out; the rest are
de-duplicated, sorted newest first, capped per publisher, and trimmed to the target count.
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

# Source labels stored on every headline so the output shows where each one came from.
SOURCE_YFINANCE = "yfinance"
SOURCE_GOOGLE_RSS = "google_rss"
SOURCE_SNAPSHOT = "snapshot"
# Regexes used to normalise titles for duplicate detection (strip punctuation, collapse spaces).
_NON_ALNUM = re.compile(r"[^a-z0-9 ]+")
_WHITESPACE = re.compile(r"\s+")
# Low-signal title patterns from config, compiled once at import time (case-insensitive).
_EXCLUDE_PATTERNS = [re.compile(pattern, re.IGNORECASE) for pattern in config.NEWS_EXCLUDE_TITLE_PATTERNS]


class NewsItem(BaseModel):
    """One headline in a source-independent schema. Pydantic rejects items without a usable title."""

    title: str
    publisher: Optional[str] = None
    link: Optional[str] = None
    published_at: Optional[datetime] = None
    source: str

    @field_validator("title")
    @classmethod
    def _title_not_blank(cls, value: str) -> str:
        # Collapse newlines / repeated spaces from feeds; an empty title makes the item invalid.
        value = _WHITESPACE.sub(" ", value).strip()
        if not value:
            raise ValueError("title is blank")
        return value


class NewsResult(BaseModel):
    """Final curated headlines plus provenance: which sources worked, snapshot use, and warnings."""

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
    """Parse an ISO-8601 timestamp (newer yfinance schema) to a UTC-aware datetime; None if invalid."""
    if not value:
        return None
    try:
        # Python 3.10's fromisoformat does not accept a trailing 'Z', so convert it to '+00:00'.
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    # Assume UTC when no timezone is given, so all timestamps can be compared and sorted.
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _parse_epoch(value: Any) -> Optional[datetime]:
    """Parse a Unix timestamp in seconds (legacy yfinance schema) to a UTC datetime; None if invalid."""
    try:
        return datetime.fromtimestamp(int(value), tz=timezone.utc)
    except (TypeError, ValueError, OverflowError, OSError):
        return None


def _nested(mapping: Any, *keys: str) -> Any:
    """Safe nested lookup: _nested(d, "a", "b") == d["a"]["b"], or None if any level is missing."""
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
        # Newer yfinance versions put the fields under a "content" key; older ones use flat keys.
        # Detect which format each item uses and map both onto the same NewsItem fields.
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
        # Items that fail validation (e.g. no title) are skipped rather than failing the whole batch.
        try:
            items.append(NewsItem(source=SOURCE_YFINANCE, **fields))
        except ValidationError:
            logger.debug("Skipping malformed yfinance news item: %s", raw)
    return items


def _split_google_title(title: str, publisher: Optional[str]) -> Tuple[str, Optional[str]]:
    """Google News appends ' - Publisher' to titles; move it into the publisher field."""
    separator = config.GOOGLE_NEWS_TITLE_SEPARATOR
    # Case 1: the feed gave us the publisher, so just remove the ' - Publisher' suffix from the title.
    if publisher and title.endswith(f"{separator}{publisher}"):
        return title[: -len(separator + publisher)].strip(), publisher
    # Case 2: no publisher field, so take it from the text after the last ' - ' in the title.
    if not publisher and separator in title:
        head, tail = title.rsplit(separator, 1)
        return head.strip(), tail.strip() or None
    return title, publisher


def parse_google_news_feed(content: bytes) -> List[NewsItem]:
    """Parse a Google News RSS document (raw bytes) into NewsItems; malformed entries are skipped."""
    feed = feedparser.parse(content)
    items: List[NewsItem] = []
    for entry in feed.entries:
        # <source> holds the publisher name when present.
        publisher = _nested(entry, "source", "title") if isinstance(entry.get("source"), dict) else None
        title, publisher = _split_google_title(entry.get("title", ""), publisher)
        # feedparser gives the date as a UTC time.struct_time; calendar.timegm converts it to epoch
        # seconds without applying the local timezone (time.mktime would).
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
    """Remove items whose title matches a low-signal pattern (see config.NEWS_EXCLUDE_TITLE_PATTERNS)."""
    return [item for item in items if not is_low_signal_title(item.title)]


def cap_per_publisher(items: Iterable[NewsItem], max_per_publisher: int = config.NEWS_MAX_PER_PUBLISHER) -> List[NewsItem]:
    """Keep at most ``max_per_publisher`` items per publisher, preserving order (items without one are kept)."""
    counts: Dict[str, int] = {}  # items kept so far per publisher
    kept: List[NewsItem] = []
    for item in items:
        # Compare publishers case-insensitively so 'Reuters' and 'reuters' share one quota.
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
    """Normalised title used for duplicate detection: lower-case, punctuation removed, single spaces."""
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
    """Sort by publish time, newest first; items without a date go to the end."""
    # Undated items get the earliest possible (timezone-aware) date so they sort last without errors.
    oldest = datetime.min.replace(tzinfo=timezone.utc)
    return sorted(items, key=lambda item: item.published_at or oldest, reverse=True)


# --------------------------------------------------------------------------- network
def fetch_yfinance_news(ticker: str) -> List[NewsItem]:
    """Fetch and parse yfinance news for the ticker (network errors are retried)."""
    raw = call_with_retry(lambda: yf.Ticker(ticker).news, f"yfinance news for {ticker}")
    return parse_yfinance_news(raw or [])


def build_google_news_url(ticker: str, company_name: Optional[str]) -> str:
    """Build the Google News RSS search URL for a query like '"AAPL" OR "Apple Inc." stock when:7d'."""
    # Search by ticker OR company name, limited to the recency window in config;
    # quote() URL-encodes spaces and quotation marks.
    query = config.GOOGLE_NEWS_QUERY_TEMPLATE.format(
        ticker=ticker,
        company=company_name or ticker,
        recency=config.GOOGLE_NEWS_RECENCY_WINDOW,
    )
    return config.GOOGLE_NEWS_RSS_URL_TEMPLATE.format(query=urllib.parse.quote(query), **config.GOOGLE_NEWS_LOCALE)


def _download(url: str) -> bytes:
    """HTTP GET with a timeout (so a hung server cannot block the pipeline) and an identifying User-Agent."""
    response = requests.get(
        url,
        timeout=config.HTTP_TIMEOUT_SECONDS,
        headers={"User-Agent": config.HTTP_USER_AGENT},
    )
    # Turn HTTP 4xx/5xx responses into exceptions so call_with_retry can retry them.
    response.raise_for_status()
    return response.content


def fetch_google_news_rss(ticker: str, company_name: Optional[str] = None) -> List[NewsItem]:
    """Fetch and parse Google News RSS results for the ticker / company (no API key needed)."""
    url = build_google_news_url(ticker, company_name)
    content = call_with_retry(_download, f"Google News RSS for {ticker}", url)
    return parse_google_news_feed(content)


# --------------------------------------------------------------------------- snapshots
def _snapshot_path(ticker: str, data_dir: Path) -> Path:
    """Location of the news snapshot file for a ticker."""
    return data_dir / config.SNAPSHOT_NEWS_TEMPLATE.format(ticker=ticker)


def save_news_snapshot(result: NewsResult, data_dir: Path) -> None:
    """Save the curated headlines as JSON (mode="json" turns datetimes into ISO strings)."""
    payload = [item.model_dump(mode="json") for item in result.items]
    write_json_snapshot(_snapshot_path(result.ticker, data_dir), payload)


def load_news_snapshot(ticker: str, data_dir: Path) -> Tuple[List[NewsItem], Optional[str]]:
    """Load saved headlines; returns (items, fetched_at). Raises DataUnavailableError if no snapshot."""
    raw, fetched_at = read_json_snapshot(_snapshot_path(ticker, data_dir))
    items = []
    # Re-validate each saved item; a corrupted entry is skipped instead of discarding the whole file.
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
    # Sources in priority order. Lambdas delay the network call until the loop decides it is needed.
    sources: List[Tuple[str, Callable[[], List[NewsItem]]]] = [
        (SOURCE_YFINANCE, lambda: fetch_yfinance_news(ticker)),
        (SOURCE_GOOGLE_RSS, lambda: fetch_google_news_rss(ticker, company_name)),
    ]

    # Step 1 - query sources one by one, stopping as soon as we have enough curated headlines.
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
        # Drop quote pages / holdings filings before they can take up headline slots.
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

    # Step 2 - curate the combined list and keep the newest `target` headlines.
    curated = curate(collected)
    # For the log only: how many items the per-publisher cap removed (curated before vs after the cap).
    capped_out = len(sort_newest_first(deduplicate(collected))) - len(curated)
    if capped_out:
        logger.info("Per-publisher cap (%d) removed %d items", config.NEWS_MAX_PER_PUBLISHER, capped_out)
    result.items = curated[:target]

    # Step 3 - too few headlines: try the snapshot. Enough headlines: refresh the snapshot.
    if result.count < minimum and use_snapshot_fallback:
        _apply_snapshot_fallback(result, data_dir)
    elif result.count >= minimum and save_snapshot:
        save_news_snapshot(result, data_dir)

    # Step 4 - still short after the fallback: return what we have, with a warning.
    if result.count < minimum:
        message = f"Only {result.count} headlines found (minimum {minimum})"
        result.warnings.append(message)
        logger.warning(message)
    return result


def _apply_snapshot_fallback(result: NewsResult, data_dir: Path) -> None:
    """Replace the live items with the snapshot (in place) if the snapshot has more headlines."""
    try:
        snapshot_items, fetched_at = load_news_snapshot(result.ticker, data_dir)
    except DataUnavailableError as exc:
        result.warnings.append(f"No news snapshot available: {exc}")
        return
    # Only switch if the snapshot is actually better; otherwise keep the (fresher) live items.
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
