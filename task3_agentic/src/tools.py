"""The five research tools.

Each tool is a method on ResearchTools that returns a typed ToolResult and never raises.
build_tools() wraps them as LangChain StructuredTools with the exact names and arguments
from the specification. The agent receives compact JSON, and the typed ToolResult rides
along as the ToolMessage artifact.

Market data, indicators, headlines and headline sentiment reuse the tested Task 1 code
(task1_financial.src), including its live -> snapshot fallbacks.
"""
# AI-ASSISTED: Claude Code (claude-opus-5-5), Prompt: 'Implement the Task 3A plan (the plan approved in Entry 13)', Date: 2026-10-09 (see CITATIONS.md Entry 14)
# AI-ASSISTED: Claude Code (claude-opus-5-5), Prompt: 'Implement the Task 3B plan', Date: 2026-10-09 (see CITATIONS.md Entry 16): per-agent toolkits

import math
import re
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from task1_financial.src import config as t1_config
from task1_financial.src import data as t1_data
from task1_financial.src import indicators as t1_indicators
from task1_financial.src import llm as t1_llm
from task1_financial.src import news as t1_news
from task1_financial.src import summary as t1_summary
from task1_financial.src.retry import call_with_retry

from . import config
from . import volatility as vol
from .llm import build_sentiment_clients, load_api_keys
from .logging_utils import get_logger
from .schemas import (
    Fundamentals,
    NewsHeadline,
    NewsList,
    PriceData,
    ScoredHeadline,
    SearchHit,
    SearchResults,
    SentimentScore,
    ToolResult,
    VolatilityResult,
)
from .session import SessionContext
from .tracing import run_traced, safe_error

logger = get_logger("tools")

_TICKER_RE = re.compile(config.TICKER_PATTERN)
_NON_WORD = re.compile(r"[^\w\s.$-]+")
_SEARCH_OPERATOR = re.compile(r"\b\w+:\S+")
_NON_WORD_HEADLINE = re.compile(r"[^a-z0-9]+")

# Shown to the agent when a tool raised unexpectedly. Each one names a concrete alternative.
ERROR_HINTS = {
    "get_price_data": (
        "Retry once (network errors are often transient). If it fails again, check the ticker symbol "
        "and use get_news or web_search for qualitative evidence while prices are unavailable."
    ),
    "get_news": "Use web_search with a query such as '<company> stock news this week' to find recent headlines.",
    "calculate_volatility": "Retry once, or call get_price_data first so prices are cached for this session.",
    "llm_sentiment": (
        "Retry once with fewer headlines. If it still fails, judge sentiment from web_search commentary "
        "and record the gap."
    ),
    "web_search": (
        "Retry with a broader or reworded query. If search stays unavailable, use get_news headlines "
        "(and llm_sentiment on them) as the qualitative source."
    ),
}


# --------------------------------------------------------------------------- helpers
def _result(status: str, tool: str, **kwargs: Any) -> ToolResult:
    return ToolResult(status=status, tool=tool, **kwargs)


def validate_ticker(ticker: Any) -> str:
    """Upper-case and check the symbol. Raises ValueError for anything that cannot be a ticker."""
    if not isinstance(ticker, str):
        raise ValueError(f"ticker must be a string, got {ticker!r}")
    symbol = ticker.strip().upper()
    if not _TICKER_RE.match(symbol):
        raise ValueError(f"{ticker!r} is not a valid ticker symbol")
    return symbol


def _finite(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _round(value: Optional[float], digits: int = 2) -> Optional[float]:
    return None if value is None else round(value, digits)


def _pct(value: Optional[float]) -> Optional[float]:
    """Fraction -> percent, rounded."""
    return None if value is None else round(value * 100, 2)


def _price_column(df: pd.DataFrame) -> str:
    """Adjusted close for returns (dividends and splits included), raw close if that is missing."""
    return t1_config.PRICE_COL_ADJ if t1_config.PRICE_COL_ADJ in df.columns else t1_config.PRICE_COL_RAW


def period_start(period: str, index: pd.DatetimeIndex) -> pd.Timestamp:
    """First session inside `period`, counted back from the last session."""
    last = index[-1]
    if period == "max":
        return index[0]
    if period == "ytd":
        start = pd.Timestamp(year=last.year, month=1, day=1)
    elif period.endswith("mo"):
        start = last - pd.DateOffset(months=int(period[:-2]))
    else:  # "1y", "2y", ...
        start = last - pd.DateOffset(years=int(period[:-1]))
    inside = index[index >= start]
    return inside[0] if len(inside) else index[0]


# --------------------------------------------------------------------------- network seams (patched in tests)
def _download_info(ticker: str) -> Dict[str, Any]:
    """Full yfinance .info payload (Task 1 keeps only a few keys, the fundamentals block needs more)."""
    import yfinance as yf

    info = yf.Ticker(ticker).info
    if not isinstance(info, dict):
        raise TypeError(f"Unexpected info payload type: {type(info).__name__}")
    return info


def _ddgs_search(kind: str, query: str, timelimit: Optional[str]) -> List[Dict[str, Any]]:
    """One DuckDuckGo call through ddgs (the renamed duckduckgo-search package)."""
    from ddgs import DDGS

    with DDGS(timeout=config.SEARCH_TIMEOUT_SECONDS) as client:
        search = client.news if kind == "news" else client.text
        return search(
            query,
            region=config.SEARCH_REGION,
            safesearch=config.SEARCH_SAFESEARCH,
            timelimit=timelimit,
            max_results=config.SEARCH_MAX_RESULTS,
        )


def fetch_fundamentals(ticker: str) -> Fundamentals:
    """The fundamentals block. Never raises; a failed .info call is reported in unavailable_reason."""
    try:
        info = call_with_retry(_download_info, f"fundamentals for {ticker}", ticker)
    except Exception as exc:  # noqa: BLE001 - fundamentals are optional context
        logger.warning("Fundamentals unavailable for %s: %s", ticker, exc)
        return Fundamentals(unavailable_reason=f"yfinance .info failed: {exc}")
    values: Dict[str, Optional[float]] = {}
    for yahoo_key, name in config.FUNDAMENTAL_KEYS.items():
        number = _finite(info.get(yahoo_key))
        if name in config.FUNDAMENTAL_FRACTION_FIELDS:
            values[f"{name}_pct"] = _pct(number)
        else:
            values[name] = _round(number, 4) if number is not None and abs(number) < 1e6 else number
    if all(value is None for value in values.values()):
        return Fundamentals(unavailable_reason="yfinance .info returned none of the fundamentals fields")
    return Fundamentals(**values)


def derive_signals(df: pd.DataFrame, price: Optional[float], flags: Iterable[str]) -> List[str]:
    """Plain-language facts from the latest indicators, so the agent reasons over combinations."""
    last = df.iloc[-1]
    signals: List[str] = []
    sma_s = _finite(last.get(t1_indicators.COL_SMA_SHORT))
    sma_l = _finite(last.get(t1_indicators.COL_SMA_LONG))
    rsi = _finite(last.get(t1_indicators.COL_RSI))
    hist = _finite(last.get(t1_indicators.COL_MACD_HIST))
    pct_b = _finite(last.get(t1_indicators.COL_BB_PCT_B))
    if price is not None and sma_s is not None and sma_l is not None:
        above = [name for name, level in (("SMA-50", sma_s), ("SMA-200", sma_l)) if price > level]
        signals.append(f"Price is above {', '.join(above)}" if above else "Price is below both SMA-50 and SMA-200")
        signals.append("SMA-50 above SMA-200 (long-term uptrend)" if sma_s > sma_l else "SMA-50 below SMA-200 (long-term downtrend)")
    if rsi is not None:
        if rsi >= t1_config.RSI_OVERBOUGHT:
            signals.append(f"RSI {rsi:.1f} is overbought (>= {t1_config.RSI_OVERBOUGHT})")
        elif rsi <= t1_config.RSI_OVERSOLD:
            signals.append(f"RSI {rsi:.1f} is oversold (<= {t1_config.RSI_OVERSOLD})")
        else:
            signals.append(f"RSI {rsi:.1f} is in the neutral zone")
    if hist is not None:
        signals.append("MACD histogram positive (bullish momentum)" if hist > 0 else "MACD histogram negative (bearish momentum)")
    if pct_b is not None:
        if pct_b > t1_config.BB_PCT_B_UPPER:
            signals.append("Price is above the upper Bollinger Band")
        elif pct_b < t1_config.BB_PCT_B_LOWER:
            signals.append("Price is below the lower Bollinger Band")
    signals.extend(flag for flag in flags if flag not in signals)
    return signals


def _headline_key(text: str) -> str:
    return " ".join(_NON_WORD_HEADLINE.sub(" ", text.lower()).split())


def verify_headlines(headlines: Sequence[str], known: Sequence[str]) -> List[str]:
    """The headlines that match a title a news or search tool returned (case and punctuation ignored).

    A model may trim a long title, so containment either way also counts once both sides are long
    enough to make an accidental match unlikely.
    """
    known_keys = [_headline_key(k) for k in known]
    verified = []
    for title in headlines:
        key = _headline_key(title)
        if not key:
            continue
        for candidate in known_keys:
            if key == candidate or (
                min(len(key), len(candidate)) >= config.HEADLINE_MATCH_MIN_CHARS and (key in candidate or candidate in key)
            ):
                verified.append(title)
                break
    return verified


def simplify_query(query: str) -> str:
    """Drop search operators, quotes and punctuation, keep the first few words."""
    text = _SEARCH_OPERATOR.sub(" ", query)
    text = _NON_WORD.sub(" ", text)
    words = text.split()
    return " ".join(words[: config.SEARCH_SIMPLIFIED_MAX_WORDS])


def _to_hits(kind: str, raw: Sequence[Dict[str, Any]]) -> List[SearchHit]:
    hits = []
    for item in raw:
        title = str(item.get("title") or "").strip()
        if not title:
            continue
        snippet = str(item.get("body") or "").strip()
        hits.append(
            SearchHit(
                title=title,
                url=item.get("href") or item.get("url"),
                snippet=snippet[: config.SEARCH_SNIPPET_MAX_CHARS] or None,
                date=item.get("date"),
                source=item.get("source"),
            )
        )
    return hits


# --------------------------------------------------------------------------- the tools
class ResearchTools:
    """The five tools bound to one session. Public methods are traced and never raise.

    In 3B each agent gets its own instance: `agent_name` attributes its calls in the trace and
    `allowed` is its tool list. A call outside `allowed` is refused inside run_traced, even when
    code calls the method directly (the third enforcement layer, after tool binding and ToolNode).
    """

    def __init__(
        self,
        ctx: SessionContext,
        agent_name: Optional[str] = None,
        allowed: Optional[Iterable[str]] = None,
    ) -> None:
        self.ctx = ctx
        self.agent_name = agent_name or ctx.agent_name
        self.allowed = frozenset(allowed) if allowed is not None else None

    # Public, traced entry points ------------------------------------------------
    def get_price_data(self, ticker: str, period: str = config.DEFAULT_PERIOD) -> ToolResult:
        return self._traced("get_price_data", self._get_price_data, ticker=ticker, period=period)

    def get_news(self, ticker: str, n: int = config.DEFAULT_NEWS_COUNT) -> ToolResult:
        return self._traced("get_news", self._get_news, ticker=ticker, n=n)

    def calculate_volatility(self, ticker: str, window: int = config.DEFAULT_VOL_WINDOW) -> ToolResult:
        return self._traced("calculate_volatility", self._calculate_volatility, ticker=ticker, window=window)

    def llm_sentiment(self, headlines: List[str]) -> ToolResult:
        return self._traced("llm_sentiment", self._llm_sentiment, headlines=headlines)

    def web_search(self, query: str) -> ToolResult:
        return self._traced("web_search", self._web_search, query=query)

    def _traced(self, name: str, func: Callable[..., ToolResult], **args: Any) -> ToolResult:
        return run_traced(self.ctx, name, func, args, ERROR_HINTS[name], agent=self.agent_name, allowed=self.allowed)

    # get_price_data -------------------------------------------------------------
    def _get_price_data(self, ticker: str, period: str) -> ToolResult:
        name = "get_price_data"
        symbol = validate_ticker(ticker)
        period_key = str(period).strip().lower()
        if period_key not in config.VALID_PERIODS:
            return _result(
                config.STATUS_ERROR,
                name,
                error=f"Unsupported period {period!r}",
                hint=f"Use one of: {', '.join(config.VALID_PERIODS)}",
            )
        # SMA-200 needs warm-up, so short periods are fetched over the 3-year warm-up window.
        fetch_period = period_key if period_key in config.PERIODS_LONGER_THAN_WARMUP else config.INDICATOR_WARMUP_PERIOD
        market = t1_data.load_market_data(
            symbol,
            period=fetch_period,
            data_dir=t1_config.DATA_DIR,
            use_snapshot_fallback=True,
            save_snapshot=False,  # Task 3 never rewrites Task 1's committed snapshot
        )
        if market.ohlcv.empty:
            return _result(config.STATUS_EMPTY, name, error=f"No price rows for {symbol}", hint=ERROR_HINTS[name])

        df = t1_indicators.add_indicators(market.ohlcv)
        stock = t1_summary.build_summary(market, df)
        self.ctx.frames[symbol] = df
        if stock.company_name:
            self.ctx.company_names[symbol] = stock.company_name

        price_col = _price_column(df)
        start = period_start(period_key, df.index)
        start_price = _finite(df.loc[start, price_col])
        end_price = _finite(df[price_col].iloc[-1])
        period_return = (end_price / start_price - 1) if start_price and end_price else None
        price = stock.current_price
        from_high = (price / stock.week_52_high - 1) if price and stock.week_52_high else None

        bars = df.tail(config.RECENT_BARS)
        recent = [
            [
                index.strftime("%Y-%m-%d"),
                _round(_finite(row["Open"])),
                _round(_finite(row[t1_config.HIGH_COL])),
                _round(_finite(row[t1_config.LOW_COL])),
                _round(_finite(row[t1_config.PRICE_COL_RAW])),
                int(row[t1_config.VOLUME_COL]) if _finite(row[t1_config.VOLUME_COL]) is not None else None,
            ]
            for index, row in bars.iterrows()
        ]
        payload = PriceData(
            ticker=symbol,
            company_name=stock.company_name,
            currency=stock.currency,
            period=period_key,
            as_of_date=stock.as_of_date,
            rows_fetched=len(df),
            current_price=_round(price),
            period_return_pct=_pct(period_return),
            period_start_date=start.strftime("%Y-%m-%d"),
            ytd_return_pct=_round(stock.ytd_return_pct),
            week_52_high=_round(stock.week_52_high),
            week_52_low=_round(stock.week_52_low),
            pct_from_52w_high=_pct(from_high),
            indicators={key: _round(value) for key, value in stock.latest_indicators.items()},
            momentum_label=stock.momentum.label,
            momentum_score=_round(stock.momentum.score),
            signals=derive_signals(df, price, stock.momentum.flags),
            fundamentals=fetch_fundamentals(symbol),
            recent_bars=recent,
        )
        return _result(
            config.STATUS_OK,
            name,
            data=payload,
            source=market.quality.source,
            warnings=stock.warnings[:3],
        )

    # get_news -------------------------------------------------------------------
    def _get_news(self, ticker: str, n: int) -> ToolResult:
        name = "get_news"
        symbol = validate_ticker(ticker)
        warnings: List[str] = []
        try:
            requested = int(n)
        except (TypeError, ValueError):
            requested = config.DEFAULT_NEWS_COUNT
            warnings.append(f"n={n!r} is not an integer; used {requested}")
        count = min(max(requested, config.MIN_NEWS_COUNT), config.MAX_NEWS_COUNT)
        if count != requested:
            warnings.append(f"n={requested} clamped to {count} (allowed {config.MIN_NEWS_COUNT}-{config.MAX_NEWS_COUNT})")

        result = t1_news.get_headlines(
            symbol,
            company_name=self.ctx.company_names.get(symbol),
            target=count,
            minimum=min(count, config.NEWS_MINIMUM_FOR_FALLBACK),
            data_dir=t1_config.DATA_DIR,
            use_snapshot_fallback=True,
            save_snapshot=False,
        )
        headlines = [
            NewsHeadline(
                title=item.title,
                publisher=item.publisher,
                published_at=item.published_at.strftime("%Y-%m-%d %H:%M UTC") if item.published_at else None,
                url=item.link if item.link and len(item.link) <= config.NEWS_URL_MAX_CHARS else None,
            )
            for item in result.items
        ]
        warnings.extend(result.warnings[:3])
        if not headlines:
            return _result(
                config.STATUS_EMPTY,
                name,
                error=f"No headlines found for {symbol}",
                hint=ERROR_HINTS[name],
                warnings=warnings,
            )
        payload = NewsList(
            ticker=symbol,
            count=len(headlines),
            headlines=headlines,
            sources_used=result.sources_used,
            from_snapshot=result.from_snapshot,
            snapshot_fetched_at=result.snapshot_fetched_at,
        )
        source = t1_news.SOURCE_SNAPSHOT if result.from_snapshot else "live"
        return _result(config.STATUS_OK, name, data=payload, source=source, warnings=warnings)

    # calculate_volatility -------------------------------------------------------
    def _calculate_volatility(self, ticker: str, window: int) -> ToolResult:
        name = "calculate_volatility"
        symbol = validate_ticker(ticker)
        try:
            window = vol.validate_window(int(window) if isinstance(window, str) and window.isdigit() else window)
        except ValueError as exc:
            return _result(
                config.STATUS_ERROR,
                name,
                error=str(exc),
                hint=f"Use an integer window between {config.MIN_VOL_WINDOW} and {config.MAX_VOL_WINDOW}, e.g. 20 or 60.",
            )

        # Prices already loaded by get_price_data are reused, so this does not refetch.
        df = self.ctx.frames.get(symbol)
        source = "session"
        if df is None:
            market = t1_data.load_market_data(
                symbol,
                period=config.VOL_FETCH_PERIOD,
                data_dir=t1_config.DATA_DIR,
                use_snapshot_fallback=True,
                save_snapshot=False,
            )
            df = market.ohlcv
            source = market.quality.source
            self.ctx.frames[symbol] = df

        prices = df[_price_column(df)]
        returns = vol.log_returns(prices)
        if len(returns) < window:
            return _result(
                config.STATUS_EMPTY,
                name,
                error=f"Only {len(returns)} daily returns available for {symbol}, window is {window}",
                hint=f"Use a window of at most {len(returns)} trading days.",
                source=source,
            )
        annual = vol.annualised_vol(returns, window)
        rolling = vol.rolling_annualised_vol(returns, window)
        percentile = vol.vol_percentile(rolling, annual)
        spot = _finite(df[t1_config.PRICE_COL_RAW].iloc[-1]) or _finite(prices.iloc[-1])
        days = vol.horizon_trading_days()
        move = vol.expected_move(annual, days)
        low_1, high_1 = vol.price_band(spot, move, 1)
        low_2, _ = vol.price_band(spot, move, 2)
        comparison = {f"{w}d": _pct(vol.annualised_vol(returns, w)) for w in config.VOL_COMPARISON_WINDOWS}
        payload = VolatilityResult(
            ticker=symbol,
            window=window,
            as_of_date=df.index[-1].strftime("%Y-%m-%d"),
            annualised_vol_pct=_pct(annual),
            daily_vol_pct=_pct(annual / math.sqrt(config.TRADING_DAYS_PER_YEAR)),
            comparison_pct=comparison,
            percentile_1y=_round(percentile, 1),
            regime=vol.vol_regime(percentile),
            spot=_round(spot),
            horizon_calendar_days=config.HEDGE_HORIZON_CALENDAR_DAYS,
            horizon_trading_days=days,
            expected_move_pct=_pct(move),
            band_1sigma_low=_round(low_1),
            band_1sigma_high=_round(high_1),
            band_2sigma_low=_round(low_2),
            max_drawdown_1y_pct=_pct(vol.max_drawdown(prices)),
        )
        return _result(config.STATUS_OK, name, data=payload, source=source)

    # llm_sentiment --------------------------------------------------------------
    def _llm_sentiment(self, headlines: List[str]) -> ToolResult:
        name = "llm_sentiment"
        if isinstance(headlines, str):
            headlines = [headlines]
        seen = set()
        clean: List[str] = []
        for text in headlines or []:
            title = str(text).strip()
            if title and title.lower() not in seen:
                seen.add(title.lower())
                clean.append(title)
        if not clean:
            return _result(
                config.STATUS_EMPTY,
                name,
                error="No headlines to score",
                hint="Call get_news first and pass its headline titles as a list of strings.",
            )
        warnings: List[str] = []
        # Only headlines a news or search tool returned this session are scored. In a live 3B run
        # Agent A (no news access) invented headlines and scored them; this check stops that.
        # AI-ASSISTED: Claude Code (claude-opus-5-5), Prompt: 'Implement the Task 3B plan', Date: 2026-10-09 (see CITATIONS.md Entry 16)
        verified = verify_headlines(clean, self.ctx.known_headlines())
        if len(verified) < len(clean):
            warnings.append(f"Dropped {len(clean) - len(verified)} headline(s) not returned by get_news or web_search this session")
        if not verified:
            return _result(
                config.STATUS_ERROR,
                name,
                error="None of the headlines came from a get_news or web_search result in this session",
                hint=(
                    "Pass headline titles exactly as get_news or web_search returned them. Without news access, "
                    "wait for headlines from the agent that has it."
                ),
                warnings=warnings,
            )
        clean = verified
        if len(clean) > config.MAX_SENTIMENT_HEADLINES:
            warnings.append(f"Scored the first {config.MAX_SENTIMENT_HEADLINES} of {len(clean)} headlines")
            clean = clean[: config.MAX_SENTIMENT_HEADLINES]

        if self.ctx.llm_clients is None:
            load_api_keys()
            self.ctx.llm_clients = build_sentiment_clients()
        if not self.ctx.llm_clients:
            return _result(
                config.STATUS_ERROR,
                name,
                error="No LLM provider is configured (GROQ_API_KEY / OPENROUTER_API_KEY)",
                hint=ERROR_HINTS[name],
            )

        news = t1_news.NewsResult(
            ticker=self.ctx.ticker, items=[t1_news.NewsItem(title=title, source="agent") for title in clean]
        )
        scored = t1_llm.score_headlines(
            news,
            self.ctx.ticker,
            company_name=self.ctx.company_names.get(self.ctx.ticker),
            clients=self.ctx.llm_clients,
        )
        warnings.extend(scored.warnings[:3])
        if scored.n_scored == 0:
            return _result(
                config.STATUS_ERROR,
                name,
                error=scored.unavailable_reason or "No headline could be scored",
                hint=ERROR_HINTS[name],
                warnings=warnings,
            )

        def top(label: str) -> List[ScoredHeadline]:
            chosen = sorted((i for i in scored.items if i.sentiment == label), key=lambda i: -i.confidence)
            return [
                ScoredHeadline(headline=i.headline, sentiment=i.sentiment, confidence=i.confidence, reason=i.brief_reason)
                for i in chosen[: config.SENTIMENT_TOP_N]
            ]

        counts = {label: sum(1 for i in scored.items if i.sentiment == label) for label in t1_config.SENTIMENT_VOTE}
        payload = SentimentScore(
            score=scored.score,
            label=scored.label,
            n_scored=scored.n_scored,
            n_unscored=len(scored.unscored),
            positive=counts[t1_config.SENTIMENT_POSITIVE],
            negative=counts[t1_config.SENTIMENT_NEGATIVE],
            neutral=counts[t1_config.SENTIMENT_NEUTRAL],
            top_positive=top(t1_config.SENTIMENT_POSITIVE),
            top_negative=top(t1_config.SENTIMENT_NEGATIVE),
        )
        providers = sorted({i.provider for i in scored.items if i.provider})
        return _result(config.STATUS_OK, name, data=payload, source=",".join(providers) or None, warnings=warnings)

    # web_search -----------------------------------------------------------------
    def _web_search(self, query: str) -> ToolResult:
        name = "web_search"
        text = str(query or "").strip()
        if not text:
            return _result(config.STATUS_ERROR, name, error="Empty query", hint="Pass a search query string.")
        from ddgs.exceptions import DDGSException, RatelimitException, TimeoutException

        simplified = simplify_query(text)
        # Fallback chain: web results -> news results -> a stripped-down query without a time limit.
        plan: List[Tuple[str, str, str, Optional[str]]] = [
            ("text", "text", text, config.SEARCH_TIMELIMIT),
            ("news", "news", text, config.SEARCH_TIMELIMIT),
        ]
        if simplified and simplified.lower() != text.lower():
            plan.append(("text-simplified", "text", simplified, None))

        failures: List[str] = []
        transient = False
        for label, kind, query_text, timelimit in plan:
            for attempt in range(1, config.SEARCH_ATTEMPTS_PER_BACKEND + 1):
                try:
                    raw = _ddgs_search(kind, query_text, timelimit)
                except (RatelimitException, TimeoutException) as exc:
                    transient = True
                    failures.append(f"{label} attempt {attempt}: {safe_error(exc)}")
                    if attempt < config.SEARCH_ATTEMPTS_PER_BACKEND:
                        time.sleep(config.SEARCH_RETRY_WAIT_SECONDS * attempt)
                    continue
                except DDGSException as exc:
                    # "No results found" and similar: retrying the same backend will not help.
                    failures.append(f"{label}: {exc}")
                    break
                hits = _to_hits(kind, raw or [])
                if hits:
                    payload = SearchResults(query=text, effective_query=query_text, backend=label, hits=hits)
                    return _result(config.STATUS_OK, name, data=payload, source="ddgs", warnings=failures[-3:])
                failures.append(f"{label}: no usable results")
                break

        status = config.STATUS_ERROR if transient else config.STATUS_EMPTY
        return _result(
            status,
            name,
            error="; ".join(failures[-3:]) or "No results",
            hint=ERROR_HINTS[name],
        )


# --------------------------------------------------------------------------- LangChain wrappers
class PriceArgs(BaseModel):
    ticker: str = Field(description="Ticker symbol, e.g. AAPL")
    period: str = Field(
        default=config.DEFAULT_PERIOD,
        description=f"History window for the period return: one of {', '.join(config.VALID_PERIODS)}",
    )


class NewsArgs(BaseModel):
    ticker: str = Field(description="Ticker symbol, e.g. AAPL")
    n: int = Field(default=config.DEFAULT_NEWS_COUNT, description=f"Number of headlines, {config.MIN_NEWS_COUNT}-{config.MAX_NEWS_COUNT}")


class VolatilityArgs(BaseModel):
    ticker: str = Field(description="Ticker symbol, e.g. AAPL")
    window: int = Field(
        default=config.DEFAULT_VOL_WINDOW,
        description=f"Look-back in trading days, {config.MIN_VOL_WINDOW}-{config.MAX_VOL_WINDOW} (20 = 1 month, 60 = 1 quarter)",
    )


class SentimentArgs(BaseModel):
    headlines: List[str] = Field(description=f"Headline titles to score (at most {config.MAX_SENTIMENT_HEADLINES} are used)")


class SearchArgs(BaseModel):
    query: str = Field(description="Web search query, e.g. 'Apple stock analyst downgrade price target'")


TOOL_DESCRIPTIONS = {
    "get_price_data": (
        "Daily OHLCV prices from yfinance with computed indicators (SMA-50/200, RSI-14, MACD 12/26/9, "
        "Bollinger 20/2), 52-week range, period and YTD return, momentum label, trend signals, "
        "fundamentals (P/E, margins, growth, debt/equity, free cash flow, beta) and the last 10 bars."
    ),
    "get_news": "Recent news headlines for a ticker as a structured list (title, publisher, time, url).",
    "calculate_volatility": (
        "Annualised historical volatility (std of daily log returns x sqrt(252)) over a window, its 1-year "
        "percentile and regime, comparison windows, max drawdown, and the 90-day 1-sigma expected move "
        "and price bands for sizing a hedge."
    ),
    "llm_sentiment": (
        "Scores headlines with an LLM (positive/negative/neutral + confidence) and returns a "
        "confidence-weighted sentiment score in [-1, 1] with the strongest positive and negative headlines."
    ),
    "web_search": "DuckDuckGo web search for analyst commentary, price targets, upgrades/downgrades and recent events.",
}

_ARG_SCHEMAS = {
    "get_price_data": PriceArgs,
    "get_news": NewsArgs,
    "calculate_volatility": VolatilityArgs,
    "llm_sentiment": SentimentArgs,
    "web_search": SearchArgs,
}


def build_tools(ctx_or_tools: Any, names: Optional[Sequence[str]] = None) -> List[StructuredTool]:
    """LangChain tools for a session. `names` restricts the set (used for per-agent access in 3B).

    Each tool returns (compact JSON for the model, typed ToolResult as the message artifact).
    """
    toolkit = ctx_or_tools if isinstance(ctx_or_tools, ResearchTools) else ResearchTools(ctx_or_tools)

    def get_price_data(ticker: str, period: str = config.DEFAULT_PERIOD) -> Tuple[str, ToolResult]:
        result = toolkit.get_price_data(ticker, period)
        return result.to_llm_json(), result

    def get_news(ticker: str, n: int = config.DEFAULT_NEWS_COUNT) -> Tuple[str, ToolResult]:
        result = toolkit.get_news(ticker, n)
        return result.to_llm_json(), result

    def calculate_volatility(ticker: str, window: int = config.DEFAULT_VOL_WINDOW) -> Tuple[str, ToolResult]:
        result = toolkit.calculate_volatility(ticker, window)
        return result.to_llm_json(), result

    def llm_sentiment(headlines: List[str]) -> Tuple[str, ToolResult]:
        result = toolkit.llm_sentiment(headlines)
        return result.to_llm_json(), result

    def web_search(query: str) -> Tuple[str, ToolResult]:
        result = toolkit.web_search(query)
        return result.to_llm_json(), result

    functions = {
        "get_price_data": get_price_data,
        "get_news": get_news,
        "calculate_volatility": calculate_volatility,
        "llm_sentiment": llm_sentiment,
        "web_search": web_search,
    }
    selected = list(names) if names is not None else list(functions)
    unknown = [name for name in selected if name not in functions]
    if unknown:
        raise ValueError(f"Unknown tool names: {unknown}")
    return [
        StructuredTool.from_function(
            func=functions[name],
            name=name,
            description=TOOL_DESCRIPTIONS[name],
            args_schema=_ARG_SCHEMAS[name],
            response_format="content_and_artifact",
        )
        for name in selected
    ]
