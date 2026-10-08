"""End-to-end Task 1 pipeline: market data -> indicators -> news -> summary -> optional LLM.

``run_pipeline`` never raises: each stage is isolated, failures are logged and collected,
and partial results are returned with a status of ``ok``, ``degraded`` or ``failed``.
The LLM stage (sentiment and Buy/Hold/Sell) runs only when ``run_llm=True``, so the
Part 1A path stays offline.
"""
# AI-ASSISTED: Cursor Agent (claude-sonnet-5.5), Prompt: 'yes implement the plan @TASK1A_PLAN.md', Date: 2026-10-08 (see CITATIONS.md Entry 4)

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from . import config
from . import indicators as ind
from .data import SOURCE_LIVE, MarketData, load_market_data
from .llm import SentimentResult, SignalRecommendation, recommend_signal, score_headlines
from .logging_utils import get_logger
from .news import NewsResult, get_headlines
from .summary import StockSummary, build_summary

logger = get_logger("pipeline")

# ok       = everything live and complete
# degraded = a summary was produced, but something fell back (snapshot data, too few headlines, a failed stage)
# failed   = no price data at all, so nothing could be computed
STATUS_OK = "ok"
STATUS_DEGRADED = "degraded"
STATUS_FAILED = "failed"


@dataclass
class PipelineResult:
    """Everything one pipeline run produced. Fields stay None when their stage could not run."""

    ticker: str
    status: str
    market: Optional[MarketData] = None
    prices: Optional[pd.DataFrame] = None  # OHLCV + indicator columns
    news: Optional[NewsResult] = None
    summary: Optional[StockSummary] = None
    sentiment: Optional[SentimentResult] = None  # set only when run_llm=True
    signal: Optional[SignalRecommendation] = None
    errors: List[str] = field(default_factory=list)

    @property
    def summary_dict(self) -> Optional[Dict[str, Any]]:
        """The summary as a JSON-ready dict (dates as strings), or None if no summary was built."""
        return self.summary.model_dump(mode="json") if self.summary else None


def _with_empty_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add all indicator columns filled with NaN, so later stages still find the columns they expect."""
    out = df.copy()
    for column in ind.INDICATOR_COLUMNS:
        out[column] = np.nan
    return out


def run_pipeline(
    ticker: str = config.TICKER,
    period: str = config.HISTORY_PERIOD,
    data_dir: Path = config.DATA_DIR,
    use_snapshot_fallback: bool = True,
    save_snapshot: bool = True,
    run_llm: bool = False,
) -> PipelineResult:
    """Run the pipeline for one ticker and return the results with a status. Never raises.

    Set use_snapshot_fallback=False to disable the offline fallback, and save_snapshot=False to
    avoid overwriting the committed snapshot (used by the robustness demos).
    run_llm=False (the default) stops after the summary, so Part 1A does not call an API.
    """
    errors: List[str] = []  # stage failures that did not stop the run
    logger.info("=== Task 1 pipeline start: %s (run_llm=%s) ===", ticker, run_llm)

    # Stage 1 - market data. This is the only fatal stage: without prices nothing else can run.
    try:
        market = load_market_data(
            ticker, period, data_dir=data_dir, use_snapshot_fallback=use_snapshot_fallback, save_snapshot=save_snapshot
        )
    except Exception as exc:  # noqa: BLE001 - nothing downstream can run without prices
        logger.error("Pipeline failed for %r: %s", ticker, exc)
        return PipelineResult(ticker=str(ticker), status=STATUS_FAILED, errors=[str(exc)])

    # Stage 2 - indicators. On failure continue with empty indicator columns
    # (the summary still reports price, 52-week range, P/E and YTD).
    try:
        prices = ind.add_indicators(market.ohlcv)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"Indicator computation failed: {exc}")
        logger.error(errors[-1])
        prices = _with_empty_indicators(market.ohlcv)

    # Stage 3 - news. The company name improves the Google News search.
    try:
        news = get_headlines(
            market.ticker,
            company_name=market.info.get("longName") or market.info.get("shortName"),
            data_dir=data_dir,
            use_snapshot_fallback=use_snapshot_fallback,
            save_snapshot=save_snapshot,
        )
    except Exception as exc:  # noqa: BLE001 - defensive: get_headlines is designed not to raise
        errors.append(f"News retrieval failed: {exc}")
        logger.error(errors[-1])
        news = NewsResult(ticker=market.ticker, warnings=[errors[-1]])

    # Stage 4 - summary dictionary.
    summary: Optional[StockSummary] = None
    try:
        summary = build_summary(market, prices, news)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"Summary construction failed: {exc}")
        logger.error(errors[-1])

    # Stage 5 - LLM. Optional, and never fatal: a failed call degrades the run instead of aborting it.
    # Only missing prices (stage 1) mark the whole pipeline as failed.
    sentiment: Optional[SentimentResult] = None
    signal: Optional[SignalRecommendation] = None
    if run_llm and summary is not None:
        sentiment, signal, llm_errors = _run_llm_stage(summary, news)
        errors.extend(llm_errors)

    # Degraded if any stage failed, the data is not live, the headline minimum was missed,
    # or the optional LLM stage could not produce a real score and signal.
    llm_degraded = run_llm and (
        sentiment is None
        or sentiment.score is None
        or signal is None
        or signal.fallback
    )
    degraded = (
        bool(errors)
        or summary is None
        or market.quality.source != SOURCE_LIVE
        or news.count < config.NEWS_MIN_COUNT
        or llm_degraded
    )
    status = STATUS_DEGRADED if degraded else STATUS_OK
    logger.info("=== Task 1 pipeline finished: %s (status=%s) ===", market.ticker, status)
    return PipelineResult(
        ticker=market.ticker,
        status=status,
        market=market,
        prices=prices,
        news=news,
        summary=summary,
        sentiment=sentiment,
        signal=signal,
        errors=errors,
    )


def _run_llm_stage(
    summary: StockSummary,
    news: NewsResult,
) -> tuple:
    """Score headlines and ask for a signal. Returns (sentiment, signal, error strings)."""
    errors: List[str] = []
    try:
        sentiment = score_headlines(
            news,
            summary.ticker,
            company_name=summary.company_name,
        )
    except Exception as exc:  # noqa: BLE001 - defensive: score_headlines is designed not to raise
        errors.append(f"Sentiment scoring failed: {exc}")
        logger.error(errors[-1])
        return None, None, errors
    if sentiment.score is None:
        errors.append(sentiment.unavailable_reason or "Sentiment score unavailable")
    try:
        signal = recommend_signal(summary, sentiment)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"Signal recommendation failed: {exc}")
        logger.error(errors[-1])
        return sentiment, None, errors
    if signal.fallback:
        errors.append(signal.unavailable_reason or "Signal fell back to Hold")
    return sentiment, signal, errors
