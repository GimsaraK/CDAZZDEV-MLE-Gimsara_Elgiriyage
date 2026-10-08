"""End-to-end Task 1A pipeline: market data -> indicators -> news -> summary.

``run_pipeline`` never raises: each stage is isolated, failures are logged and collected,
and partial results are returned with a status of ``ok``, ``degraded`` or ``failed``.
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
from .logging_utils import get_logger
from .news import NewsResult, get_headlines
from .summary import StockSummary, build_summary

logger = get_logger("pipeline")

STATUS_OK = "ok"
STATUS_DEGRADED = "degraded"
STATUS_FAILED = "failed"


@dataclass
class PipelineResult:
    ticker: str
    status: str
    market: Optional[MarketData] = None
    prices: Optional[pd.DataFrame] = None  # OHLCV + indicator columns
    news: Optional[NewsResult] = None
    summary: Optional[StockSummary] = None
    errors: List[str] = field(default_factory=list)

    @property
    def summary_dict(self) -> Optional[Dict[str, Any]]:
        return self.summary.model_dump(mode="json") if self.summary else None


def _with_empty_indicators(df: pd.DataFrame) -> pd.DataFrame:
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
) -> PipelineResult:
    errors: List[str] = []
    logger.info("=== Task 1A pipeline start: %s ===", ticker)

    try:
        market = load_market_data(
            ticker, period, data_dir=data_dir, use_snapshot_fallback=use_snapshot_fallback, save_snapshot=save_snapshot
        )
    except Exception as exc:  # noqa: BLE001 - nothing downstream can run without prices
        logger.error("Pipeline failed for %r: %s", ticker, exc)
        return PipelineResult(ticker=str(ticker), status=STATUS_FAILED, errors=[str(exc)])

    try:
        prices = ind.add_indicators(market.ohlcv)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"Indicator computation failed: {exc}")
        logger.error(errors[-1])
        prices = _with_empty_indicators(market.ohlcv)

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

    summary: Optional[StockSummary] = None
    try:
        summary = build_summary(market, prices, news)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"Summary construction failed: {exc}")
        logger.error(errors[-1])

    degraded = (
        bool(errors)
        or summary is None
        or market.quality.source != SOURCE_LIVE
        or news.count < config.NEWS_MIN_COUNT
    )
    status = STATUS_DEGRADED if degraded else STATUS_OK
    logger.info("=== Task 1A pipeline finished: %s (status=%s) ===", market.ticker, status)
    return PipelineResult(
        ticker=market.ticker,
        status=status,
        market=market,
        prices=prices,
        news=news,
        summary=summary,
        errors=errors,
    )
