"""Momentum signal and the clean summary dictionary (Task 1A output, Task 1B input).

Price conventions
-----------------
- current price and 52-week high/low use RAW prices, so they match quoted market data.
- YTD return and all indicators use ADJUSTED close, so splits/dividends do not distort them.
"""
# AI-ASSISTED: Cursor Agent (claude-sonnet-5.5), Prompt: 'yes implement the plan @TASK1A_PLAN.md', Date: 2026-10-08 (see CITATIONS.md Entry 4)

import math
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from pydantic import BaseModel, Field

from . import config
from . import indicators as ind
from .data import DataQualityReport, MarketData
from .logging_utils import get_logger
from .news import NewsResult

logger = get_logger("summary")

BULLISH, BEARISH, NO_VOTE = 1, -1, 0


# --------------------------------------------------------------------------- models
class MomentumComponent(BaseModel):
    name: str
    value: Optional[float] = None
    vote: Optional[int] = None  # +1 bullish, -1 bearish, 0 neutral/stretched, None unavailable
    rationale: str


class MomentumSignal(BaseModel):
    score: Optional[float] = None  # mean of available votes, in [-1, 1]
    label: str
    components: List[MomentumComponent] = Field(default_factory=list)
    flags: List[str] = Field(default_factory=list)
    unavailable: List[str] = Field(default_factory=list)


class StockSummary(BaseModel):
    # --- required by the spec --------------------------------------------
    ticker: str
    current_price: Optional[float] = None
    week_52_high: Optional[float] = None
    week_52_low: Optional[float] = None
    pe_ratio: Optional[float] = None
    ytd_return: Optional[float] = None  # fraction, e.g. 0.12 = +12%
    momentum: MomentumSignal
    # --- context and provenance ------------------------------------------
    company_name: Optional[str] = None
    currency: Optional[str] = None
    as_of_date: Optional[str] = None
    week_52_high_date: Optional[str] = None
    week_52_low_date: Optional[str] = None
    pe_source: Optional[str] = None
    pe_unavailable_reason: Optional[str] = None
    forward_pe: Optional[float] = None
    ytd_return_pct: Optional[float] = None
    ytd_base_date: Optional[str] = None
    ytd_base_price: Optional[float] = None
    ytd_unavailable_reason: Optional[str] = None
    latest_indicators: Dict[str, Optional[float]] = Field(default_factory=dict)
    news_count: int = 0
    news_sources: List[str] = Field(default_factory=list)
    data_quality: Optional[DataQualityReport] = None
    warnings: List[str] = Field(default_factory=list)

    def core_dict(self) -> Dict[str, Any]:
        """Only the fields the spec asks for, as a plain JSON-ready dict."""
        return {
            "ticker": self.ticker,
            "current_price": self.current_price,
            "week_52_high": self.week_52_high,
            "week_52_low": self.week_52_low,
            "pe_ratio": self.pe_ratio,
            "ytd_return": self.ytd_return,
            "momentum_signal": self.momentum.label,
            "momentum_score": self.momentum.score,
        }


# --------------------------------------------------------------------------- helpers
def _finite(value: Any) -> Optional[float]:
    """Float if ``value`` is a finite number, else None (handles None, NaN, 'Infinity', strings)."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _round(value: Optional[float], decimals: int) -> Optional[float]:
    return None if value is None else round(value, decimals)


def _sign_vote(value: Optional[float]) -> Optional[int]:
    if value is None:
        return None
    return BULLISH if value > 0 else BEARISH if value < 0 else NO_VOTE


def _last_value(series: pd.Series) -> Optional[float]:
    return _finite(series.iloc[-1]) if len(series) else None


# --------------------------------------------------------------------------- momentum
def _relative_gap(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
    if numerator is None or denominator in (None, 0):
        return None
    return numerator / denominator - 1.0


def _crossover_flag(df: pd.DataFrame) -> Optional[str]:
    """Report a golden/death cross of SMA-50 vs SMA-200 within the lookback window."""
    spread = (df[ind.COL_SMA_SHORT] - df[ind.COL_SMA_LONG]).dropna()
    window = spread.iloc[-(config.CROSSOVER_LOOKBACK_DAYS + 1) :]
    signs = window.apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0)
    changes = signs[signs.diff().fillna(0) != 0]
    if changes.empty:
        return None
    last_change_date = changes.index[-1]
    sessions_ago = len(window) - 1 - window.index.get_loc(last_change_date)
    kind = "Golden cross" if signs.iloc[-1] > 0 else "Death cross"
    return f"{kind} (SMA-{config.SMA_SHORT_WINDOW} vs SMA-{config.SMA_LONG_WINDOW}) {sessions_ago} sessions ago"


def momentum_label(score: Optional[float]) -> str:
    if score is None:
        return config.MOMENTUM_LABEL_UNAVAILABLE
    if score >= config.MOMENTUM_STRONG_THRESHOLD:
        return config.MOMENTUM_LABEL_STRONG_BULLISH
    if score >= config.MOMENTUM_MILD_THRESHOLD:
        return config.MOMENTUM_LABEL_BULLISH
    if score <= -config.MOMENTUM_STRONG_THRESHOLD:
        return config.MOMENTUM_LABEL_STRONG_BEARISH
    if score <= -config.MOMENTUM_MILD_THRESHOLD:
        return config.MOMENTUM_LABEL_BEARISH
    return config.MOMENTUM_LABEL_NEUTRAL


def compute_momentum(df: pd.DataFrame, price_col: str = config.PRICE_COL_ADJ) -> MomentumSignal:
    """Composite momentum: each rule votes +1 / -1 / 0; the score is the mean of available votes."""
    if df is None or df.empty:
        return MomentumSignal(label=config.MOMENTUM_LABEL_UNAVAILABLE, unavailable=["all (no data)"])

    price = _last_value(df[price_col])
    sma_short = _last_value(df[ind.COL_SMA_SHORT])
    sma_long = _last_value(df[ind.COL_SMA_LONG])
    macd_value = _last_value(df[ind.COL_MACD])
    macd_signal = _last_value(df[ind.COL_MACD_SIGNAL])
    hist = df[ind.COL_MACD_HIST].dropna()
    rsi_value = _last_value(df[ind.COL_RSI])
    pct_b = _last_value(df[ind.COL_BB_PCT_B])

    components: List[MomentumComponent] = []
    flags: List[str] = []

    gap_short = _relative_gap(price, sma_short)
    components.append(
        MomentumComponent(
            name=f"price_vs_sma_{config.SMA_SHORT_WINDOW}",
            value=_round(gap_short, config.RATIO_DECIMALS),
            vote=_sign_vote(gap_short),
            rationale=f"Price relative to the {config.SMA_SHORT_WINDOW}-day SMA (short-term trend)",
        )
    )

    gap_long = _relative_gap(price, sma_long)
    components.append(
        MomentumComponent(
            name=f"price_vs_sma_{config.SMA_LONG_WINDOW}",
            value=_round(gap_long, config.RATIO_DECIMALS),
            vote=_sign_vote(gap_long),
            rationale=f"Price relative to the {config.SMA_LONG_WINDOW}-day SMA (long-term trend)",
        )
    )

    regime = _relative_gap(sma_short, sma_long)
    components.append(
        MomentumComponent(
            name="trend_regime",
            value=_round(regime, config.RATIO_DECIMALS),
            vote=_sign_vote(regime),
            rationale=f"SMA-{config.SMA_SHORT_WINDOW} above SMA-{config.SMA_LONG_WINDOW} = golden-cross regime",
        )
    )
    cross = _crossover_flag(df) if regime is not None else None
    if cross:
        flags.append(cross)

    macd_gap = None if macd_value is None or macd_signal is None else macd_value - macd_signal
    components.append(
        MomentumComponent(
            name="macd_vs_signal",
            value=_round(macd_gap, config.INDICATOR_DECIMALS),
            vote=_sign_vote(macd_gap),
            rationale="MACD line above its signal line = positive momentum",
        )
    )

    hist_change = None
    if len(hist) > config.MACD_HIST_LOOKBACK:
        hist_change = _finite(hist.iloc[-1] - hist.iloc[-1 - config.MACD_HIST_LOOKBACK])
    components.append(
        MomentumComponent(
            name="macd_histogram_direction",
            value=_round(hist_change, config.INDICATOR_DECIMALS),
            vote=_sign_vote(hist_change),
            rationale=f"MACD histogram change over {config.MACD_HIST_LOOKBACK} sessions (accelerating vs fading)",
        )
    )

    rsi_vote: Optional[int] = None
    if rsi_value is not None:
        if rsi_value > config.RSI_OVERBOUGHT:
            rsi_vote = NO_VOTE
            flags.append(f"RSI overbought ({rsi_value:.1f} > {config.RSI_OVERBOUGHT})")
        elif rsi_value < config.RSI_OVERSOLD:
            rsi_vote = NO_VOTE
            flags.append(f"RSI oversold ({rsi_value:.1f} < {config.RSI_OVERSOLD})")
        else:
            rsi_vote = BULLISH if rsi_value >= config.RSI_MIDLINE else BEARISH
    components.append(
        MomentumComponent(
            name="rsi_zone",
            value=_round(rsi_value, config.INDICATOR_DECIMALS),
            vote=rsi_vote,
            rationale=(
                f"RSI {config.RSI_MIDLINE}-{config.RSI_OVERBOUGHT} bullish, {config.RSI_OVERSOLD}-{config.RSI_MIDLINE} "
                "bearish; beyond the bands = stretched (no vote, flagged)"
            ),
        )
    )

    bb_vote: Optional[int] = None
    if pct_b is not None:
        if pct_b > config.BB_PCT_B_UPPER:
            bb_vote = NO_VOTE
            flags.append(f"Price above upper Bollinger Band (%B = {pct_b:.2f})")
        elif pct_b < config.BB_PCT_B_LOWER:
            bb_vote = NO_VOTE
            flags.append(f"Price below lower Bollinger Band (%B = {pct_b:.2f})")
        else:
            bb_vote = BULLISH if pct_b >= config.BB_PCT_B_MID else BEARISH
    components.append(
        MomentumComponent(
            name="bollinger_pct_b",
            value=_round(pct_b, config.INDICATOR_DECIMALS),
            vote=bb_vote,
            rationale="%B in the upper half of the bands bullish, lower half bearish; outside = stretched",
        )
    )

    votes = [c.vote for c in components if c.vote is not None]
    unavailable = [c.name for c in components if c.vote is None]
    score = round(sum(votes) / len(votes), config.RATIO_DECIMALS) if votes else None
    if unavailable:
        logger.warning("Momentum components unavailable (insufficient data): %s", unavailable)
    return MomentumSignal(
        score=score,
        label=momentum_label(score),
        components=components,
        flags=flags,
        unavailable=unavailable,
    )


# --------------------------------------------------------------------------- summary fields
def compute_week_52_range(
    df: pd.DataFrame,
) -> Tuple[Optional[float], Optional[str], Optional[float], Optional[str]]:
    """Raw High/Low over a calendar 52-week window ending at the last bar."""
    if df is None or df.empty:
        return None, None, None, None
    window_start = df.index[-1] - pd.DateOffset(weeks=config.WEEKS_IN_52W_WINDOW)
    window = df[df.index > window_start]
    highs = window[config.HIGH_COL].dropna()
    lows = window[config.LOW_COL].dropna()
    high = _finite(highs.max()) if not highs.empty else None
    low = _finite(lows.min()) if not lows.empty else None
    high_date = highs.idxmax().date().isoformat() if high is not None else None
    low_date = lows.idxmin().date().isoformat() if low is not None else None
    return high, high_date, low, low_date


def check_week_52_against_source(
    computed_high: Optional[float],
    computed_low: Optional[float],
    info: Dict[str, Any],
) -> List[str]:
    warnings = []
    for label, ours, key in (("high", computed_high, "fiftyTwoWeekHigh"), ("low", computed_low, "fiftyTwoWeekLow")):
        theirs = _finite(info.get(key))
        if ours is None or theirs is None or theirs == 0:
            continue
        gap_pct = abs(ours / theirs - 1.0) * 100
        if gap_pct > config.WEEK52_MISMATCH_TOLERANCE_PCT:
            warnings.append(
                f"Computed 52-week {label} {ours:.2f} differs from Yahoo's {theirs:.2f} by {gap_pct:.1f}%"
            )
    return warnings


def compute_ytd_return(
    df: pd.DataFrame,
) -> Tuple[Optional[float], Optional[str], Optional[float], Optional[str]]:
    """YTD return on adjusted close vs the last close of the previous calendar year.

    Returns (ytd_return, base_date, base_price, unavailable_reason).
    """
    if df is None or df.empty:
        return None, None, None, "No price data"
    adjusted = df[config.PRICE_COL_ADJ].dropna()
    if adjusted.empty:
        return None, None, None, "No adjusted close data"
    current_year = adjusted.index[-1].year
    prior = adjusted[adjusted.index.year < current_year]
    if prior.empty:
        return None, None, None, "No trading data before the start of the current year"
    base_price = _finite(prior.iloc[-1])
    if not base_price:
        return None, None, None, "Invalid base price"
    ytd = adjusted.iloc[-1] / base_price - 1.0
    return float(ytd), prior.index[-1].date().isoformat(), base_price, None


def compute_pe(
    info: Dict[str, Any],
    current_price: Optional[float],
) -> Tuple[Optional[float], Optional[str], Optional[str], Optional[float]]:
    """P/E fallback chain: trailingPE -> price / trailingEps -> None with a reason.

    Returns (pe_ratio, pe_source, unavailable_reason, forward_pe).
    """
    forward_pe = _finite(info.get("forwardPE"))
    forward_pe = forward_pe if forward_pe is not None and forward_pe > 0 else None

    trailing_pe = _finite(info.get("trailingPE"))
    if trailing_pe is not None and trailing_pe > 0:
        return trailing_pe, "trailingPE", None, forward_pe

    eps = _finite(info.get("trailingEps"))
    if eps is not None and eps <= 0:
        return None, None, "Trailing EPS is zero or negative, so P/E is not meaningful", forward_pe
    if eps is not None and current_price:
        return current_price / eps, "computed (price / trailingEps)", None, forward_pe
    return None, None, "P/E and trailing EPS unavailable from the data source", forward_pe


def latest_indicator_values(df: pd.DataFrame) -> Dict[str, Optional[float]]:
    if df is None or df.empty:
        return {column: None for column in ind.INDICATOR_COLUMNS}
    last = df.iloc[-1]
    return {
        column: _round(_finite(last.get(column)), config.INDICATOR_DECIMALS) for column in ind.INDICATOR_COLUMNS
    }


# --------------------------------------------------------------------------- build
def build_summary(
    market: MarketData,
    df_with_indicators: pd.DataFrame,
    news: Optional[NewsResult] = None,
) -> StockSummary:
    """Assemble the validated summary. Missing inputs become None fields with reasons, never errors."""
    df = df_with_indicators
    info = market.info or {}
    warnings: List[str] = list(market.quality.warnings)

    current_price = _last_value(df[config.PRICE_COL_RAW]) if not df.empty else None
    high, high_date, low, low_date = compute_week_52_range(df)
    warnings += check_week_52_against_source(high, low, info)

    pe_ratio, pe_source, pe_reason, forward_pe = compute_pe(info, current_price)
    ytd, ytd_base_date, ytd_base_price, ytd_reason = compute_ytd_return(df)

    if news is not None:
        warnings += news.warnings

    for warning in warnings[len(market.quality.warnings) :]:
        logger.warning("Summary: %s", warning)

    return StockSummary(
        ticker=market.ticker,
        current_price=_round(current_price, config.PRICE_DECIMALS),
        week_52_high=_round(high, config.PRICE_DECIMALS),
        week_52_low=_round(low, config.PRICE_DECIMALS),
        pe_ratio=_round(pe_ratio, config.PRICE_DECIMALS),
        ytd_return=_round(ytd, config.RATIO_DECIMALS),
        momentum=compute_momentum(df),
        company_name=info.get("longName") or info.get("shortName"),
        currency=info.get("currency"),
        as_of_date=df.index[-1].date().isoformat() if not df.empty else None,
        week_52_high_date=high_date,
        week_52_low_date=low_date,
        pe_source=pe_source,
        pe_unavailable_reason=pe_reason,
        forward_pe=_round(forward_pe, config.PRICE_DECIMALS),
        ytd_return_pct=_round(ytd * 100, config.PRICE_DECIMALS) if ytd is not None else None,
        ytd_base_date=ytd_base_date,
        ytd_base_price=_round(ytd_base_price, config.PRICE_DECIMALS),
        ytd_unavailable_reason=ytd_reason,
        latest_indicators=latest_indicator_values(df),
        news_count=news.count if news is not None else 0,
        news_sources=news.sources_used if news is not None else [],
        data_quality=market.quality,
        warnings=warnings,
    )
