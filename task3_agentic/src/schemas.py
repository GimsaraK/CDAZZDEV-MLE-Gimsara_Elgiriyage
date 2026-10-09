"""Pydantic models: the tool envelope, the five tool payloads, and the research report.

Every tool returns ToolResult[<payload>]. The agent sees it as compact JSON with a
status of ok / empty / error, and a hint when something went wrong.
"""
# AI-ASSISTED: Claude Code (claude-opus-5-5), Prompt: 'Implement the Task 3A plan (the plan approved in Entry 13)', Date: 2026-10-09 (see CITATIONS.md Entry 14)

import json
from typing import Dict, Generic, List, Literal, Optional, TypeVar

from pydantic import BaseModel, Field, field_validator

from . import config

PayloadT = TypeVar("PayloadT", bound=BaseModel)

ToolStatus = Literal["ok", "empty", "error"]
ToolName = Literal["get_price_data", "get_news", "calculate_volatility", "llm_sentiment", "web_search"]
TOOL_NAMES: tuple = ToolName.__args__  # type: ignore[attr-defined]


# --------------------------------------------------------------------------- envelope
class ToolResult(BaseModel, Generic[PayloadT]):
    """What every tool returns. Tools never raise; failures become status='error' with a hint."""

    status: ToolStatus
    tool: str
    data: Optional[PayloadT] = None
    error: Optional[str] = None
    # What the agent could try instead (other arguments, another tool).
    hint: Optional[str] = None
    warnings: List[str] = Field(default_factory=list)
    # live / snapshot / session / fallback: where the data came from.
    source: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.status == config.STATUS_OK

    def to_llm_json(self, max_chars: int = config.TOOL_MESSAGE_MAX_CHARS) -> str:
        """Compact JSON for the agent's context, cut to `max_chars` so one tool cannot flood it."""
        text = json.dumps(self.model_dump(mode="json", exclude_none=True), separators=(",", ":"), default=str)
        if len(text) <= max_chars:
            return text
        return text[: max_chars - len(config.TRUNCATION_MARKER)] + config.TRUNCATION_MARKER


# --------------------------------------------------------------------------- get_price_data
class Fundamentals(BaseModel):
    """Balance-sheet and earnings fields from yfinance .info. Yahoo omits some fields, so all are optional."""

    market_cap: Optional[float] = None
    trailing_pe: Optional[float] = None
    forward_pe: Optional[float] = None
    profit_margin_pct: Optional[float] = None
    revenue_growth_pct: Optional[float] = None
    earnings_growth_pct: Optional[float] = None
    return_on_equity_pct: Optional[float] = None
    debt_to_equity_pct: Optional[float] = None  # percent, as Yahoo reports it (78 = 0.78x)
    current_ratio: Optional[float] = None
    free_cash_flow: Optional[float] = None
    total_cash: Optional[float] = None
    total_debt: Optional[float] = None
    beta: Optional[float] = None
    unavailable_reason: Optional[str] = None


class PriceData(BaseModel):
    """Price snapshot, latest indicators, fundamentals, and the last few daily bars."""

    ticker: str
    company_name: Optional[str] = None
    currency: Optional[str] = None
    period: str
    as_of_date: Optional[str] = None
    rows_fetched: int
    current_price: Optional[float] = None
    period_return_pct: Optional[float] = None
    period_start_date: Optional[str] = None
    ytd_return_pct: Optional[float] = None
    week_52_high: Optional[float] = None
    week_52_low: Optional[float] = None
    pct_from_52w_high: Optional[float] = None
    # SMA_50, SMA_200, RSI_14, MACD, MACD_Signal, MACD_Hist, BB_Upper/Middle/Lower, BB_PctB, BB_Bandwidth.
    indicators: Dict[str, Optional[float]] = Field(default_factory=dict)
    momentum_label: Optional[str] = None
    momentum_score: Optional[float] = None
    # Plain-language facts derived from the indicators (trend, crossover, overbought, ...).
    signals: List[str] = Field(default_factory=list)
    fundamentals: Fundamentals = Field(default_factory=Fundamentals)
    # Columns: date, open, high, low, close, volume. Arrays rather than dicts keep the JSON small.
    recent_bars_columns: List[str] = Field(default_factory=lambda: ["date", "open", "high", "low", "close", "volume"])
    recent_bars: List[list] = Field(default_factory=list)


# --------------------------------------------------------------------------- get_news
class NewsHeadline(BaseModel):
    title: str
    publisher: Optional[str] = None
    published_at: Optional[str] = None
    url: Optional[str] = None


class NewsList(BaseModel):
    ticker: str
    count: int
    headlines: List[NewsHeadline] = Field(default_factory=list)
    sources_used: List[str] = Field(default_factory=list)
    from_snapshot: bool = False
    snapshot_fetched_at: Optional[str] = None


# --------------------------------------------------------------------------- calculate_volatility
class VolatilityResult(BaseModel):
    """Annualised close-to-close volatility plus the numbers a hedge can be sized from."""

    ticker: str
    window: int
    as_of_date: Optional[str] = None
    method: str = "std(daily log returns, ddof=1) x sqrt(252)"
    annualised_vol_pct: float
    daily_vol_pct: float
    # Same estimator over other windows, e.g. {"20d": 22.1, "60d": 25.3, "252d": 27.8}.
    comparison_pct: Dict[str, Optional[float]] = Field(default_factory=dict)
    percentile_1y: Optional[float] = None
    regime: Optional[str] = None
    spot: float
    horizon_calendar_days: int
    horizon_trading_days: int
    # One standard deviation of the log price move over the horizon, in percent.
    expected_move_pct: float
    band_1sigma_low: float
    band_1sigma_high: float
    band_2sigma_low: float
    max_drawdown_1y_pct: Optional[float] = None


# --------------------------------------------------------------------------- llm_sentiment
class ScoredHeadline(BaseModel):
    headline: str
    sentiment: str
    confidence: float
    reason: str


class SentimentScore(BaseModel):
    """Confidence-weighted sentiment in [-1, 1] (Task 1 aggregation) with the most extreme headlines."""

    score: Optional[float] = None
    label: str
    n_scored: int
    n_unscored: int
    positive: int = 0
    negative: int = 0
    neutral: int = 0
    top_positive: List[ScoredHeadline] = Field(default_factory=list)
    top_negative: List[ScoredHeadline] = Field(default_factory=list)
    method: str = "confidence-weighted mean of +1/0/-1 votes"


# --------------------------------------------------------------------------- web_search
class SearchHit(BaseModel):
    title: str
    url: Optional[str] = None
    snippet: Optional[str] = None
    date: Optional[str] = None
    source: Optional[str] = None


class SearchResults(BaseModel):
    query: str
    # The query that actually returned results (it differs when the simplified fallback was used).
    effective_query: str
    backend: str
    hits: List[SearchHit] = Field(default_factory=list)


# --------------------------------------------------------------------------- research report
class Evidence(BaseModel):
    """One fact behind a claim, tied to the tool that produced it."""

    source_tool: ToolName = Field(description="The tool whose result contains this fact.")
    detail: str = Field(description="The specific number, headline or quote, e.g. 'RSI_14 = 71.3'.")
    url: Optional[str] = Field(default=None, description="Source URL for headlines or search results, if any.")


class Metric(BaseModel):
    name: str
    value: str
    source_tool: ToolName


class FinancialHealth(BaseModel):
    summary: str = Field(description="3-5 sentences on trend, valuation, profitability, balance sheet and momentum.")
    market_sentiment: str = Field(description="2-3 sentences on news sentiment and analyst commentary.")
    key_metrics: List[Metric] = Field(description="The numbers the summary relies on, each with its source tool.")


class Risk(BaseModel):
    # Not called "title": Pydantic puts a "title" key inside every property's JSON schema,
    # and gpt-oss-20b dropped a property with that name in structured-output tests.
    name: str = Field(description="Short name of the risk, e.g. 'Valuation compression'.")
    description: str = Field(description="Why this could move the share price in the next 90 days.")
    evidence: List[Evidence] = Field(min_length=1, description="At least one fact from a tool result.")


class HedgeStrategy(BaseModel):
    strategy: str = Field(description="Name of the hedge, e.g. protective put, collar, position trim.")
    instrument: str = Field(description="What is traded, e.g. 'AAPL 90-day put, strike near the 1-sigma low'.")
    rationale: str = Field(description="Which risks it covers and why it suits the current volatility.")
    sizing: str = Field(description="Strike, hedge ratio or trim size, derived from the volatility numbers.")
    annualised_vol_pct: float = Field(description="Annualised volatility used, copied from calculate_volatility.")
    expected_move_90d_pct: float = Field(description="90-day 1-sigma move used, copied from calculate_volatility.")
    trade_offs: str = Field(description="Cost and what the hedge gives up.")


class ResearchReport(BaseModel):
    """The structure the report writer must fill. Exactly three risks."""

    ticker: str
    financial_health: FinancialHealth
    risks: List[Risk] = Field(min_length=config.REQUIRED_RISKS, max_length=config.REQUIRED_RISKS)
    hedge: HedgeStrategy
    data_gaps: List[str] = Field(default_factory=list, description="Evidence that could not be gathered, and why.")
    # 3B only (the single agent in 3A leaves it empty): how Agent A's answer to the clarification changed the report.
    # AI-ASSISTED: Claude Code (claude-opus-5-5), Prompt: 'Implement the Task 3B plan', Date: 2026-10-09 (see CITATIONS.md Entry 16)
    clarification_used: Optional[str] = Field(
        default=None,
        description="Multi-agent runs only: what was asked of the Data Analyst, the values it returned, "
        "and where the report uses them. Leave null in single-agent runs.",
    )

    @field_validator("ticker")
    @classmethod
    def _upper(cls, value: str) -> str:
        return value.strip().upper()

    @field_validator("data_gaps")
    @classmethod
    def _drop_empty_gaps(cls, value: List[str]) -> List[str]:
        # Models sometimes emit fragments such as ':{' here; keep only entries with words in them.
        return [gap.strip() for gap in value if any(ch.isalpha() for ch in gap)]


class FinalReport(BaseModel):
    """The report plus run metadata that the model does not write."""

    ticker: str
    query: str
    session_id: str
    generated_at: str
    status: str
    report: Optional[ResearchReport] = None
    validation_warnings: List[str] = Field(default_factory=list)
    tools_used: List[str] = Field(default_factory=list)
    revisions: int = 0
    # Which models actually answered: agent turns per model, and the report writer's model.
    agent_models: Dict[str, int] = Field(default_factory=dict)
    report_model: Optional[str] = None
    # 3B: tools each agent called successfully, e.g. {"data_analyst": [...], "research_writer": [...]}.
    agent_tools: Dict[str, List[str]] = Field(default_factory=dict)
    error: Optional[str] = None
