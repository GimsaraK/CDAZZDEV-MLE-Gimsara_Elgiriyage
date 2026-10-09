"""3B inter-agent messages: typed payloads, the code that fills their numbers, and the incorporation check.

Every message between the agents is a Pydantic model wrapped in an AgentMessage envelope.
Nothing is passed as a raw string. The numeric fields of the Data Analyst's brief and of its
clarification answer are filled from that agent's own tool results in code, so they cannot be
hallucinated; the LLMs write only the interpretive text around them.
"""
# AI-ASSISTED: Claude Code (claude-opus-5-5), Prompt: 'Implement the Task 3B plan', Date: 2026-10-09 (see CITATIONS.md Entry 16)

import json
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Type, TypeVar

from pydantic import BaseModel, Field, field_validator, model_validator

from . import config
from .schemas import Fundamentals, PriceData, ResearchReport, SentimentScore, VolatilityResult
from .session import SessionContext

ModelT = TypeVar("ModelT", bound=BaseModel)

RequestKind = Literal["score_headlines", "volatility_window", "price_period", "metric_check"]
MessageKind = Literal["task", "data_brief", "clarification_request", "clarification_response", "final_report"]
REQUEST_KINDS: Tuple[str, ...] = RequestKind.__args__  # type: ignore[attr-defined]

_NUMBER = re.compile(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?")
# Characters models use for minus signs and thin spaces; normalised before numbers are read.
_TEXT_FIXES = {"−": "-", "‑": "-", "–": "-", " ": " ", " ": " ", " ": " "}
_NON_ALNUM = re.compile(r"[^a-z0-9]")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --------------------------------------------------------------------------- orchestrator -> A
class AnalystTask(BaseModel):
    ticker: str
    query: str
    focus: str
    tools: List[str]


# --------------------------------------------------------------------------- A -> B: the data brief
class PriceBlock(BaseModel):
    """get_price_data's result without the raw bars (B needs the analysis, not the rows)."""

    source_tool: Literal["get_price_data"] = "get_price_data"
    company_name: Optional[str] = None
    currency: Optional[str] = None
    as_of_date: Optional[str] = None
    period: str
    current_price: Optional[float] = None
    period_return_pct: Optional[float] = None
    ytd_return_pct: Optional[float] = None
    week_52_high: Optional[float] = None
    week_52_low: Optional[float] = None
    pct_from_52w_high: Optional[float] = None
    momentum_label: Optional[str] = None
    momentum_score: Optional[float] = None
    indicators: Dict[str, Optional[float]] = Field(default_factory=dict)
    signals: List[str] = Field(default_factory=list)
    fundamentals: Fundamentals = Field(default_factory=Fundamentals)

    @classmethod
    def from_price(cls, data: PriceData) -> "PriceBlock":
        return cls(**data.model_dump(exclude={"ticker", "rows_fetched", "period_start_date", "recent_bars", "recent_bars_columns"}))


class RiskFlag(BaseModel):
    metric: str = Field(description="The brief field the concern rests on, e.g. 'forward_pe' or 'RSI_14'.")
    value: str = Field(description="Its value, copied from the brief.")
    concern: str = Field(description="Why it matters for the share price over the next 90 days.")


class AnalystInterpretation(BaseModel):
    """What Agent A's LLM adds to the brief. Numbers stay in the code-filled blocks."""

    key_findings: List[str] = Field(min_length=1, description="2-5 quantitative findings that combine several metrics.")
    quant_risk_flags: List[RiskFlag] = Field(default_factory=list, description="Up to 5 quantitative risk flags.")
    confidence_notes: str = Field(description="What the numbers cannot tell you, and any data quality caveats.")

    @field_validator("key_findings")
    @classmethod
    def _cap_findings(cls, value: List[str]) -> List[str]:
        return [v.strip() for v in value if v.strip()][: config.BRIEF_MAX_FINDINGS]

    @field_validator("quant_risk_flags")
    @classmethod
    def _cap_flags(cls, value: List[RiskFlag]) -> List[RiskFlag]:
        return value[: config.BRIEF_MAX_FINDINGS]


class DataBrief(BaseModel):
    """Agent A's structured handoff to Agent B."""

    ticker: str
    as_of: str = Field(description="UTC time the brief was assembled.")
    prepared_by: str = config.AGENT_A_NAME
    price: Optional[PriceBlock] = Field(default=None, description="[code] get_price_data result without raw bars.")
    volatility: Optional[VolatilityResult] = Field(default=None, description="[code] calculate_volatility result.")
    sentiment: Optional[SentimentScore] = Field(default=None, description="[code] llm_sentiment result; empty until B sends headlines.")
    sentiment_unavailable_reason: Optional[str] = Field(default=None, description="[code] why there is no sentiment yet.")
    tool_status: Dict[str, str] = Field(default_factory=dict, description="[code] per A tool: ok / empty / error / not_called.")
    data_gaps: List[str] = Field(default_factory=list, description="[code] failed tools and missing blocks.")
    key_findings: List[str] = Field(default_factory=list, description="[A's LLM] findings combining several metrics.")
    quant_risk_flags: List[RiskFlag] = Field(default_factory=list, description="[A's LLM] risks tied to a named metric.")
    confidence_notes: Optional[str] = Field(default=None, description="[A's LLM] what the numbers cannot tell.")
    interpretation_model: Optional[str] = Field(default=None, description="Model that wrote the interpretive fields.")


# --------------------------------------------------------------------------- B -> A: the critique request
class ClarificationRequest(BaseModel):
    """Agent B's one question to Agent A. The kind decides which parameters must be present."""

    request_kind: RequestKind = Field(
        description="score_headlines: A scores B's headlines with llm_sentiment. volatility_window: A computes "
        "volatility over another window. price_period: A computes the return over another period. "
        "metric_check: A confirms one metric's current value."
    )
    question: str = Field(min_length=10, description="The specific question for Agent A.")
    why_needed: str = Field(description="Which part of the report depends on the answer.")
    headlines: List[str] = Field(default_factory=list, description="score_headlines only: the headline titles to score.")
    window: Optional[int] = Field(default=None, description="volatility_window only: look-back in trading days.")
    period: Optional[str] = Field(default=None, description="price_period only: one of the yfinance periods.")
    metric: Optional[str] = Field(default=None, description="metric_check only: the brief field to confirm.")

    @model_validator(mode="after")
    def _parameters_for_kind(self) -> "ClarificationRequest":
        if self.request_kind == "score_headlines":
            seen, clean = set(), []
            for title in self.headlines:
                text = str(title).strip()
                if text and text.lower() not in seen:
                    seen.add(text.lower())
                    clean.append(text)
            if not clean:
                raise ValueError("score_headlines needs at least one headline")
            self.headlines = clean[: config.MAX_SENTIMENT_HEADLINES]
        elif self.request_kind == "volatility_window":
            if self.window is None or not config.MIN_VOL_WINDOW <= self.window <= config.MAX_VOL_WINDOW:
                raise ValueError(f"volatility_window needs a window between {config.MIN_VOL_WINDOW} and {config.MAX_VOL_WINDOW}")
        elif self.request_kind == "price_period":
            if (self.period or "").strip().lower() not in config.VALID_PERIODS:
                raise ValueError(f"price_period needs a period in {config.VALID_PERIODS}")
            self.period = self.period.strip().lower()
        elif self.request_kind == "metric_check" and not (self.metric or "").strip():
            raise ValueError("metric_check needs a metric name")
        return self


# --------------------------------------------------------------------------- A -> B: the answer
class PeriodReturn(BaseModel):
    source_tool: Literal["get_price_data"] = "get_price_data"
    period: str
    period_return_pct: Optional[float] = None
    period_start_date: Optional[str] = None
    current_price: Optional[float] = None


class ClarificationResponse(BaseModel):
    request_kind: RequestKind
    status: Literal["ok", "partial", "failed"] = Field(description="[code] ok when A's new tool results hold the requested data.")
    answer: str = Field(description="[A's LLM] 1-3 sentence reply citing the values.")
    sentiment: Optional[SentimentScore] = Field(default=None, description="[code] score_headlines: llm_sentiment on B's headlines.")
    volatility: Optional[VolatilityResult] = Field(default=None, description="[code] volatility_window: the requested window.")
    period_return: Optional[PeriodReturn] = Field(default=None, description="[code] price_period: the requested period's return.")
    metric_name: Optional[str] = Field(default=None, description="[code] metric_check: the metric looked up.")
    metric_value: Optional[float] = Field(default=None, description="[code] metric_check: its value.")
    tools_used: List[str] = Field(default_factory=list, description="[code] A's tool calls after the request.")
    # "fallback" when the orchestrator ran A's tool on A's behalf because A's LLM did not.
    fulfilled_by: Literal["agent", "fallback"] = Field(default="agent", description="agent, or fallback if A's own toolkit was run for it.")
    error: Optional[str] = None


# --------------------------------------------------------------------------- the envelope
class AgentMessage(BaseModel):
    """One inter-agent message. `payload` is the dump of a validated model; receivers validate it again."""

    id: str
    sender: str = Field(description="data_analyst, research_writer or orchestrator.")
    recipient: str
    kind: MessageKind = Field(description="task / data_brief / clarification_request / clarification_response / final_report.")
    summary: str = Field(description="One line for the trace.")
    payload: Dict[str, Any] = Field(description="The validated model, dumped; the receiver validates it again.")
    timestamp: str
    fallback: bool = Field(default=False, description="True when the orchestrator's fallback produced this message.")

    @classmethod
    def wrap(cls, sender: str, recipient: str, kind: str, model: BaseModel, summary: str, fallback: bool = False) -> "AgentMessage":
        return cls(
            id=uuid.uuid4().hex[:8],
            sender=sender,
            recipient=recipient,
            kind=kind,
            summary=summary,
            payload=model.model_dump(mode="json", exclude_none=True),
            timestamp=_now(),
            fallback=fallback,
        )

    def open(self, schema: Type[ModelT]) -> ModelT:
        """The receiver's side: re-validate the payload against the schema it expects."""
        return schema.model_validate(self.payload)


# --------------------------------------------------------------------------- code-filled numbers
def _latest(ctx: SessionContext, tool: str, agent: str, since: int = 0):
    records = ctx.results_for(tool, agent=agent, since=since)
    return records[0] if records else None


def tool_status(ctx: SessionContext, agent: str, tools: Sequence[str]) -> Dict[str, str]:
    status = {}
    for tool in tools:
        if ctx.results_for(tool, agent=agent):
            status[tool] = config.STATUS_OK
        else:
            attempts = ctx.results_for(tool, ok_only=False, agent=agent)
            status[tool] = attempts[0].result.status if attempts else "not_called"
    return status


def build_brief_numbers(ctx: SessionContext, agent: str = config.AGENT_A_NAME) -> DataBrief:
    """The brief's numeric blocks from A's latest successful tool results. Gaps are named, not invented."""
    statuses = tool_status(ctx, agent, config.AGENT_A_TOOLS)
    price_rec = _latest(ctx, "get_price_data", agent)
    vol_rec = _latest(ctx, "calculate_volatility", agent)
    sent_rec = _latest(ctx, "llm_sentiment", agent)
    gaps: List[str] = []
    for tool, status in statuses.items():
        if status == "not_called" and tool == "llm_sentiment":
            continue  # expected: A has no headlines until B sends some
        if status != config.STATUS_OK:
            last = ctx.results_for(tool, ok_only=False, agent=agent)
            reason = last[0].result.error if last else "not called"
            gaps.append(f"{tool}: {status} ({reason})")
    if sent_rec is None:
        reason = "Agent A has no news access; headlines must come from Agent B"
        if statuses.get("llm_sentiment") not in ("not_called", None):
            reason = f"llm_sentiment {statuses['llm_sentiment']}"
    else:
        reason = None
    return DataBrief(
        ticker=ctx.ticker,
        as_of=_now(),
        prepared_by=agent,
        price=PriceBlock.from_price(price_rec.result.data) if price_rec else None,
        volatility=vol_rec.result.data if vol_rec else None,
        sentiment=sent_rec.result.data if sent_rec else None,
        sentiment_unavailable_reason=reason,
        tool_status=statuses,
        data_gaps=gaps,
    )


def merge_interpretation(brief: DataBrief, interpretation: Optional[AnalystInterpretation], model: Optional[str]) -> DataBrief:
    if interpretation is None:
        return brief.model_copy(update={"data_gaps": [*brief.data_gaps, "Agent A's interpretation step failed; numbers only"]})
    return brief.model_copy(
        update={
            "key_findings": interpretation.key_findings,
            "quant_risk_flags": interpretation.quant_risk_flags,
            "confidence_notes": interpretation.confidence_notes,
            "interpretation_model": model,
        }
    )


def _key(name: str) -> str:
    return _NON_ALNUM.sub("", name.lower())


def lookup_metric(name: str, price: Optional[PriceData], volatility: Optional[VolatilityResult]) -> Optional[float]:
    """Find a numeric metric by (loose) name in A's latest price and volatility results."""
    wanted = _key(name)
    candidates: Dict[str, Any] = {}
    if price is not None:
        candidates.update(price.model_dump(exclude={"indicators", "fundamentals", "recent_bars", "recent_bars_columns"}))
        candidates.update(price.indicators)
        candidates.update(price.fundamentals.model_dump())
    if volatility is not None:
        candidates.update(volatility.model_dump(exclude={"comparison_pct"}))
    for key, value in candidates.items():
        if _key(key) == wanted and isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return None


def build_response(
    ctx: SessionContext,
    request: ClarificationRequest,
    answer: str,
    since: int,
    agent: str = config.AGENT_A_NAME,
    fulfilled_by: str = "agent",
) -> ClarificationResponse:
    """A's typed answer: values come from A's tool results made after the request (index `since`)."""
    tools_used = [r.tool for r in ctx.history[since:] if r.agent == agent and not r.denied]
    values: Dict[str, Any] = {}
    if request.request_kind == "score_headlines":
        rec = _latest(ctx, "llm_sentiment", agent, since)
        if rec:
            values["sentiment"] = rec.result.data
    elif request.request_kind == "volatility_window":
        records = ctx.results_for("calculate_volatility", agent=agent, since=since)
        exact = [r for r in records if r.result.data.window == request.window]
        if records:
            values["volatility"] = (exact or records)[0].result.data
    elif request.request_kind == "price_period":
        records = ctx.results_for("get_price_data", agent=agent, since=since)
        exact = [r for r in records if r.result.data.period == request.period]
        if records:
            data = (exact or records)[0].result.data
            values["period_return"] = PeriodReturn(
                period=data.period,
                period_return_pct=data.period_return_pct,
                period_start_date=data.period_start_date,
                current_price=data.current_price,
            )
    else:
        price_rec = _latest(ctx, "get_price_data", agent)
        vol_rec = _latest(ctx, "calculate_volatility", agent)
        value = lookup_metric(request.metric or "", price_rec.result.data if price_rec else None, vol_rec.result.data if vol_rec else None)
        if value is not None:
            values.update({"metric_name": request.metric, "metric_value": value})
    found = bool(values)
    return ClarificationResponse(
        request_kind=request.request_kind,
        status="ok" if found else "failed",
        answer=answer[: config.CLARIFICATION_ANSWER_MAX_CHARS],
        tools_used=tools_used,
        fulfilled_by=fulfilled_by,
        error=None if found else "Agent A did not produce the requested data",
        **values,
    )


# --------------------------------------------------------------------------- incorporation check
def expected_value(response: ClarificationResponse) -> Optional[Tuple[float, float, str]]:
    """(value, tolerance, label) that B's report must contain, or None when A had nothing to give."""
    if response.status == "failed":
        return None
    if response.sentiment is not None and response.sentiment.score is not None:
        return response.sentiment.score, config.INCORPORATION_SCORE_TOLERANCE, "sentiment score"
    if response.volatility is not None:
        return response.volatility.annualised_vol_pct, config.INCORPORATION_PCT_TOLERANCE, "annualised volatility (%)"
    if response.period_return is not None and response.period_return.period_return_pct is not None:
        return response.period_return.period_return_pct, config.INCORPORATION_PCT_TOLERANCE, f"{response.period_return.period} return (%)"
    if response.metric_value is not None:
        tolerance = max(abs(response.metric_value) * config.INCORPORATION_RELATIVE_TOLERANCE, config.INCORPORATION_SCORE_TOLERANCE)
        return response.metric_value, tolerance, f"{response.metric_name} value"
    return None


def report_numbers(report: ResearchReport) -> List[float]:
    """Every number written anywhere in the report."""
    text = report.model_dump_json()
    for bad, good in _TEXT_FIXES.items():
        text = text.replace(bad, good)
    numbers = []
    for match in _NUMBER.findall(text):
        try:
            numbers.append(float(match.replace(",", "")))
        except ValueError:
            continue
    return numbers


def check_incorporation(report: ResearchReport, response: Optional[ClarificationResponse]) -> List[str]:
    """Problems with how B used A's answer. An empty list means the answer is in the report."""
    if response is None:
        return []
    issues: List[str] = []
    if not (report.clarification_used or "").strip():
        issues.append(
            "clarification_used is empty: say what you asked the Data Analyst, what it returned, and where the report uses it."
        )
    target = expected_value(response)
    if target is not None:
        value, tolerance, label = target
        if not any(abs(number - value) <= tolerance for number in report_numbers(report)):
            issues.append(
                f"The report does not use the Data Analyst's {label} ({value}). Cite this value where it changes "
                "the analysis (market sentiment, a risk's evidence, or the hedge)."
            )
    return issues


def handoffs_to_json(handoffs: Sequence[AgentMessage]) -> str:
    return json.dumps([h.model_dump(mode="json") for h in handoffs], indent=2, ensure_ascii=False)
