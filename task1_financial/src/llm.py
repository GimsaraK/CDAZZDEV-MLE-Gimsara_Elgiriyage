"""Task 1B: per-headline sentiment and a Buy / Hold / Sell signal.

Each headline is one LLM call. The response must be JSON and is validated with
Pydantic before it is used. A validation failure gets one repair call, then the
other provider. Headlines that still fail are left unscored. If the signal call
fails on both providers, the result is Hold with fallback=True and a reason,
which is not presented as the model's own thesis.

Network errors are retried by call_with_retry. Validation errors are not.
"""
# AI-ASSISTED: Cursor Agent (claude-sonnet-5.5), Prompt: 'Implement the Task 1B plan', Date: 2026-10-08 (see CITATIONS.md Entry 6)

import json
import os
import re
import sys
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Type, TypeVar

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from . import config
from . import prompts
from .errors import LLMUnavailableError
from .logging_utils import get_logger
from .news import NewsResult
from .retry import call_with_retry
from .summary import StockSummary

logger = get_logger("llm")

ModelT = TypeVar("ModelT", bound=BaseModel)

# A sentence ends at . ! or ? only when the next word starts with a capital letter.
# That keeps decimals such as 0.43 from counting as a sentence break.
_SENTENCE_BOUNDARY = re.compile(r"[.!?](?=\s+[A-Z])")
_JSON_FENCE_START = re.compile(r"^```(?:json)?\s*", re.IGNORECASE)
_JSON_FENCE_END = re.compile(r"\s*```$")

SentimentLabel = Literal["positive", "negative", "neutral"]
SignalLabel = Literal["Buy", "Hold", "Sell"]


# --------------------------------------------------------------------------- models
class HeadlineSentiment(BaseModel):
    """One headline scored by the LLM. `provider` is filled by us, not by the model."""

    headline: str
    sentiment: SentimentLabel
    confidence: float = Field(ge=0, le=1)
    brief_reason: str
    provider: Optional[str] = None

    @field_validator("sentiment", mode="before")
    @classmethod
    def _normalise_sentiment(cls, value: Any) -> Any:
        # Accept "Positive" as well as "positive"; the allowed set is still the three labels.
        if isinstance(value, str):
            return value.strip().lower()
        return value

    @field_validator("brief_reason")
    @classmethod
    def _reason_not_blank(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("brief_reason is blank")
        return cleaned


class SentimentResult(BaseModel):
    """Scored headlines plus the confidence-weighted overall score."""

    items: List[HeadlineSentiment] = Field(default_factory=list)
    unscored: List[str] = Field(default_factory=list)
    score: Optional[float] = None
    label: str = config.SENTIMENT_LABEL_UNAVAILABLE
    n_scored: int = 0
    weight_sum: float = 0.0
    unavailable_reason: Optional[str] = None
    warnings: List[str] = Field(default_factory=list)


class SignalRecommendation(BaseModel):
    """Buy / Hold / Sell. A fallback Hold is flagged so it is not read as model analysis."""

    signal: SignalLabel
    justification: str = ""
    fallback: bool = False
    unavailable_reason: Optional[str] = None
    provider: Optional[str] = None

    @field_validator("signal", mode="before")
    @classmethod
    def _normalise_signal(cls, value: Any) -> Any:
        # "buy" and "BUY" both become "Buy". Anything else still fails the Literal check.
        if isinstance(value, str):
            return value.strip().capitalize()
        return value

    @model_validator(mode="after")
    def _justification_length(self) -> "SignalRecommendation":
        # The Hold fallback is built by us and has no model text to count.
        if self.fallback and not self.justification.strip():
            return self
        count = count_sentences(self.justification)
        if count < config.JUSTIFICATION_MIN_SENTENCES or count > config.JUSTIFICATION_MAX_SENTENCES:
            raise ValueError(
                f"justification has {count} sentences; write between "
                f"{config.JUSTIFICATION_MIN_SENTENCES} and {config.JUSTIFICATION_MAX_SENTENCES}, "
                "each ending with a period"
            )
        return self


# --------------------------------------------------------------------------- pure helpers
def count_sentences(text: str) -> int:
    """How many sentences `text` has. Periods inside numbers (0.43) are not breaks."""
    cleaned = " ".join((text or "").split())
    if not cleaned:
        return 0
    return len(_SENTENCE_BOUNDARY.split(cleaned))


def sentiment_label(score: Optional[float]) -> str:
    """Map a score in [-1, 1] to a label using the same cut-offs as the momentum score."""
    if score is None:
        return config.SENTIMENT_LABEL_UNAVAILABLE
    if score >= config.MOMENTUM_STRONG_THRESHOLD:
        return config.SENTIMENT_LABEL_STRONG_POSITIVE
    if score >= config.MOMENTUM_MILD_THRESHOLD:
        return config.SENTIMENT_LABEL_POSITIVE
    if score <= -config.MOMENTUM_STRONG_THRESHOLD:
        return config.SENTIMENT_LABEL_STRONG_NEGATIVE
    if score <= -config.MOMENTUM_MILD_THRESHOLD:
        return config.SENTIMENT_LABEL_NEGATIVE
    return config.SENTIMENT_LABEL_NEUTRAL


def aggregate_sentiment(
    items: Sequence[HeadlineSentiment],
) -> Tuple[Optional[float], str, int, float, Optional[str]]:
    """Confidence-weighted mean of votes. Returns (score, label, n, weight_sum, reason).

    positive = +1, negative = -1, neutral = 0. The weight is the headline's confidence.
    No scored items, or a total weight of 0, yields score=None rather than a fake 0.
    """
    if not items:
        return None, config.SENTIMENT_LABEL_UNAVAILABLE, 0, 0.0, "No headlines were scored"
    weight_sum = sum(item.confidence for item in items)
    if weight_sum == 0:
        return None, config.SENTIMENT_LABEL_UNAVAILABLE, len(items), 0.0, "Every scored headline has zero confidence"
    weighted = sum(config.SENTIMENT_VOTE[item.sentiment] * item.confidence for item in items)
    score = round(weighted / weight_sum, config.RATIO_DECIMALS)
    return score, sentiment_label(score), len(items), round(weight_sum, config.RATIO_DECIMALS), None


def build_signal_context(summary: StockSummary, sentiment: SentimentResult) -> Dict[str, Any]:
    """The user-message payload: summary, every momentum vote, indicators, and sentiments.

    Sending the component votes (not only the momentum label) is what lets the model
    talk about combinations and contradictions.
    """
    return {
        "summary": summary.core_dict(),
        "momentum": {
            "score": summary.momentum.score,
            "label": summary.momentum.label,
            "flags": summary.momentum.flags,
            "components": [component.model_dump() for component in summary.momentum.components],
        },
        "latest_indicators": summary.latest_indicators,
        "news_sentiment": {
            "score": sentiment.score,
            "label": sentiment.label,
            "headlines": [
                item.model_dump(exclude={"provider"}) for item in sentiment.items
            ],
        },
    }


# --------------------------------------------------------------------------- client
class ChatClient:
    """One OpenAI-compatible chat client (Groq or OpenRouter). Never logs the API key."""

    def __init__(self, provider: str, model: str, api_key: str, base_url: str) -> None:
        if not api_key or not str(api_key).strip():
            raise LLMUnavailableError(f"{provider} API key is not set")
        # Imported here so a missing optional extra fails at client build time, not at import.
        from openai import OpenAI

        self.provider = provider
        self.model = model
        headers = config.OPENROUTER_HEADERS if provider == config.LLM_PROVIDER_OPENROUTER else None
        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=config.LLM_TIMEOUT_SECONDS,
            default_headers=headers,
        )

    def complete(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int,
        response_format: Dict[str, str],
    ) -> str:
        """Send `messages` in JSON mode and return the assistant text. Retries network errors."""
        from openai import APIStatusError, AuthenticationError, RateLimitError

        def _call() -> Any:
            try:
                return self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=config.LLM_TEMPERATURE,
                    max_tokens=max_tokens,
                    response_format=response_format,
                )
            except AuthenticationError as exc:
                # A bad key will not start working on a retry. The message stays generic:
                # provider bodies can contain account ids.
                raise LLMUnavailableError(f"{self.provider} rejected the API key") from exc
            except RateLimitError as exc:
                # Retryable. Replace the body so logs do not keep the provider's account id.
                raise RuntimeError(f"{self.provider} rate limited (HTTP 429)") from exc
            except APIStatusError as exc:
                # 400 and 404 (unknown model) will fail the same way on every retry.
                raise ValueError(
                    f"{self.provider} rejected the request (HTTP {exc.status_code})"
                ) from exc

        completion = call_with_retry(_call, f"{self.provider} chat completion")
        content = completion.choices[0].message.content if completion.choices else None
        if not content or not str(content).strip():
            raise ValueError(f"{self.provider} returned an empty completion")
        return str(content)


def load_api_keys() -> Dict[str, bool]:
    """Load Groq and OpenRouter keys into the environment. Returns whether each one is set.

    Colab reads the Secrets panel. A local run reads the repo-root `.env`.
    The return value is only True/False, so a caller can print it without leaking a key.
    """
    names = (config.ENV_GROQ_API_KEY, config.ENV_OPENROUTER_API_KEY)
    if "google.colab" in sys.modules:
        # Colab's userdata.get raises when the secret is missing; that is not fatal.
        from google.colab import userdata

        for name in names:
            if os.environ.get(name):
                continue
            try:
                value = userdata.get(name)
            except Exception as exc:  # noqa: BLE001 - missing secret or Colab UI error
                logger.info("Colab secret %s not available: %s", name, exc)
                value = None
            if value:
                os.environ[name] = value
    else:
        from dotenv import load_dotenv

        # The repo root is the parent of task1_financial/, which is where .env lives.
        load_dotenv(config.TASK_DIR.parent / ".env")
    return {name: bool(os.environ.get(name, "").strip()) for name in names}


def build_clients() -> Dict[str, ChatClient]:
    """One client per provider whose key is set. Missing keys are logged, not raised."""
    specs = (
        (config.LLM_PROVIDER_GROQ, config.LLM_MODEL_GROQ, config.ENV_GROQ_API_KEY, config.GROQ_BASE_URL),
        (
            config.LLM_PROVIDER_OPENROUTER,
            config.LLM_MODEL_OPENROUTER,
            config.ENV_OPENROUTER_API_KEY,
            config.OPENROUTER_BASE_URL,
        ),
    )
    clients: Dict[str, ChatClient] = {}
    for provider, model, env_name, base_url in specs:
        try:
            clients[provider] = ChatClient(provider, model, os.environ.get(env_name, ""), base_url)
        except LLMUnavailableError:
            logger.warning("LLM provider %s skipped: %s is not set", provider, env_name)
    return clients


# --------------------------------------------------------------------------- validate and repair
def _strip_json_fence(raw: str) -> str:
    """Drop a ```json fence if the model added one despite JSON mode."""
    text = (raw or "").strip()
    if text.startswith("```"):
        text = _JSON_FENCE_START.sub("", text)
        text = _JSON_FENCE_END.sub("", text)
    return text.strip()


def _validation_message(exc: ValidationError) -> str:
    """A short, single-line description of a Pydantic error, safe to send back to the model."""
    parts = []
    for err in exc.errors():
        loc = ".".join(str(piece) for piece in err.get("loc", ())) or "response"
        parts.append(f"{loc}: {err.get('msg')}")
    return ("; ".join(parts) or str(exc))[:500]


def parse_or_repair(
    client: Any,
    system: str,
    user: str,
    model_cls: Type[ModelT],
    max_tokens: int,
) -> Tuple[Optional[ModelT], bool, Optional[str]]:
    """Validate one completion. On failure, send the error back once and try again.

    Returns (model, repaired, error). `repaired` is True when the second call succeeded.
    `error` is set only when both calls fail. Transport errors are not repaired: there is
    no response to correct, and the caller moves on to the next provider.
    """
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    last_error = "no response"
    for attempt in range(1 + config.LLM_REPAIR_ATTEMPTS):
        try:
            raw = client.complete(messages, max_tokens, config.JSON_RESPONSE_FORMAT)
        except Exception as exc:  # noqa: BLE001 - one provider failing must not abort the other
            last_error = f"{client.provider} request failed: {exc}"
            logger.warning(last_error)
            return None, False, last_error
        try:
            parsed = model_cls.model_validate_json(_strip_json_fence(raw))
            return parsed, attempt > 0, None
        except ValidationError as exc:
            last_error = _validation_message(exc)
            logger.warning(
                "%s response failed validation (attempt %d): %s",
                client.provider,
                attempt + 1,
                last_error,
            )
        if attempt < config.LLM_REPAIR_ATTEMPTS:
            # Second user turn: the error plus the bad text, so the model can correct it.
            messages = messages + [{
                "role": "user",
                "content": prompts.REPAIR_SUFFIX.format(error=last_error, raw=(raw or "")[:500]),
            }]
    return None, False, last_error


def _ordered_clients(clients: Optional[Dict[str, Any]]) -> List[Any]:
    """Providers in config order. None builds them from the environment (tests pass fakes)."""
    chosen = build_clients() if clients is None else clients
    return [chosen[name] for name in config.LLM_PROVIDER_ORDER if chosen.get(name) is not None]


# --------------------------------------------------------------------------- scoring
def score_headlines(
    news: NewsResult,
    ticker: str,
    company_name: Optional[str] = None,
    clients: Optional[Dict[str, Any]] = None,
) -> SentimentResult:
    """Score each headline, then aggregate. Never raises; failures become unscored titles."""
    ordered = _ordered_clients(clients)
    titles = [item.title for item in news.items]
    warnings: List[str] = []
    if not ordered:
        warnings.append("No LLM provider is configured (set GROQ_API_KEY or OPENROUTER_API_KEY)")
        return SentimentResult(unscored=titles, warnings=warnings, unavailable_reason=warnings[0])
    if not titles:
        reason = "No headlines to score"
        return SentimentResult(warnings=[reason], unavailable_reason=reason)

    scored: List[HeadlineSentiment] = []
    unscored: List[str] = []
    for title in titles:
        parsed, note = _score_one(title, ticker, company_name or ticker, ordered)
        if parsed is None:
            unscored.append(title)
            warnings.append(note or f"Unscored headline: {title}")
            logger.warning("Headline left unscored: %s", title)
        else:
            scored.append(parsed)
            if note:
                warnings.append(note)

    score, label, n_scored, weight_sum, reason = aggregate_sentiment(scored)
    logger.info(
        "Sentiment for %s: %s scored, %s unscored, score=%s (%s)",
        ticker,
        n_scored,
        len(unscored),
        score,
        label,
    )
    return SentimentResult(
        items=scored,
        unscored=unscored,
        score=score,
        label=label,
        n_scored=n_scored,
        weight_sum=weight_sum,
        unavailable_reason=reason,
        warnings=warnings,
    )


def _score_one(
    title: str,
    ticker: str,
    company_name: str,
    clients: Sequence[Any],
) -> Tuple[Optional[HeadlineSentiment], Optional[str]]:
    """Try each provider, with repair, until one returns a valid headline score."""
    user = prompts.SENTIMENT_USER.format(ticker=ticker, company_name=company_name, headline=title)
    failures: List[str] = []
    for client in clients:
        parsed, repaired, error = parse_or_repair(
            client, prompts.SENTIMENT_SYSTEM, user, HeadlineSentiment, config.LLM_MAX_TOKENS_SENTIMENT
        )
        if parsed is not None:
            # Keep the source headline. A paraphrased title would not match the news table.
            parsed.headline = title
            parsed.provider = client.provider
            note = f"Repaired invalid response for headline: {title}" if repaired else None
            if repaired:
                logger.info("Repaired sentiment response for %r via %s", title, client.provider)
            return parsed, note
        failures.append(f"{client.provider}: {error}")
    return None, f"Unscored headline {title!r} ({'; '.join(failures)})"


def recommend_signal(
    summary: StockSummary,
    sentiment: SentimentResult,
    clients: Optional[Dict[str, Any]] = None,
) -> SignalRecommendation:
    """Ask for Buy / Hold / Sell. If every provider fails, return a flagged Hold fallback."""
    ordered = _ordered_clients(clients)
    if not ordered:
        reason = "No LLM provider is configured (set GROQ_API_KEY or OPENROUTER_API_KEY)"
        logger.warning(reason)
        return _signal_fallback(reason)

    context = json.dumps(build_signal_context(summary, sentiment), indent=2, default=str)
    user = prompts.SIGNAL_USER.format(context_json=context)
    failures: List[str] = []
    for client in ordered:
        parsed, repaired, error = parse_or_repair(
            client, prompts.SIGNAL_SYSTEM, user, SignalRecommendation, config.LLM_MAX_TOKENS_SIGNAL
        )
        if parsed is not None:
            parsed.provider = client.provider
            parsed.fallback = False
            if repaired:
                logger.info("Repaired signal response via %s", client.provider)
            logger.info("Signal for %s: %s (provider=%s)", summary.ticker, parsed.signal, client.provider)
            return parsed
        failures.append(f"{client.provider}: {error}")

    reason = "All providers failed signal validation: " + "; ".join(failures)
    logger.warning(reason)
    return _signal_fallback(reason)


def _signal_fallback(reason: str) -> SignalRecommendation:
    """Hold with an explicit fallback flag. The justification is left empty on purpose."""
    return SignalRecommendation(
        signal=config.SIGNAL_FALLBACK,
        justification="",
        fallback=True,
        unavailable_reason=reason,
        provider=None,
    )
