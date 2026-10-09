"""Chat models for the agent: Groq gpt-oss-120b first, then Groq gpt-oss-20b, then free OpenRouter models.

Fallbacks are chained after tools or a schema are bound, because a RunnableWithFallbacks
has no bind_tools(). Each provider retries its own short 429s first (the SDKs honour
retry-after); the next model is used only when that one still fails.

llm_sentiment reuses Task 1's headline scorer, which takes Task 1 ChatClients. The same
model order is given to it through ModelChainClient.
"""
# AI-ASSISTED: Claude Code (claude-sonnet-5.5), Prompt: 'Implement the Task 3A plan (the plan approved in Entry 13)', Date: 2026-10-09 (see CITATIONS.md Entry 14)

import os
from typing import Any, Dict, List, Optional, Sequence, Set, Type

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from pydantic import BaseModel

from task1_financial.src import llm as t1_llm
from task1_financial.src.errors import LLMUnavailableError

from . import config
from .logging_utils import get_logger
from .tracing import DAILY_LIMIT_REASON, safe_error

logger = get_logger("llm")


def load_api_keys() -> Dict[str, bool]:
    """Load GROQ_API_KEY / OPENROUTER_API_KEY from Colab Secrets or the repo .env. Returns booleans only."""
    return t1_llm.load_api_keys()


def _key(name: str) -> str:
    return os.environ.get(name, "").strip()


def build_chat_models(
    max_tokens: int = config.AGENT_MAX_TOKENS,
    reasoning_effort: str = config.AGENT_REASONING_EFFORT,
) -> List[BaseChatModel]:
    """Models in fallback order. Providers without a key are skipped. Raises if none is configured."""
    models: List[BaseChatModel] = []
    groq_key = _key(config.ENV_GROQ_API_KEY)
    if groq_key:
        from langchain_groq import ChatGroq

        for model_id in config.GROQ_MODELS:
            models.append(
                ChatGroq(
                    model=model_id,
                    api_key=groq_key,
                    temperature=config.AGENT_TEMPERATURE,
                    max_tokens=max_tokens,
                    reasoning_effort=reasoning_effort,
                    timeout=config.LLM_TIMEOUT_SECONDS,
                    max_retries=config.GROQ_MAX_RETRIES,
                )
            )
    else:
        logger.warning("%s is not set; Groq skipped", config.ENV_GROQ_API_KEY)

    openrouter_key = _key(config.ENV_OPENROUTER_API_KEY)
    if openrouter_key:
        from langchain_openai import ChatOpenAI

        for model_id in config.OPENROUTER_FALLBACK_MODELS:
            models.append(
                ChatOpenAI(
                    model=model_id,
                    api_key=openrouter_key,
                    base_url=config.OPENROUTER_BASE_URL,
                    temperature=config.AGENT_TEMPERATURE,
                    max_tokens=max_tokens,
                    timeout=config.LLM_TIMEOUT_SECONDS,
                    max_retries=config.OPENROUTER_MAX_RETRIES,
                    default_headers=config.OPENROUTER_HEADERS,
                )
            )
    else:
        logger.warning("%s is not set; OpenRouter fallbacks skipped", config.ENV_OPENROUTER_API_KEY)

    if not models:
        raise LLMUnavailableError("No agent LLM is configured: set GROQ_API_KEY or OPENROUTER_API_KEY")
    return models


class AllModelsFailed(RuntimeError):
    """Every model in a FallbackChain failed or is skipped. The message lists each model's (safe) error."""


class FallbackChain(Runnable):
    """Try each model in order; skip a model for the rest of the session once it reports a daily limit.

    Used instead of Runnable.with_fallbacks for two reasons: with_fallbacks re-raises only the
    first model's error, which hides why the later ones failed, and it retries a model that has
    hit its daily quota on every turn, wasting the retry wait each time.
    """

    def __init__(self, runnables: Sequence[Runnable], names: Sequence[str], skipped: Optional[Set[str]] = None) -> None:
        self.runnables = list(runnables)
        self.names = list(names)
        # Shared between the agent chain and the report chain of one session.
        self.skipped: Set[str] = skipped if skipped is not None else set()

    def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        # Implementing invoke() is all LangChain needs: the graph calls chain.invoke(messages) as it would
        # on a single model. Each model already retried its own short 429s inside its SDK before failing here.
        errors: List[str] = []
        for name, runnable in zip(self.names, self.runnables):
            if name in self.skipped:
                continue
            try:
                return runnable.invoke(input, config, **kwargs)
            except Exception as exc:  # noqa: BLE001 - move on to the next model
                message = safe_error(exc)
                errors.append(f"{name}: {message}")
                logger.warning("Model %s failed: %s", name, message)
                # A daily cap will not reset during this run, so stop wasting retries on this model.
                # `skipped` is a set shared by every chain in the session, so all agents skip it.
                if DAILY_LIMIT_REASON in message:
                    self.skipped.add(name)
                    logger.warning("%s skipped for the rest of this session (daily limit)", name)
        # Every model failed (or was skipped): raise one error that lists each model's reason.
        raise AllModelsFailed("; ".join(errors) or "every model is skipped for this session (daily limits)")


def _chain(runnables: Sequence[Runnable], models: Sequence[BaseChatModel], skipped: Optional[Set[str]]) -> Runnable:
    return FallbackChain(runnables, model_names(models), skipped)


def with_tools(models: Sequence[BaseChatModel], tools: Sequence[BaseTool], skipped: Optional[Set[str]] = None) -> Runnable:
    """The agent's model: tools bound to every provider, then chained as fallbacks."""
    return _chain([model.bind_tools(list(tools)) for model in models], models, skipped)


def structured_method(model: BaseChatModel) -> str:
    """Groq gets JSON-schema output: gpt-oss-20b names its function call 'functions.<Schema>' (a harmony-format
    leak) which LangChain's tool parser rejects. OpenRouter models use function calling."""
    from langchain_groq import ChatGroq

    return "json_schema" if isinstance(model, ChatGroq) else "function_calling"


def with_structure(
    models: Sequence[BaseChatModel], schema: Type[BaseModel], skipped: Optional[Set[str]] = None
) -> Runnable:
    """The report writer's model: Pydantic structured output on every provider, chained as fallbacks.

    include_raw=True returns {"raw", "parsed", "parsing_error"}, so the caller can see which model answered.
    """
    return _chain(
        [model.with_structured_output(schema, method=structured_method(model), include_raw=True) for model in models],
        models,
        skipped,
    )


def model_names(models: Sequence[BaseChatModel]) -> List[str]:
    return [getattr(m, "model_name", None) or getattr(m, "model", "?") for m in models]


# --------------------------------------------------------------------------- llm_sentiment clients
class ModelChainClient:
    """Task 1 ChatClient interface over several models of one provider, tried in order.

    A model that is still rate limited after Task 1's retries is skipped for the rest of the
    session: at that point it is almost always the daily cap, which will not reset mid-run.
    """

    def __init__(self, name: str, clients: Sequence[Any]) -> None:
        self.name = name
        self.clients = list(clients)
        self.skipped: Set[str] = set()
        self.last_model: Optional[str] = None

    @property
    def provider(self) -> str:
        # Task 1 stores this on every scored headline, so it names the model that answered.
        return f"{self.name}/{self.last_model}" if self.last_model else self.name

    def complete(self, messages: List[Dict[str, str]], max_tokens: int, response_format: Dict[str, str]) -> str:
        # Same idea as FallbackChain, but with Task 1's simpler client interface (complete() returns text),
        # because llm_sentiment reuses Task 1's scorer, which calls client.complete(...).
        errors: List[str] = []
        for client in self.clients:
            if client.model in self.skipped:
                continue
            try:
                text = client.complete(messages, max_tokens, response_format)
            except Exception as exc:  # noqa: BLE001 - try the next model of this provider
                message = safe_error(exc)
                errors.append(f"{client.model}: {message}")
                # Task 1's client turns every 429 into "rate limited (HTTP 429)" after its own retries,
                # so a 429 reaching this point is treated as the daily cap.
                if "429" in message or "rate limit" in message.lower():
                    self.skipped.add(client.model)
                    logger.warning("%s skipped for this session after repeated 429s", client.model)
                continue
            self.last_model = client.model
            return text
        raise RuntimeError("; ".join(errors) or f"every {self.name} model is skipped for this session")


def build_sentiment_clients() -> Dict[str, Any]:
    """Clients for Task 1's score_headlines: Groq 120b -> Groq 20b, then the first OpenRouter fallback."""
    clients: Dict[str, Any] = {}
    groq_key = _key(config.ENV_GROQ_API_KEY)
    if groq_key:
        clients[config.LLM_PROVIDER_GROQ] = ModelChainClient(
            config.LLM_PROVIDER_GROQ,
            [t1_llm.ChatClient(config.LLM_PROVIDER_GROQ, m, groq_key, config.GROQ_BASE_URL) for m in config.GROQ_MODELS],
        )
    openrouter_key = _key(config.ENV_OPENROUTER_API_KEY)
    if openrouter_key:
        clients[config.LLM_PROVIDER_OPENROUTER] = t1_llm.ChatClient(
            config.LLM_PROVIDER_OPENROUTER,
            config.OPENROUTER_FALLBACK_MODELS[0],
            openrouter_key,
            config.OPENROUTER_BASE_URL,
        )
    return clients
