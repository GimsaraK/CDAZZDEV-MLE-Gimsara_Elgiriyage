"""Groq-first chat client for the teacher. OpenRouter is used only when Groq's reply fails validation.

HTTP errors are re-raised with the status code only. Provider bodies can contain
an account id, and that id must not land in a log or a notebook output.
"""
# AI-ASSISTED: Cursor Agent (claude-sonnet-5.5), Prompt: 'Implement the Task 2A dataset plan', Date: 2026-10-08 (see CITATIONS.md Entry 8)

import logging
import os
import sys
from typing import Any, Callable, Dict, List, TypeVar

from tenacity import Retrying, retry_if_not_exception_type, stop_after_attempt, wait_random_exponential

from . import config

logger = logging.getLogger("task2.client")
T = TypeVar("T")

# A bad request or a schema error will fail the same way on every retry.
NON_RETRYABLE = (ValueError, KeyError, TypeError)


class TeacherUnavailable(RuntimeError):
    """The provider rejected the key, or no provider is configured."""


def _log_retry(description: str) -> Callable:
    def _log(state) -> None:
        exc = state.outcome.exception() if state.outcome else None
        logger.warning(
            "%s failed (attempt %d/%d): %s - retrying",
            description,
            state.attempt_number,
            config.RETRY_MAX_ATTEMPTS,
            exc,
        )

    return _log


def call_with_retry(func: Callable[..., T], description: str) -> T:
    """Retry network and rate-limit errors. ValueError is not retried."""
    retrying = Retrying(
        # At most RETRY_MAX_ATTEMPTS calls in total (the first call counts as attempt 1).
        stop=stop_after_attempt(config.RETRY_MAX_ATTEMPTS),
        # Exponential backoff with random jitter, so retries spread out instead of hitting the API in lockstep.
        wait=wait_random_exponential(
            multiplier=config.RETRY_BACKOFF_MIN_SECONDS,
            max=config.RETRY_BACKOFF_MAX_SECONDS,
        ),
        # complete() turns a bad request into ValueError precisely so this rule skips the retry.
        retry=retry_if_not_exception_type(NON_RETRYABLE),
        before_sleep=_log_retry(description),
        # Raise the original exception, not tenacity's RetryError wrapper.
        reraise=True,
    )
    return retrying(func)


def load_api_keys() -> Dict[str, bool]:
    """Put Groq and OpenRouter keys in the environment. The return value is only set/missing."""
    names = (config.ENV_GROQ_API_KEY, config.ENV_OPENROUTER_API_KEY)
    if "google.colab" in sys.modules:
        from google.colab import userdata

        for name in names:
            if os.environ.get(name):
                continue
            try:
                value = userdata.get(name)
            except Exception as exc:  # noqa: BLE001 - a missing Colab secret is not fatal
                logger.info("Colab secret %s not available: %s", name, exc)
                value = None
            if value:
                os.environ[name] = value
    else:
        from dotenv import load_dotenv

        load_dotenv(config.REPO_ROOT / ".env")
    return {name: bool(os.environ.get(name, "").strip()) for name in names}


class ChatClient:
    """One OpenAI-compatible endpoint. complete() returns the assistant text."""

    def __init__(self, provider: str, model: str, api_key: str, base_url: str, temperature=None, max_tokens=None):
        if not (api_key or "").strip():
            raise TeacherUnavailable(f"{provider} API key is blank")
        from openai import OpenAI

        headers = config.OPENROUTER_HEADERS if provider == config.LLM_PROVIDER_OPENROUTER else None
        self.provider = provider
        self.model = model
        # The teacher values stay the default. The Task 2C judge passes its own.
        self.temperature = config.TEACHER_TEMPERATURE if temperature is None else temperature
        self.max_tokens = config.TEACHER_MAX_TOKENS if max_tokens is None else max_tokens
        self._client = OpenAI(
            api_key=api_key.strip(),
            base_url=base_url,
            timeout=config.TEACHER_TIMEOUT_SECONDS,
            default_headers=headers,
        )

    def complete(self, messages: List[Dict[str, str]]) -> str:
        """JSON-mode completion. Rate limits retry. 400 and 404 do not."""
        from openai import APIStatusError, AuthenticationError, RateLimitError

        def _call() -> Any:
            try:
                return self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    response_format={"type": "json_object"},
                )
            # Each provider error is re-raised as a type the retry rule understands, with a message
            # that holds only the status code (the provider's body can include an account id):
            except AuthenticationError as exc:
                # Bad key: not in NON_RETRYABLE, but a RuntimeError subclass the caller can catch by name.
                raise TeacherUnavailable(f"{self.provider} rejected the API key") from exc
            except RateLimitError as exc:
                # 429: worth retrying after a backoff.
                raise RuntimeError(f"{self.provider} rate limited (HTTP 429)") from exc
            except APIStatusError as exc:
                # Other 4xx/5xx (e.g. 400 bad request, 404 unknown model): ValueError, so no retry.
                raise ValueError(f"{self.provider} rejected the request (HTTP {exc.status_code})") from exc

        completion = call_with_retry(_call, f"{self.provider} chat completion")
        content = completion.choices[0].message.content if completion.choices else None
        if not content or not str(content).strip():
            raise ValueError(f"{self.provider} returned an empty completion")
        return str(content)


def build_clients() -> Dict[str, ChatClient]:
    """Groq when its key is set, then OpenRouter. A missing key skips that provider."""
    specs = (
        (config.LLM_PROVIDER_GROQ, config.TEACHER_MODEL_GROQ, config.ENV_GROQ_API_KEY, config.GROQ_BASE_URL),
        (
            config.LLM_PROVIDER_OPENROUTER,
            config.TEACHER_MODEL_OPENROUTER,
            config.ENV_OPENROUTER_API_KEY,
            config.OPENROUTER_BASE_URL,
        ),
    )
    clients = {}
    for provider, model, env_name, base_url in specs:
        try:
            clients[provider] = ChatClient(provider, model, os.environ.get(env_name, ""), base_url)
        except TeacherUnavailable:
            logger.warning("Teacher provider %s skipped: %s is not set", provider, env_name)
    return clients


# AI-ASSISTED: Cursor Agent (claude-sonnet-5.5), Prompt: 'Implement the Task 2C evaluation plan' (see CITATIONS.md Entry 12)
def build_judge_clients() -> List[ChatClient]:
    """Task 2C judges in the order they are tried: Gemma, then Nemotron, both on OpenRouter.

    Neither is the teacher, so no judge grades answers against its own references.
    An empty list means OPENROUTER_API_KEY is not set.
    """
    from . import eval_config

    clients = []
    for model in (eval_config.JUDGE_MODEL_PRIMARY, eval_config.JUDGE_MODEL_BACKUP):
        try:
            clients.append(
                ChatClient(
                    config.LLM_PROVIDER_OPENROUTER,
                    model,
                    os.environ.get(config.ENV_OPENROUTER_API_KEY, ""),
                    config.OPENROUTER_BASE_URL,
                    temperature=eval_config.JUDGE_TEMPERATURE,
                    max_tokens=eval_config.JUDGE_MAX_TOKENS,
                )
            )
        except TeacherUnavailable:
            logger.warning("Judge %s skipped: %s is not set", model, config.ENV_OPENROUTER_API_KEY)
    return clients
