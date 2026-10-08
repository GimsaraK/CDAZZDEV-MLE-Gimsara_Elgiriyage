"""Retry policy for network calls (yfinance, RSS): exponential backoff with jitter."""
# AI-ASSISTED: Cursor Agent (claude-sonnet-5.5), Prompt: 'yes implement the plan @TASK1A_PLAN.md', Date: 2026-10-08 (see CITATIONS.md Entry 4)

from typing import Any, Callable, TypeVar

from tenacity import (
    RetryCallState,
    Retrying,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from . import config
from .errors import PipelineError
from .logging_utils import get_logger

logger = get_logger("retry")

T = TypeVar("T")

# Deterministic failures (e.g. unknown ticker, bad arguments) are not worth retrying.
NON_RETRYABLE_EXCEPTIONS = (PipelineError, ValueError, KeyError, TypeError)


def _log_before_sleep(description: str) -> Callable[[RetryCallState], None]:
    """Build a tenacity callback that logs each failed attempt before the retry wait starts."""

    def _log(state: RetryCallState) -> None:
        # state.outcome holds the result of the attempt that just failed; extract its exception.
        exc = state.outcome.exception() if state.outcome else None
        logger.warning(
            "%s failed (attempt %d/%d): %s - retrying",
            description,
            state.attempt_number,
            config.RETRY_MAX_ATTEMPTS,
            exc,
        )

    return _log


def call_with_retry(func: Callable[..., T], description: str, *args: Any, **kwargs: Any) -> T:
    """Call ``func`` with the configured retry policy; re-raises the last error if all attempts fail.

    The policy is built at call time from ``config`` so it can be tuned (or disabled in tests).
    """
    retrying = Retrying(
        # Give up after RETRY_MAX_ATTEMPTS calls in total (the first call counts as attempt 1).
        stop=stop_after_attempt(config.RETRY_MAX_ATTEMPTS),
        # Exponential backoff spreads retries out; random jitter avoids retrying in lock-step.
        # Full jitter: sleep ~ U(0, min(max, multiplier * 2**attempt)).
        wait=wait_random_exponential(
            multiplier=config.RETRY_BACKOFF_MIN_SECONDS,
            max=config.RETRY_BACKOFF_MAX_SECONDS,
        ),
        # Retry everything except deterministic errors (retrying an unknown ticker cannot help).
        retry=retry_if_not_exception_type(NON_RETRYABLE_EXCEPTIONS),
        before_sleep=_log_before_sleep(description),
        # Re-raise the original exception (not tenacity's RetryError) so callers see the real cause.
        reraise=True,
    )
    return retrying(func, *args, **kwargs)
