"""LLM client layer for worker graphs (GRAPH_NEAT_DESIGN.md sections 1.4 and 9.1).

Transient API failures are absorbed here, never by graph redundancy:
:class:`RetryingClient` wraps any object exposing ``chat.completions.create`` and retries
connection errors, timeouts, rate limits (429) and 5xx responses with exponential backoff
(``backoff * 2**attempt`` seconds, capped at 30). Other 4xx errors are raised at once.

Deviation from the contract layout: :class:`ModelSettings` is specified in
``graph/runner.py``; it is defined here instead to avoid a circular import (runner
imports this module) and ``runner.py`` re-exports it.
"""
from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Awaitable, Callable

import openai

MAX_BACKOFF_SECONDS = 30.0

Sleep = Callable[[float], Awaitable[Any]]


@dataclass
class ModelSettings:
    """Model and endpoint used by every worker in a graph run. Re-exported by ``runner.py``."""

    model: str
    api_base_url: str | None = None
    api_key_env_var: str | None = "ANTHROPIC_API_KEY"


def is_retryable(exc: BaseException) -> bool:
    """True for transient API failures: connection/timeout, 429, and any status >= 500."""
    if isinstance(exc, (openai.APIConnectionError, openai.APITimeoutError, openai.RateLimitError)):
        return True
    if isinstance(exc, openai.APIStatusError):
        return exc.status_code >= 500
    return False


class _RetryingCompletions:
    def __init__(self, owner: RetryingClient) -> None:
        self._owner = owner

    async def create(self, **kwargs: Any) -> Any:
        return await self._owner._create(**kwargs)


class RetryingClient:
    """Proxy around an AsyncOpenAI-compatible client whose ``chat.completions.create`` retries.

    Attribute access other than ``chat`` falls through to the wrapped client.
    ``max_retries`` is the number of retries after the first attempt, so the inner
    ``create`` is called at most ``max_retries + 1`` times before the last error is raised.
    """

    def __init__(
        self,
        inner: Any,
        *,
        max_retries: int = 5,
        backoff: float = 1.0,
        sleep: Sleep = asyncio.sleep,
    ) -> None:
        self.inner = inner
        self.max_retries = max_retries
        self.backoff = backoff
        self.sleep = sleep
        self.calls = 0
        self.chat = SimpleNamespace(completions=_RetryingCompletions(self))

    def delay(self, attempt: int) -> float:
        """Seconds to wait before retry number ``attempt`` (0-based)."""
        return min(self.backoff * 2**attempt, MAX_BACKOFF_SECONDS)

    async def _create(self, **kwargs: Any) -> Any:
        attempt = 0
        while True:
            self.calls += 1
            try:
                return await self.inner.chat.completions.create(**kwargs)
            except Exception as exc:
                if attempt >= self.max_retries or not is_retryable(exc):
                    raise
                await self.sleep(self.delay(attempt))
                attempt += 1

    def __getattr__(self, name: str) -> Any:
        if name == "inner":
            raise AttributeError(name)
        return getattr(self.inner, name)


def make_client_factory(
    settings: ModelSettings,
    *,
    max_retries: int = 5,
    backoff: float = 1.0,
    sleep: Sleep = asyncio.sleep,
) -> Callable[[], RetryingClient]:
    """Return a zero-argument factory building a fresh retrying ``AsyncOpenAI`` client per call."""

    def factory() -> RetryingClient:
        api_key = os.environ.get(settings.api_key_env_var) if settings.api_key_env_var else None
        # Retries are RetryingClient's job; the SDK's own retries (which also cover 408/409)
        # would multiply attempts and break the "no retry on 4xx except 429" rule.
        inner = openai.AsyncOpenAI(api_key=api_key, base_url=settings.api_base_url, max_retries=0)
        return RetryingClient(inner, max_retries=max_retries, backoff=backoff, sleep=sleep)

    return factory
