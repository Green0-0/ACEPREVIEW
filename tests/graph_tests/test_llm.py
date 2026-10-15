"""Tests for ace.graph.llm: retry policy and client factory."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import httpx
import openai
import pytest

from ace.graph.llm import (
    MAX_BACKOFF_SECONDS,
    ModelSettings,
    RetryingClient,
    is_retryable,
    make_client_factory,
)

REQUEST = httpx.Request("POST", "https://example.invalid/v1/chat/completions")


def status_error(code: int) -> openai.APIStatusError:
    response = httpx.Response(code, request=REQUEST)
    if code == 429:
        return openai.RateLimitError("rate limited", response=response, body=None)
    if code >= 500:
        return openai.InternalServerError("server error", response=response, body=None)
    return openai.APIStatusError(f"status {code}", response=response, body=None)


class FlakyInner:
    """Raises the given errors in order, then returns ``"ok"``."""

    def __init__(self, errors: list[BaseException]):
        self.errors = list(errors)
        self.calls = 0
        self.kwargs: list[dict] = []
        self.marker = "inner-attribute"
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    async def create(self, **kwargs):
        self.calls += 1
        self.kwargs.append(kwargs)
        if self.errors:
            raise self.errors.pop(0)
        return "ok"


class Sleeps:
    def __init__(self):
        self.delays: list[float] = []

    async def __call__(self, delay: float) -> None:
        self.delays.append(delay)


def test_retries_rate_limit_then_succeeds():
    inner = FlakyInner([status_error(429), status_error(429)])
    sleeps = Sleeps()
    client = RetryingClient(inner, max_retries=5, backoff=1.0, sleep=sleeps)
    result = asyncio.run(client.chat.completions.create(model="m", messages=[]))
    assert result == "ok"
    assert inner.calls == 3 and client.calls == 3
    assert sleeps.delays == [1.0, 2.0]
    assert inner.kwargs[-1] == {"model": "m", "messages": []}


def test_retries_connection_timeout_and_5xx():
    errors = [
        openai.APIConnectionError(request=REQUEST),
        openai.APITimeoutError(request=REQUEST),
        status_error(503),
    ]
    inner = FlakyInner(errors)
    client = RetryingClient(inner, max_retries=3, backoff=0.5, sleep=Sleeps())
    assert asyncio.run(client.chat.completions.create(model="m")) == "ok"
    assert inner.calls == 4


def test_does_not_retry_400():
    inner = FlakyInner([status_error(400)])
    sleeps = Sleeps()
    client = RetryingClient(inner, sleep=sleeps)
    with pytest.raises(openai.APIStatusError) as info:
        asyncio.run(client.chat.completions.create(model="m"))
    assert info.value.status_code == 400
    assert inner.calls == 1 and sleeps.delays == []


def test_does_not_retry_arbitrary_exceptions():
    inner = FlakyInner([ValueError("boom")])
    client = RetryingClient(inner, sleep=Sleeps())
    with pytest.raises(ValueError):
        asyncio.run(client.chat.completions.create(model="m"))
    assert inner.calls == 1


def test_gives_up_after_max_retries():
    inner = FlakyInner([status_error(429) for _ in range(10)])
    sleeps = Sleeps()
    client = RetryingClient(inner, max_retries=3, backoff=1.0, sleep=sleeps)
    with pytest.raises(openai.RateLimitError):
        asyncio.run(client.chat.completions.create(model="m"))
    assert inner.calls == 4
    assert sleeps.delays == [1.0, 2.0, 4.0]


def test_backoff_is_capped():
    client = RetryingClient(FlakyInner([]), backoff=1.0)
    assert client.delay(0) == 1.0
    assert client.delay(10) == MAX_BACKOFF_SECONDS


def test_is_retryable():
    assert is_retryable(status_error(429))
    assert is_retryable(status_error(500))
    assert is_retryable(status_error(502))
    assert is_retryable(openai.APIConnectionError(request=REQUEST))
    assert is_retryable(openai.APITimeoutError(request=REQUEST))
    assert not is_retryable(status_error(400))
    assert not is_retryable(status_error(401))
    assert not is_retryable(status_error(404))
    assert not is_retryable(RuntimeError("x"))


def test_proxy_forwards_other_attributes():
    inner = FlakyInner([])
    client = RetryingClient(inner)
    assert client.marker == "inner-attribute"


def test_make_client_factory_builds_retrying_async_openai(monkeypatch):
    monkeypatch.setenv("GRAPH_TEST_KEY", "sk-test")
    settings = ModelSettings(model="m", api_base_url="https://example.invalid/v1/",
                             api_key_env_var="GRAPH_TEST_KEY")
    factory = make_client_factory(settings, max_retries=2, backoff=0.25)
    first, second = factory(), factory()
    assert first is not second
    assert isinstance(first, RetryingClient)
    assert isinstance(first.inner, openai.AsyncOpenAI)
    assert first.inner.api_key == "sk-test"
    assert str(first.inner.base_url).startswith("https://example.invalid/v1")
    assert first.max_retries == 2 and first.backoff == 0.25
    assert callable(first.chat.completions.create)


def test_model_settings_defaults():
    settings = ModelSettings(model="m")
    assert settings.api_base_url is None
    assert settings.api_key_env_var == "ANTHROPIC_API_KEY"
