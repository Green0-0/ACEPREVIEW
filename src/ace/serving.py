"""Client side of the vLLM fleet: endpoint discovery, sticky load balancing, token accounting.

vLLM server jobs (``slurm/vllm_server.sbatch``) write ``<endpoints_dir>/<jobid>.json``
(``{"url", "model", ...}``) once healthy and delete it on exit. :class:`EndpointPool`
re-reads that directory (cached for a few seconds) and health-checks nothing itself: a
server that stops answering is put on a cooldown list by the clients that saw it fail.

:func:`make_pool_client_factory` returns a zero-argument factory (the ``client_factory`` of
``RunContext`` / ``TaskEvaluator`` and the ``client`` of a baseline agent2 ``Agent``). Each
client it builds is *sticky*: it picks the least-loaded endpoint when created and keeps it
for every call, so a multi-step agent conversation hits one server and reuses its prefix
cache. On a connection error, timeout or 5xx the client moves to another endpoint; when no
endpoint is up it waits (up to ``wait_for_endpoint_s``) instead of failing the task, because
a failed task would score 0 and corrupt fitness when the cluster merely preempted a server.

Every response's ``usage`` is added to the :class:`UsageMeter` found in the
:data:`USAGE` context variable at call time. Callers set a fresh meter before running one
task (``with metered() as meter: ...``); asyncio tasks created inside inherit it.

Defaults applied to every request unless the caller set them: ``max_tokens``
(``max_tokens`` argument) so a degenerate generation cannot run to the context limit, and
the vLLM ``chat_template_kwargs`` ``{"enable_thinking": false}`` (Gemma-4 lowest thinking
level). Sampling parameters are left to the model's generation config (standard samplers).
"""
from __future__ import annotations

import asyncio
import contextlib
import contextvars
import json
import logging
import random
import time
import weakref
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Iterator

import openai

logger = logging.getLogger(__name__)

DEFAULT_ENDPOINTS_DIR = Path("/usr/project/xtmp/kg349/ace/endpoints")


# ------------------------------------------------------------------------------ usage


@dataclass
class UsageMeter:
    """Token and call counts of one task run (or any scope)."""

    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    errors: int = 0
    latency_s: float = 0.0

    def add(self, usage: Any, latency: float) -> None:
        self.calls += 1
        self.latency_s += latency
        if usage is None:
            return
        self.prompt_tokens += int(getattr(usage, "prompt_tokens", 0) or 0)
        self.completion_tokens += int(getattr(usage, "completion_tokens", 0) or 0)
        details = getattr(usage, "prompt_tokens_details", None)
        cached = getattr(details, "cached_tokens", None) if details is not None else None
        self.cached_tokens += int(cached or 0)

    def to_json(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
            "cached_tokens": self.cached_tokens,
            "errors": self.errors,
            "latency_s": round(self.latency_s, 3),
        }


USAGE: contextvars.ContextVar[UsageMeter | None] = contextvars.ContextVar("ace_usage", default=None)


@contextlib.contextmanager
def metered() -> Iterator[UsageMeter]:
    """Install a fresh :class:`UsageMeter` for the current context."""
    meter = UsageMeter()
    token = USAGE.set(meter)
    try:
        yield meter
    finally:
        USAGE.reset(token)


# -------------------------------------------------------------------------- endpoints


@dataclass
class Endpoint:
    url: str
    model: str
    name: str
    client: Any = None
    pinned: Any = field(default_factory=weakref.WeakSet)  # PooledClients pinned here

    def active(self, idle_release_s: float) -> int:
        """Pinned clients with a call in flight or one that finished < idle_release_s ago."""
        now = time.monotonic()
        return sum(1 for c in list(self.pinned)
                   if c.in_flight or now - c.last_done < idle_release_s)


@dataclass
class EndpointPool:
    """Endpoints discovered from a directory of JSON files, or a fixed list of URLs."""

    endpoints_dir: Path | None = DEFAULT_ENDPOINTS_DIR
    static_urls: list[str] = field(default_factory=list)
    model: str | None = None
    request_timeout: float = 900.0
    refresh_s: float = 10.0
    cooldown_s: float = 60.0
    # Admission control: at most this many *active* conversations per endpoint from this
    # process (None: unlimited). Keeps the live conversations' KV prefixes resident in the
    # server's prefix cache; beyond that, cached blocks are evicted between agent turns and
    # every step re-prefills its whole context.
    per_endpoint_cap: int | None = None
    idle_release_s: float = 120.0
    # Optional live override: JSON file {cap_key: cap} re-read on every scan, so admission
    # can be tuned for running processes (default <endpoints_dir>/../admission.json).
    cap_key: str | None = None
    cap_file: Path | None = None
    _known: dict[str, Endpoint] = field(default_factory=dict)
    _down_until: dict[str, float] = field(default_factory=dict)
    _last_scan: float = 0.0

    def _client_for(self, url: str) -> Any:
        return openai.AsyncOpenAI(
            api_key="EMPTY", base_url=url, max_retries=0, timeout=self.request_timeout
        )

    def scan(self, force: bool = False) -> list[Endpoint]:
        now = time.monotonic()
        if not force and now - self._last_scan < self.refresh_s and self._known:
            return self.live()
        self._last_scan = now
        if self.cap_key and self.cap_file is not None and self.cap_file.is_file():
            try:
                caps = json.loads(self.cap_file.read_text())
                if self.cap_key in caps and int(caps[self.cap_key]) != self.per_endpoint_cap:
                    logger.warning("admission cap %s: %s -> %s", self.cap_key,
                                   self.per_endpoint_cap, caps[self.cap_key])
                    self.per_endpoint_cap = int(caps[self.cap_key])
            except (OSError, ValueError, TypeError):
                pass
        found: dict[str, tuple[str, str]] = {}
        for url in self.static_urls:
            found[url] = (url, self.model or "")
        if self.endpoints_dir is not None and self.endpoints_dir.is_dir():
            for path in sorted(self.endpoints_dir.glob("*.json")):
                try:
                    data = json.loads(path.read_text())
                except (OSError, json.JSONDecodeError):
                    continue
                if self.model and data.get("model") != self.model:
                    continue
                found[data["url"]] = (path.stem, data.get("model", ""))
        for url, (name, model) in found.items():
            if url not in self._known:
                self._known[url] = Endpoint(url, model, name, self._client_for(url))
                logger.info("endpoint up: %s (%s)", url, name)
        for url in list(self._known):
            if url not in found:
                logger.info("endpoint gone: %s", url)
                del self._known[url]
        return self.live()

    def live(self) -> list[Endpoint]:
        now = time.monotonic()
        return [e for u, e in self._known.items() if self._down_until.get(u, 0.0) <= now]

    def mark_down(self, endpoint: Endpoint) -> None:
        self._down_until[endpoint.url] = time.monotonic() + self.cooldown_s

    def pick(self, exclude: Endpoint | None = None, *, admit: bool = True) -> Endpoint | None:
        live = self.scan()
        if exclude is not None and len(live) > 1:
            live = [e for e in live if e.url != exclude.url]
        if not live:
            return None
        loads = {e.url: e.active(self.idle_release_s) for e in live}
        if admit and self.per_endpoint_cap is not None:
            live = [e for e in live if loads[e.url] < self.per_endpoint_cap]
            if not live:
                return None
        low = min(loads[e.url] for e in live)
        return random.choice([e for e in live if loads[e.url] == low])


def _retryable(exc: BaseException) -> bool:
    if isinstance(exc, (openai.APIConnectionError, openai.APITimeoutError, openai.RateLimitError)):
        return True
    if isinstance(exc, openai.APIStatusError):
        return exc.status_code >= 500 or exc.status_code == 404  # 404: model gone after restart
    return False


class _Completions:
    def __init__(self, owner: PooledClient) -> None:
        self._owner = owner

    async def create(self, **kwargs: Any) -> Any:
        return await self._owner._create(**kwargs)


class PooledClient:
    """Sticky AsyncOpenAI-like client over an :class:`EndpointPool`."""

    def __init__(
        self,
        pool: EndpointPool,
        *,
        model: str,
        max_tokens: int,
        extra_body: dict[str, Any] | None,
        wait_for_endpoint_s: float,
        max_transient_retries: int,
    ) -> None:
        self.pool = pool
        self.model = model
        self.max_tokens = max_tokens
        self.extra_body = extra_body or {}
        self.wait_for_endpoint_s = wait_for_endpoint_s
        self.max_transient_retries = max_transient_retries
        self.endpoint: Endpoint | None = None
        self.in_flight = False
        self.last_done = time.monotonic()
        self.chat = SimpleNamespace(completions=_Completions(self))

    def _release(self) -> None:
        if self.endpoint is not None:
            self.endpoint.pinned.discard(self)
            self.endpoint = None

    def release(self) -> None:
        """Give up this client's admission slot (call when its agent has finished)."""
        self._release()

    async def _acquire(self, exclude: Endpoint | None = None, *, admit: bool = True) -> Endpoint:
        deadline = time.monotonic() + self.wait_for_endpoint_s
        waited = 0.0
        while True:
            ep = self.pool.pick(exclude=exclude, admit=admit)
            if ep is not None:
                self._release()
                ep.pinned.add(self)
                self.endpoint = ep
                return ep
            if time.monotonic() > deadline:
                raise RuntimeError("no LLM endpoint available")
            if not self.pool.live():
                if waited == 0.0:
                    logger.warning("no live LLM endpoint; waiting")
                delay = 15.0
            else:  # all endpoints at their admission cap: wait for a slot
                delay = 2.0
            waited += delay
            await asyncio.sleep(delay)
            if not self.pool.live():
                self.pool.scan(force=True)

    async def _create(self, **kwargs: Any) -> Any:
        kwargs["model"] = self.model
        kwargs.setdefault("max_tokens", self.max_tokens)
        if self.extra_body:
            body = dict(self.extra_body)
            body.update(kwargs.get("extra_body") or {})
            kwargs["extra_body"] = body
        meter = USAGE.get()
        attempt = 0
        while True:
            ep = self.endpoint or await self._acquire()
            started = time.monotonic()
            self.in_flight = True
            try:
                response = await ep.client.chat.completions.create(**kwargs)
            except Exception as exc:
                self.in_flight = False
                self.last_done = time.monotonic()
                if meter is not None:
                    meter.errors += 1
                if not _retryable(exc) or attempt >= self.max_transient_retries:
                    raise
                attempt += 1
                logger.warning("LLM call failed on %s (%s); retry %d", ep.url,
                               type(exc).__name__, attempt)
                if isinstance(exc, (openai.APIConnectionError, openai.APIStatusError)) and not isinstance(
                    exc, openai.RateLimitError
                ):
                    self.pool.mark_down(ep)
                    await self._acquire(exclude=ep, admit=False)
                await asyncio.sleep(min(2.0 * attempt, 30.0))
                continue
            self.in_flight = False
            self.last_done = time.monotonic()
            if meter is not None:
                meter.add(getattr(response, "usage", None), time.monotonic() - started)
            return response


def make_pool_client_factory(
    model: str,
    *,
    endpoints_dir: Path | str | None = DEFAULT_ENDPOINTS_DIR,
    static_urls: list[str] | None = None,
    max_tokens: int = 4096,
    disable_thinking: bool = True,
    wait_for_endpoint_s: float = 6 * 3600.0,
    max_transient_retries: int = 30,
    request_timeout: float = 900.0,
    per_endpoint_cap: int | None = None,
    cap_key: str | None = None,
) -> Callable[[], PooledClient]:
    """A factory of sticky pooled clients (see module docstring)."""
    pool = EndpointPool(
        endpoints_dir=Path(endpoints_dir) if endpoints_dir is not None else None,
        static_urls=list(static_urls or []),
        model=model,
        request_timeout=request_timeout,
        per_endpoint_cap=per_endpoint_cap,
        cap_key=cap_key,
        cap_file=(Path(endpoints_dir).parent / "admission.json") if endpoints_dir is not None else None,
    )
    extra = {"chat_template_kwargs": {"enable_thinking": False}} if disable_thinking else None

    def factory() -> PooledClient:
        return PooledClient(
            pool,
            model=model,
            max_tokens=max_tokens,
            extra_body=extra,
            wait_for_endpoint_s=wait_for_endpoint_s,
            max_transient_retries=max_transient_retries,
        )

    factory.pool = pool  # type: ignore[attr-defined]
    return factory


__all__ = [
    "DEFAULT_ENDPOINTS_DIR",
    "EndpointPool",
    "PooledClient",
    "USAGE",
    "UsageMeter",
    "make_pool_client_factory",
    "metered",
]
