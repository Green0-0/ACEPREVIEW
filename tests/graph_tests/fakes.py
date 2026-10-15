"""Shared fakes for graph and neat tests.

``FakeClient`` mirrors ``tests/core_tests/test_lifecycle.py`` so workers can be driven
without network access. ``ScriptedClient`` returns canned assistant messages in order;
``RoutingClient`` picks a script per worker by inspecting the system prompt.
"""
from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace
from typing import Callable, Iterable


class FakeFunction:
    def __init__(self, name: str, arguments: str):
        self.name = name
        self.arguments = arguments


class FakeToolCall:
    def __init__(self, call_id: str, name: str, arguments: str):
        self.id = call_id
        self.type = "function"
        self.function = FakeFunction(name, arguments)


class FakeMessage:
    def __init__(self, *, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls

    def model_dump(self, exclude_unset=True):
        message = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            message["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.function.name,
                        "arguments": call.function.arguments,
                    },
                }
                for call in self.tool_calls
            ]
        return message


def tool_call_message(*calls: tuple[str, str], prefix: str = "call") -> FakeMessage:
    """Build an assistant message that calls tools: ``(name, json_arguments)`` pairs."""
    return FakeMessage(
        tool_calls=[FakeToolCall(f"{prefix}{i}", name, args) for i, (name, args) in enumerate(calls)]
    )


def text_message(content: str) -> FakeMessage:
    return FakeMessage(content=content)


class FakeClient:
    """Returns the scripted messages in order; records every request."""

    def __init__(self, messages: Iterable[FakeMessage]):
        self.messages = iter(messages)
        self.requests: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    async def create(self, **kwargs):
        self.requests.append(deepcopy(kwargs))
        try:
            message = next(self.messages)
        except StopIteration as exc:
            raise AssertionError("FakeClient ran out of scripted messages") from exc
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


ScriptedClient = FakeClient


class RoutingClient:
    """Chooses a script by predicate on the request (e.g. the system prompt).

    ``routes`` is a list of ``(predicate, messages)``; the first predicate that returns
    True for a request's kwargs selects that script, and the script is consumed in order
    across calls. Use it when one client factory must serve several workers.
    """

    def __init__(self, routes: list[tuple[Callable[[dict], bool], Iterable[FakeMessage]]]):
        self.routes = [(predicate, iter(messages)) for predicate, messages in routes]
        self.requests: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    async def create(self, **kwargs):
        self.requests.append(deepcopy(kwargs))
        for predicate, messages in self.routes:
            if predicate(kwargs):
                try:
                    message = next(messages)
                except StopIteration as exc:
                    raise AssertionError("RoutingClient script exhausted") from exc
                return SimpleNamespace(choices=[SimpleNamespace(message=message)])
        raise AssertionError("RoutingClient has no route for request")


def system_prompt_of(kwargs: dict) -> str:
    for message in kwargs.get("messages", []):
        if message.get("role") == "system":
            return message.get("content") or ""
    return ""
