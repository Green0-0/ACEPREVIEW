"""Spec-compliance audit checks that span every worker type.

Requirement 15: every worker receives the original task prompt in its first user message.
Requirement 16: only VIEWER and EDITOR get tools, never a forbidden one (delegate,
message_subagent, cancel_subagent, bash, tools); tool-less workers send no tool schema.
The graphs and scripted replies are the end-to-end ones; together they cover all seven
worker types.
"""
from __future__ import annotations

import pytest

import test_end_to_end as e2e
from ace.graph import prompts
from ace.graph.types import WorkerType

T = WorkerType


def _pick_routes():
    merged = e2e.ace((f"{e2e.UTILS}::clamp", e2e.EDITOR_CLAMP))
    first = prompts.INSTRUCTIONS[T.EDITOR][0]
    second = prompts.INSTRUCTIONS[T.EDITOR][e2e.SECOND_INSTRUCTION]
    return [
        (e2e.is_type(T.EDITOR, user_contains=first),
         e2e.editor_script(e2e.EDITOR_CLAMP, "first", prefix="x")),
        (e2e.is_type(T.EDITOR, user_contains=second),
         e2e.editor_script(e2e.SECOND_EDITOR_CLAMP, "second", prefix="y")),
        (e2e.is_type(T.PICKMERGER), [e2e.text_message(merged)]),
    ]


def _critic_routes():
    return [
        (e2e.is_type(T.EDITOR), e2e.editor_script(e2e.EDITOR_CLAMP, e2e.EDITOR_REPORT)),
        (e2e.is_type(T.CRITIC), [e2e.text_message(e2e.ace((f"{e2e.UTILS}::clamp", e2e.CRITIC_CLAMP)))]),
    ]


CASES = {
    "chain": (e2e.chain_graph, e2e.chain_routes),
    "diamond": (e2e.diamond_graph, e2e.diamond_routes),
    "critic": (e2e.critic_graph, _critic_routes),
    "pick": (e2e.pick_graph, _pick_routes),
}


def _type_of(kwargs: dict) -> WorkerType:
    (match,) = [t for t in WorkerType if e2e.marker(t) in e2e.system_prompt_of(kwargs)]
    return match


def test_cases_cover_every_worker_type():
    covered = {n.type for build, _ in CASES.values() for n in build().nodes.values()}
    assert covered == set(WorkerType)


@pytest.mark.parametrize("name", sorted(CASES))
def test_every_worker_gets_prompt_and_only_permitted_tools(session, name):
    build, routes = CASES[name]
    ctx, client = e2e.make_ctx(session, routes())
    e2e.run_graph(build(), ctx)
    assert client.requests
    for kwargs in client.requests:
        worker_type = _type_of(kwargs)
        assert e2e.PROMPT in e2e.first_user(kwargs), worker_type
        names = {t["function"]["name"] for t in kwargs.get("tools") or []}
        assert not names & prompts.FORBIDDEN_TOOLS, worker_type
        if worker_type in (T.VIEWER, T.EDITOR):
            assert names
        else:
            assert not names, worker_type
