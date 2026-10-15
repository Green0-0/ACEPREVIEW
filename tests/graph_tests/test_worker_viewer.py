"""Tests for graph/workers/viewer.py: the tool-using, read-only VIEWER worker."""
from __future__ import annotations

import asyncio
import json

import pytest

from ace.graph import prompts
from ace.graph.graph import WorkerNode
from ace.graph.properties import PROPERTIES, default_properties, resolve
from ace.graph.runner import ModelSettings, RunContext, WorkerInputs, WorkerOutputs
from ace.graph.types import GraphFailure, WorkerType
from ace.graph.views import Level
from ace.graph.workers import _REGISTRY
from ace.graph.workers.viewer import ViewerWorker
from fakes import FakeClient, system_prompt_of, text_message, tool_call_message

T = WorkerType
PROMPT = "Make slug lowercase."


def make_ctx(session, messages=()) -> tuple[RunContext, list[FakeClient]]:
    clients: list[FakeClient] = []

    def factory() -> FakeClient:
        client = FakeClient(list(messages))
        clients.append(client)
        return client

    settings = ModelSettings(model="fake-model", api_base_url="http://x", api_key_env_var="FAKE_KEY")
    return RunContext(session, PROMPT, settings, factory, run_id="r"), clients


def option_index(name: str, value: object) -> int:
    return PROPERTIES[T.VIEWER][name].options.index(value)


def node_with(**overrides: int) -> WorkerNode:
    indices = default_properties(T.VIEWER)
    indices.update(overrides)
    return WorkerNode(4, T.VIEWER, indices)


def tool_names(request: dict) -> list[str]:
    return [tool["function"]["name"] for tool in request.get("tools") or []]


def user_prompt_of(request: dict) -> str:
    return next(m["content"] for m in request["messages"] if m["role"] == "user")


def run(ctx: RunContext, node: WorkerNode) -> WorkerOutputs:
    props = resolve(T.VIEWER, node.properties)
    return asyncio.run(ViewerWorker().run(ctx, node, WorkerInputs(), props))


def test_registered():
    assert _REGISTRY[T.VIEWER] is ViewerWorker
    assert ViewerWorker.type is T.VIEWER


def test_viewer_records_view_and_report(session):
    script = [
        tool_call_message(("view", json.dumps({"paths": "pkg/models.py::Item.slug"}))),
        tool_call_message(("view_file", json.dumps({"file_path": "pkg/utils.py"})), prefix="b"),
        text_message("Item.slug in pkg/models.py calls slugify."),
    ]
    ctx, clients = make_ctx(session, script)
    node = node_with(tool_set=option_index("tool_set", "viewer_full"), role=2)
    props = resolve(T.VIEWER, node.properties)

    out = run(ctx, node)

    assert out.report == "Item.slug in pkg/models.py calls slugify."
    assert out.edit is None
    assert out.view is not None
    assert out.view.files["pkg/models.py"].levels.get("Item.slug") == Level.FULL
    assert out.view.files["pkg/utils.py"].levels.get("") == Level.COLLAPSED
    assert len(out.rollout_ids) == 1 and out.rollout_ids[0]

    (client,) = clients
    assert len(client.requests) == 3
    first = client.requests[0]
    assert sorted(tool_names(first)) == sorted(prompts.TOOL_SETS["viewer_full"])
    system = system_prompt_of(first)
    assert props["role"].strip() in system
    assert "{{FILES*}}" not in system and "pkg/models.py" in system
    user = user_prompt_of(first)
    assert PROMPT in user
    assert user.startswith(props["instruction"].strip())
    assert prompts.VIEWS_HEADER not in user and prompts.REPORTS_HEADER not in user
    assert first["model"] == "fake-model"


def test_viewer_tool_set_has_no_editing_tools(session):
    ctx, clients = make_ctx(session, [text_message("Nothing to see.")])
    node = node_with(tool_set=option_index("tool_set", "viewer_minimal"))
    out = run(ctx, node)
    assert out.report == "Nothing to see."
    assert out.view is not None and out.view.is_empty()
    names = tool_names(clients[0].requests[0])
    assert sorted(names) == sorted(prompts.TOOL_SETS["viewer_minimal"])
    assert not any(name.startswith(("replace_", "open_")) for name in names)
    assert "make_empty_files" not in names and "move_files" not in names


@pytest.mark.parametrize("reply", [None, "", "   \n"])
def test_viewer_empty_report_fails(session, reply):
    ctx, _ = make_ctx(session, [text_message(reply)])  # type: ignore[arg-type]
    with pytest.raises(GraphFailure, match=r"node 4 \(viewer\)"):
        run(ctx, node_with())
