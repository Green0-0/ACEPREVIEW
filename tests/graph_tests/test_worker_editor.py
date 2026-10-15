"""Tests for graph/workers/editor.py: the tool-using EDITOR worker."""
from __future__ import annotations

import asyncio
import json

import pytest

from ace.graph import prompts
from ace.graph.graph import WorkerNode
from ace.graph.properties import PROPERTIES, default_properties, resolve
from ace.graph.runner import ModelSettings, RunContext, WorkerInputs, WorkerOutputs
from ace.graph.types import GraphFailure, WorkerType
from ace.graph.views import Level, View
from ace.graph.workers import _REGISTRY
from ace.graph.workers.editor import EditorWorker
from fakes import FakeClient, system_prompt_of, text_message, tool_call_message

T = WorkerType
PROMPT = "Make clamp handle reversed bounds."

NEW_CLAMP = (
    "def clamp(value, low, high):\n"
    "    if low > high:\n"
    "        low, high = high, low\n"
    "    return max(low, min(high, value))\n"
)


def make_ctx(session, messages=()) -> tuple[RunContext, list[FakeClient]]:
    clients: list[FakeClient] = []

    def factory() -> FakeClient:
        client = FakeClient(list(messages))
        clients.append(client)
        return client

    settings = ModelSettings(model="fake-model", api_base_url="http://x", api_key_env_var="FAKE_KEY")
    return RunContext(session, PROMPT, settings, factory, run_id="r"), clients


def node_with(**overrides: int) -> WorkerNode:
    indices = default_properties(T.EDITOR)
    indices.update(overrides)
    return WorkerNode(11, T.EDITOR, indices)


def tool_names(request: dict) -> list[str]:
    return [tool["function"]["name"] for tool in request.get("tools") or []]


def user_prompt_of(request: dict) -> str:
    return next(m["content"] for m in request["messages"] if m["role"] == "user")


def run(ctx: RunContext, node: WorkerNode, inputs: WorkerInputs | None = None) -> WorkerOutputs:
    props = resolve(T.EDITOR, node.properties)
    return asyncio.run(EditorWorker().run(ctx, node, inputs or WorkerInputs(), props))


def slug_view() -> View:
    view = View()
    view.add_element("pkg/models.py", "Item.slug", Level.FULL)
    return view


def test_registered():
    assert _REGISTRY[T.EDITOR] is EditorWorker
    assert EditorWorker.type is T.EDITOR


def test_editor_edits_and_reports(session):
    script = [
        tool_call_message(("view", json.dumps({"paths": "pkg/utils.py::clamp"}))),
        tool_call_message(
            (
                "replace_element",
                json.dumps(
                    {"file_path": "pkg/utils.py", "element_path": "clamp", "new_content": NEW_CLAMP}
                ),
            ),
            prefix="b",
        ),
        text_message("Swapped reversed bounds in pkg/utils.py::clamp."),
    ]
    ctx, clients = make_ctx(session, script)
    node = node_with()
    props = resolve(T.EDITOR, node.properties)
    assert props["tool_set"] == "editor_standard"

    out = run(ctx, node)

    assert out.report == "Swapped reversed bounds in pkg/utils.py::clamp."
    assert out.edit is not None
    assert set(out.edit.files) == {"pkg/utils.py"}
    new_text = out.edit.files["pkg/utils.py"].decode()
    assert "low, high = high, low" in new_text
    assert "def slugify(text):" in new_text  # rest of the file kept
    diffs = out.edit.per_file_diff(session)
    assert [path for path, _ in diffs] == ["pkg/utils.py"]
    assert "+        low, high = high, low" in diffs[0][1]
    # The view holds only what was viewed (replace_element is not a view).
    assert out.view is not None
    assert set(out.view.files) == {"pkg/utils.py"}
    assert out.view.files["pkg/utils.py"].levels.get("clamp") == Level.FULL
    assert out.view.files["pkg/utils.py"].levels.get("slugify") is None
    assert len(out.rollout_ids) == 1 and out.rollout_ids[0]

    (client,) = clients
    assert len(client.requests) == 3
    first = client.requests[0]
    assert sorted(tool_names(first)) == sorted(prompts.TOOL_SETS["editor_standard"])
    system = system_prompt_of(first)
    assert props["role"].strip() in system
    assert "{{FILES*}}" not in system and "pkg/models.py" in system
    user = user_prompt_of(first)
    assert PROMPT in user
    assert prompts.VIEWS_HEADER not in user and prompts.REPORTS_HEADER not in user


def test_editor_tool_set_property(session):
    index = PROPERTIES[T.EDITOR]["tool_set"].options.index("editor_interactive")
    ctx, clients = make_ctx(session, [text_message("No change needed.")])
    out = run(ctx, node_with(tool_set=index))
    assert out.edit is not None and out.edit.is_empty()  # empty edit is not a failure
    assert out.report == "No change needed."
    assert sorted(tool_names(clients[0].requests[0])) == sorted(
        prompts.TOOL_SETS["editor_interactive"]
    )


def test_editor_renders_input_views(session):
    ctx, clients = make_ctx(session, [text_message("Read the slug code; no change.")])
    inputs = WorkerInputs(views={1: slug_view()}, source_types={1: T.VIEWER})
    out = run(ctx, node_with(), inputs)

    user = user_prompt_of(clients[0].requests[0])
    assert PROMPT in user
    assert prompts.VIEWS_HEADER in user
    assert "def slug" in user
    assert user.index(prompts.TASK_HEADER) < user.index(prompts.VIEWS_HEADER)
    assert prompts.REPORTS_HEADER not in user
    # Forwarded view includes what it was shown although it viewed nothing itself.
    assert out.view is not None
    assert out.view.files["pkg/models.py"].levels.get("Item.slug") == Level.FULL
    assert out.edit is not None and out.edit.is_empty()


def test_editor_renders_input_reports(session):
    ctx, clients = make_ctx(session, [text_message("Done.")])
    inputs = WorkerInputs(
        reports={5: "The slug lives in pkg/models.py::Item.slug."},
        source_types={5: T.VIEWER},
    )
    out = run(ctx, node_with(), inputs)

    user = user_prompt_of(clients[0].requests[0])
    assert prompts.REPORTS_HEADER in user
    assert (
        "REPORT from worker 5 (viewer)\nThe slug lives in pkg/models.py::Item.slug." in user
    )
    assert prompts.VIEWS_HEADER not in user
    assert out.report == "Done."
    assert out.view is not None and out.view.is_empty()


@pytest.mark.parametrize("reply", [None, "  "])
def test_editor_empty_report_fails(session, reply):
    ctx, _ = make_ctx(session, [text_message(reply)])  # type: ignore[arg-type]
    with pytest.raises(GraphFailure, match=r"node 11 \(editor\)"):
        run(ctx, node_with())
