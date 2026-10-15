"""Tests for graph/workers/vieweditor.py: rendered views in, ACE edit out."""
from __future__ import annotations

import asyncio

import pytest

from ace.graph import prompts
from ace.graph.ace import AceError, AceFormat, ace_instructions
from ace.graph.graph import WorkerNode
from ace.graph.properties import PROPERTIES, default_properties, resolve
from ace.graph.runner import ModelSettings, RunContext, WorkerInputs
from ace.graph.types import GraphFailure, WorkerType
from ace.graph.views import Level, View
from ace.graph.workers import WORKERS
from ace.graph.workers.vieweditor import ViewEditorWorker
from fakes import FakeClient, system_prompt_of, text_message

T = WorkerType

NEW_CLAMP = "def clamp(value, low, high):\n    return min(high, max(low, value))  # reordered\n"
ACE_REPLY = f"I will reorder the calls.\n```python\n@@ pkg/utils.py::clamp\n{NEW_CLAMP}```\n"


def make_ctx(session, messages) -> tuple[RunContext, list[FakeClient]]:
    clients: list[FakeClient] = []

    def factory() -> FakeClient:
        client = FakeClient(list(messages))
        clients.append(client)
        return client

    return RunContext(session, "Reorder clamp.", ModelSettings(model="fake"), factory, run_id="r"), clients


def clamp_inputs(reports: dict[int, str] | None = None) -> WorkerInputs:
    view = View()
    view.add_element("pkg/utils.py", "clamp", Level.FULL)
    return WorkerInputs(
        views={2: view},
        reports=dict(reports or {}),
        source_types={2: T.VIEWER},
    )


def props_with(ace_format: str = "element") -> tuple[WorkerNode, dict]:
    indices = default_properties(T.VIEWEDITOR)
    indices["ace_format"] = PROPERTIES[T.VIEWEDITOR]["ace_format"].options.index(ace_format)
    node = WorkerNode(6, T.VIEWEDITOR, indices)
    return node, resolve(T.VIEWEDITOR, indices)


def run_worker(session, messages, inputs=None, ace_format="element"):
    ctx, clients = make_ctx(session, messages)
    node, props = props_with(ace_format)
    outputs = asyncio.run(
        ViewEditorWorker().run(ctx, node, inputs if inputs is not None else clamp_inputs(), props)
    )
    return outputs, clients


def test_registered():
    assert WORKERS[T.VIEWEDITOR] is ViewEditorWorker


def test_ace_edit(session):
    outputs, clients = run_worker(session, [text_message(ACE_REPLY)])
    request = clients[0].requests[0]
    assert len(clients[0].requests) == 1
    assert not request.get("tools")
    user = [m["content"] for m in request["messages"] if m["role"] == "user"][0]
    assert prompts.VIEWS_HEADER in user
    assert "def clamp" in user
    assert "Reorder clamp." in user
    assert prompts.REPORTS_HEADER not in user

    system = system_prompt_of(request)
    assert "@@ " in system
    assert ace_instructions(AceFormat.ELEMENT) in system
    assert prompts.ACE_PLACEHOLDER not in system

    new_utils = outputs.edit.files["pkg/utils.py"].decode()
    assert "return min(high, max(low, value))  # reordered" in new_utils
    assert "def slugify" in new_utils
    assert outputs.view.files["pkg/utils.py"].levels["clamp"] == Level.FULL
    assert outputs.report is None
    assert len(outputs.rollout_ids) == 1


def test_reports_are_included(session):
    inputs = clamp_inputs({2: "clamp has its min/max nested the other way round."})
    _, clients = run_worker(session, [text_message(ACE_REPLY)], inputs)
    user = [m["content"] for m in clients[0].requests[0]["messages"] if m["role"] == "user"][0]
    assert prompts.REPORTS_HEADER in user
    assert "nested the other way round" in user
    assert prompts.REPORT_LABEL.format(id=2, type="viewer") in user


def test_reply_without_ace_block(session):
    with pytest.raises(AceError):
        run_worker(session, [text_message("I think clamp is fine as it is.")])
    assert issubclass(AceError, GraphFailure)


def test_format_gate_rejects_whole_file_header(session):
    reply = "```\n@@ pkg/utils.py\nprint('whole file')\n```"
    with pytest.raises(AceError):
        run_worker(session, [text_message(reply)], ace_format="element")


def test_whole_file_header_allowed_by_element_file(session):
    reply = "```\n@@ pkg/utils.py\nVALUE = 1\n```"
    outputs, clients = run_worker(session, [text_message(reply)], ace_format="element_file")
    assert outputs.edit.files["pkg/utils.py"].decode().strip() == "VALUE = 1"
    assert ace_instructions(AceFormat.ELEMENT_FILE) in system_prompt_of(clients[0].requests[0])


def test_no_input_views_raises(session):
    with pytest.raises(GraphFailure, match="no input code"):
        run_worker(session, [text_message(ACE_REPLY)], WorkerInputs())
