"""Tests for graph/workers/pickmerger.py."""
from __future__ import annotations

import asyncio

import pytest

from ace.graph import prompts
from ace.graph.ace import AceError, ace_instructions, AceFormat
from ace.graph.edits import Edit
from ace.graph.graph import WorkerNode
from ace.graph.properties import default_properties, resolve
from ace.graph.runner import ModelSettings, RunContext, WorkerInputs
from ace.graph.types import GraphFailure, WorkerType
from ace.graph.views import Level, View
from ace.graph.workers import WORKERS
from ace.graph.workers.pickmerger import PickMergerWorker
from fakes import FakeClient, system_prompt_of, text_message
from sample_project import SAMPLE_FILES

T = WorkerType
UTILS = "pkg/utils.py"
ORIGINAL_LINE = "return max(low, min(high, value))"

MERGE_REPLY = (
    "Combining both candidates.\n"
    "```python\n"
    "@@ pkg/utils.py::clamp\n"
    "def clamp(value, low, high):\n"
    "    if low > high:\n"
    "        raise ValueError('MERGED bounds')\n"
    "    return min(max(value, low), high)\n"
    "```\n"
)


def make_ctx(session, messages=()) -> tuple[RunContext, list[FakeClient]]:
    clients: list[FakeClient] = []

    def factory() -> FakeClient:
        client = FakeClient(list(messages))
        clients.append(client)
        return client

    settings = ModelSettings(model="fake-model")
    return RunContext(session, "Make clamp validate its bounds.", settings, factory, run_id="r"), clients


def props() -> dict:
    return resolve(T.PICKMERGER, default_properties(T.PICKMERGER))


def clamp_edit(marker: str) -> Edit:
    text = SAMPLE_FILES[UTILS].replace(ORIGINAL_LINE, f"return min(max(value, low), high)  # {marker}")
    return Edit(files={UTILS: text.encode()})


def make_inputs(edits: dict[int, Edit] | None = None) -> WorkerInputs:
    view = View()
    view.add_element(UTILS, "clamp", Level.FULL)
    if edits is None:
        edits = {2: clamp_edit("TWO"), 3: clamp_edit("THREE")}
    return WorkerInputs(
        views={1: view},
        edits=edits,
        reports={2: "Editor two's account."},
        source_types={1: T.VIEWER, 2: T.EDITOR, 3: T.CRITIC},
    )


def run(session, reply: str, inputs: WorkerInputs | None = None):
    ctx, clients = make_ctx(session, [text_message(reply)])
    node = WorkerNode(7, T.PICKMERGER, default_properties(T.PICKMERGER))
    inputs = inputs or make_inputs()
    out = asyncio.run(PickMergerWorker().run(ctx, node, inputs, props()))
    return out, inputs, clients


def test_registered():
    assert WORKERS[T.PICKMERGER] is PickMergerWorker


def test_prompt_and_merged_edit(session):
    out, inputs, clients = run(session, MERGE_REPLY)
    request = clients[0].requests[0]
    user = request["messages"][-1]["content"]
    assert "def clamp" in user
    c1 = user.index(prompts.CANDIDATE_LABEL.format(n=1, id=2))
    c2 = user.index(prompts.CANDIDATE_LABEL.format(n=2, id=3))
    assert c1 < c2
    assert "# TWO" in user[c1:c2] and "# THREE" in user[c2:]
    assert prompts.REPORT_LABEL.format(id=2, type="editor") in user
    assert "Editor two's account." in user
    system = system_prompt_of(request)
    assert prompts.ACE_PLACEHOLDER not in system
    assert ace_instructions(AceFormat(props()["ace_format"])) in system
    assert not request.get("tools")

    assert out.report is None
    assert out.view is not None and out.view.files[UTILS].levels.get("clamp") == Level.FULL
    assert len(out.rollout_ids) == 1
    assert out.edit is not None
    assert all(out.edit is not edit for edit in inputs.edits.values())
    content = out.edit.files[UTILS].decode()
    assert "MERGED bounds" in content
    assert "# TWO" not in content and "# THREE" not in content


def test_no_ace_block(session):
    with pytest.raises(AceError):
        run(session, "Candidate 2 is best.")


def test_requires_two_edits(session):
    with pytest.raises(GraphFailure, match=r"node 7 \(pickmerger\)"):
        run(session, MERGE_REPLY, make_inputs({2: clamp_edit("TWO")}))


def test_requires_views(session):
    inputs = make_inputs()
    inputs.views = {}
    with pytest.raises(GraphFailure, match="view"):
        run(session, MERGE_REPLY, inputs)
