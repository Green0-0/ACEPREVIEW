"""Tests for graph/workers/critic.py."""
from __future__ import annotations

import asyncio

import pytest

from ace.graph import prompts
from ace.graph.ace import AceError
from ace.graph.edits import Edit
from ace.graph.graph import WorkerNode
from ace.graph.properties import default_properties, resolve
from ace.graph.runner import ModelSettings, RunContext, WorkerInputs
from ace.graph.types import GraphFailure, WorkerType
from ace.graph.views import Level, View
from ace.graph.workers import WORKERS
from ace.graph.workers.critic import CriticWorker
from fakes import FakeClient, system_prompt_of, text_message
from sample_project import SAMPLE_FILES

T = WorkerType
UTILS = "pkg/utils.py"


def make_ctx(session, messages=()) -> tuple[RunContext, list[FakeClient]]:
    clients: list[FakeClient] = []

    def factory() -> FakeClient:
        client = FakeClient(list(messages))
        clients.append(client)
        return client

    settings = ModelSettings(model="fake-model")
    return RunContext(session, "Make clamp validate its bounds.", settings, factory, run_id="r"), clients


def props() -> dict:
    return resolve(T.CRITIC, default_properties(T.CRITIC))


def clamp_view() -> View:
    view = View()
    view.add_element(UTILS, "clamp", Level.FULL)
    return view


def input_edit(session) -> Edit:
    """The author's edit: clamp rewritten with an ``# AUTHOR`` marker."""
    text = SAMPLE_FILES[UTILS]
    new = text.replace(
        "return max(low, min(high, value))", "return min(max(value, low), high)  # AUTHOR"
    )
    assert new != text
    return Edit(files={UTILS: new.encode()})


CRITIC_REPLY = (
    "The candidate misses validation.\n"
    "```python\n"
    "@@ pkg/utils.py::clamp\n"
    "def clamp(value, low, high):\n"
    "    if low > high:\n"
    "        raise ValueError('CRITIC bounds')\n"
    "    return max(low, min(high, value))\n"
    "```\n"
)


def make_inputs(session, *, edits=None) -> WorkerInputs:
    return WorkerInputs(
        views={1: clamp_view()},
        edits={2: input_edit(session)} if edits is None else edits,
        reports={1: "Viewer notes about clamp.", 2: "I swapped min and max."},
        source_types={1: T.VIEWER, 2: T.EDITOR},
    )


def test_registered():
    assert WORKERS[T.CRITIC] is CriticWorker


def test_critic_prompt_and_output(session):
    ctx, clients = make_ctx(session, [text_message(CRITIC_REPLY)])
    node = WorkerNode(5, T.CRITIC, default_properties(T.CRITIC))
    inputs = make_inputs(session)
    out = asyncio.run(CriticWorker().run(ctx, node, inputs, props()))

    request = clients[0].requests[0]
    user = request["messages"][-1]["content"]
    assert request["messages"][-1]["role"] == "user"
    assert "def clamp" in user
    assert prompts.VIEWS_HEADER in user
    assert prompts.EDIT_UNDER_REVIEW_HEADER in user
    section = user.split(prompts.EDIT_UNDER_REVIEW_HEADER, 1)[1]
    assert section.lstrip("\n").startswith("from worker 2 (editor)")
    assert section.lstrip("\n").startswith("from worker 2 (editor)\n```diff\n")
    assert "+    return min(max(value, low), high)  # AUTHOR" in section
    author_label = prompts.REPORT_LABEL.format(id=2, type="editor")
    diff_pos = section.index("+    return min(max(value, low), high)")
    author_pos = section.index(author_label)
    assert diff_pos < author_pos
    assert section.index("I swapped min and max.") > author_pos
    # Other reports live under the reports header, after the edit section; the author's
    # report is not repeated there.
    reports = user.split(prompts.REPORTS_HEADER, 1)[1]
    assert "Viewer notes about clamp." in reports
    assert prompts.REPORT_LABEL.format(id=1, type="viewer") in reports
    assert "I swapped min and max." not in reports
    assert user.index(prompts.EDIT_UNDER_REVIEW_HEADER) < user.index(prompts.REPORTS_HEADER)

    system = system_prompt_of(request)
    assert prompts.ACE_PLACEHOLDER not in system
    assert "COMPLETE" in system

    assert out.report is None
    assert out.view is not None and out.view.files[UTILS].levels.get("clamp") == Level.FULL
    assert out.rollout_ids and len(out.rollout_ids) == 1
    assert out.edit is not None and out.edit is not inputs.edits[2]
    content = out.edit.files[UTILS].decode()
    assert "CRITIC bounds" in content
    assert "# AUTHOR" not in content
    assert set(out.edit.files) == {UTILS}
    assert clients[0].requests and len(clients[0].requests) == 1
    assert not request.get("tools")


def test_critic_no_author_report(session):
    ctx, clients = make_ctx(session, [text_message(CRITIC_REPLY)])
    node = WorkerNode(5, T.CRITIC, {})
    inputs = make_inputs(session)
    inputs.reports = {}
    asyncio.run(CriticWorker().run(ctx, node, inputs, props()))
    user = clients[0].requests[0]["messages"][-1]["content"]
    assert "REPORT from worker" not in user
    assert prompts.REPORTS_HEADER not in user


def test_critic_empty_input_edit(session):
    ctx, clients = make_ctx(session, [text_message(CRITIC_REPLY)])
    node = WorkerNode(5, T.CRITIC, {})
    inputs = make_inputs(session, edits={2: Edit()})
    out = asyncio.run(CriticWorker().run(ctx, node, inputs, props()))
    user = clients[0].requests[0]["messages"][-1]["content"]
    assert "from worker 2 (editor)\n(no changes)" in user
    assert b"CRITIC bounds" in out.edit.files[UTILS]


@pytest.mark.parametrize("count", [0, 2])
def test_critic_requires_exactly_one_edit(session, count):
    ctx, _ = make_ctx(session, [text_message(CRITIC_REPLY)])
    node = WorkerNode(5, T.CRITIC, {})
    edits = {sid: input_edit(session) for sid in range(2, 2 + count)}
    with pytest.raises(GraphFailure, match=r"node 5 \(critic\)"):
        asyncio.run(CriticWorker().run(ctx, node, make_inputs(session, edits=edits), props()))


def test_critic_requires_views(session):
    ctx, _ = make_ctx(session, [text_message(CRITIC_REPLY)])
    node = WorkerNode(5, T.CRITIC, {})
    inputs = make_inputs(session)
    inputs.views = {}
    with pytest.raises(GraphFailure, match="view"):
        asyncio.run(CriticWorker().run(ctx, node, inputs, props()))


def test_critic_no_ace_block(session):
    ctx, _ = make_ctx(session, [text_message("Looks good to me, no changes needed.")])
    node = WorkerNode(5, T.CRITIC, {})
    with pytest.raises(AceError):
        asyncio.run(CriticWorker().run(ctx, node, make_inputs(session), props()))
