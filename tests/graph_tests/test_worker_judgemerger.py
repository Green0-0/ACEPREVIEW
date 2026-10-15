"""Tests for graph/workers/judgemerger.py."""
from __future__ import annotations

import asyncio

import pytest

from ace.graph import prompts
from ace.graph.edits import Edit
from ace.graph.graph import WorkerNode
from ace.graph.properties import default_properties, resolve
from ace.graph.runner import ModelSettings, RunContext, WorkerInputs
from ace.graph.types import GraphFailure, WorkerType
from ace.graph.views import Level, View
from ace.graph.workers import WORKERS
from ace.graph.workers.judgemerger import JudgeMergerWorker
from fakes import FakeClient, system_prompt_of, text_message
from sample_project import SAMPLE_FILES

T = WorkerType
UTILS = "pkg/utils.py"
ORIGINAL_LINE = "return max(low, min(high, value))"


def make_ctx(session, messages=()) -> tuple[RunContext, list[FakeClient]]:
    clients: list[FakeClient] = []

    def factory() -> FakeClient:
        client = FakeClient(list(messages))
        clients.append(client)
        return client

    settings = ModelSettings(model="fake-model")
    return RunContext(session, "Make clamp validate its bounds.", settings, factory, run_id="r"), clients


def props() -> dict:
    return resolve(T.JUDGEMERGER, default_properties(T.JUDGEMERGER))


def clamp_edit(marker: str) -> Edit:
    text = SAMPLE_FILES[UTILS].replace(ORIGINAL_LINE, f"return min(max(value, low), high)  # {marker}")
    return Edit(files={UTILS: text.encode()})


def make_inputs(edits: dict[int, Edit] | None = None) -> WorkerInputs:
    view = View()
    view.add_element(UTILS, "clamp", Level.FULL)
    if edits is None:
        edits = {3: clamp_edit("THREE"), 2: clamp_edit("TWO")}
    return WorkerInputs(
        views={1: view},
        edits=edits,
        reports={1: "Viewer notes about clamp."},
        source_types={1: T.VIEWER, 2: T.EDITOR, 3: T.CRITIC},
    )


def run(session, reply: str, inputs: WorkerInputs | None = None, worker=None):
    ctx, clients = make_ctx(session, [text_message(reply)])
    node = WorkerNode(6, T.JUDGEMERGER, default_properties(T.JUDGEMERGER))
    worker = worker or JudgeMergerWorker()
    inputs = inputs or make_inputs()
    out = asyncio.run(worker.run(ctx, node, inputs, props()))
    return out, inputs, clients, worker


def test_registered():
    assert WORKERS[T.JUDGEMERGER] is JudgeMergerWorker


def test_prompt_lists_candidates(session):
    out, inputs, clients, worker = run(session, "2")
    request = clients[0].requests[0]
    user = request["messages"][-1]["content"]
    assert "def clamp" in user
    assert prompts.CANDIDATES_HEADER in user
    c1 = user.index(prompts.CANDIDATE_LABEL.format(n=1, id=2))
    c2 = user.index(prompts.CANDIDATE_LABEL.format(n=2, id=3))
    assert c1 < c2
    assert "+    return min(max(value, low), high)  # TWO" in user[c1:c2]
    assert "+    return min(max(value, low), high)  # THREE" in user[c2:]
    assert "Viewer notes about clamp." in user
    assert user.index(prompts.CANDIDATES_HEADER) < user.index(prompts.REPORTS_HEADER)
    system = system_prompt_of(request)
    assert "candidate number" in system
    assert not request.get("tools")
    assert out.report is None
    assert out.view is not None and out.view.files[UTILS].levels.get("clamp") == Level.FULL
    assert len(out.rollout_ids) == 1


def test_reply_two_picks_source_three(session):
    out, inputs, _, worker = run(session, "2")
    assert out.edit is inputs.edits[3]
    assert worker.last_choice == 3


def test_reply_sentence_picks_first(session):
    out, inputs, _, worker = run(session, "Candidate 1 is best.")
    assert out.edit is inputs.edits[2]
    assert worker.last_choice == 2


@pytest.mark.parametrize("reply", ["7", "0", "none"])
def test_bad_reply_fails(session, reply):
    with pytest.raises(GraphFailure, match=r"node 6 \(judgemerger\)") as info:
        run(session, reply)
    assert repr(reply) in str(info.value)


def test_requires_two_edits(session):
    with pytest.raises(GraphFailure, match="two input edits"):
        run(session, "1", make_inputs({2: clamp_edit("TWO")}))


def test_requires_views(session):
    inputs = make_inputs()
    inputs.views = {}
    with pytest.raises(GraphFailure, match="view"):
        run(session, "1", inputs)
