"""Tests for graph/workers/listviewer.py: candidate listing, selection parsing, report turn."""
from __future__ import annotations

import asyncio

import pytest

from ace.graph import prompts
from ace.graph.graph import WorkerNode
from ace.graph.properties import default_properties, resolve
from ace.graph.runner import ModelSettings, RunContext, WorkerInputs
from ace.graph.types import GraphFailure, WorkerType
from ace.graph.views import Level, View
from ace.graph.workers import WORKERS
from ace.graph.workers.listviewer import ListViewerWorker, parse_selection, selection_entries
from fakes import FakeClient, text_message

T = WorkerType

SELECTION = "Here is my pick:\n```\npkg/utils.py::slugify\npkg/models.py\n```\n"
REPORT = "slugify lowercases text and joins words with dashes; Item.slug delegates to it."


def make_ctx(session, messages) -> tuple[RunContext, list[FakeClient]]:
    clients: list[FakeClient] = []

    def factory() -> FakeClient:
        client = FakeClient(list(messages))
        clients.append(client)
        return client

    settings = ModelSettings(model="fake-model")
    return RunContext(session, "Make slugs use underscores.", settings, factory, run_id="r"), clients


def slug_inputs() -> WorkerInputs:
    view = View()
    view.add_element("pkg/models.py", "Item.slug", Level.FULL)
    return WorkerInputs(views={1: view}, source_types={1: T.VIEWER})


def run_worker(session, messages, inputs=None):
    ctx, clients = make_ctx(session, messages)
    node = WorkerNode(4, T.LISTVIEWER, default_properties(T.LISTVIEWER))
    props = resolve(T.LISTVIEWER, node.properties)
    outputs = asyncio.run(ListViewerWorker().run(ctx, node, inputs or slug_inputs(), props))
    return outputs, clients


def user_messages(request: dict) -> list[str]:
    return [m["content"] for m in request["messages"] if m.get("role") == "user"]


def test_registered():
    assert WORKERS[T.LISTVIEWER] is ListViewerWorker


def test_select_then_report(session):
    outputs, clients = run_worker(session, [text_message(SELECTION), text_message(REPORT)])
    client = clients[0]
    assert len(client.requests) == 2

    first = user_messages(client.requests[0])
    assert len(first) == 1
    assert prompts.LISTVIEWER_CANDIDATES_HEADER in first[0]
    assert "Make slugs use underscores." in first[0]
    candidate_lines = first[0].splitlines()
    assert any(line.startswith("pkg/utils.py::slugify") for line in candidate_lines)
    # the seed itself is ranked first, the referenced slugify right after it
    listed = [line.split()[0] for line in candidate_lines if "::" in line]
    assert listed.index("pkg/models.py::Item.slug") < listed.index("pkg/utils.py::slugify")
    assert listed.index("pkg/utils.py::slugify") < listed.index("pkg/models.py::Cart.total")

    second = user_messages(client.requests[1])
    assert len(second) == 2
    assert second[1].startswith(prompts.LISTVIEWER_FOLLOWUP)
    assert "def slugify" in second[1]

    assert outputs.report == REPORT
    assert outputs.edit is None
    assert len(outputs.rollout_ids) == 1
    view = outputs.view
    assert view.files["pkg/utils.py"].levels["slugify"] == Level.FULL
    assert view.files["pkg/models.py"].levels[""] >= Level.COLLAPSED


def test_system_prompt_has_no_tools(session):
    _, clients = run_worker(session, [text_message(SELECTION), text_message(REPORT)])
    request = clients[0].requests[0]
    assert not request.get("tools")


def test_unresolvable_entries_are_dropped(session):
    reply = (
        "```text\n"
        "- `pkg/utils.py::slugify`\n"
        "- pkg/missing.py\n"
        "- pkg/utils.py::nope\n"
        "- notes.txt::Something\n"
        "```"
    )
    outputs, clients = run_worker(session, [text_message(reply), text_message(REPORT)])
    view = outputs.view
    assert view.files["pkg/utils.py"].levels["slugify"] == Level.FULL
    assert "pkg/missing.py" not in view.files
    followup = user_messages(clients[0].requests[1])[1]
    assert "def slugify" in followup


def test_all_unresolvable_raises(session):
    reply = "```\npkg/missing.py\npkg/utils.py::nope\n```"
    with pytest.raises(GraphFailure, match="no selected location"):
        run_worker(session, [text_message(reply), text_message(REPORT)])


def test_reply_without_code_block(session):
    reply = "1. pkg/utils.py::clamp  def clamp(value, low, high):\n2. `pkg/service.py`\n"
    outputs, clients = run_worker(session, [text_message(reply), text_message(REPORT)])
    assert outputs.view.files["pkg/utils.py"].levels["clamp"] == Level.FULL
    assert outputs.view.files["pkg/service.py"].levels[""] >= Level.COLLAPSED
    assert "def clamp" in user_messages(clients[0].requests[1])[1]


def test_empty_report_raises(session):
    with pytest.raises(GraphFailure, match="empty report"):
        run_worker(session, [text_message(SELECTION), text_message("   \n")])


def test_empty_input_views_still_list_candidates(session):
    inputs = WorkerInputs(views={1: View()}, source_types={1: T.VIEWER})
    outputs, clients = run_worker(session, [text_message(SELECTION), text_message(REPORT)], inputs)
    first = user_messages(clients[0].requests[0])[0]
    assert prompts.LISTVIEWER_CANDIDATES_HEADER in first
    assert "pkg/utils.py" in first
    assert outputs.report == REPORT


def test_empty_selection_falls_back_to_ranked_listing(session):
    ctx, clients = make_ctx(session, [text_message(SELECTION), text_message(REPORT)])
    node = WorkerNode(4, T.LISTVIEWER, default_properties(T.LISTVIEWER))
    props = resolve(T.LISTVIEWER, node.properties)
    graph = ctx.codegraph
    original = graph.select

    def select(seeds, budget, sess):  # the candidate selection comes back empty
        return View() if budget == props["list_budget"] else original(seeds, budget, sess)

    graph.select = select
    outputs = asyncio.run(ListViewerWorker().run(ctx, node, slug_inputs(), props))
    first = user_messages(clients[0].requests[0])[0]
    assert "pkg/utils.py::slugify" in first and "pkg/service.py" in first
    assert outputs.view.files["pkg/utils.py"].levels["slugify"] == Level.FULL


def test_selection_parsing_rules(session):
    reply = (
        "Intro prose\n"
        "```\n"
        "* `pkg/models.py::Item.slug`\n"
        "(2) ./pkg/utils.py::slugify,\n"
        "\n"
        "pkg/utils.py::\n"
        "README.md\n"
        "```\n"
        "pkg/service.py (outside the block: ignored)\n"
    )
    assert selection_entries(reply) == [
        "pkg/models.py::Item.slug",
        "pkg/utils.py::slugify",
        "pkg/utils.py::",
        "README.md",
    ]
    view = parse_selection(reply, session)
    assert view.files["pkg/models.py"].levels == {"Item.slug": Level.FULL}
    assert view.files["pkg/utils.py"].levels == {"slugify": Level.FULL, "": Level.COLLAPSED}
    assert view.files["README.md"].levels == {"": Level.COLLAPSED}
    assert "pkg/service.py" not in view.files
