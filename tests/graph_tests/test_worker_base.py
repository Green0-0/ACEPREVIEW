"""Tests for graph/workers/base.py: agent construction, run helpers and prompt rendering."""
from __future__ import annotations

import asyncio
import json

import pytest

from agent2.core.agent import Agent
from agent2.core.objects import WorkspacePolicy
from ace.graph import prompts
from ace.graph.ace import AceError, AceFormat, ace_instructions
from ace.graph.edits import Edit
from ace.graph.graph import WorkerNode
from ace.graph.properties import default_properties, resolve
from ace.graph.rendering import measure
from ace.graph.runner import ModelSettings, RunContext, WorkerInputs, WorkerOutputs
from ace.graph.types import GraphFailure, WorkerType
from ace.graph.views import Level, View
from ace.graph.workers.base import (
    Worker,
    build_user_prompt,
    first_int,
    followup_agent,
    parse_ace_output,
    render_edits,
    render_inputs_views,
    render_reports,
    run_agent,
    system_prompt_for,
)
from fakes import FakeClient, text_message, tool_call_message

T = WorkerType


class DummyWorker(Worker):
    type = T.EDITOR

    async def run(self, ctx, node, inputs, props):  # noqa: ANN001
        return WorkerOutputs()


def make_ctx(session, messages=()) -> tuple[RunContext, list[FakeClient]]:
    clients: list[FakeClient] = []

    def factory() -> FakeClient:
        client = FakeClient(list(messages))
        clients.append(client)
        return client

    settings = ModelSettings(model="fake-model", api_base_url="http://x", api_key_env_var="FAKE_KEY")
    return RunContext(session, "Make slug lowercase.", settings, factory, run_id="r"), clients


def props_of(t: WorkerType) -> dict:
    return resolve(t, default_properties(t))


def test_make_agent_with_tools(session):
    ctx, clients = make_ctx(session)
    node = WorkerNode(7, T.EDITOR, default_properties(T.EDITOR))
    agent = DummyWorker().make_agent(
        ctx, node, props_of(T.EDITOR),
        tools=["view", "view_element", "replace_element"],
        system_prompt="SYS\n{{FILES*}}",
        user_prompt="hello",
        max_steps=4,
    )
    assert isinstance(agent, Agent)
    cfg = agent.config
    assert cfg.path == "graph/editor/7"
    assert cfg.subagent_name == "editor_7"
    assert cfg.model == "fake-model"
    assert cfg.api_base_url == "http://x"
    assert cfg.api_key_env_var == "FAKE_KEY"
    assert cfg.prompt_wrap_bottom == "{{PROMPT}}"
    assert cfg.followup_prompt_wrapper == "{{PROMPT}}"
    assert cfg.max_steps == 4
    assert cfg.constructors == ["default", "directory"]
    assert cfg.hooks == ["tool_call"]
    assert cfg.primary_tools == ["view", "view_element", "replace_element"]
    assert cfg.secondary_tools == [] and cfg.subagent_names == []
    assert cfg.workspace_policy is WorkspacePolicy.ISOLATED
    assert agent.client is clients[0]
    assert agent.workspace is not None and agent.workspace.files == {}
    assert agent.view_recorder is not None
    assert set(agent.tools) == {"view", "view_element", "replace_element"}
    assert all(hasattr(tool, "func") and hasattr(tool, "openai_schema") for tool in agent.tools.values())
    system, user = agent.rollout.raw_chat
    assert system["content"].startswith("SYS\n") and "pkg/models.py" in system["content"]
    assert user == {"role": "user", "content": "hello"}


def test_make_agent_without_tools(session):
    ctx, _ = make_ctx(session)
    node = WorkerNode(3, T.VIEWEDITOR, default_properties(T.VIEWEDITOR))
    agent = DummyWorker().make_agent(
        ctx, node, {}, tools=[], system_prompt="S", user_prompt="U", max_steps=1
    )
    assert agent.config.constructors == ["default"]
    assert agent.config.hooks == []
    assert agent.config.primary_tools == []
    assert agent.view_recorder is None
    assert agent.tools == {}


def test_make_agent_rejects_forbidden_tools(session):
    ctx, _ = make_ctx(session)
    node = WorkerNode(1, T.EDITOR, {})
    with pytest.raises(ValueError, match="bash"):
        DummyWorker().make_agent(
            ctx, node, {}, tools=["view", "bash"], system_prompt="S", user_prompt="U", max_steps=2
        )


def test_tool_views_are_recorded(session):
    args = json.dumps({"file_path": "pkg/models.py", "element_path": "Item.slug"})
    ctx, clients = make_ctx(session, [tool_call_message(("view_element", args)), text_message("done")])
    node = WorkerNode(2, T.VIEWER, default_properties(T.VIEWER))
    agent = DummyWorker().make_agent(
        ctx, node, {}, tools=["view_element"], system_prompt="S", user_prompt="U", max_steps=3
    )
    assert asyncio.run(run_agent(agent)) == "done"
    view = agent.view_recorder.finalize(agent.workspace)
    assert view.files["pkg/models.py"].levels.get("Item.slug") == Level.FULL
    assert len(clients[0].requests) == 2


def test_run_agent_and_followup(session):
    ctx, clients = make_ctx(session, [text_message("first"), text_message("second")])
    node = WorkerNode(5, T.LISTVIEWER, {})
    agent = DummyWorker().make_agent(
        ctx, node, {}, tools=[], system_prompt="S", user_prompt="U", max_steps=1
    )
    assert asyncio.run(run_agent(agent)) == "first"
    assert asyncio.run(followup_agent(agent, "more please")) == "second"
    chat = agent.rollout.raw_chat
    assert [m["role"] for m in chat] == ["system", "user", "assistant", "user", "assistant"]
    assert chat[3]["content"] == "more please"
    last_request = clients[0].requests[-1]["messages"]
    assert last_request[-1] == {"role": "user", "content": "more please"}


def test_run_agent_failures_name_node(session):
    ctx, _ = make_ctx(session, [text_message(None)])  # type: ignore[arg-type]
    node = WorkerNode(9, T.VIEWEDITOR, {})
    agent = DummyWorker().make_agent(
        ctx, node, {}, tools=[], system_prompt="S", user_prompt="U", max_steps=1
    )
    with pytest.raises(GraphFailure, match=r"node 9 \(vieweditor\)"):
        asyncio.run(run_agent(agent))

    ctx, _ = make_ctx(session, [])  # client runs out of messages -> AssertionError
    agent = DummyWorker().make_agent(
        ctx, node, {}, tools=[], system_prompt="S", user_prompt="U", max_steps=1
    )
    with pytest.raises(GraphFailure, match=r"node 9 \(vieweditor\)") as info:
        asyncio.run(run_agent(agent))
    assert isinstance(info.value.__cause__, AssertionError)


def test_render_reports_order_and_labels():
    inputs = WorkerInputs(
        reports={4: "  second  ", 1: "first"},
        source_types={1: T.VIEWER, 4: T.EDITOR},
    )
    text = render_reports(inputs)
    assert text == (
        "REPORT from worker 1 (viewer)\nfirst\n\nREPORT from worker 4 (editor)\nsecond"
    )
    assert render_reports(WorkerInputs()) == ""


def test_render_edits_numbering_and_order(session):
    ctx, _ = make_ctx(session)
    inputs = WorkerInputs(
        edits={
            8: Edit({"notes.txt": b"line one\nline TWO\nline three\nline four\n"}),
            3: Edit(),
        }
    )
    text, ids = render_edits(ctx, inputs)
    assert ids == [3, 8]
    first, second = text.split("\n\n", 1)
    assert first == "CANDIDATE 1 (worker 3)\n(no changes)"
    assert second.startswith("CANDIDATE 2 (worker 8)\n```diff\n")
    assert "-line two" in second and "+line TWO" in second
    assert second.rstrip().endswith("```")
    assert render_edits(ctx, WorkerInputs()) == ("", [])


def test_first_int():
    assert first_int("I pick candidate 2, not 3") == 2
    assert first_int("  12\n") == 12
    assert first_int("none") is None
    assert first_int("") is None
    assert first_int(None) is None


def test_system_prompt_for_fills_ace_placeholder():
    props = props_of(T.CRITIC)
    text = system_prompt_for(T.CRITIC, props, ace_fmt=AceFormat.ELEMENT_FILE)
    assert prompts.ACE_PLACEHOLDER not in text
    assert ace_instructions(AceFormat.ELEMENT_FILE) in text
    assert text.startswith(props["role"].strip())
    # Falls back to the ace_format property.
    fallback = system_prompt_for(T.CRITIC, props)
    assert ace_instructions(AceFormat(props["ace_format"])) in fallback
    # Non-ACE type: no placeholder involved.
    editor = system_prompt_for(T.EDITOR, props_of(T.EDITOR))
    assert prompts.ACE_PLACEHOLDER not in editor and "{{FILES*}}" in editor


def test_build_user_prompt(session):
    ctx, _ = make_ctx(session)
    props = props_of(T.VIEWEDITOR)
    text = build_user_prompt(props, ctx, [(prompts.VIEWS_HEADER, "CODE"), ("# Empty", "  ")])
    assert text.startswith(props["instruction"].strip())
    assert f"{prompts.TASK_HEADER}\nMake slug lowercase." in text
    assert text.endswith(f"{prompts.VIEWS_HEADER}\nCODE")
    assert "# Empty" not in text


def test_parse_ace_output(session):
    ctx, _ = make_ctx(session)
    reply = (
        "Here is the fix.\n```python\n@@ pkg/models.py::Item.slug\n"
        "def slug(self):\n    return slugify(self.name).lower()\n```\n"
    )
    edit = parse_ace_output(reply, "element", ctx)
    assert set(edit.files) == {"pkg/models.py"}
    assert b"slugify(self.name).lower()" in edit.files["pkg/models.py"]
    with pytest.raises(AceError):
        parse_ace_output("no blocks here", AceFormat.ELEMENT, ctx)
    with pytest.raises(AceError):
        parse_ace_output("```\n@@ pkg/models.py::Nope\nx\n```", AceFormat.ELEMENT, ctx)


def test_render_inputs_views(session):
    ctx, _ = make_ctx(session)
    assert render_inputs_views(ctx, WorkerInputs(), 5000, 2) == ("", View())

    seen = View()
    seen.add_element("pkg/models.py", "Item.slug", Level.FULL)
    inputs = WorkerInputs(views={1: seen}, source_types={1: T.VIEWER})
    budget = 2000
    text, view = render_inputs_views(ctx, inputs, budget, 2)
    assert text
    assert measure(text) <= budget
    assert "def slug" in text
    assert view.files["pkg/models.py"].levels.get("Item.slug") == Level.FULL
