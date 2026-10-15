"""Tests for ace.graph.recording (GRAPH_NEAT_DESIGN.md section 6)."""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from agent2.core.agent import Agent
from agent2.core.globals import TOOLS
from agent2.core.workspace import Workspace
from ace.graph.recording import RecordingTool, ViewRecorder, map_ranges, recording_tools
from ace.graph.views import Level, View

from fakes import FakeClient, text_message, tool_call_message

FULL, COLLAPSED = Level.FULL, Level.COLLAPSED

ALL_RECORDING = [
    "view", "view_path", "view_files", "view_file", "views_at", "view_at", "view_element",
    "view_multi_lines", "view_lines", "open_file", "open_lines", "open_element",
    "open_element_at",
]


@pytest.fixture
def workspace(session):
    return Workspace(session)


@pytest.fixture
def agent(workspace):
    return SimpleNamespace(workspace=workspace, rollout=SimpleNamespace(raw_chat=[]))


@pytest.fixture
def recorder(session):
    return ViewRecorder(session)


@pytest.fixture
def tools(recorder):
    return recording_tools(TOOLS.keys(), recorder)


def call(tools, name, session, agent, **kwargs):
    return tools[name](session=session, agent=agent, **kwargs)


def snapshot(view: View) -> dict:
    return {
        path: (dict(fv.levels), list(fv.line_ranges))
        for path, fv in view.files.items()
        if not fv.is_empty()
    }


# --------------------------------------------------------------------------- per-tool rules

CASES = [
    ("view", {"paths": "pkg/models.py::Item.slug"},
     {"pkg/models.py": ({"Item.slug": FULL}, [])}),
    ("view", {"paths": "pkg/utils.py"}, {"pkg/utils.py": ({"": COLLAPSED}, [])}),
    ("view_path", {"path": "pkg/utils.py"}, {"pkg/utils.py": ({"": COLLAPSED}, [])}),
    ("view_path", {"path": "pkg/models.py::Item.slug.13"},
     {"pkg/models.py": ({"Item.slug": FULL}, [])}),
    ("view_files", {"file_paths": "pkg/utils.py*, notes.txt"},
     {"pkg/utils.py": ({"": FULL}, []), "notes.txt": ({"": COLLAPSED}, [])}),
    ("view_file", {"file_path": "pkg/utils.py", "full": True}, {"pkg/utils.py": ({"": FULL}, [])}),
    ("view_file", {"file_path": "pkg/utils.py"}, {"pkg/utils.py": ({"": COLLAPSED}, [])}),
    ("views_at", {"paths": "pkg/models.py::14, pkg/utils.py::8"},
     {"pkg/models.py": ({"Item.slug": FULL}, []), "pkg/utils.py": ({"slugify": FULL}, [])}),
    ("view_at", {"file_path": "pkg/models.py", "line": 10},
     {"pkg/models.py": ({"Item.__init__": FULL}, [])}),
    # No element at line 2: the 50-line fallback window, clamped to the 31-line file.
    ("view_at", {"file_path": "pkg/models.py", "line": 2}, {"pkg/models.py": ({}, [(2, 31)])}),
    ("view_at", {"file_path": "notes.txt", "line": 2, "full": True}, {"notes.txt": ({}, [(2, 4)])}),
    ("view_element", {"file_path": "pkg/models.py", "element_path": "Cart.add"},
     {"pkg/models.py": ({"Cart.add": FULL}, [])}),
    ("view_element", {"file_path": "pkg/models.py", "element_path": "Cart.21", "full": True},
     {"pkg/models.py": ({"Cart": FULL}, [])}),
    ("view_multi_lines", {"file_line_paths": "notes.txt::1::2, pkg/utils.py::3::4"},
     {"notes.txt": ({}, [(1, 2)]), "pkg/utils.py": ({}, [(3, 4)])}),
    ("view_lines", {"file_path": "notes.txt", "line_start": 2, "line_end": 3},
     {"notes.txt": ({}, [(2, 3)])}),
    ("view_lines", {"file_path": "notes.txt", "line_start": 3, "line_end": 99},
     {"notes.txt": ({}, [(3, 4)])}),
    ("open_file", {"file_path": "pkg/utils.py"}, {"pkg/utils.py": ({"": FULL}, [])}),
    ("open_lines", {"file_path": "notes.txt", "line_start": 1, "line_end": 3},
     {"notes.txt": ({}, [(1, 3)])}),
    ("open_element", {"file_path": "pkg/models.py", "element_path": "Item.to_json"},
     {"pkg/models.py": ({"Item.to_json": FULL}, [])}),
    ("open_element_at", {"file_path": "pkg/models.py", "line": 28},
     {"pkg/models.py": ({"Cart.add": FULL}, [])}),
]


@pytest.mark.parametrize(("name", "kwargs", "expected"), CASES,
                         ids=[f"{c[0]}-{i}" for i, c in enumerate(CASES)])
def test_viewing_tool_records_expected_view(tools, recorder, session, agent, name, kwargs, expected):
    result = call(tools, name, session, agent, **kwargs)
    content = result[0] if isinstance(result, tuple) else result
    assert not content.startswith("Error"), content
    assert snapshot(recorder.view) == expected
    assert snapshot(recorder.finalize(agent.workspace)) == expected


ERROR_CASES = [
    ("view", {"paths": "missing.py"}),
    ("view_path", {"path": "pkg/models.py::Nope"}),
    ("view_files", {"file_paths": "missing.py*"}),
    ("view_file", {"file_path": "missing.py"}),
    ("views_at", {"paths": "missing.py::3, pkg/models.py"}),
    ("view_at", {"file_path": "missing.py", "line": 3}),
    ("view_element", {"file_path": "notes.txt", "element_path": "x"}),
    ("view_element", {"file_path": "pkg/models.py", "element_path": "Item.nope"}),
    ("view_multi_lines", {"file_line_paths": "missing.py::1::2, notes.txt::1"}),
    ("view_lines", {"file_path": "missing.py", "line_start": 1, "line_end": 2}),
    ("open_file", {"file_path": "missing.py"}),
    ("open_lines", {"file_path": "notes.txt", "line_start": 3, "line_end": 99}),
    ("open_element", {"file_path": "pkg/models.py", "element_path": "Nope"}),
    ("open_element_at", {"file_path": "pkg/models.py", "line": 2}),
]


@pytest.mark.parametrize(("name", "kwargs"), ERROR_CASES,
                         ids=[f"{c[0]}-{i}" for i, c in enumerate(ERROR_CASES)])
def test_error_results_are_not_recorded(tools, recorder, session, agent, name, kwargs):
    result = call(tools, name, session, agent, **kwargs)
    content = result[0] if isinstance(result, tuple) else result
    assert "Error" in content
    assert recorder.view.is_empty()
    assert recorder.sightings == []


def test_multi_path_view_records_both_paths(tools, recorder, session, agent):
    call(tools, "view", session, agent, paths="pkg/utils.py, pkg/models.py::Cart.total")
    assert snapshot(recorder.view) == {
        "pkg/utils.py": ({"": COLLAPSED}, []),
        "pkg/models.py": ({"Cart.total": FULL}, []),
    }


def test_multi_path_view_records_successes_among_errors(tools, recorder, session, agent):
    result = call(tools, "view", session, agent,
                  paths="missing.py, pkg/utils.py, pkg/models.py::Nope, pkg/models.py::Cart")
    assert result.startswith("Error")  # first section failed; later ones still recorded
    assert snapshot(recorder.view) == {
        "pkg/utils.py": ({"": COLLAPSED}, []),
        "pkg/models.py": ({"Cart": FULL}, []),
    }


def test_open_element_records_full_and_returns_continuation(tools, recorder, session, agent):
    result = call(tools, "open_element", session, agent,
                  file_path="pkg/models.py", element_path="Item.slug")
    assert isinstance(result, tuple) and len(result) == 2
    assert "def slug(self):" in result[0]
    assert callable(result[1])
    assert recorder.view.files["pkg/models.py"].levels == {"Item.slug": FULL}


def test_wrappers_expose_tool_interface_and_pass_through(tools, recorder, session, agent):
    for name, tool in TOOLS.items():
        wrapper = tools[name]
        assert isinstance(wrapper, RecordingTool)
        assert wrapper.func is tool.func
        assert wrapper.name == name
        assert wrapper.description == tool.description
        assert wrapper.openai_schema == tool.openai_schema
        assert wrapper.examples == tool.examples
    result = call(tools, "replace_file", session, agent, file_path="notes.txt", new_content="x\n")
    assert result == "Replaced content of notes.txt."
    assert recorder.sightings == []


def test_recording_tools_accepts_mapping_and_rejects_unknown(recorder):
    wrapped = recording_tools({"view": TOOLS["view"]}, recorder)
    assert wrapped["view"].tool is TOOLS["view"]
    with pytest.raises(KeyError):
        recording_tools(["no_such_tool"], recorder)


def test_view_at_uses_workspace_state(tools, recorder, session, agent, workspace):
    # Insert two lines at the top: line 12 of the edited file is inside Item.__init__.
    original = session.files["pkg/models.py"].current_state.bytes
    workspace.write_file("pkg/models.py", b"# a\n# b\n" + original)
    call(tools, "view_at", session, agent, file_path="pkg/models.py", line=12)
    assert recorder.view.files["pkg/models.py"].levels == {"Item.__init__": FULL}


# --------------------------------------------------------------------------- finalize


def test_map_ranges_drops_replaced_lines_and_shifts_equal_blocks():
    pristine = b"a\nb\nc\nd\n"
    assert map_ranges([(1, 4)], pristine, pristine) == [(1, 4)]
    assert map_ranges([(1, 4)], pristine, b"a\nB\nC\nd\n") == [(1, 1), (4, 4)]
    assert map_ranges([(3, 6)], pristine, b"x\ny\na\nb\nc\nd\n") == [(1, 4)]
    assert map_ranges([(2, 2)], pristine, b"a\nnew\nb\nc\nd\n") == []


def test_finalize_remaps_line_ranges_after_edit(tools, recorder, session, agent, workspace):
    workspace.write_file("notes.txt", b"line one\nLINE TWO\nLINE THREE\nline four\n")
    call(tools, "view_lines", session, agent, file_path="notes.txt", line_start=1, line_end=4)
    assert recorder.view.files["notes.txt"].line_ranges == [(1, 4)]
    final = recorder.finalize(workspace)
    assert final.files["notes.txt"].line_ranges == [(1, 1), (4, 4)]


def test_finalize_without_snapshot_uses_final_workspace(recorder, workspace):
    workspace.write_file("notes.txt", b"inserted\nline one\nline two\nline three\nline four\n")
    recorder.record_lines("notes.txt", 2, 3)
    assert recorder.finalize(workspace).files["notes.txt"].line_ranges == [(1, 2)]


def test_finalize_uses_state_seen_at_call_time(tools, recorder, session, agent, workspace):
    # Lines viewed on the pristine file stay put even if the file is edited afterwards.
    call(tools, "view_lines", session, agent, file_path="notes.txt", line_start=2, line_end=3)
    workspace.write_file("notes.txt", b"new\nnew\nnew\nline one\nline two\nline three\n")
    assert recorder.finalize(workspace).files["notes.txt"].line_ranges == [(2, 3)]


def test_finalize_drops_created_file_and_renamed_element(tools, recorder, session, agent, workspace):
    call(tools, "make_empty_files", session, agent, file_paths="pkg/new.py")
    workspace.write_file("pkg/new.py", b"def fresh():\n    return 1\n")
    call(tools, "view_file", session, agent, file_path="pkg/new.py", full=True)
    call(tools, "view_element", session, agent, file_path="pkg/new.py", element_path="fresh")

    call(tools, "view_element", session, agent, file_path="pkg/models.py", element_path="Item.slug")
    result = call(tools, "replace_element", session, agent, file_path="pkg/models.py",
                  element_path="Item.slug",
                  new_content='def make_slug(self):\n        return slugify(self.name)')
    assert not result.startswith("Error"), result
    call(tools, "view_element", session, agent,
         file_path="pkg/models.py", element_path="Item.make_slug")

    assert set(recorder.view.files) == {"pkg/new.py", "pkg/models.py"}
    assert "Item.make_slug" in recorder.view.files["pkg/models.py"].levels

    final = recorder.finalize(workspace)
    assert snapshot(final) == {"pkg/models.py": ({"Item.slug": FULL}, [])}


def test_finalize_drops_moved_file_and_keeps_file_levels(tools, recorder, session, agent, workspace):
    call(tools, "view_file", session, agent, file_path="pkg/utils.py")
    call(tools, "move_files", session, agent, paths="pkg/utils.py>pkg/helpers.py")
    call(tools, "view_file", session, agent, file_path="pkg/helpers.py", full=True)
    final = recorder.finalize(workspace)
    assert snapshot(final) == {"pkg/utils.py": ({"": COLLAPSED}, [])}


def test_finalize_returns_restricted_json_roundtrip(tools, recorder, session, agent, workspace):
    call(tools, "views_at", session, agent, paths="pkg/service.py::8, notes.txt::1", full=False)
    final = recorder.finalize(workspace)
    assert View.from_json(json.loads(json.dumps(final.to_json()))) == final
    assert final.files["pkg/service.py"].levels == {"checkout": FULL}
    assert final.files["notes.txt"].line_ranges == [(1, 4)]


# --------------------------------------------------------------------------- agent-driven


def test_agent_run_records_views_through_tool_hook(session):
    base = session.get_config("config/agents/basic_agent.toml")
    banned = {"delegate", "message_subagent", "cancel_subagent", "bash", "tools"}
    config = base.model_copy(update={
        "primary_tools": [name for name in base.primary_tools if name not in banned],
        "secondary_tools": [name for name in base.secondary_tools if name not in banned],
        "subagent_names": [],
        "max_steps": 5,
    })
    client = FakeClient([
        tool_call_message(("view", json.dumps({"paths": "pkg/models.py::Item.slug"}))),
        text_message("Item.slug delegates to slugify."),
    ])
    workspace = Workspace(session)
    agent = Agent(session, workspace, config, "Explain Item.slug", client=client)
    recorder = ViewRecorder(session)
    agent.tools = recording_tools(agent.tools.keys(), recorder)

    final_message = asyncio.run(agent.run())

    assert final_message == "Item.slug delegates to slugify."
    tool_messages = [m for m in agent.rollout.raw_chat if m.get("role") == "tool"]
    assert tool_messages[0]["content"].startswith("--- pkg/models.py::Item.slug ---")
    # the recording wrappers are what the model was offered
    offered = {tool["function"]["name"] for tool in client.requests[0]["tools"]}
    assert "view" in offered and not offered & banned
    final = recorder.finalize(agent.workspace)
    assert final.files["pkg/models.py"].levels.get("Item.slug") == Level.FULL
