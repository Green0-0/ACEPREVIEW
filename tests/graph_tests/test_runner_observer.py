"""``RunContext.observer`` sees every worker start and finish, and can never break a run."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from fakes import FakeClient, text_message, tool_call_message

from agent2.core.session import Session
from ace.graph.graph import minimal_graph
from ace.graph.llm import ModelSettings
from ace.graph.runner import GraphRunner, RunContext, WorkerOutputs
from ace.graph.types import GraphFailure

NEW_CLAMP = "def clamp(value, low, high):\n    return max(low, min(high, value))\n"


def editing_client() -> FakeClient:
    return FakeClient([
        tool_call_message(("replace_element", json.dumps(
            {"file_path": "pkg/utils.py", "element_path": "clamp", "new_content": NEW_CLAMP}))),
        text_message("done"),
    ])


def context(project_dir: Path, tmp_path: Path, factory, observer) -> RunContext:
    session = Session(str(project_dir), rollout_dir=str(tmp_path / "rollouts"))
    return RunContext(session=session, prompt="fix clamp", settings=ModelSettings(model="fake"),
                      client_factory=factory, rollout_dir=tmp_path / "rollouts", observer=observer)


def test_observer_sees_start_and_done(project_dir: Path, tmp_path: Path):
    events = []
    ctx = context(project_dir, tmp_path, editing_client, lambda e, nid, p: events.append((e, nid, p)))
    result = asyncio.run(GraphRunner(minimal_graph()).run(ctx))
    (nid,) = result.outputs
    assert [(e, n) for e, n, _ in events] == [("start", nid), ("done", nid)]
    assert events[0][2] is None
    assert isinstance(events[1][2], WorkerOutputs) and events[1][2] is result.outputs[nid]


def test_observer_sees_failure(project_dir: Path, tmp_path: Path):
    events = []
    ctx = context(project_dir, tmp_path, lambda: FakeClient([]), lambda e, nid, p: events.append((e, nid, p)))
    with pytest.raises(GraphFailure):
        asyncio.run(GraphRunner(minimal_graph()).run(ctx))
    assert [e for e, _, _ in events] == ["start", "failed"]
    assert isinstance(events[1][2], GraphFailure)


def test_observer_errors_are_ignored(project_dir: Path, tmp_path: Path):
    def broken(event, nid, payload):
        raise RuntimeError("observer bug")

    ctx = context(project_dir, tmp_path, editing_client, broken)
    result = asyncio.run(GraphRunner(minimal_graph()).run(ctx))
    assert "clamp" in result.diff
