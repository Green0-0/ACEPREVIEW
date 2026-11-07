"""Demo runs from a genome page: start, poll, rollouts and views, without any network."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "graph_tests"))
from fakes import FakeClient, text_message, tool_call_message  # noqa: E402
from sample_project import write_sample_project  # noqa: E402

from ace.webui.app import create_app
from ace.webui.demo import DemoSettings

NEW_CLAMP = (
    "def clamp(value, low, high):\n"
    "    if low > high:\n"
    "        low, high = high, low\n"
    "    return max(low, min(high, value))\n"
)


def editor_script() -> FakeClient:
    return FakeClient([
        tool_call_message(("replace_element", json.dumps(
            {"file_path": "pkg/utils.py", "element_path": "clamp", "new_content": NEW_CLAMP}))),
        text_message("Swapped reversed bounds."),
    ])


def wait_for(client: TestClient, demo_id: str, timeout: float = 20.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        body = client.get(f"/api/demos/{demo_id}").json()
        if body["status"] != "running":
            return body
        time.sleep(0.05)
    raise AssertionError("demo did not finish")


@pytest.fixture
def demo_project(tmp_path: Path) -> Path:
    return write_sample_project(tmp_path / "demo_project")


@pytest.fixture
def demo_client(runs: dict[str, Any], demo_project: Path) -> TestClient:
    settings = DemoSettings(project_dir=demo_project, client_factory=editor_script)
    with TestClient(create_app(runs["root"], settings)) as client:
        yield client


def test_demo_info(demo_client: TestClient, demo_project: Path):
    info = demo_client.get("/api/demo").json()
    assert info["enabled"] is True and info["key_set"] is True and info["reason"] is None
    assert info["project_dir"] == str(demo_project)
    assert info["model"] == "claude-haiku-5-5" and info["thinking"] == "off"
    assert info["running"] is None


def test_demo_runs_graph_and_serves_rollouts(demo_client: TestClient, demo_project: Path):
    before = {p: p.read_text() for p in demo_project.rglob("*.py")}
    started = demo_client.post("/api/runs/run-a/genomes/g0a/demo", json={"prompt": "fix clamp"})
    assert started.status_code == 202, started.text
    demo_id = started.json()["id"]
    assert started.json()["status"] == "running"
    assert started.json()["workers_total"] == 1

    body = wait_for(demo_client, demo_id)
    assert body["status"] == "done", body.get("error")
    assert body["genome_id"] == "g0a" and body["run_id"] == "run-a" and body["prompt"] == "fix clamp"
    assert "diff --git a/pkg/utils.py b/pkg/utils.py" in body["diff"]
    assert body["touched_paths"] == ["pkg/utils.py"] and body["diff_chars"] == len(body["diff"])
    assert body["llm_calls"] >= 1 and body["rollouts"] >= 1
    assert body["failure"] is None
    workers = body["workers"]
    assert workers and all("view" not in w for w in workers)
    ok = [w for w in workers if w["status"] == "ok"]
    assert ok and ok[0]["rollout_ids"]
    # live fields: the worker's own diff, its call counts, the execution order and the log
    assert "clamp" in ok[0]["diff"] and ok[0]["edit_touched_paths"] == ["pkg/utils.py"]
    assert ok[0]["llm_calls"] == 2 and ok[0]["steps"] == 2 and ok[0]["tool_calls"] == 1
    assert ok[0]["started_at"] and ok[0]["finished_at"] and ok[0]["properties"]
    assert body["order"] == [ok[0]["node_id"]] and body["final_node"] == ok[0]["node_id"]
    assert body["inputs"][str(ok[0]["node_id"])] == {"views_from": [], "edits_from": [], "reports_from": []}
    texts = [l["text"] for l in body["log"]]
    assert any("started" in t for t in texts) and any("finished" in t for t in texts)
    assert body["workers_done"] == 1 and body["worker_counts"]["ok"] == 1
    # the project on disk is untouched; rollouts went to a temporary folder outside it
    assert {p: p.read_text() for p in demo_project.rglob("*.py")} == before
    assert not (demo_project / ".agent2").exists()
    assert not Path(ok[0]["rollout_dir"]).resolve().is_relative_to(demo_project.resolve())

    rid = ok[0]["rollout_ids"][0]
    rollout = demo_client.get(f"/api/demos/{demo_id}/rollouts/{rid}").json()
    assert rollout["id"] == rid and rollout["raw_chat"]
    assert demo_client.get(f"/api/demos/{demo_id}/rollouts/nope").status_code == 404
    assert demo_client.get(f"/api/demos/{demo_id}/rollouts/..%2F..%2Fx").status_code in {400, 404}

    rollouts = demo_client.get(f"/api/demos/{demo_id}/workers/{ok[0]['node_id']}/rollouts").json()
    assert [r["id"] for r in rollouts] == [rid] and rollouts[0]["raw_chat"]
    assert demo_client.get(f"/api/demos/{demo_id}/workers/999/rollouts").status_code == 404

    view = demo_client.get(f"/api/demos/{demo_id}/workers/{ok[0]['node_id']}/view").json()
    assert "view" in view and "rendered" in view
    assert demo_client.get(f"/api/demos/{demo_id}/workers/999/view").status_code == 404

    listed = demo_client.get("/api/demos").json()
    assert [d["id"] for d in listed] == [demo_id]
    assert [d["id"] for d in demo_client.get("/api/demos?run=run-a&genome=g0a").json()] == [demo_id]
    assert demo_client.get("/api/demos?genome=g2a").json() == []
    assert demo_client.get("/api/demos/unknown").status_code == 404


def test_demo_request_validation(demo_client: TestClient):
    assert demo_client.post("/api/runs/run-a/genomes/g0a/demo", json={"prompt": "   "}).status_code == 422
    assert demo_client.post("/api/runs/run-a/genomes/g0a/demo", json={"prompt": "x" * 5000}).status_code == 422
    assert demo_client.post("/api/runs/run-a/genomes/nope/demo", json={"prompt": "fix"}).status_code == 404
    assert demo_client.post("/api/runs/nope/genomes/g0a/demo", json={"prompt": "fix"}).status_code == 404


def test_demo_request_adds_model_defaults(runs: dict[str, Any], demo_project: Path):
    seen: list[FakeClient] = []

    def factory() -> FakeClient:
        client = editor_script()
        seen.append(client)
        return client

    settings = DemoSettings(project_dir=demo_project, client_factory=factory, max_tokens=1234)
    with TestClient(create_app(runs["root"], settings)) as client:
        demo_id = client.post("/api/runs/run-a/genomes/g0a/demo", json={"prompt": "fix clamp"}).json()["id"]
        assert wait_for(client, demo_id)["status"] == "done"
    requests = [r for c in seen for r in c.requests]
    assert requests
    assert all(r["max_tokens"] == 1234 for r in requests)
    assert all(r["extra_body"] == {"thinking": {"type": "disabled"}} for r in requests)


def test_demo_unavailable_without_project_or_key(runs: dict[str, Any], demo_project: Path, monkeypatch):
    with TestClient(create_app(runs["root"], DemoSettings(project_dir=None))) as client:
        info = client.get("/api/demo").json()
        assert info["enabled"] is False and "demo project" in info["reason"]
        res = client.post("/api/runs/run-a/genomes/g0a/demo", json={"prompt": "fix"})
        assert res.status_code == 409

    monkeypatch.delenv("ACE_TEST_DEMO_KEY", raising=False)
    settings = DemoSettings(project_dir=demo_project, api_key_env_var="ACE_TEST_DEMO_KEY")
    with TestClient(create_app(runs["root"], settings)) as client:
        info = client.get("/api/demo").json()
        assert info["enabled"] is True and info["key_set"] is False and "ACE_TEST_DEMO_KEY" in info["reason"]
        res = client.post("/api/runs/run-a/genomes/g0a/demo", json={"prompt": "fix"})
        assert res.status_code == 409 and "ACE_TEST_DEMO_KEY" in res.json()["detail"]


def test_demo_failure_is_reported(runs: dict[str, Any], demo_project: Path):
    def broken() -> FakeClient:
        return FakeClient([])  # runs out of script immediately → worker failure

    with TestClient(create_app(runs["root"], DemoSettings(project_dir=demo_project, client_factory=broken))) as client:
        demo_id = client.post("/api/runs/run-a/genomes/g0a/demo", json={"prompt": "fix clamp"}).json()["id"]
        body = wait_for(client, demo_id)
    assert body["status"] == "failed"
    assert body["error"] and body["diff"] is None
    assert any(w["status"] == "failed" for w in body["workers"])


def test_one_demo_at_a_time(runs: dict[str, Any], demo_project: Path):
    import asyncio

    class SlowClient(FakeClient):
        async def create(self, **kwargs):
            await asyncio.sleep(0.3)
            return await super().create(**kwargs)

    def slow() -> SlowClient:
        return SlowClient([
            tool_call_message(("replace_element", json.dumps(
                {"file_path": "pkg/utils.py", "element_path": "clamp", "new_content": NEW_CLAMP}))),
            text_message("done"),
        ])

    with TestClient(create_app(runs["root"], DemoSettings(project_dir=demo_project, client_factory=slow))) as client:
        first = client.post("/api/runs/run-a/genomes/g0a/demo", json={"prompt": "fix clamp"}).json()["id"]
        live = client.get(f"/api/demos/{first}").json()
        assert live["status"] == "running" and len(live["workers"]) == 1
        assert live["workers"][0]["status"] in {"pending", "running"} and live["workers"][0]["diff"] is None
        second = client.post("/api/runs/run-a/genomes/g0a/demo", json={"prompt": "fix clamp"})
        assert second.status_code == 409 and first in second.json()["detail"]
        assert client.get("/api/demo").json()["running"] == first
        cancelled = client.delete(f"/api/demos/{first}").json()
        assert cancelled["id"] == first
        body = wait_for(client, first)
        assert body["status"] in {"cancelled", "done"}
        assert all(w["status"] in {"ok", "skipped"} for w in body["workers"])
        if body["status"] == "cancelled":
            assert any("cancel" in l["text"] for l in body["log"])
