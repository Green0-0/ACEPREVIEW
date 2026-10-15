"""Endpoint tests for the analysis WebUI (spec section 12.1)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from ace.neat.genome import Genome
from ace.neat.speciation import distance

from conftest import append_jsonl

# ----------------------------------------------------------------------------- meta / runs


def test_health(client, runs):
    body = client.get("/api/health").json()
    assert body == {"ok": True, "runs_root": str(runs["root"])}


def test_index_and_static(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "<html" in r.text.lower()
    assert client.get("/static/does-not-exist.js").status_code == 404
    assert client.get("/static/../app.py").status_code == 404
    assert client.get("/static/%2E%2E/app.py").status_code == 404


def test_list_runs(client):
    rows = client.get("/api/runs").json()
    assert [r["run_id"] for r in rows] == ["run-b", "run-a"]  # newest first; empty-dir skipped
    a = rows[1]
    assert set(a) == {"run_id", "status", "started_at", "finished_at", "generations_completed",
                      "best_fitness", "population_size", "evaluator_type"}
    assert a["status"] == "finished"
    assert a["generations_completed"] == 3
    assert a["best_fitness"] == 0.9
    assert a["population_size"] == 4
    assert a["evaluator_type"] == "task"
    assert rows[0]["status"] == "running"
    assert rows[0]["evaluator_type"] == "mock"


def test_run_detail_counts(client):
    body = client.get("/api/runs/run-a").json()
    assert body["run_id"] == "run-a"
    assert body["config"]["speciation"]["c_excess"] == 2.0
    assert body["counts"] == {"genomes": 5, "evaluations": 6, "generations": 3, "species": 2}


def test_unknown_run_404(client):
    for path in ("", "/progress", "/species", "/genomes", "/operators", "/failures", "/innovations"):
        assert client.get(f"/api/runs/nope{path}").status_code == 404
    assert client.get("/api/runs/empty-dir").status_code == 404


@pytest.mark.parametrize("bad", ["%2E%2E", "..", ".hidden", "a%2F..%2Fb", "..%2Frun-a"])
def test_run_id_traversal_rejected(client, bad):
    r = client.get(f"/api/runs/{bad}/progress")
    assert r.status_code in (400, 404)
    assert "generations" not in r.text


# ----------------------------------------------------------------------------- progress / generations


def test_progress_shape(client):
    body = client.get("/api/runs/run-a/progress").json()
    assert body["status"] == "finished"
    assert [g["generation"] for g in body["generations"]] == [0, 1, 2]
    assert all("population" not in g for g in body["generations"])
    assert body["generations"][2]["best_genome_id"] == "g2a"
    assert body["events_tail"][-1]["event"] == "run_end"


def test_progress_tails_appended_generation(client, runs):
    run_b = runs["b"]
    first = client.get("/api/runs/run-b/progress").json()
    assert [g["generation"] for g in first["generations"]] == [0]
    assert first["status"] == "running"
    record = dict(json.loads((run_b / "generations.jsonl").read_text().splitlines()[0]))
    record.update(generation=1, best_fitness=0.4)
    append_jsonl(run_b / "generations.jsonl", record)
    append_jsonl(run_b / "events.jsonl", {"ts": "t", "event": "generation_end", "generation": 1,
                                          "duration_s": 1.0, "best_fitness": 0.4})
    second = client.get("/api/runs/run-b/progress").json()
    assert [g["generation"] for g in second["generations"]] == [0, 1]
    assert second["generations"][1]["best_fitness"] == 0.4
    assert second["events_tail"][-1]["event"] == "generation_end"
    assert client.get("/api/runs/run-b/generations/1").status_code == 200


def test_progress_ignores_partial_line_until_complete(client, runs):
    path = runs["b"] / "generations.jsonl"
    client.get("/api/runs/run-b/progress")
    with path.open("a") as fh:
        fh.write('{"generation": 5, "best_fi')
    assert len(client.get("/api/runs/run-b/progress").json()["generations"]) == 1
    with path.open("a") as fh:
        fh.write('tness": 1.0}\n')
    gens = client.get("/api/runs/run-b/progress").json()["generations"]
    assert [g["generation"] for g in gens] == [0, 5]


def test_generation_detail(client):
    body = client.get("/api/runs/run-a/generations/1").json()
    assert body["generation"] == 1
    assert [p["genome_id"] for p in body["population"]] == ["g0a", "g1a", "g1b"]
    assert body["population"][1]["operators"] == ["add_worker"]
    assert client.get("/api/runs/run-a/generations/9").status_code == 404


# ----------------------------------------------------------------------------- species


def test_species(client):
    body = client.get("/api/runs/run-a/species").json()
    assert [t["generation"] for t in body["timeline"]] == [0, 1, 2]
    assert body["timeline"][1]["species"] == [
        {"id": 0, "size": 2, "best_fitness": 0.5, "representative_id": "g0a"},
        {"id": 1, "size": 1, "best_fitness": 0.7, "representative_id": "g1a"},
    ]
    assert body["species"]["0"] == {"created": 0, "last_seen": 1, "best_fitness": 0.5, "max_size": 2}
    assert body["species"]["1"] == {"created": 1, "last_seen": 2, "best_fitness": 0.9, "max_size": 2}


# ----------------------------------------------------------------------------- genomes


def test_genomes_list_filters_and_pagination(client):
    body = client.get("/api/runs/run-a/genomes").json()
    assert body["total"] == 5
    assert all("genome" not in item for item in body["items"])
    assert [i["id"] for i in body["items"]] == ["g0a", "g0b", "g1a", "g1b", "g2a"]

    gen1 = client.get("/api/runs/run-a/genomes?generation=1").json()
    assert {i["id"] for i in gen1["items"]} == {"g1a", "g1b"}
    sp1 = client.get("/api/runs/run-a/genomes?species=1").json()
    assert {i["id"] for i in sp1["items"]} == {"g1a", "g2a"}
    fit = client.get("/api/runs/run-a/genomes?min_fitness=0.5").json()
    assert {i["id"] for i in fit["items"]} == {"g0a", "g1a", "g2a"}

    desc = client.get("/api/runs/run-a/genomes?sort=-fitness").json()
    assert [i["id"] for i in desc["items"]] == ["g2a", "g1a", "g0a", "g0b", "g1b"]
    asc = client.get("/api/runs/run-a/genomes?sort=adjusted_fitness&limit=2&offset=1").json()
    assert asc["total"] == 5
    assert [i["id"] for i in asc["items"]] == ["g0b", "g0a"]


def test_genome_detail(client, runs):
    info = runs["a"]
    body = client.get("/api/runs/run-a/genomes/g2a").json()
    assert body["id"] == "g2a"
    assert body["genome"]["id"] == "g2a"
    assert body["graph"] == info["genomes"]["g2a"].to_graph().to_json()
    assert [r["id"] for r in body["lineage"]] == ["g1a", "g0a"]
    assert all("genome" not in r for r in body["lineage"])
    assert body["children"] == []
    assert [e["task_index"] for e in body["evaluations"]] == [0, 1]
    for ev in body["evaluations"]:
        assert all("view" not in w for w in ev["workers"])

    layout = body["layout"]
    nodes = {n["id"]: n for n in layout["nodes"]}
    assert set(nodes) == {info["editor"], info["viewer"], info["branch"], info["merger"], "END"}
    assert nodes["END"]["level"] == max(n["level"] for n in layout["nodes"])
    for edge in layout["edges"]:
        assert nodes[edge["src"]]["level"] < nodes[edge["dst"]]["level"]
    assert nodes[info["viewer"]]["level"] == 0
    assert nodes[info["merger"]]["level"] == 2
    assert nodes["END"]["level"] == 3
    for n in layout["nodes"]:
        assert set(n) == {"id", "type", "level", "x", "y", "properties_resolved"}
        assert 0.0 <= n["x"] <= 1.0 and 0.0 <= n["y"] <= 1.0
    assert nodes[info["editor"]]["properties_resolved"]["max_steps"] == 30
    assert {e["kind"] for e in layout["edges"]} == {"view", "edit"}

    parent = client.get("/api/runs/run-a/genomes/g0a").json()
    assert sorted(parent["children"]) == ["g1a", "g1b"]
    assert parent["lineage"] == []


def test_genome_404_and_traversal(client):
    assert client.get("/api/runs/run-a/genomes/nope").status_code == 404
    assert client.get("/api/runs/run-a/genomes/nope/dot").status_code == 404
    assert client.get("/api/runs/run-a/genomes/..%2Frun.json").status_code in (400, 404)


def test_genome_dot(client, runs):
    info = runs["a"]
    r = client.get("/api/runs/run-a/genomes/g2a/dot")
    assert r.status_code == 200
    text = r.text
    assert text.startswith("digraph")
    for nid in (info["editor"], info["viewer"], info["branch"], info["merger"]):
        assert f"n{nid} [" in text
    assert "END" in text
    assert f"n{info['merger']} -> END" in text
    assert 'color="#1f77b4"' in text and 'color="#d62728"' in text


# ----------------------------------------------------------------------------- evaluations


def test_evaluations_list_filters(client):
    body = client.get("/api/runs/run-a/evaluations").json()
    assert body["total"] == 6
    assert all("workers" not in i for i in body["items"])
    assert client.get("/api/runs/run-a/evaluations?genome=g2a").json()["total"] == 2
    assert client.get("/api/runs/run-a/evaluations?task=1").json()["total"] == 1
    failed = client.get("/api/runs/run-a/evaluations?failed=true").json()
    assert {(i["genome_id"], i["task_index"]) for i in failed["items"]} == {("g0b", 0), ("g1b", 0), ("g2a", 1)}
    assert client.get("/api/runs/run-a/evaluations?failed=false").json()["total"] == 3
    page = client.get("/api/runs/run-a/evaluations?limit=2&offset=4").json()
    assert page["total"] == 6
    assert [(i["genome_id"], i["task_index"]) for i in page["items"]] == [("g2a", 0), ("g2a", 1)]


def test_evaluation_detail(client, runs):
    body = client.get("/api/runs/run-a/evaluations/g2a/1").json()
    statuses = {w["node_id"]: w["status"] for w in body["workers"]}
    info = runs["a"]
    assert statuses == {info["viewer"]: "ok", info["editor"]: "failed",
                        info["branch"]: "skipped", info["merger"]: "skipped"}
    assert body["workers"][0]["view"]["files"]["mod.py"]["levels"] == {"": 2}
    assert client.get("/api/runs/run-a/evaluations/g2a/7").status_code == 404
    assert client.get("/api/runs/run-a/evaluations/nope/0").status_code == 404


def test_worker_view_rendered(client, runs):
    info, project = runs["a"], runs["project"]
    before = sorted(p.relative_to(project) for p in project.rglob("*"))
    body = client.get(f"/api/runs/run-a/evaluations/g2a/0/workers/{info['viewer']}/view").json()
    assert body["view"]["files"]["mod.py"]["levels"] == {"": 2}
    assert body["rendered"] is not None
    assert "--- mod.py ---" in body["rendered"]
    assert "def f(a):" in body["rendered"]
    # nothing written into the task project
    assert sorted(p.relative_to(project) for p in project.rglob("*")) == before


def test_worker_view_missing_project_and_no_view(client, runs):
    info = runs["a"]
    body = client.get(f"/api/runs/run-a/evaluations/g2a/1/workers/{info['viewer']}/view").json()
    assert body["view"] is not None and body["rendered"] is None
    body = client.get(f"/api/runs/run-a/evaluations/g2a/0/workers/{info['merger']}/view").json()
    assert body == {"view": None, "rendered": None}
    assert client.get("/api/runs/run-a/evaluations/g2a/0/workers/999/view").status_code == 404


# ----------------------------------------------------------------------------- rollouts


def test_rollout(client):
    body = client.get("/api/runs/run-a/rollouts/g2a/0/r1").json()
    assert body["id"] == "r1"
    assert [m["role"] for m in body["raw_chat"]] == ["system", "user", "assistant", "tool", "assistant"]
    assert client.get("/api/runs/run-a/rollouts/g2a/0/r1.json").status_code == 200
    assert client.get("/api/runs/run-a/rollouts/g2a/0/nope").status_code == 404
    assert client.get("/api/runs/run-a/rollouts/g2a/1/r1").status_code == 404
    assert client.get("/api/runs/run-a/rollouts/nope/0/r1").status_code == 404


@pytest.mark.parametrize(
    "path",
    [
        "/api/runs/run-a/rollouts/g2a/0/..",
        "/api/runs/run-a/rollouts/g2a/0/%2E%2E",
        "/api/runs/run-a/rollouts/g2a/0/..%2F..%2F..%2Frun",
        "/api/runs/run-a/rollouts/..%2F..%2Frun-a/0/r1",
        "/api/runs/run-a/rollouts/%2E%2E/0/r1",
        "/api/runs/run-a/rollouts/g2a/0/.hidden",
    ],
)
def test_rollout_traversal_rejected(client, path):
    r = client.get(path)
    assert r.status_code in (400, 404)
    assert "raw_chat" not in r.text and "run_id" not in r.text


# ----------------------------------------------------------------------------- analysis endpoints


def test_operators_hand_computed(client):
    rows = client.get("/api/runs/run-a/operators").json()
    assert [r["operator"] for r in rows] == ["mutate_properties", "add_worker", "crossover", "insert_merger"]
    by_op = {r["operator"]: r for r in rows}
    expected = {
        # g1b (0.20 vs best parent g0a 0.48) and g2a (0.80 vs g1a 0.66)
        "mutate_properties": (2, 0.50, 0.57, -0.07, 0.5),
        "add_worker": (1, 0.66, 0.48, 0.18, 1.0),  # g1a vs g0a
        "crossover": (1, 0.20, 0.48, -0.28, 0.0),  # g1b vs max(g0a, g0b)
        "insert_merger": (1, 0.80, 0.66, 0.14, 1.0),  # g2a vs g1a
    }
    for op, (count, child, parent, delta, success) in expected.items():
        row = by_op[op]
        assert row["count"] == count
        assert row["mean_child_fitness"] == pytest.approx(child)
        assert row["mean_parent_fitness"] == pytest.approx(parent)
        assert row["mean_delta"] == pytest.approx(delta)
        assert row["success_rate"] == pytest.approx(success)


def test_compare(client, runs):
    info = runs["a"]
    cfg = info["config"].speciation
    genomes = info["genomes"]
    body = client.get("/api/runs/run-a/compare?a=g0b&b=g0a").json()
    # same structure; the editor differs in role and max_steps (2 of 7 properties)
    assert body["excess"] == 0 and body["disjoint"] == 0
    assert body["property_difference"] == pytest.approx(2 / 7)
    assert body["distance"] == pytest.approx(0.25 * 2 / 7)
    assert body["distance"] == pytest.approx(distance(genomes["g0b"], genomes["g0a"], cfg))
    diff = body["property_diffs"][0]
    assert diff["node"] == info["editor"]
    assert set(diff["diffs"]) == {"role", "max_steps"}
    assert diff["diffs"]["max_steps"] == {"a": 2, "b": 0, "a_resolved": 60, "b_resolved": 30}
    assert body["a"]["genome"]["id"] == "g0b"
    assert "layout" in body["a"] and "graph" in body["b"]

    body = client.get("/api/runs/run-a/compare?a=g0a&b=g2a").json()
    g0a = Genome.from_json(genomes["g0a"].to_json())
    g2a = Genome.from_json(genomes["g2a"].to_json())
    # g2a has 11 genes; 9 of them (3 nodes, 6 connections) lie beyond g0a's largest innovation 1
    assert body["excess"] == len(set(g2a.nodes) | set(g2a.connections)) - 2 == 9
    assert body["disjoint"] == 0
    assert body["matched_nodes"] == [info["editor"]]
    assert body["distance"] == pytest.approx(distance(g0a, g2a, cfg))
    assert body["distance"] == pytest.approx(2.0 * body["excess"] + 0.25 * (1 / 7))
    assert body["property_diffs"][0]["diffs"]["role"]["b"] == 2

    assert client.get("/api/runs/run-a/compare?a=g0a&b=nope").status_code == 404
    assert client.get("/api/runs/run-a/compare?a=g0a").status_code == 422


def test_innovations(client, runs):
    body = client.get("/api/runs/run-a/innovations").json()
    db = runs["a"]["db"]
    assert len(body["nodes"]) == len(db.node_entries())
    assert len(body["connections"]) == len(db.connection_entries())
    first = body["nodes"][0]
    assert first == {"id": 0, "key": ["init", 0, "editor"], "type": "editor"}
    assert {"id": 1, "src": 0, "dst": "END", "kind": "edit"} in body["connections"]


def test_failures(client):
    rows = client.get("/api/runs/run-a/failures").json()
    assert rows == [
        {"failure_type": "timeout", "node_type": "editor", "count": 2, "examples": [
            {"genome_id": "g0b", "task_index": 0, "message": "editor timed out"},
            {"genome_id": "g1b", "task_index": 0, "message": "editor timed out again"},
        ]},
        {"failure_type": "GraphFailure", "node_type": "editor", "count": 1, "examples": [
            {"genome_id": "g2a", "task_index": 1, "message": "GraphFailure: node failed"},
        ]},
    ]


# ----------------------------------------------------------------------------- partial run


def test_partial_running_run(client):
    assert client.get("/api/runs/run-b").json()["counts"] == {
        "genomes": 0, "evaluations": 0, "generations": 1, "species": 0}
    assert client.get("/api/runs/run-b/genomes").json() == {"total": 0, "items": []}
    assert client.get("/api/runs/run-b/evaluations").json() == {"total": 0, "items": []}
    assert client.get("/api/runs/run-b/species").json() == {"timeline": [], "species": {}}
    assert client.get("/api/runs/run-b/operators").json() == []
    assert client.get("/api/runs/run-b/failures").json() == []
    assert client.get("/api/runs/run-b/innovations").json() == {"nodes": [], "connections": []}
    assert client.get("/api/runs/run-b/generations/0").json()["population"] == []


def test_run_without_run_json(client, runs):
    run = runs["root"] / "run-c"
    run.mkdir()
    (run / "generations.jsonl").write_text(json.dumps({"generation": 0, "best_fitness": 0.2, "population": [{}, {}]}) + "\n")
    rows = {r["run_id"]: r for r in client.get("/api/runs").json()}
    assert rows["run-c"]["population_size"] == 2
    assert rows["run-c"]["best_fitness"] == 0.2
    assert rows["run-c"]["status"] == "unknown"
    detail = client.get("/api/runs/run-c").json()
    assert detail["run_id"] == "run-c" and detail["counts"]["generations"] == 1


def test_never_writes(client, runs):
    root: Path = runs["root"]
    snapshot = {p: p.stat().st_mtime_ns for p in root.rglob("*")}
    for url in ("/api/runs", "/api/runs/run-a", "/api/runs/run-a/progress", "/api/runs/run-a/genomes/g2a",
                "/api/runs/run-a/innovations", "/api/runs/run-a/rollouts/g2a/0/r1"):
        assert client.get(url).status_code == 200
    assert {p: p.stat().st_mtime_ns for p in root.rglob("*")} == snapshot
