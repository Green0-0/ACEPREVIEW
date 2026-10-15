"""WebUI fixtures: hand-built run directories following spec section 11.

``runs_root`` contains:

* ``run-a`` (finished): five genomes over three generations forming a lineage
  (g0a, g0b -> g1a [add_worker], g1b [crossover, mutate_properties] -> g2a
  [insert_merger, mutate_properties], a diamond VIEWER -> EDITOR x2 -> JUDGEMERGER -> END),
  generation/species/event records, evaluations (successes, a failure with skipped workers,
  failures for aggregation), innovations.json and one rollout file.
* ``run-b`` (running): only ``run.json``, ``events.jsonl`` and one generation record.
* ``empty-dir``: no run files (must not be listed).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from ace.graph.types import END, Kind, WorkerType
from ace.neat.config import NeatConfig, SpeciationConfig
from ace.neat.genome import Genome, minimal_genome
from ace.neat.innovation import InnovationDB
from ace.webui.app import create_app

TS = "2026-10-15T10:00:{:02d}+00:00"

# Adjusted fitness values used by the hand-computed operator statistics.
FITNESS = {
    "g0a": (0.50, 0.48),
    "g0b": (0.30, 0.28),
    "g1a": (0.70, 0.66),
    "g1b": (0.25, 0.20),
    "g2a": (0.90, 0.80),
}


def ts(second: int) -> str:
    return TS.format(second % 60)


def _set_meta(genome: Genome, gid: str, generation: int, parents: tuple[str, ...], operators: list[str], species: int) -> Genome:
    genome.id = gid
    genome.generation = generation
    genome.parents = parents
    genome.operators = operators
    genome.fitness, genome.adjusted_fitness = FITNESS[gid]
    genome.species_id = species
    genome.validate()
    return genome


def build_genomes(db: InnovationDB) -> dict[str, Genome]:
    g0a = _set_meta(minimal_genome(db), "g0a", 0, (), [], 0)
    g0b = minimal_genome(db)
    editor = next(iter(g0b.nodes))
    g0b.nodes[editor].properties["role"] = 1
    g0b.nodes[editor].properties["max_steps"] = 2
    _set_meta(g0b, "g0b", 0, (), [], 0)

    # add_worker: VIEWER -> EDITOR
    g1a = g0a.copy()
    viewer = g1a.add_node(db, ("source", editor, Kind.VIEW, WorkerType.VIEWER))
    g1a.add_connection(db, viewer.innovation, editor, Kind.VIEW)
    _set_meta(g1a, "g1a", 1, ("g0a",), ["add_worker"], 1)

    g1b = g0b.copy()
    g1b.nodes[editor].properties["strategy"] = 3
    _set_meta(g1b, "g1b", 1, ("g0a", "g0b"), ["crossover", "mutate_properties"], 0)

    # insert_merger: diamond VIEWER -> {EDITOR, EDITOR} -> JUDGEMERGER -> END
    g2a = g1a.copy()
    end_conn = g2a.find_connection(editor, END, Kind.EDIT)
    branch = g2a.add_node(db, ("branch", viewer.innovation, WorkerType.EDITOR))
    merger = g2a.add_node(db, ("merge", end_conn.innovation, WorkerType.JUDGEMERGER))
    g2a.add_connection(db, viewer.innovation, branch.innovation, Kind.VIEW)
    g2a.add_connection(db, editor, merger.innovation, Kind.EDIT)
    g2a.add_connection(db, branch.innovation, merger.innovation, Kind.EDIT)
    g2a.add_connection(db, viewer.innovation, merger.innovation, Kind.VIEW)
    g2a.add_connection(db, merger.innovation, END, Kind.EDIT)
    end_conn.enabled = False
    g2a.nodes[editor].properties["role"] = 2
    _set_meta(g2a, "g2a", 2, ("g1a",), ["insert_merger", "mutate_properties"], 1)
    return {g.id: g for g in (g0a, g0b, g1a, g1b, g2a)}


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record) + "\n")


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


def genome_record(g: Genome, second: int) -> dict[str, Any]:
    graph = g.to_graph()
    types: dict[str, int] = {}
    for node in graph.nodes.values():
        types[node.type.value] = types.get(node.type.value, 0) + 1
    return {
        "id": g.id,
        "generation": g.generation,
        "parents": list(g.parents),
        "operators": list(g.operators),
        "species_id": g.species_id,
        "fitness": g.fitness,
        "adjusted_fitness": g.adjusted_fitness,
        "structure_hash": graph.structure_hash(),
        "num_workers": len(graph.nodes),
        "genome": g.to_json(),
        "created_at": ts(second),
        "evaluated_at": ts(second + 1),
        "eval_duration_s": 1.0,
        "worker_types": types,
        "task_scores": [g.fitness],
    }


def population_entry(g: Genome, elite: bool = False) -> dict[str, Any]:
    record = genome_record(g, 0)
    return {
        "genome_id": g.id,
        "species_id": g.species_id,
        "fitness": g.fitness,
        "adjusted_fitness": g.adjusted_fitness,
        "num_workers": record["num_workers"],
        "structure_hash": record["structure_hash"],
        "is_elite": elite,
        "parents": list(g.parents),
        "operators": list(g.operators),
        "worker_types": record["worker_types"],
    }


def generation_record(gen: int, members: list[Genome], second: int) -> dict[str, Any]:
    fits = [g.fitness for g in members]
    adjs = [g.adjusted_fitness for g in members]
    best = max(members, key=lambda g: g.fitness)
    ops: dict[str, int] = {}
    for g in members:
        for op in g.operators:
            ops[op] = ops.get(op, 0) + 1
    return {
        "generation": gen,
        "started_at": ts(second),
        "finished_at": ts(second + 5),
        "duration_s": 5.0,
        "evaluated": len(members),
        "elites": 1,
        "species_count": len({g.species_id for g in members}),
        "threshold": 3.0 - gen * 0.5,
        "best_fitness": best.fitness,
        "mean_fitness": sum(fits) / len(fits),
        "median_fitness": sorted(fits)[len(fits) // 2],
        "best_adjusted": max(adjs),
        "mean_adjusted": sum(adjs) / len(adjs),
        "mean_workers": sum(len(g.nodes) for g in members) / len(members),
        "best_genome_id": best.id,
        "operator_counts": ops,
        "population": [population_entry(g, elite=(g is best)) for g in members],
    }


def worker(node_id: int, wtype: str, status: str, *, view: dict | None = None, rollout_dir: str = "", error: str | None = None) -> dict[str, Any]:
    return {
        "node_id": node_id,
        "type": wtype,
        "properties": {"role": 0},
        "started_at": ts(1) if status != "skipped" else None,
        "finished_at": ts(2) if status != "skipped" else None,
        "duration_s": 1.0 if status != "skipped" else 0.0,
        "status": status,
        "error": error,
        "rollout_ids": ["r1"] if status == "ok" else [],
        "rollout_dir": rollout_dir,
        "llm_calls": 2 if status == "ok" else 0,
        "tool_calls": 1 if status == "ok" else 0,
        "report": "looked at mod.py" if status == "ok" else None,
        "view": view,
        "view_summary": {"files": 1 if view else 0, "entries": 1 if view else 0, "rendered_chars": 10},
        "edit_touched_paths": [],
        "edit_chars": 0,
    }


def evaluation(gid: str, gen: int, task_index: int, project_dir: str, *, score: float, failure: str | None = None,
               failure_type: str | None = None, failure_node_id: int | None = None,
               workers: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {
        "genome_id": gid,
        "generation": gen,
        "task_index": task_index,
        "task": {"project_dir": project_dir, "prompt": "fix the bug", "scorer": "nonempty_diff"},
        "started_at": ts(10),
        "finished_at": ts(12),
        "duration_s": 2.0,
        "score": score,
        "failure": failure,
        "failure_node_id": failure_node_id,
        "failure_type": failure_type,
        "diff": None if failure else "--- a/mod.py\n+++ b/mod.py\n@@\n-x = 1\n+x = 2\n",
        "touched_paths": [] if failure else ["mod.py"],
        "diff_chars": 0 if failure else 40,
        "order": [w["node_id"] for w in workers or []],
        "workers": workers or [],
        "inputs": {},
    }


VIEW_JSON = {"files": {"mod.py": {"levels": {"": 2}, "line_ranges": []}}}


def build_run_a(root: Path, project: Path) -> dict[str, Any]:
    run = root / "run-a"
    run.mkdir(parents=True)
    db = InnovationDB()
    genomes = build_genomes(db)
    config = NeatConfig(
        population_size=4,
        speciation=SpeciationConfig(c_excess=2.0, c_disjoint=1.5, c_properties=0.25, target_species=2),
    )
    run_json = {
        "run_id": "run-a",
        "ace_version": "0",
        "status": "finished",
        "started_at": ts(0),
        "finished_at": ts(59),
        "duration_s": 59.0,
        "resumed_from": None,
        "seed": 7,
        "config": config.to_toml_dict(),
        "model": {"model": "fake-model", "api_base_url": None, "api_key_env_var": "FAKE_KEY"},
        "evaluator": {"type": "task", "tasks": [
            {"index": 0, "project_dir": str(project), "prompt": "fix the bug", "scorer": "nonempty_diff"},
            {"index": 1, "project_dir": str(root / "missing-project"), "prompt": "other", "scorer": "nonempty_diff"},
        ]},
        "generations_completed": 3,
        "best_genome_id": "g2a",
        "best_fitness": 0.9,
        "error": None,
    }
    (run / "run.json").write_text(json.dumps(run_json), encoding="utf-8")

    g = genomes
    gens = [
        generation_record(0, [g["g0a"], g["g0b"]], 0),
        generation_record(1, [g["g0a"], g["g1a"], g["g1b"]], 10),
        generation_record(2, [g["g1a"], g["g2a"]], 20),
    ]
    write_jsonl(run / "generations.jsonl", gens)
    write_jsonl(run / "genomes.jsonl", [genome_record(x, i * 5) for i, x in enumerate(genomes.values())])

    def sp(sid: int, members: list[Genome], created: int) -> dict[str, Any]:
        return {"id": sid, "size": len(members), "member_ids": [m.id for m in members],
                "representative_id": members[0].id, "best_fitness": max(m.fitness for m in members),
                "created": created, "last_improved": created}

    write_jsonl(run / "species.jsonl", [
        {"generation": 0, "threshold": 3.0, "species": [sp(0, [g["g0a"], g["g0b"]], 0)]},
        {"generation": 1, "threshold": 2.5, "species": [sp(0, [g["g0a"], g["g1b"]], 0), sp(1, [g["g1a"]], 1)]},
        {"generation": 2, "threshold": 2.0, "species": [sp(1, [g["g1a"], g["g2a"]], 1)]},
    ])

    events = [{"ts": ts(0), "event": "run_start", "run_id": "run-a"}]
    for gen in range(3):
        events.append({"ts": ts(gen * 10), "event": "generation_start", "generation": gen})
        events.append({"ts": ts(gen * 10 + 5), "event": "generation_end", "generation": gen,
                       "duration_s": 5.0, "best_fitness": gens[gen]["best_fitness"]})
    events.append({"ts": ts(59), "event": "run_end", "status": "finished", "duration_s": 59.0})
    write_jsonl(run / "events.jsonl", events)

    rollout_dir = str(run / "rollouts" / "g2a" / "0")
    diamond = g["g2a"].to_graph()
    order = diamond.topological_order()
    viewer_id = next(n for n in order if diamond.nodes[n].type == WorkerType.VIEWER)
    merger_id = next(n for n in order if diamond.nodes[n].type == WorkerType.JUDGEMERGER)
    editors = sorted(n for n in order if diamond.nodes[n].type == WorkerType.EDITOR)
    ok_workers = [
        worker(viewer_id, "viewer", "ok", view=VIEW_JSON, rollout_dir=rollout_dir),
        worker(editors[0], "editor", "ok", view=VIEW_JSON, rollout_dir=rollout_dir),
        worker(editors[1], "editor", "ok", rollout_dir=rollout_dir),
        worker(merger_id, "judgemerger", "ok", rollout_dir=rollout_dir),
    ]
    failed_workers = [
        worker(viewer_id, "viewer", "ok", view=VIEW_JSON),
        worker(editors[0], "editor", "failed", error="boom"),
        worker(editors[1], "editor", "skipped"),
        worker(merger_id, "judgemerger", "skipped"),
    ]
    missing = str(root / "missing-project")
    editor0 = editors[0]
    evaluations = [
        evaluation("g0a", 0, 0, str(project), score=0.5, workers=[worker(editor0, "editor", "ok", view=VIEW_JSON)]),
        evaluation("g0b", 0, 0, str(project), score=0.0, failure="editor timed out", failure_type="timeout",
                   failure_node_id=editor0, workers=[worker(editor0, "editor", "failed", error="timeout")]),
        evaluation("g1a", 1, 0, str(project), score=0.7, workers=[]),
        evaluation("g1b", 1, 0, str(project), score=0.0, failure="editor timed out again", failure_type="timeout",
                   failure_node_id=None, workers=[worker(editor0, "editor", "failed", error="timeout")]),
        evaluation("g2a", 2, 0, str(project), score=0.9, workers=ok_workers),
        evaluation("g2a", 2, 1, missing, score=0.0, failure="GraphFailure: node failed", failure_type="GraphFailure",
                   failure_node_id=editor0, workers=failed_workers),
    ]
    write_jsonl(run / "evaluations.jsonl", evaluations)
    db.save(run / "innovations.json")

    rollout = {
        "id": "r1",
        "raw_chat": [
            {"role": "system", "content": "You are a viewer."},
            {"role": "user", "content": "fix the bug"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "view_file", "arguments": "{\"path\": \"mod.py\"}"}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "x = 1"},
            {"role": "assistant", "content": "done"},
        ],
    }
    (run / "rollouts" / "g2a" / "0").mkdir(parents=True)
    (run / "rollouts" / "g2a" / "0" / "r1.json").write_text(json.dumps(rollout), encoding="utf-8")
    return {
        "dir": run,
        "db": db,
        "genomes": genomes,
        "config": config,
        "editor": editor0,
        "viewer": viewer_id,
        "merger": merger_id,
        "branch": editors[1],
    }


def build_run_b(root: Path) -> Path:
    run = root / "run-b"
    run.mkdir(parents=True)
    (run / "run.json").write_text(json.dumps({
        "run_id": "run-b", "ace_version": "0", "status": "running", "started_at": ts(30),
        "finished_at": None, "duration_s": None, "resumed_from": None, "seed": 1,
        "config": NeatConfig(population_size=6, speciation=SpeciationConfig(target_species=2)).to_toml_dict(),
        "model": {"model": "fake-model", "api_base_url": None, "api_key_env_var": None},
        "evaluator": {"type": "mock", "tasks": []},
        "generations_completed": 1, "best_genome_id": None, "best_fitness": None, "error": None,
    }), encoding="utf-8")
    write_jsonl(run / "events.jsonl", [
        {"ts": ts(30), "event": "run_start", "run_id": "run-b"},
        {"ts": ts(31), "event": "generation_start", "generation": 0},
    ])
    write_jsonl(run / "generations.jsonl", [{
        "generation": 0, "started_at": ts(31), "finished_at": ts(32), "duration_s": 1.0,
        "evaluated": 6, "elites": 0, "species_count": 1, "threshold": 3.0, "best_fitness": 0.1,
        "mean_fitness": 0.05, "median_fitness": 0.05, "best_adjusted": 0.08, "mean_adjusted": 0.03,
        "mean_workers": 1.0, "best_genome_id": "x0", "operator_counts": {}, "population": [],
    }])
    return run


@pytest.fixture
def project(tmp_path: Path) -> Path:
    proj = tmp_path / "project"
    proj.mkdir()
    (proj / "mod.py").write_text('"""Module."""\n\n\ndef f(a):\n    return a + 1\n\n\nx = 1\n', encoding="utf-8")
    return proj


@pytest.fixture
def runs(tmp_path: Path, project: Path) -> dict[str, Any]:
    root = tmp_path / "runs"
    root.mkdir()
    info = build_run_a(root, project)
    build_run_b(root)
    (root / "empty-dir").mkdir()
    return {"root": root, "a": info, "b": root / "run-b", "project": project}


@pytest.fixture
def client(runs: dict[str, Any]) -> TestClient:
    return TestClient(create_app(runs["root"]))
