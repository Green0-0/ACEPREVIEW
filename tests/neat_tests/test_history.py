"""Tests for ace.neat.history (driven by lightweight stand-in genomes)."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from ace.graph.types import END, Kind, WorkerType
from ace.neat.history import History
from ace.neat.innovation import InnovationDB


@dataclass
class FakeGraph:
    nodes: dict[int, Any]
    tag: str

    def structure_hash(self) -> str:
        return f"hash-{self.tag}"


@dataclass
class FakeGenome:
    id: str
    generation: int = 0
    parents: tuple[str, ...] = ()
    operators: list[str] = field(default_factory=list)
    fitness: float | None = None
    adjusted_fitness: float | None = None
    species_id: int | None = None
    workers: int = 1

    def to_graph(self) -> FakeGraph:
        return FakeGraph(nodes={i: object() for i in range(self.workers)}, tag=self.id)

    def to_json(self) -> dict:
        return {"id": self.id, "kind": WorkerType.EDITOR, "dst": END}


@dataclass
class FakeSpecies:
    id: int
    members: list[FakeGenome]
    representative: FakeGenome
    best_fitness: float
    created: int
    last_improved: int


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_directory_created_on_demand(tmp_path: Path) -> None:
    directory = tmp_path / "nested" / "hist"
    history = History(directory)
    assert not directory.exists()
    history.log_generation(0, {"best": 1.0})
    assert (directory / "generations.jsonl").exists()


def test_log_genome_record(tmp_path: Path) -> None:
    history = History(tmp_path)
    g = FakeGenome("a", generation=2, parents=("p", "q"), operators=["crossover", "add_worker"],
                   fitness=0.5, adjusted_fitness=0.44, species_id=3, workers=3)
    history.log_genome(g)
    (record,) = read_jsonl(tmp_path / "genomes.jsonl")
    assert record == {
        "id": "a",
        "generation": 2,
        "parents": ["p", "q"],
        "operators": ["crossover", "add_worker"],
        "species_id": 3,
        "fitness": 0.5,
        "adjusted_fitness": 0.44,
        "structure_hash": "hash-a",
        "num_workers": 3,
        "genome": {"id": "a", "kind": "editor", "dst": "END"},
        "created_at": None,
        "evaluated_at": None,
        "eval_duration_s": None,
        "worker_types": {},
        "task_scores": [],
    }
    assert history.is_logged("a")


def test_minimal_stand_in(tmp_path: Path) -> None:
    class Bare:
        id = "bare"

    history = History(tmp_path)
    history.log_genome(Bare())  # type: ignore[arg-type]
    (record,) = history.iter_genomes()
    assert record["id"] == "bare"
    assert record["parents"] == [] and record["operators"] == []
    assert record["structure_hash"] is None and record["num_workers"] is None
    assert record["genome"] is None


def test_genome_logged_once_including_resume(tmp_path: Path) -> None:
    history = History(tmp_path)
    g = FakeGenome("a")
    history.log_genome(g)
    g.fitness = 1.0
    history.log_genome(g)
    assert len(list(history.iter_genomes())) == 1
    resumed = History(tmp_path)
    assert resumed.is_logged("a")
    resumed.log_genome(g)
    resumed.log_genome(FakeGenome("b"))
    ids = [r["id"] for r in resumed.iter_genomes()]
    assert ids == ["a", "b"]
    assert set(resumed.load_genome_records()) == {"a", "b"}


def test_log_species(tmp_path: Path) -> None:
    history = History(tmp_path)
    a, b, c = FakeGenome("a"), FakeGenome("b"), FakeGenome("c")
    species = [
        FakeSpecies(1, [a, b], a, 0.9, 0, 2),
        FakeSpecies(4, [c], c, 0.1, 1, 1),
    ]
    history.log_species(2, species, 2.5)
    (record,) = list(history.iter_species())
    assert record["generation"] == 2
    assert record["threshold"] == 2.5
    assert record["species"] == [
        {"id": 1, "size": 2, "member_ids": ["a", "b"], "representative_id": "a",
         "best_fitness": 0.9, "created": 0, "last_improved": 2},
        {"id": 4, "size": 1, "member_ids": ["c"], "representative_id": "c",
         "best_fitness": 0.1, "created": 1, "last_improved": 1},
    ]


def test_log_generation(tmp_path: Path) -> None:
    history = History(tmp_path)
    history.log_generation(0, {"best": 0.5, "mean": 0.2, "species": 3})
    history.log_generation(1, {"best": 0.7, "generation": 99, "kinds": (Kind.VIEW,)})
    records = list(history.iter_generations())
    assert records[0] == {"generation": 0, "best": 0.5, "mean": 0.2, "species": 3}
    assert records[1] == {"generation": 1, "best": 0.7, "kinds": ["view"]}
    assert list(records[0])[0] == "generation"


def test_innovations_roundtrip(tmp_path: Path) -> None:
    history = History(tmp_path / "h")
    assert history.load_innovations() is None
    db = InnovationDB()
    n = db.node(("init", 0, WorkerType.EDITOR))
    db.connection(n, END, Kind.EDIT)
    history.save_innovations(db)
    assert (tmp_path / "h" / "innovations.json").exists()
    assert history.load_innovations() == db


def test_lineage(tmp_path: Path) -> None:
    history = History(tmp_path)
    # g0: r1, r2, r3 ; g1: m1 = r1 x r2, m2 = mutate(r2) ; g2: c = m1 x m2 ; g3: d = c x r3
    for g in [
        FakeGenome("r1", 0), FakeGenome("r2", 0), FakeGenome("r3", 0),
        FakeGenome("m1", 1, ("r1", "r2"), ["crossover"]),
        FakeGenome("m2", 1, ("r2",), ["add_worker"]),
        FakeGenome("c", 2, ("m1", "m2"), ["crossover"]),
        FakeGenome("d", 3, ("c", "r3"), ["crossover"]),
    ]:
        history.log_genome(g)
    chain = [r["id"] for r in history.lineage("c")]
    assert chain == ["c", "m1", "m2", "r1", "r2"]
    # r3 (gen 0) is a direct parent of d but must still come after the gen-1/2 ancestors.
    chain_d = [r["id"] for r in history.lineage("d")]
    assert chain_d[0] == "d"
    assert sorted(chain_d) == sorted(["d", "c", "r3", "m1", "m2", "r1", "r2"])
    gens = [r["generation"] for r in history.lineage("d")]
    assert gens == sorted(gens, reverse=True)
    assert [r["id"] for r in history.lineage("r1")] == ["r1"]
    assert history.lineage("missing") == []


def test_lineage_skips_unlogged_parents(tmp_path: Path) -> None:
    history = History(tmp_path)
    history.log_genome(FakeGenome("x", 1, ("ghost", "y")))
    history.log_genome(FakeGenome("y", 0))
    assert [r["id"] for r in history.lineage("x")] == ["x", "y"]


def test_truncated_trailing_line_ignored(tmp_path: Path) -> None:
    history = History(tmp_path)
    history.log_genome(FakeGenome("a"))
    with (tmp_path / "genomes.jsonl").open("a") as fh:
        fh.write('{"id": "b", "gen')
    with pytest.warns(UserWarning, match="malformed"):
        resumed = History(tmp_path)
        assert [r["id"] for r in resumed.iter_genomes()] == ["a"]
    resumed.log_genome(FakeGenome("b"))
    with pytest.warns(UserWarning):
        assert [r["id"] for r in History(tmp_path).iter_genomes()] == ["a", "b"]


# ------------------------------------------------------------ run recording (spec 11)


def test_now_is_iso_utc() -> None:
    from datetime import datetime

    stamp = History.now()
    parsed = datetime.fromisoformat(stamp)
    assert parsed.utcoffset() is not None and parsed.utcoffset().total_seconds() == 0


def test_log_run_rewrites_atomically(tmp_path: Path) -> None:
    history = History(tmp_path / "run")
    assert history.load_run() is None
    history.log_run({"run_id": "r", "status": "running", "path": Path("x")})
    assert history.load_run() == {"run_id": "r", "status": "running", "path": "x"}
    history.log_run({"run_id": "r", "status": "finished"})
    assert history.load_run() == {"run_id": "r", "status": "finished"}
    assert sorted(p.name for p in (tmp_path / "run").iterdir()) == ["run.json"]  # no temp left


def test_log_event_and_iter_events(tmp_path: Path) -> None:
    history = History(tmp_path)
    history.log_event("run_start", run_id="r")
    history.log_event("generation_start", generation=0, kind=Kind.VIEW)
    events = list(history.iter_events())
    assert [e["event"] for e in events] == ["run_start", "generation_start"]
    assert events[0]["run_id"] == "r"
    assert events[1]["kind"] == "view"
    assert list(events[0])[:2] == ["ts", "event"]
    assert all(isinstance(e["ts"], str) for e in events)


def test_log_evaluation_and_lookup(tmp_path: Path) -> None:
    history = History(tmp_path)
    history.log_evaluation({"genome_id": "a", "task_index": 1, "score": 0.5})
    history.log_evaluation({"genome_id": "b", "task_index": 0, "score": 1.0})
    history.log_evaluation({"genome_id": "a", "task_index": 0, "score": 0.0})
    assert [r["genome_id"] for r in history.iter_evaluations()] == ["a", "b", "a"]
    assert [r["task_index"] for r in history.evaluations_for("a")] == [0, 1]
    assert history.evaluations_for("missing") == []
    assert (tmp_path / "evaluations.jsonl").exists()


def test_genome_record_recording_keys(tmp_path: Path) -> None:
    from ace.neat.genome import minimal_genome

    history = History(tmp_path)
    genome = minimal_genome(InnovationDB())
    genome.created_at = "2026-01-01T00:00:00+00:00"
    genome.evaluated_at = "2026-01-01T00:00:01+00:00"
    genome.eval_duration_s = 1.0
    genome.task_scores = [1.0, 0.0]
    history.log_genome(genome)
    (record,) = history.iter_genomes()
    assert record["created_at"] == genome.created_at
    assert record["evaluated_at"] == genome.evaluated_at
    assert record["eval_duration_s"] == 1.0
    assert record["worker_types"] == {"editor": 1}
    assert record["task_scores"] == [1.0, 0.0]
    assert record["genome"]["created_at"] == genome.created_at
