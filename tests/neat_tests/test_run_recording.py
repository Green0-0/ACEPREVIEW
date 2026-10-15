"""Tests for ace.neat.recording.RunRecorder (recording spec 11.1, 11.2, 11.6)."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from ace.graph.llm import ModelSettings
from ace.neat.config import NeatConfig, SpeciationConfig
from ace.neat.evaluation import MockEvaluator
from ace.neat.history import History
from ace.neat.population import Population
from ace.neat.recording import RunRecorder, ace_version

RUN_KEYS = {
    "run_id", "ace_version", "status", "started_at", "finished_at", "duration_s",
    "resumed_from", "seed", "config", "model", "evaluator", "generations_completed",
    "best_genome_id", "best_fitness", "error",
}


def make_recorder(tmp_path: Path, **kwargs) -> tuple[RunRecorder, History, NeatConfig]:
    cfg = NeatConfig(population_size=6, generations=2, seed=3, history_dir=tmp_path,
                     speciation=SpeciationConfig(target_species=2))
    history = History(tmp_path)
    recorder = RunRecorder(
        history,
        run_id="run-1",
        config=cfg,
        model=kwargs.pop("model", None),
        evaluator_info={"type": "mock", "tasks": []},
        seed=cfg.seed,
        **kwargs,
    )
    return recorder, history, cfg


def test_lifecycle_writes_run_json_and_events(tmp_path: Path) -> None:
    recorder, history, cfg = make_recorder(tmp_path)
    recorder.start()
    run = history.load_run()
    assert set(run) == RUN_KEYS
    assert run["status"] == "running" and run["finished_at"] is None
    assert run["duration_s"] is None and run["generations_completed"] == 0
    assert run["config"] == cfg.to_toml_dict()
    assert run["ace_version"] == ace_version() and isinstance(run["ace_version"], str)
    assert run["evaluator"] == {"type": "mock", "tasks": []}
    assert run["seed"] == 3 and run["resumed_from"] is None

    recorder.generation_started(0)
    recorder.generation_finished(0, {"best_fitness": 1.5, "best_genome_id": "g0", "duration_s": 0.25})
    run = history.load_run()
    assert run["generations_completed"] == 0
    assert run["best_genome_id"] == "g0" and run["best_fitness"] == 1.5

    recorder.generation_started(1)
    recorder.generation_finished(1, {"best_fitness": 2.0, "best_id": "g1"})
    recorder.generation_started(2)
    recorder.generation_finished(2, {"best_fitness": 1.0, "best_genome_id": "g2"})
    run = history.load_run()
    assert run["generations_completed"] == 2
    assert run["best_genome_id"] == "g1" and run["best_fitness"] == 2.0  # best ever

    recorder.finish()
    run = history.load_run()
    assert run["status"] == "finished" and run["error"] is None
    assert run["finished_at"] >= run["started_at"]
    assert isinstance(run["duration_s"], float) and run["duration_s"] >= 0

    events = list(history.iter_events())
    assert [e["event"] for e in events] == [
        "run_start",
        "generation_start", "generation_end",
        "generation_start", "generation_end",
        "generation_start", "generation_end",
        "run_end",
    ]
    assert events[0]["run_id"] == "run-1"
    assert events[2] == {**events[2], "generation": 0, "duration_s": 0.25, "best_fitness": 1.5}
    assert isinstance(events[4]["duration_s"], float)
    assert events[-1]["status"] == "finished" and events[-1]["duration_s"] >= 0
    stamps = [e["ts"] for e in events]
    assert stamps == sorted(stamps)


def test_failed_run_logs_error_then_run_end(tmp_path: Path) -> None:
    recorder, history, _ = make_recorder(tmp_path, resumed_from="old/population_state.json")
    recorder.start()
    recorder.finish("failed", error="RuntimeError: boom")
    run = history.load_run()
    assert run["status"] == "failed" and run["error"] == "RuntimeError: boom"
    assert run["resumed_from"] == "old/population_state.json"
    events = [e["event"] for e in history.iter_events()]
    assert events == ["run_start", "error", "run_end"]
    error = list(history.iter_events())[1]
    assert error["message"] == "RuntimeError: boom"


def test_run_json_never_contains_api_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ACE_TEST_SECRET_KEY", "sk-very-secret-value")
    settings = ModelSettings(model="m", api_base_url="http://x/v1",
                             api_key_env_var="ACE_TEST_SECRET_KEY")
    recorder, history, _ = make_recorder(tmp_path, model=settings)
    recorder.start()
    recorder.finish()
    text = (tmp_path / "run.json").read_text()
    assert "sk-very-secret-value" not in text
    run = json.loads(text)
    assert run["model"] == {"model": "m", "api_base_url": "http://x/v1",
                            "api_key_env_var": "ACE_TEST_SECRET_KEY"}
    assert "api_key" not in run["model"]
    for line in (tmp_path / "events.jsonl").read_text().splitlines():
        assert "sk-very-secret-value" not in line


def test_population_run_drives_recorder(tmp_path: Path) -> None:
    recorder, history, cfg = make_recorder(tmp_path)
    pop = Population(cfg, MockEvaluator(noise=0.1), history=history, recorder=recorder)
    best = asyncio.run(pop.run())
    run = history.load_run()
    assert run["status"] == "finished"
    assert run["generations_completed"] == cfg.generations
    assert run["best_fitness"] == pytest.approx(best.fitness)
    names = [e["event"] for e in history.iter_events()]
    assert names[0] == "run_start" and names[-1] == "run_end"
    assert names.count("generation_start") == names.count("generation_end") == 3
    # every generation_start precedes its generation_end
    gens = [(e["event"], e.get("generation")) for e in history.iter_events()
            if e["event"].startswith("generation_")]
    assert gens == [("generation_start", 0), ("generation_end", 0),
                    ("generation_start", 1), ("generation_end", 1),
                    ("generation_start", 2), ("generation_end", 2)]


def test_population_run_failure_marks_run_failed(tmp_path: Path) -> None:
    class Exploding:
        history = None

        async def evaluate(self, genome) -> float:
            raise RuntimeError("evaluator exploded")

    recorder, history, cfg = make_recorder(tmp_path)
    pop = Population(cfg, Exploding(), history=history, recorder=recorder)
    with pytest.raises(RuntimeError):
        asyncio.run(pop.run())
    run = history.load_run()
    assert run["status"] == "failed" and "evaluator exploded" in run["error"]
    assert [e["event"] for e in history.iter_events()][-2:] == ["error", "run_end"]
