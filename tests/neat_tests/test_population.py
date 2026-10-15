"""Tests for ace.neat.population driven by the structural MockEvaluator."""
from __future__ import annotations

import asyncio
import json
from collections import Counter
from pathlib import Path

import pytest

from ace.neat.config import NeatConfig, SpeciationConfig
from ace.neat.evaluation import MockEvaluator
from ace.neat.genome import Genome
from ace.neat.history import History
from ace.neat.innovation import InnovationDB
from ace.neat.population import Population


def make_config(tmp_path: Path, **overrides) -> NeatConfig:
    speciation = overrides.pop("speciation", SpeciationConfig(target_species=3))
    data = dict(
        population_size=12,
        generations=3,
        seed=7,
        initial_min_workers=1,
        initial_max_workers=5,
        max_workers=10,
        node_cost=0.05,
        history_dir=tmp_path / "hist",
        speciation=speciation,
    )
    data.update(overrides)
    return NeatConfig(**data)


def make_population(tmp_path: Path, evaluator: MockEvaluator | None = None, **overrides) -> Population:
    cfg = make_config(tmp_path, **overrides)
    return Population(
        cfg, evaluator or MockEvaluator(noise=0.2), history=History(cfg.history_dir)
    )


def bootstrapped(pop: Population) -> Population:
    pop.initialize()
    asyncio.run(pop.evaluate())
    asyncio.run(pop.bootstrap_species())
    return pop


def genome_lines(pop: Population) -> list[dict]:
    path = pop.history.genomes_path
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# ------------------------------------------------------------------------ setup


def test_initialize(tmp_path: Path):
    pop = make_population(tmp_path)
    pop.initialize()
    cfg = pop.cfg
    assert len(pop.genomes) == cfg.population_size
    assert len({g.id for g in pop.genomes}) == cfg.population_size
    for g in pop.genomes:
        g.validate()
        assert cfg.initial_min_workers <= g.num_workers() <= cfg.initial_max_workers
        assert g.generation == 0 and g.parents == () and g.operators == ["init"]
        assert g.fitness is None
    assert pop.generation == 0
    assert pop.history.innovations_path.exists()


def test_initialize_avoids_exact_duplicates(tmp_path: Path):
    pop = make_population(tmp_path, initial_max_workers=3, population_size=8)
    pop.initialize()
    keys = {
        (g.structure_hash(), json.dumps(g.to_json()["nodes"], sort_keys=True))
        for g in pop.genomes
    }
    assert len(keys) == len(pop.genomes)


def test_evaluate_sets_fitness_with_node_cost(tmp_path: Path):
    ev = MockEvaluator(noise=0.2)
    pop = make_population(tmp_path, ev)
    pop.initialize()
    asyncio.run(pop.evaluate())
    for g in pop.genomes:
        assert g.fitness == pytest.approx(ev.score(g))
        assert g.adjusted_fitness == pytest.approx(g.fitness - 0.05 * g.num_workers())
    assert ev.calls == len(pop.genomes)
    asyncio.run(pop.evaluate())  # nothing left to evaluate
    assert ev.calls == len(pop.genomes)
    assert [r["id"] for r in genome_lines(pop)] == [g.id for g in pop.genomes]


# -------------------------------------------------------------------- bootstrap


def test_bootstrap_species_top_k(tmp_path: Path):
    pop = make_population(tmp_path)
    pop.initialize()
    asyncio.run(pop.evaluate())
    ranked = sorted(pop.genomes, key=lambda g: (-g.adjusted_fitness, g.num_workers()))
    top = [g.id for g in ranked[:3]]
    asyncio.run(pop.bootstrap_species())

    assert len(pop.genomes) == pop.cfg.population_size
    species = pop.speciator.species
    assert len(species) == 3
    assert {sp.representative.id for sp in species.values()} == set(top)
    assert sorted(sum((sp.members for sp in species.values()), []), key=id) == sorted(
        pop.genomes, key=id
    )
    for g in pop.genomes:
        g.validate()
        assert g.fitness is not None and g.species_id in species
        if g.id not in top:
            assert g.generation == 0 and len(g.parents) == 1 and g.parents[0] in top
            assert g.operators  # mutated at least once
    # every genome ever created is logged once (initial ones and regenerated ones)
    ids = [r["id"] for r in genome_lines(pop)]
    assert len(ids) == len(set(ids)) == 2 * pop.cfg.population_size - 3
    assert len(list(pop.history.iter_species())) == 1
    (gen0,) = list(pop.history.iter_generations())
    assert gen0["generation"] == 0 and gen0["num_species"] == 3


# ------------------------------------------------------------------------- step


def test_step(tmp_path: Path):
    ev = MockEvaluator(noise=0.2)
    pop = bootstrapped(make_population(tmp_path, ev))
    before = {g.id: g for g in pop.genomes}
    fitness_before = {g.id: g.fitness for g in pop.genomes}
    calls_before = ev.calls
    asyncio.run(pop.step())

    assert pop.generation == 1
    assert len(pop.genomes) == pop.cfg.population_size
    assert len({g.id for g in pop.genomes}) == len(pop.genomes)
    elites = [g for g in pop.genomes if g.id in before]
    assert elites and {g.id for g in elites} == set(pop.last_elites)
    for g in elites:
        assert g is before[g.id] and g.fitness == fitness_before[g.id]
    children = [g for g in pop.genomes if g.id not in before]
    assert ev.calls - calls_before == len(children)
    assert set(ev.evaluated_ids[calls_before:]) == {g.id for g in children}
    for g in pop.genomes:
        g.validate()
        assert g.num_workers() <= pop.cfg.max_workers
        assert g.fitness is not None
        assert g.adjusted_fitness == pytest.approx(g.fitness - 0.05 * g.num_workers())
    for g in children:
        assert g.generation == 1
        assert 1 <= len(g.parents) <= 2 and all(p in before for p in g.parents)
        if len(g.parents) == 1:
            assert g.operators and "crossover" not in g.operators
        else:
            assert g.operators[0] in ("crossover", "crossover_fallback")
            assert len(g.operators) >= 2

    lines = genome_lines(pop)
    ids = [r["id"] for r in lines]
    assert len(ids) == len(set(ids))
    assert {g.id for g in pop.genomes} <= set(ids)
    assert pop.history.load_innovations() == pop.innovations
    stats = list(pop.history.iter_generations())
    assert [s["generation"] for s in stats] == [0, 1]
    assert stats[1]["num_elites"] == len(elites)
    assert stats[1]["best_fitness"] == pytest.approx(pop.best().fitness)


def test_run_three_generations(tmp_path: Path):
    pop = make_population(tmp_path)
    best = asyncio.run(pop.run())
    assert pop.generation == 3
    assert best.fitness is not None
    assert best.fitness == max(g.fitness for g in pop.genomes)
    assert [s["generation"] for s in pop.stats] == [0, 1, 2, 3]
    assert len(list(pop.history.iter_species())) == 4


def test_run_speciation_disabled(tmp_path: Path):
    pop = make_population(
        tmp_path, speciation=SpeciationConfig(enabled=False, target_species=3)
    )
    best = asyncio.run(pop.run())
    assert best.fitness is not None
    assert pop.generation == 3
    assert list(pop.speciator.species) == [0]
    assert all(g.species_id == 0 for g in pop.speciator.species[0].members)
    assert len(pop.genomes) == pop.cfg.population_size
    for g in pop.genomes:
        g.validate()


def test_best_prefers_fewer_workers(tmp_path: Path):
    pop = make_population(tmp_path)
    pop.initialize()
    with pytest.raises(ValueError):
        pop.best()
    for g in pop.genomes:
        g.fitness = 1.0
    smallest = min(g.num_workers() for g in pop.genomes)
    assert pop.best().num_workers() == smallest


# ------------------------------------------------------------------------ state


def test_save_load_resume(tmp_path: Path):
    pop = bootstrapped(make_population(tmp_path))
    asyncio.run(pop.step())
    state = tmp_path / "state.json"
    pop.save_state(state)

    restored = make_population(tmp_path / "other")
    restored.load_state(state)
    assert restored.generation == 1
    assert [g.id for g in restored.genomes] == [g.id for g in pop.genomes]
    assert [g.fitness for g in restored.genomes] == [g.fitness for g in pop.genomes]
    assert restored.innovations == pop.innovations
    assert restored.speciator.threshold == pop.speciator.threshold
    assert sorted(restored.speciator.species) == sorted(pop.speciator.species)

    asyncio.run(pop.step())
    asyncio.run(restored.step())
    assert restored.generation == pop.generation == 2
    # Same RNG state, innovations and species: the continuation is identical.
    assert [g.structure_hash() for g in restored.genomes] == [
        g.structure_hash() for g in pop.genomes
    ]
    assert [g.fitness for g in restored.genomes] == [g.fitness for g in pop.genomes]

    best = asyncio.run(restored.run())  # continues to cfg.generations
    assert restored.generation == 3 and best.fitness is not None


# ------------------------------------------------------------------ invariants


def test_innovations_never_mix_types(tmp_path: Path):
    pop = make_population(tmp_path, generations=4)
    asyncio.run(pop.run())
    node_types: dict[int, set[str]] = {}
    conn_keys: dict[int, set[tuple]] = {}
    records = genome_lines(pop)
    assert len(records) > pop.cfg.population_size * 4
    for record in records:
        genome = Genome.from_json(record["genome"])
        genome.validate()
        for inn, node in genome.nodes.items():
            node_types.setdefault(inn, set()).add(node.type.value)
        for inn, conn in genome.connections.items():
            conn_keys.setdefault(inn, set()).add((conn.src, conn.dst, conn.kind.value))
    assert all(len(types) == 1 for types in node_types.values())
    assert all(len(keys) == 1 for keys in conn_keys.values())
    assert not node_types.keys() & conn_keys.keys()
    # every logged genome's parents were logged too
    ids = {r["id"] for r in records}
    assert all(p in ids for r in records for p in r["parents"])
    assert Counter(r["id"] for r in records).most_common(1)[0][1] == 1
    assert isinstance(pop.history.load_innovations(), InnovationDB)


# ------------------------------------------------------------- regression fixes


def test_crossover_receives_worker_cap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import ace.neat.population as population_module

    calls: list[dict] = []
    real = population_module.crossover

    def spy(a, b, rng, **kwargs):
        calls.append(kwargs)
        return real(a, b, rng, **kwargs)

    monkeypatch.setattr(population_module, "crossover", spy)
    pop = bootstrapped(make_population(tmp_path, crossover_rate=1.0))
    asyncio.run(pop.step())
    assert calls
    assert all(kw["max_workers"] == pop.cfg.max_workers for kw in calls)
    assert all(kw["innovations"] is pop.innovations for kw in calls)


def test_oversize_crossover_child_still_discarded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import ace.neat.population as population_module

    def oversize(a, b, rng, **kwargs):
        child = a.copy()
        child.operators = ["crossover"]
        child.num_workers = lambda: 10_000  # type: ignore[method-assign]
        return child

    monkeypatch.setattr(population_module, "crossover", oversize)
    pop = bootstrapped(make_population(tmp_path, crossover_rate=1.0))
    before = {g.id for g in pop.genomes}
    asyncio.run(pop.step())
    children = [g for g in pop.genomes if g.id not in before]
    assert children
    for g in children:  # every crossover was replaced by a mutation-only child
        assert len(g.parents) == 1 and "crossover" not in g.operators
        assert g.num_workers() <= pop.cfg.max_workers


class _FixedEvaluator:
    def __init__(self, scores: dict[str, float]) -> None:
        self.scores = scores

    async def evaluate(self, genome: Genome) -> float:
        return self.scores.get(genome.id, 0.0)


def _viewers_genome(db: InnovationDB, viewers: tuple[int, ...]) -> Genome:
    from ace.graph.types import Kind, WorkerType
    from ace.neat.genome import minimal_genome

    g = minimal_genome(db)
    for i in viewers:
        v = g.add_node(db, ("source", 0, Kind.VIEW, i, WorkerType.VIEWER))
        g.add_connection(db, v.innovation, 0, Kind.VIEW)
    g.validate()
    return g


def test_bootstrap_representatives_found_species_before_others(tmp_path: Path):
    from ace.neat.speciation import Speciator, distance

    db = InnovationDB()
    a = _viewers_genome(db, (0, 1))
    b = _viewers_genome(db, (2, 3))
    middle = _viewers_genome(db, (0, 2))  # halfway between a and b
    cfg = make_config(
        tmp_path, population_size=3, node_cost=0.0, speciation=SpeciationConfig(target_species=2)
    )
    sc = cfg.speciation
    assert distance(a, b, sc) == pytest.approx(8.0)
    assert distance(a, middle, sc) == distance(b, middle, sc) == pytest.approx(4.0)

    # The trap: with the bootstrap threshold (just below d(a, b)) and the in-between genome
    # processed first, plain first-match speciation absorbs both representatives.
    naive = Speciator(sc)
    naive.set_threshold_for([a, b])
    naive.speciate([middle.copy(), a.copy(), b.copy()], 0)
    assert len(naive.species) == 1

    pop = Population(cfg, _FixedEvaluator({a.id: 5.0, b.id: 4.0, middle.id: 1.0}),
                     innovations=db, history=History(cfg.history_dir))
    pop.genomes = [middle, a, b]  # the non-representative comes first
    regrown = middle.copy()
    pop._mutated_child = lambda parent: (regrown, ["mutate_properties"])  # type: ignore[method-assign]
    asyncio.run(pop.bootstrap_species())

    assert pop.speciator.threshold == pytest.approx(8.0 - 1e-6)
    species = pop.speciator.species
    assert len(species) == 2
    assert {sp.representative.id for sp in species.values()} == {a.id, b.id}
    assert a.species_id != b.species_id
    assert regrown in species[regrown.species_id].members
    assert sorted(len(sp.members) for sp in species.values()) == [1, 2]


# --------------------------------------------------------------- run recording (11.3)

GEN_KEYS = {
    "generation", "started_at", "finished_at", "duration_s", "evaluated", "elites",
    "species_count", "threshold", "best_fitness", "mean_fitness", "median_fitness",
    "best_adjusted", "mean_adjusted", "mean_workers", "best_genome_id", "operator_counts",
    "population",
}
POP_KEYS = {
    "genome_id", "species_id", "fitness", "adjusted_fitness", "num_workers",
    "structure_hash", "is_elite", "parents", "operators", "worker_types",
}


def test_generation_records_hold_population(tmp_path: Path):
    pop = make_population(tmp_path)
    asyncio.run(pop.run())
    records = list(pop.history.iter_generations())
    assert [r["generation"] for r in records] == [0, 1, 2, 3]
    genome_ids = {r["id"] for r in genome_lines(pop)}
    for record in records:
        assert GEN_KEYS <= set(record)
        assert record["species_count"] == record["num_species"]
        assert record["duration_s"] >= 0 and record["started_at"] <= record["finished_at"]
        population = record["population"]
        assert len(population) == pop.cfg.population_size
        for entry in population:
            assert set(entry) == POP_KEYS
            assert entry["genome_id"] in genome_ids
            assert sum(entry["worker_types"].values()) == entry["num_workers"]
            assert entry["adjusted_fitness"] == pytest.approx(
                entry["fitness"] - 0.05 * entry["num_workers"]
            )
        elites = [e for e in population if e["is_elite"]]
        assert record["elites"] == len(elites) == record["num_elites"]
        assert record["best_genome_id"] in {e["genome_id"] for e in population}
        best = max(population, key=lambda e: e["fitness"])
        assert record["best_fitness"] == pytest.approx(best["fitness"])
        expected_ops = Counter(op for e in population if not e["is_elite"] for op in e["operators"])
        assert record["operator_counts"] == dict(expected_ops)
        assert record["operator_counts"]
    assert records[0]["elites"] == 0 and "init" in records[0]["operator_counts"]
    assert all(r["elites"] >= 1 for r in records[1:])
    # elites of generation g were members of generation g - 1
    for prev, cur in zip(records, records[1:]):
        prev_ids = {e["genome_id"] for e in prev["population"]}
        assert all(e["genome_id"] in prev_ids for e in cur["population"] if e["is_elite"])


def test_genomes_get_recording_timestamps(tmp_path: Path):
    pop = make_population(tmp_path, generations=1)
    asyncio.run(pop.run())
    for record in genome_lines(pop):
        assert record["created_at"] and record["evaluated_at"]
        assert record["created_at"] <= record["evaluated_at"]
        assert record["eval_duration_s"] >= 0
        assert record["task_scores"] == []  # MockEvaluator has no per-task results
        assert sum(record["worker_types"].values()) == record["num_workers"]
    # the evaluator received the population's history and wrote mock evaluations
    assert pop.evaluator.history is pop.history
    evaluations = list(pop.history.iter_evaluations())
    assert len(evaluations) == len(genome_lines(pop))
    assert all(e["mock"] for e in evaluations)


def test_task_scores_from_evaluator_results(tmp_path: Path):
    class PerTask:
        history = None

        def __init__(self) -> None:
            self.results_by_genome: dict[str, list[dict]] = {}

        async def evaluate(self, genome: Genome) -> float:
            self.results_by_genome[genome.id] = [{"score": 1.0}, {"score": 0.0}]
            return 0.5

    cfg = make_config(tmp_path, population_size=4, speciation=SpeciationConfig(target_species=2))
    pop = Population(cfg, PerTask(), history=History(cfg.history_dir))
    pop.initialize()
    asyncio.run(pop.evaluate())
    for g in pop.genomes:
        assert g.task_scores == [1.0, 0.0] and g.evaluated_at is not None
    assert all(r["task_scores"] == [1.0, 0.0] for r in genome_lines(pop))


def test_speciated_and_genome_evaluated_events(tmp_path: Path):
    pop = make_population(tmp_path)
    asyncio.run(pop.run())
    events = list(pop.history.iter_events())
    speciated = [e for e in events if e["event"] == "speciated"]
    assert [e["generation"] for e in speciated] == [0, 1, 2, 3]
    species_log = list(pop.history.iter_species())
    for event, snapshot in zip(speciated, species_log):
        assert event["species_count"] == len(snapshot["species"])
        assert event["threshold"] == snapshot["threshold"]
    evaluated = [e for e in events if e["event"] == "genome_evaluated"]
    assert len(evaluated) == len(genome_lines(pop))
    assert all({"genome_id", "fitness", "adjusted_fitness", "duration_s"} <= set(e) for e in evaluated)


def test_run_json_finished_with_recorder(tmp_path: Path):
    from ace.neat.recording import RunRecorder

    cfg = make_config(tmp_path)
    history = History(cfg.history_dir)
    ev = MockEvaluator(noise=0.2)
    recorder = RunRecorder(history, run_id="r", config=cfg, model=None,
                           evaluator_info=ev.describe(), seed=cfg.seed)
    pop = Population(cfg, ev, history=history, recorder=recorder)
    best = asyncio.run(pop.run())
    run = history.load_run()
    assert run["status"] == "finished"
    assert run["generations_completed"] == cfg.generations == 3
    assert run["best_fitness"] == pytest.approx(best.fitness)
    assert run["evaluator"] == {"type": "mock", "tasks": []}
