"""Stress test: long random sequences of mutation and crossover keep every invariant.

For seeds 0..19 a small population is built with ``random_valid_genome`` and then put
through 300 rounds of randomly chosen steps (``mutate``, a single named operator, or
``crossover`` of a random pair with random / tied / missing fitness). After every step
the produced genome must:

* validate (gene-level checks + ``to_graph().validate()``),
* have valid properties for each node's worker type,
* have at most ``max_workers`` workers,
* have END fed by exactly one enabled connection gene, of kind EDIT,
* never reuse a node innovation with a different worker type (across every genome seen),
  and agree with the InnovationDB on each node innovation's type,

operators that report failure must leave the genome untouched, and the parents of a step
must never be modified. Finally two independent runs with the same seed must produce
byte-identical JSON for the final population (uuid4 is replaced by a seeded generator so
ids are reproducible; a third run with the real uuid4 must agree on everything but ids).
"""
from __future__ import annotations

import json
import random
import uuid

import pytest

from ace.graph.properties import validate_properties
from ace.graph.types import END, Kind, WorkerType
from ace.neat.config import MutationConfig
from ace.neat.crossover import crossover
from ace.neat.genome import Genome
from ace.neat.innovation import InnovationDB
from ace.neat.mutation import OPERATORS, MutationContext, mutate, random_valid_genome

SEEDS = range(20)
ROUNDS = 300
POPULATION = 6
MAX_WORKERS = 8

STRESS_CONFIG = MutationConfig(
    p_property=0.8,
    p_property_per_gene=0.2,
    p_add_connection=0.5,
    p_add_worker=0.5,
    p_insert_merger=0.3,
    p_remove_connection=0.3,
    p_remove_worker=0.3,
    p_toggle_enable=0.3,
    max_attempts=10,
)


def _dump(genome: Genome) -> str:
    return json.dumps(genome.to_json(), sort_keys=True, separators=(",", ":"))


def _check(genome: Genome, db: InnovationDB, seen_types: dict[int, WorkerType],
           db_types: dict[int, WorkerType]) -> None:
    genome.validate()
    for node in genome.nodes.values():
        validate_properties(node.type, node.properties)
    assert genome.num_workers() <= MAX_WORKERS, (
        f"{genome.num_workers()} workers > max_workers {MAX_WORKERS} ({genome.operators})"
    )
    into_end = genome.connections_into(END, enabled_only=True)
    assert len(into_end) == 1 and into_end[0].kind is Kind.EDIT
    graph = genome.to_graph()
    assert graph.end_source() == into_end[0].src
    assert set(graph.nodes) == set(genome.nodes)
    for inn, node in genome.nodes.items():
        assert inn not in genome.connections
        assert seen_types.setdefault(inn, node.type) is node.type, (
            f"node innovation {inn} seen as {seen_types[inn]} and {node.type}"
        )
        assert db_types[inn] is WorkerType(node.type), f"innovation {inn} disagrees with DB"
    for inn, conn in genome.connections.items():
        assert db.peek_connection(conn.src, conn.dst, conn.kind) == inn


def _run(seed: int) -> tuple[list[Genome], dict[str, int]]:
    rng = random.Random(seed)
    db = InnovationDB()
    ctx = MutationContext(rng=rng, innovations=db, config=STRESS_CONFIG,
                          max_workers=MAX_WORKERS)
    seen: dict[int, WorkerType] = {}
    stats = {"mutate": 0, "operator": 0, "crossover": 0, "crossover_fallback": 0}

    def db_types() -> dict[int, WorkerType]:
        return {inn: WorkerType(key[-1]) for key, inn in db.node_entries().items()}

    population = [
        random_valid_genome(db, rng, STRESS_CONFIG, min_workers=1, max_workers=5)
        for _ in range(POPULATION)
    ]
    for g in population:
        _check(g, db, seen, db_types())

    for _ in range(ROUNDS):
        roll = rng.random()
        if roll < 0.35:
            parent = rng.choice(population)
            before = _dump(parent)
            child = mutate(parent, ctx)
            assert _dump(parent) == before, "mutate modified its input"
            stats["mutate"] += 1
        elif roll < 0.65:
            parent = rng.choice(population)
            child = parent.copy()
            name = rng.choice(sorted(OPERATORS))
            before = _dump(child)
            ok = OPERATORS[name](child, ctx)
            if not ok:
                assert _dump(child) == before, f"failed {name} modified the genome"
            stats["operator"] += 1
        else:
            i, j = rng.sample(range(len(population)), 2)
            a, b = population[i], population[j]
            before_a, before_b = _dump(a), _dump(b)
            child = crossover(a, b, rng, innovations=db, max_workers=MAX_WORKERS)
            assert (_dump(a), _dump(b)) == (before_a, before_b), "crossover modified a parent"
            stats["crossover"] += 1
            if "crossover_fallback" in child.operators:
                stats["crossover_fallback"] += 1
        _check(child, db, seen, db_types())
        # Random fitness, including ties and missing values, to exercise every branch of
        # the crossover gene inheritance (fitter-only vs both parents).
        child.fitness = rng.choice([None, 0.0, 0.5, 1.0, rng.random()])
        child.adjusted_fitness = None if child.fitness is None else child.fitness - 0.01 * child.num_workers()
        population[rng.randrange(len(population))] = child

    for g in population:
        _check(g, db, seen, db_types())
    return population, stats


class _SeededUuid:
    def __init__(self, seed: int) -> None:
        self._rng = random.Random(f"uuid-{seed}")

    def __call__(self) -> uuid.UUID:
        return uuid.UUID(int=self._rng.getrandbits(128), version=4)


def _strip_ids(genome: Genome) -> str:
    data = genome.to_json()
    del data["id"], data["parents"]
    return json.dumps(data, sort_keys=True, separators=(",", ":"))


@pytest.mark.parametrize("seed", SEEDS)
def test_stress_invariants_and_determinism(seed: int, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(uuid, "uuid4", _SeededUuid(seed))
    first, stats = _run(seed)
    monkeypatch.setattr(uuid, "uuid4", _SeededUuid(seed))
    second, _ = _run(seed)
    assert [_dump(g) for g in first] == [_dump(g) for g in second]
    assert stats["crossover"] > 0 and stats["mutate"] > 0 and stats["operator"] > 0

    # Ids must not influence any random choice: with real uuid4 everything else agrees.
    monkeypatch.undo()
    third, _ = _run(seed)
    assert [_strip_ids(g) for g in first] == [_strip_ids(g) for g in third]
