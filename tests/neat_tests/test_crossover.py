"""Tests for ace.neat.crossover."""
from __future__ import annotations

import inspect
import random
from collections.abc import Callable

import pytest

from ace.graph.properties import random_properties
from ace.graph.types import END, InvalidGraph, Kind, WorkerType
from ace.neat.crossover import align, compare_fitness, crossover, repair
from ace.neat.genome import ConnectionGene, Genome, NodeGene, minimal_genome
from ace.neat.innovation import InnovationDB

try:  # mutation.py is written concurrently; use it when it is available.
    from ace.neat import mutation as _mutation
except ImportError:  # pragma: no cover - depends on the build state of the repo
    _mutation = None


# ------------------------------------------------------------------ builders


def genes_json(g: Genome) -> dict:
    """The gene part of a genome's JSON (no id / metadata)."""
    data = g.to_json()
    return {"nodes": data["nodes"], "connections": data["connections"]}


def viewer_genome(db: InnovationDB, key: tuple | None = None) -> Genome:
    """Minimal genome plus a VIEWER source feeding the editor with a VIEW edge."""
    g = minimal_genome(db)
    v = g.add_node(db, key or ("source", 0, Kind.VIEW, WorkerType.VIEWER))
    g.add_connection(db, v.innovation, 0, Kind.VIEW)
    return g


def diamond_genome(db: InnovationDB) -> Genome:
    """viewer -> {editor0, editor1} -> judgemerger -> END; editor0 -> END disabled."""
    g = minimal_genome(db)
    e0 = 0
    e1 = g.add_node(db, ("init", 1, WorkerType.EDITOR)).innovation
    v = g.add_node(db, ("source", e0, Kind.VIEW, WorkerType.VIEWER)).innovation
    m = g.add_node(db, ("merge", e0, e1, WorkerType.JUDGEMERGER)).innovation
    g.add_connection(db, v, e0, Kind.VIEW)
    g.add_connection(db, v, e1, Kind.VIEW)
    g.add_connection(db, e0, m, Kind.EDIT)
    g.add_connection(db, e1, m, Kind.EDIT)
    g.add_connection(db, v, m, Kind.VIEW)
    g.add_connection(db, m, END, Kind.EDIT)
    g.find_connection(e0, END, Kind.EDIT).enabled = False
    return g


def critic_chain_genome(db: InnovationDB) -> Genome:
    """editor -> critic (EDIT + VIEW) -> END; editor -> END disabled."""
    g = minimal_genome(db)
    end_conn = db.connection(0, END, Kind.EDIT)
    c = g.add_node(db, ("critic", end_conn, WorkerType.CRITIC)).innovation
    g.add_connection(db, 0, c, Kind.EDIT)
    g.add_connection(db, 0, c, Kind.VIEW)
    g.add_connection(db, c, END, Kind.EDIT)
    g.find_connection(0, END, Kind.EDIT).enabled = False
    return g


HAND_BUILT: list[Callable[[InnovationDB], Genome]] = [
    minimal_genome,
    viewer_genome,
    diamond_genome,
    critic_chain_genome,
]


def randomise_properties(g: Genome, rng: random.Random) -> Genome:
    for node in g.nodes.values():
        node.properties = random_properties(node.type, rng)
    return g


def _call_with_known_args(func: Callable, **available: object) -> object:
    """Call ``func`` passing only the keyword arguments its signature names."""
    params = inspect.signature(func).parameters
    kwargs = {name: value for name, value in available.items() if name in params}
    return func(**kwargs)


def mutation_pair_factory() -> Callable[[random.Random, InnovationDB], tuple[Genome, Genome]] | None:
    """Pairs built through ``ace.neat.mutation`` if its API is available, else None."""
    if _mutation is None:
        return None
    needed = ("random_valid_genome", "mutate", "MutationContext")
    if not all(hasattr(_mutation, name) for name in needed):
        return None
    from ace.neat.config import MutationConfig

    def make(rng: random.Random, db: InnovationDB) -> tuple[Genome, Genome]:
        ctx = _call_with_known_args(
            _mutation.MutationContext,
            rng=rng,
            innovations=db,
            config=MutationConfig(),
            max_workers=8,
        )
        base = _call_with_known_args(
            _mutation.random_valid_genome,
            rng=rng,
            innovations=db,
            db=db,
            ctx=ctx,
            context=ctx,
            config=MutationConfig(),
            max_workers=8,
        )
        a, b = base, base
        for _ in range(rng.randint(0, 3)):
            a = _mutation.mutate(a, ctx)
        for _ in range(rng.randint(0, 3)):
            b = _mutation.mutate(b, ctx)
        return a, b

    try:  # the signature is guessed; fall back to hand-built pairs if it does not fit
        make(random.Random(0), InnovationDB())
    except TypeError:
        return None
    return make


def hand_built_pair(rng: random.Random, db: InnovationDB) -> tuple[Genome, Genome]:
    a = randomise_properties(rng.choice(HAND_BUILT)(db), rng)
    b = randomise_properties(rng.choice(HAND_BUILT)(db), rng)
    return a, b


# --------------------------------------------------------------------- tests


def test_self_crossover_reproduces_structure() -> None:
    rng = random.Random(1)
    for build in HAND_BUILT:
        db = InnovationDB()
        g = randomise_properties(build(db), rng)
        g.fitness = 1.0
        for _ in range(20):
            child = crossover(g, g, rng, innovations=db)
            assert child.operators == ["crossover"]
            assert genes_json(child) == genes_json(g)
            child.validate()


def test_align() -> None:
    db = InnovationDB()
    a = diamond_genome(db)
    b = critic_chain_genome(db)
    matched_nodes, matched_conns, only_a, only_b = align(a, b)
    assert [ga.innovation for ga, _ in matched_nodes] == [0]
    assert [ca.innovation for ca, _ in matched_conns] == [1]
    assert all(ga.innovation == gb.innovation for ga, gb in matched_nodes + matched_conns)
    assert {g.innovation for g in only_a} == (set(a.nodes) | set(a.connections)) - {0, 1}
    assert {g.innovation for g in only_b} == (set(b.nodes) | set(b.connections)) - {0, 1}
    nodes_a = [g for g in only_a if isinstance(g, NodeGene)]
    assert only_a[: len(nodes_a)] == nodes_a  # node genes first
    assert [g.innovation for g in nodes_a] == sorted(g.innovation for g in nodes_a)


def test_compare_fitness() -> None:
    db = InnovationDB()
    a, b = minimal_genome(db), minimal_genome(db)
    assert compare_fitness(a, b) == 0
    a.fitness, b.fitness = 2.0, 1.0
    assert compare_fitness(a, b) == 1
    a.adjusted_fitness, b.adjusted_fitness = 0.1, 0.5  # adjusted wins when both set
    assert compare_fitness(a, b) == -1
    b.adjusted_fitness = None  # falls back to raw fitness
    assert compare_fitness(a, b) == 1
    a.fitness = None
    assert compare_fitness(a, b) == 0
    a.fitness = 1.0
    assert compare_fitness(a, b) == 0


def test_matched_properties_come_from_one_parent_per_property() -> None:
    rng = random.Random(2)
    db = InnovationDB()
    a = diamond_genome(db)
    b = diamond_genome(db)
    for node in a.nodes.values():
        node.properties = {name: 0 for name in node.properties}
        node.properties["only_in_a"] = 7
    for node in b.nodes.values():
        node.properties = {name: 1 for name in node.properties}
        node.properties["only_in_b"] = 5
    seen: dict[tuple[int, str], set[int]] = {}
    for _ in range(200):
        child = crossover(a, b, rng, innovations=db)
        for inn, node in child.nodes.items():
            assert node.type is a.nodes[inn].type
            assert node.properties["only_in_a"] == 7
            assert node.properties["only_in_b"] == 5
            for name, value in node.properties.items():
                if name.startswith("only_in"):
                    continue
                assert value in (a.nodes[inn].properties[name], b.nodes[inn].properties[name])
                seen.setdefault((inn, name), set()).add(value)
    # Gene-wise choice: every property took each parent's value at least once.
    assert seen and all(values == {0, 1} for values in seen.values())


def test_child_does_not_alias_parents() -> None:
    rng = random.Random(3)
    db = InnovationDB()
    a = viewer_genome(db)
    b = minimal_genome(db)
    a.fitness, b.fitness = 2.0, 1.0
    before_a, before_b = a.to_json(), b.to_json()
    child = crossover(a, b, rng, innovations=db)
    for node in child.nodes.values():
        node.properties["role"] = 99
    for conn in child.connections.values():
        conn.enabled = not conn.enabled
    assert a.to_json() == before_a and b.to_json() == before_b


def test_disjoint_genes_from_fitter_parent_only() -> None:
    rng = random.Random(4)
    db = InnovationDB()
    a = viewer_genome(db)
    b = minimal_genome(db)
    viewer = next(i for i, n in a.nodes.items() if n.type is WorkerType.VIEWER)
    for _ in range(20):
        a.fitness, b.fitness = 2.0, 1.0
        child = crossover(a, b, rng, innovations=db)
        assert set(child.nodes) == set(a.nodes)
        assert set(child.connections) == set(a.connections)
        a.fitness, b.fitness = 1.0, 2.0
        child = crossover(a, b, rng, innovations=db)
        assert viewer not in child.nodes
        assert set(child.nodes) == set(b.nodes)
        assert set(child.connections) == set(b.connections)
        # adjusted fitness takes precedence over raw fitness
        a.adjusted_fitness, b.adjusted_fitness = 0.5, 0.1
        child = crossover(a, b, rng, innovations=db)
        assert viewer in child.nodes
        a.adjusted_fitness = b.adjusted_fitness = None


def test_disjoint_genes_from_both_parents_when_equal() -> None:
    rng = random.Random(5)
    db = InnovationDB()
    a = viewer_genome(db, ("source", 0, Kind.VIEW, WorkerType.VIEWER))
    b = viewer_genome(db, ("init", 1, WorkerType.VIEWER))
    _, _, only_a, only_b = align(a, b)
    assert only_a and only_b
    for fa, fb in [(None, None), (1.0, 1.0), (None, 3.0)]:
        a.fitness, b.fitness = fa, fb
        child = crossover(a, b, rng, innovations=db)
        assert child.operators == ["crossover"]
        assert set(child.nodes) == set(a.nodes) | set(b.nodes)
        assert set(child.connections) == set(a.connections) | set(b.connections)
        child.validate()


def test_disabled_in_either_parent_is_disabled_75_percent() -> None:
    rng = random.Random(6)
    db = InnovationDB()
    a = viewer_genome(db)
    v = next(i for i, n in a.nodes.items() if n.type is WorkerType.VIEWER)
    a.add_connection(db, v, 0, Kind.REPORT, enabled=False)
    b = a.copy()
    b.find_connection(v, 0, Kind.REPORT).enabled = True
    a.validate()
    b.validate()
    report = db.connection(v, 0, Kind.REPORT)
    runs = 4000
    for parents in [(a, b), (b, a), (a, a)]:
        disabled = 0
        for _ in range(runs):
            child = crossover(*parents, rng, innovations=db)
            disabled += not child.connections[report].enabled
        assert abs(disabled / runs - 0.75) < 0.03, parents
    # Enabled in both parents: always enabled.
    for _ in range(200):
        assert crossover(b, b, rng, innovations=db).connections[report].enabled


def test_child_metadata() -> None:
    rng = random.Random(7)
    db = InnovationDB()
    a = diamond_genome(db)
    b = critic_chain_genome(db)
    a.generation, b.generation = 3, 5
    a.fitness, b.fitness = 1.0, 0.5
    a.adjusted_fitness, b.adjusted_fitness = 0.9, 0.4
    a.species_id = b.species_id = 2
    a.operators, b.operators = ["mutate"], ["add_worker"]
    child = crossover(a, b, rng, innovations=db)
    assert child.id not in (a.id, b.id) and len(child.id) == 32
    assert child.generation == 6
    assert child.parents == (a.id, b.id)
    assert child.operators == ["crossover"]
    assert child.fitness is None and child.adjusted_fitness is None and child.species_id is None
    child.validate()


def test_type_mismatch_raises() -> None:
    rng = random.Random(8)
    db = InnovationDB()
    a = diamond_genome(db)
    b = a.copy()
    merger = next(i for i, n in b.nodes.items() if n.type is WorkerType.JUDGEMERGER)
    b.nodes[merger].type = WorkerType.PICKMERGER
    with pytest.raises(InvalidGraph):
        crossover(a, b, rng, innovations=db)


def test_child_always_validates_across_random_pairs() -> None:
    rng = random.Random(9)
    make_pair = mutation_pair_factory() or hand_built_pair
    fitness_choices = [None, 0.0, 0.5, 0.5, 1.0]
    fallbacks = 0
    for _ in range(300):
        db = InnovationDB()
        a, b = make_pair(rng, db)
        a.validate()
        b.validate()
        a.fitness, b.fitness = rng.choice(fitness_choices), rng.choice(fitness_choices)
        child = crossover(a, b, rng, innovations=db)
        child.validate()
        assert child.parents == (a.id, b.id)
        assert child.generation == max(a.generation, b.generation) + 1
        assert child.operators in (["crossover"], ["crossover_fallback"])
        fallbacks += child.operators == ["crossover_fallback"]
        a.assert_types_match(child)
        b.assert_types_match(child)
        assert set(child.nodes) <= set(a.nodes) | set(b.nodes)
        assert set(child.connections) <= set(a.connections) | set(b.connections)
    assert fallbacks < 300


# -------------------------------------------------------------------- repair


def test_repair_reenables_single_disabled_end_edge() -> None:
    db = InnovationDB()
    g = viewer_genome(db)
    g.connections_into(END)[0].enabled = False
    assert not g.is_valid()
    assert repair(g, g.copy()) is True
    assert [c.src for c in g.connections_into(END, enabled_only=True)] == [0]
    g.validate()


def test_repair_end_prefers_fitter_source() -> None:
    db = InnovationDB()
    fitter = diamond_genome(db)
    merger = fitter.connections_into(END, enabled_only=True)[0].src
    # Several disabled END inputs: enable the fitter's END source.
    g = fitter.copy()
    for c in g.connections_into(END):
        c.enabled = False
    assert repair(g, fitter)
    assert [c.src for c in g.connections_into(END, enabled_only=True)] == [merger]
    # Several enabled END inputs: keep the fitter's END source, disable the rest.
    g = fitter.copy()
    for c in g.connections_into(END):
        c.enabled = True
    assert repair(g, fitter)
    assert [c.src for c in g.connections_into(END, enabled_only=True)] == [merger]
    # No preference available: lowest innovation wins (editor 0 -> END), merger is pruned.
    g = fitter.copy()
    for c in g.connections_into(END):
        c.enabled = True
    assert repair(g, Genome("no-end-source", {}, {}))
    assert [c.src for c in g.connections_into(END, enabled_only=True)] == [0]
    assert merger not in g.nodes
    g.validate()


def test_repair_removes_dangling_connections_and_orphans() -> None:
    db = InnovationDB()
    g = viewer_genome(db)
    v = next(i for i, n in g.nodes.items() if n.type is WorkerType.VIEWER)
    # Dangling connection gene (missing source) is dropped.
    g.connections[500] = ConnectionGene(500, src=404, dst=0, kind=Kind.VIEW, enabled=False)
    # The viewer becomes an orphan once its only edge is disabled.
    g.find_connection(v, 0, Kind.VIEW).enabled = False
    assert not g.is_valid()
    assert repair(g, g)
    assert 500 not in g.connections
    assert v not in g.nodes
    assert all(c.src != v for c in g.connections.values())
    g.validate()


def test_repair_removes_nodes_with_unsatisfied_slots() -> None:
    db = InnovationDB()
    g = minimal_genome(db)
    # A critic with a VIEW input but no EDIT input feeds the editor: it reaches END but
    # its required EDIT slot is empty, so step 4 removes it.
    c = g.add_node(db, ("source", 0, Kind.VIEW, WorkerType.CRITIC)).innovation
    g.add_connection(db, c, 0, Kind.VIEW)
    # A viewer feeding only the critic is stranded once the critic goes.
    v = g.add_node(db, ("source", c, Kind.VIEW, WorkerType.VIEWER)).innovation
    g.add_connection(db, v, c, Kind.VIEW)
    assert not g.is_valid()
    assert repair(g, g)
    assert set(g.nodes) == {0}
    g.validate()


def test_fallback_returns_copy_of_fitter() -> None:
    rng = random.Random(10)
    db = InnovationDB()

    def two_editors(view_src: int, view_dst: int) -> Genome:
        g = minimal_genome(db)
        e1 = g.add_node(db, ("init", 1, WorkerType.EDITOR)).innovation
        m = g.add_node(db, ("merge", 0, e1, WorkerType.JUDGEMERGER)).innovation
        g.add_connection(db, 0, m, Kind.EDIT)
        g.add_connection(db, e1, m, Kind.EDIT)
        g.add_connection(db, 0, m, Kind.VIEW)
        g.add_connection(db, m, END, Kind.EDIT)
        g.find_connection(0, END, Kind.EDIT).enabled = False
        g.add_connection(db, [0, e1][view_src], [0, e1][view_dst], Kind.VIEW)
        g.validate()
        return g

    a = two_editors(0, 1)
    b = two_editors(1, 0)
    a.generation, b.generation = 4, 2
    a.fitness = b.fitness = 1.0  # equal: union of disjoint genes creates a cycle
    a.species_id = 3
    child = crossover(a, b, rng, innovations=db)
    assert child.operators == ["crossover_fallback"]
    assert child.id not in (a.id, b.id)
    assert genes_json(child) == genes_json(a)
    assert child.parents == (a.id, b.id) and child.generation == 5
    assert child.fitness is None and child.adjusted_fitness is None and child.species_id is None
    child.validate()
    # With b fitter there is no cycle and crossover succeeds normally.
    b.fitness = 2.0
    child = crossover(a, b, rng, innovations=db)
    assert child.operators == ["crossover"]
    child.validate()


def test_max_workers_caps_union_of_equally_fit_parents() -> None:
    db = InnovationDB()
    a = viewer_genome(db, ("source", 0, Kind.VIEW, WorkerType.VIEWER))
    b = viewer_genome(db, ("init", 1, WorkerType.VIEWER))
    a.fitness = b.fitness = 1.0  # equal: disjoint genes of both parents are inherited
    child = crossover(a, b, random.Random(0), innovations=db)
    assert child.num_workers() == 3 and child.operators == ["crossover"]
    capped = crossover(a, b, random.Random(0), innovations=db, max_workers=2)
    assert capped.operators == ["crossover_fallback"]
    assert genes_json(capped) == genes_json(a) and capped.num_workers() == 2
    capped.validate()
