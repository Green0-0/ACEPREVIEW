"""Tests for ace.neat.mutation."""
from __future__ import annotations

import random

import pytest

from ace.graph.properties import validate_properties
from ace.graph.types import END, MERGER_TYPES, Kind, WorkerType
from ace.neat.config import MutationConfig
from ace.neat.genome import Genome, minimal_genome
from ace.neat.innovation import InnovationDB
from ace.neat.mutation import (
    ADD_WORKER_ROWS,
    OPERATORS,
    MutationContext,
    add_worker,
    insert_merger,
    mutate,
    random_valid_genome,
    remove_worker,
)


def make_ctx(seed: int, db: InnovationDB | None = None, *, max_workers: int = 12,
             **cfg: float) -> MutationContext:
    return MutationContext(
        rng=random.Random(seed),
        innovations=db if db is not None else InnovationDB(),
        config=MutationConfig(**cfg),
        max_workers=max_workers,
    )


def viewer_genome(db: InnovationDB) -> Genome:
    """Minimal genome plus an init-keyed VIEWER feeding the editor with a VIEW edge."""
    g = minimal_genome(db)
    v = g.add_node(db, ("init", 1, WorkerType.VIEWER))
    g.add_connection(db, v.innovation, 0, Kind.VIEW)
    return g


def types_of(g: Genome) -> list[WorkerType]:
    return sorted((n.type for n in g.nodes.values()), key=lambda t: t.value)


def record_types(seen: dict[int, WorkerType], g: Genome) -> None:
    for inn, node in g.nodes.items():
        assert seen.setdefault(inn, node.type) is node.type, f"innovation {inn} mixes types"


# ------------------------------------------------------------------ validity loop


@pytest.mark.parametrize("name", sorted(OPERATORS))
def test_operator_keeps_validity(name: str) -> None:
    op = OPERATORS[name]
    db = InnovationDB()
    rng = random.Random(1234)
    pool = [
        random_valid_genome(db, rng, MutationConfig(), min_workers=1, max_workers=6)
        for _ in range(10)
    ]
    ctx = MutationContext(rng=rng, innovations=db, config=MutationConfig(), max_workers=10)
    successes = 0
    seen: dict[int, WorkerType] = {}
    for _ in range(200):
        g = rng.choice(pool)
        before = g.to_json()
        ok = op(g, ctx)
        g.validate()
        record_types(seen, g)
        if ok:
            successes += 1
            assert g.num_workers() <= max(10, len(before["nodes"]))
        else:
            assert g.to_json() == before
        for node in g.nodes.values():
            validate_properties(node.type, node.properties)
    assert successes > 0, f"{name} never succeeded"


# ----------------------------------------------------------------- insertion table


def test_add_worker_rows_from_minimal_genome() -> None:
    """From the minimal genome only the CRITIC, VIEWER and EDITOR rows have sites."""
    found: set[str] = set()
    for seed in range(60):
        db = InnovationDB()
        g = minimal_genome(db)
        assert add_worker(g, make_ctx(seed, db))
        g.validate()
        types = set(types_of(g))
        if WorkerType.CRITIC in types:
            found.add("critic")
        elif types & MERGER_TYPES:
            assert sum(n.type is WorkerType.EDITOR for n in g.nodes.values()) == 2
            found.add("editor")
        elif WorkerType.VIEWER in types:
            found.add("viewer")
    assert found == {"critic", "viewer", "editor"}


def test_add_worker_every_row_from_genome_with_view_edge() -> None:
    found: set[str] = set()
    for seed in range(100):
        db = InnovationDB()
        g = viewer_genome(db)
        assert add_worker(g, make_ctx(seed, db))
        g.validate()
        types = types_of(g)
        if WorkerType.LISTVIEWER in types:
            found.add("listviewer")
        if WorkerType.CRITIC in types:
            found.add("critic")
        if WorkerType.VIEWEDITOR in types:
            found.add("vieweditor")
            assert set(types) & MERGER_TYPES
        if types.count(WorkerType.EDITOR) == 2:
            found.add("editor")
        if types.count(WorkerType.VIEWER) == 2:
            found.add("viewer")
    assert found == set(ADD_WORKER_ROWS)


@pytest.mark.parametrize("row", ADD_WORKER_ROWS)
def test_add_worker_named_row(row: str) -> None:
    db = InnovationDB()
    g = viewer_genome(db)
    assert add_worker(g, make_ctx(0, db), row=row)
    g.validate()
    assert g.num_workers() >= 3


def test_vieweditor_row_mirrors_view_sources_and_reaches_end() -> None:
    db = InnovationDB()
    g = viewer_genome(db)
    assert add_worker(g, make_ctx(3, db), row="vieweditor")
    ve = next(n for n in g.nodes.values() if n.type is WorkerType.VIEWEDITOR).innovation
    assert ve == db.peek_node(("mirror", 0, WorkerType.VIEWEDITOR))
    srcs = {c.src for c in g.connections_into(ve, enabled_only=True) if c.kind is Kind.VIEW}
    assert srcs == {c.src for c in g.connections_into(0) if c.kind is Kind.VIEW}
    assert g.nodes[g.to_graph().end_source()].type in MERGER_TYPES
    # A second vieweditor goes into the existing merger instead of creating a new one.
    assert add_worker(g, make_ctx(4, db), row="vieweditor")
    assert sum(n.type in MERGER_TYPES for n in g.nodes.values()) == 1
    g.validate()


def test_listviewer_split_disables_edge_and_innovations_consistent() -> None:
    db = InnovationDB()
    base = viewer_genome(db)
    view_conn = next(c for c in base.connections.values() if c.kind is Kind.VIEW)
    results = []
    for seed in (0, 1):
        g = base.copy()
        assert add_worker(g, make_ctx(seed, db), row="listviewer")
        results.append(g)
        assert g.connections[view_conn.innovation].enabled is False
        lv = db.peek_node(("split", view_conn.innovation, WorkerType.LISTVIEWER))
        assert lv in g.nodes and g.nodes[lv].type is WorkerType.LISTVIEWER
        assert g.has_connection(view_conn.src, lv, Kind.VIEW, enabled_only=True)
        assert g.has_connection(lv, view_conn.dst, Kind.VIEW, enabled_only=True)
        g.validate()
    a, b = results
    assert set(a.nodes) == set(b.nodes)
    assert set(a.connections) == set(b.connections)
    assert base.connections[view_conn.innovation].enabled is True


def test_critic_split_disables_edge_and_innovations_consistent() -> None:
    db = InnovationDB()
    base = minimal_genome(db)
    end_conn = base.find_connection(0, END, Kind.EDIT)
    assert end_conn is not None
    copies = []
    for seed in (5, 6):
        g = base.copy()
        assert add_worker(g, make_ctx(seed, db), row="critic")
        copies.append(g)
        critic = db.peek_node(("critic", end_conn.innovation, WorkerType.CRITIC))
        assert g.nodes[critic].type is WorkerType.CRITIC
        assert g.connections[end_conn.innovation].enabled is False
        assert g.has_connection(0, critic, Kind.EDIT, enabled_only=True)
        assert g.has_connection(0, critic, Kind.VIEW, enabled_only=True)
        assert g.has_connection(critic, END, Kind.EDIT, enabled_only=True)
        g.validate()
    assert set(copies[0].nodes) == set(copies[1].nodes)
    assert set(copies[0].connections) == set(copies[1].connections)


def test_viewer_source_key() -> None:
    db = InnovationDB()
    g = minimal_genome(db)
    assert add_worker(g, make_ctx(0, db), row="viewer")
    v = db.peek_node(("source", 0, Kind.VIEW, WorkerType.VIEWER))
    assert v in g.nodes and g.has_connection(v, 0, Kind.VIEW)
    # The same event cannot happen twice in one genome (same key -> same innovation).
    assert not add_worker(g, make_ctx(1, db), row="viewer")


# ------------------------------------------------------------------ merger + removal


def test_insert_merger_on_minimal_genome() -> None:
    for seed in range(10):
        db = InnovationDB()
        g = minimal_genome(db)
        assert insert_merger(g, make_ctx(seed, db))
        g.validate()
        graph = g.to_graph()
        m = graph.end_source()
        assert g.nodes[m].type in MERGER_TYPES
        edit_in = [e for e in graph.incoming(m) if e.kind is Kind.EDIT]
        assert len(edit_in) == 2
        assert any(e.kind is Kind.VIEW for e in graph.incoming(m))
        assert g.connections[db.peek_connection(0, END, Kind.EDIT)].enabled is False
        cand = next(e.src for e in edit_in if e.src != 0)
        assert m == db.peek_node(("merge", 0, cand, g.nodes[m].type))


def test_insert_merger_with_given_candidate() -> None:
    db = InnovationDB()
    g = minimal_genome(db)
    e1 = g.add_node(db, ("init", 2, WorkerType.EDITOR)).innovation
    g.add_connection(db, e1, 0, Kind.VIEW)  # e1's edit does not reach END
    g.validate()
    ctx = make_ctx(0, db)
    assert not insert_merger(g, ctx, candidate=0)  # END source is not a candidate
    assert insert_merger(g, ctx, candidate=e1)
    m = g.to_graph().end_source()
    assert {c.src for c in g.connections_into(m, enabled_only=True) if c.kind is Kind.EDIT} == {0, e1}
    # e1 now reaches END via edits: no further candidate unless a new one is created.
    assert not insert_merger(g, ctx, candidate=e1)


def test_remove_worker_collapses_merger() -> None:
    for seed in range(10):
        db = InnovationDB()
        g = minimal_genome(db)
        ctx = make_ctx(seed, db)
        assert insert_merger(g, ctx)
        assert g.num_workers() == 3
        assert remove_worker(g, ctx)
        g.validate()
        assert g.num_workers() == 1
        (only,) = g.nodes.values()
        assert only.type is WorkerType.EDITOR
        assert g.to_graph().end_source() == only.innovation


def test_remove_worker_never_removes_end_source_or_orphans() -> None:
    db = InnovationDB()
    g = minimal_genome(db)
    assert not remove_worker(g, make_ctx(0, db))
    g = viewer_genome(db)
    assert add_worker(g, make_ctx(0, db), row="listviewer")
    # Removing the viewer would empty the listviewer's required VIEW slot; removing the
    # listviewer would orphan the viewer: every proposal is rejected.
    before = g.to_json()
    assert not remove_worker(g, make_ctx(1, db))
    assert g.to_json() == before


def test_max_workers_respected() -> None:
    db = InnovationDB()
    g = viewer_genome(db)
    ctx = make_ctx(0, db, max_workers=2)
    before = g.to_json()
    assert not add_worker(g, ctx)
    assert not insert_merger(g, ctx)
    assert g.to_json() == before
    ctx = make_ctx(0, db, max_workers=3)
    assert not insert_merger(g, ctx)  # needs an editor candidate + merger = 4 workers
    for seed in range(30):
        h = viewer_genome(db)
        c = make_ctx(seed, db, max_workers=3)
        for _ in range(5):
            add_worker(h, c)
            assert h.num_workers() <= 3


# ------------------------------------------------------------------------ mutate


def test_mutate_records_operators_and_resets_fitness() -> None:
    db = InnovationDB()
    rng = random.Random(7)
    g = random_valid_genome(db, rng, MutationConfig(), min_workers=3, max_workers=5)
    g.fitness, g.adjusted_fitness, g.species_id = 1.0, 0.5, 2
    g.operators = ["crossover"]
    before = g.to_json()
    ctx = make_ctx(
        7, db,
        p_property=1.0, p_add_connection=1.0, p_add_worker=1.0, p_insert_merger=1.0,
        p_remove_connection=1.0, p_remove_worker=1.0, p_toggle_enable=1.0,
    )
    child = mutate(g, ctx)
    child.validate()
    assert child.id != g.id
    assert child.fitness is None and child.adjusted_fitness is None and child.species_id is None
    assert child.operators[0] == "crossover"
    applied = child.operators[1:]
    assert "mutate_properties" in applied
    assert set(applied) <= set(OPERATORS)
    assert g.to_json() == before  # parent untouched

    quiet = mutate(g, make_ctx(
        0, db, p_property=0.0, p_add_connection=0.0, p_add_worker=0.0,
        p_insert_merger=0.0, p_remove_connection=0.0, p_remove_worker=0.0,
        p_toggle_enable=0.0,
    ))
    assert quiet.operators == ["crossover"]
    assert quiet.fitness is None


def test_mutate_chain_stays_valid_and_type_consistent() -> None:
    db = InnovationDB()
    rng = random.Random(99)
    ctx = MutationContext(rng=rng, innovations=db,
                          config=MutationConfig(p_add_worker=0.5, p_insert_merger=0.2,
                                                p_remove_worker=0.2),
                          max_workers=12)
    seen: dict[int, WorkerType] = {}
    for _ in range(10):
        g = random_valid_genome(db, rng, MutationConfig(), min_workers=1, max_workers=5)
        for _ in range(30):
            g = mutate(g, ctx)
            g.validate()
            record_types(seen, g)
            assert g.num_workers() <= 12


# ----------------------------------------------------------- random_valid_genome


def test_random_valid_genome_counts_and_validity() -> None:
    db = InnovationDB()
    seen: dict[int, WorkerType] = {}
    counts = set()
    for seed in range(50):
        rng = random.Random(seed)
        g = random_valid_genome(db, rng, MutationConfig(), min_workers=2, max_workers=6,
                                generation=3)
        g.validate()
        assert 2 <= g.num_workers() <= 6
        assert g.generation == 3
        counts.add(g.num_workers())
        record_types(seen, g)
        for node in g.nodes.values():
            validate_properties(node.type, node.properties)
    assert len(counts) >= 3
    # Init genomes use ("init", index, type) keys only.
    init_ids = {i for k, i in db.node_entries().items() if k[0] == "init"}
    assert set(seen) <= init_ids


def test_random_valid_genome_accepts_neat_config_and_bounds() -> None:
    from ace.neat.config import NeatConfig

    db = InnovationDB()
    g = random_valid_genome(db, random.Random(0), NeatConfig(), min_workers=1, max_workers=1)
    assert g.num_workers() == 1
    with pytest.raises(ValueError):
        random_valid_genome(db, random.Random(0), MutationConfig(), min_workers=3, max_workers=2)


def test_no_worker_type_mixing_across_genomes() -> None:
    db = InnovationDB()
    seen: dict[int, WorkerType] = {}
    ctx = make_ctx(42, db, p_add_worker=0.6, p_insert_merger=0.3, p_add_connection=0.5)
    genomes = []
    for seed in range(20):
        g = random_valid_genome(db, random.Random(seed), MutationConfig(),
                                min_workers=1, max_workers=5)
        genomes.append(g)
        for _ in range(10):
            g = mutate(g, ctx)
            genomes.append(g)
    for g in genomes:
        record_types(seen, g)
    for a, b in zip(genomes, genomes[1:]):
        a.assert_types_match(b)
