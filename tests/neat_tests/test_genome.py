"""Tests for ace.neat.genome."""
from __future__ import annotations

import json
import uuid

import pytest

from ace.graph.graph import minimal_graph
from ace.graph.types import END, InvalidGraph, Kind, WorkerType
from ace.neat.genome import ConnectionGene, Genome, NodeGene, minimal_genome
from ace.neat.innovation import InnovationDB


def viewer_genome(db: InnovationDB) -> Genome:
    """Minimal genome plus a VIEWER source feeding the editor with a VIEW edge."""
    g = minimal_genome(db)
    v = g.add_node(db, ("source", 0, Kind.VIEW, WorkerType.VIEWER))
    g.add_connection(db, v.innovation, 0, Kind.VIEW)
    return g


def diamond_genome(db: InnovationDB) -> tuple[Genome, dict[str, int]]:
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
    return g, {"e0": e0, "e1": e1, "v": v, "m": m}


def test_minimal_genome_matches_minimal_graph() -> None:
    db = InnovationDB()
    g = minimal_genome(db)
    assert set(g.nodes) == {0}
    assert g.nodes[0].type is WorkerType.EDITOR
    assert db.peek_node(("init", 0, WorkerType.EDITOR)) == 0
    assert list(g.connections) == [db.peek_connection(0, END, Kind.EDIT)] == [1]
    assert g.to_graph().to_json() == minimal_graph().to_json()
    assert g.structure_hash() == minimal_graph().structure_hash()
    g.validate()
    assert g.num_workers() == 1
    assert len(uuid.UUID(hex=g.id).hex) == 32


def test_minimal_genome_properties_and_generation() -> None:
    db = InnovationDB()
    g = minimal_genome(db, {"role": 2}, generation=3)
    assert g.nodes[0].properties == {"role": 2}
    assert g.generation == 3
    graph = g.to_graph()
    graph.nodes[0].properties["role"] = 9
    assert g.nodes[0].properties == {"role": 2}  # to_graph copies the properties dict


def test_to_graph_uses_enabled_connections_only() -> None:
    db = InnovationDB()
    g, ids = diamond_genome(db)
    g.validate()
    graph = g.to_graph()
    assert set(graph.nodes) == set(g.nodes)
    assert all(graph.nodes[i].id == i for i in graph.nodes)
    assert len(graph.edges) == len(g.enabled_connections()) == len(g.connections) - 1
    assert graph.end_source() == ids["m"]
    assert g.num_workers() == 4


def test_queries() -> None:
    db = InnovationDB()
    g, ids = diamond_genome(db)
    into_m = g.connections_into(ids["m"])
    assert {(c.src, c.kind) for c in into_m} == {
        (ids["e0"], Kind.EDIT), (ids["e1"], Kind.EDIT), (ids["v"], Kind.VIEW)
    }
    assert [c.innovation for c in into_m] == sorted(c.innovation for c in into_m)
    assert len(g.connections_into(END)) == 2
    assert len(g.connections_into(END, enabled_only=True)) == 1
    assert {c.dst for c in g.connections_from(ids["e0"])} == {ids["m"], END}
    assert [c.dst for c in g.connections_from(ids["e0"], enabled_only=True)] == [ids["m"]]
    assert g.has_connection(ids["v"], ids["m"], Kind.VIEW)
    assert not g.has_connection(ids["v"], ids["m"], Kind.REPORT)
    assert g.has_connection(ids["e0"], END, Kind.EDIT)
    assert not g.has_connection(ids["e0"], END, Kind.EDIT, enabled_only=True)
    assert all(c.enabled for c in g.enabled_connections())


def test_validate_rejects_unknown_node() -> None:
    db = InnovationDB()
    g = viewer_genome(db)
    g.validate()
    g.connections[50] = ConnectionGene(50, src=99, dst=0, kind=Kind.VIEW)
    with pytest.raises(InvalidGraph, match="99"):
        g.validate()
    g2 = minimal_genome(InnovationDB())
    g2.connections[50] = ConnectionGene(50, src=0, dst=77, kind=Kind.EDIT, enabled=False)
    with pytest.raises(InvalidGraph, match="77"):  # disabled genes are checked too
        g2.validate()


def test_validate_rejects_disabled_end_edge() -> None:
    db = InnovationDB()
    g = minimal_genome(db)
    g.connections[1].enabled = False
    with pytest.raises(InvalidGraph, match="END"):
        g.validate()
    assert not g.is_valid()


def test_validate_rejects_innovation_collisions_and_bad_types() -> None:
    db = InnovationDB()
    g = minimal_genome(db)
    g.connections[0] = ConnectionGene(0, src=0, dst=END, kind=Kind.EDIT)
    with pytest.raises(InvalidGraph):
        g.validate()

    g = minimal_genome(InnovationDB())
    g.connections[7] = ConnectionGene(8, src=0, dst=END, kind=Kind.EDIT)
    with pytest.raises(InvalidGraph, match="innovation"):
        g.validate()

    g = minimal_genome(InnovationDB())
    g.nodes[0].type = "editor"  # type: ignore[assignment]
    with pytest.raises(InvalidGraph, match="worker type"):
        g.validate()


def test_assert_types_match() -> None:
    db = InnovationDB()
    a, ids = diamond_genome(db)
    b = a.copy()
    a.assert_types_match(b)
    b.nodes[ids["m"]].type = WorkerType.PICKMERGER
    with pytest.raises(InvalidGraph, match=str(ids["m"])):
        a.assert_types_match(b)
    with pytest.raises(InvalidGraph):
        b.assert_types_match(a)


def test_json_round_trip() -> None:
    db = InnovationDB()
    g, ids = diamond_genome(db)
    g.fitness = 0.5
    g.adjusted_fitness = 0.25
    g.species_id = 3
    g.parents = ("p1", "p2")
    g.operators = ["crossover", "add_worker"]
    g.generation = 4
    data = json.loads(json.dumps(g.to_json()))
    end_conns = [c for c in data["connections"] if c["dst"] == "END"]
    assert len(end_conns) == 2
    assert sorted(c["enabled"] for c in end_conns) == [False, True]
    assert {n["type"] for n in data["nodes"]} == {"editor", "viewer", "judgemerger"}
    back = Genome.from_json(data)
    assert back.to_json() == g.to_json()
    assert back.id == g.id
    assert back.parents == ("p1", "p2")
    assert all(c.dst == END for c in back.connections_into(END))
    assert all(isinstance(c.kind, Kind) for c in back.connections.values())
    assert all(isinstance(n.type, WorkerType) for n in back.nodes.values())
    assert back.find_connection(ids["e0"], END, Kind.EDIT).enabled is False
    back.validate()
    assert back.structure_hash() == g.structure_hash()


def test_copy_is_independent() -> None:
    db = InnovationDB()
    g, ids = diamond_genome(db)
    g.operators.append("x")
    g.parents = ("p",)
    g.generation = 2
    c = g.copy()
    assert c.id != g.id
    assert c.generation == 2 and c.parents == ("p",) and c.operators == ["x"]
    assert c.to_json() | {"id": g.id} == g.to_json()
    c.nodes[0].properties["role"] = 3
    c.connections[1].enabled = True
    c.operators.append("y")
    del c.nodes[ids["v"]]
    assert g.nodes[0].properties.get("role") != 3
    assert g.connections[1].enabled is False
    assert g.operators == ["x"]
    assert ids["v"] in g.nodes
    assert g.copy(new_id=False).id == g.id


def test_add_node_and_connection_are_idempotent() -> None:
    db = InnovationDB()
    g = minimal_genome(db)
    n1 = g.add_node(db, ("source", 0, Kind.VIEW, WorkerType.VIEWER), {"role": 1})
    n2 = g.add_node(db, ("source", 0, Kind.VIEW, WorkerType.VIEWER))
    assert n1 is n2 and isinstance(n1, NodeGene)
    c1 = g.add_connection(db, n1.innovation, 0, Kind.VIEW)
    c2 = g.add_connection(db, n1.innovation, 0, Kind.VIEW, enabled=False)
    assert c1 is c2 and c1.enabled is False
    # Same structural event in another genome yields the same innovations.
    h = viewer_genome(db)
    assert set(h.nodes) == set(g.nodes)
    assert set(h.connections) == set(g.connections)
