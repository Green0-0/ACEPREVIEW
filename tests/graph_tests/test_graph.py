"""Tests for ace.graph.graph (GRAPH_NEAT_DESIGN.md 1.3 and 7.2)."""
from __future__ import annotations

import json

import pytest

from ace.graph.graph import WorkerEdge, WorkerGraph, WorkerNode, minimal_graph
from ace.graph.types import END, InvalidGraph, Kind, WorkerType

V, E, R = Kind.VIEW, Kind.EDIT, Kind.REPORT
T = WorkerType


def make(nodes: dict[int, WorkerType], edges: list[tuple[int, int | str, Kind]]) -> WorkerGraph:
    return WorkerGraph(
        nodes={nid: WorkerNode(nid, t, {"role": nid % 3}) for nid, t in nodes.items()},
        edges=[WorkerEdge(s, d, k) for s, d, k in edges],
    )


def diamond(ids: tuple[int, int, int, int] = (1, 2, 3, 4)) -> WorkerGraph:
    """VIEWER -> {EDITOR, VIEWEDITOR} -> JUDGEMERGER -> END."""
    v, ed, ve, jm = ids
    return make(
        {v: T.VIEWER, ed: T.EDITOR, ve: T.VIEWEDITOR, jm: T.JUDGEMERGER},
        [
            (v, ed, V),
            (v, ed, R),
            (v, ve, V),
            (v, jm, V),
            (ed, jm, E),
            (ve, jm, E),
            (jm, END, E),
        ],
    )


def assert_invalid(graph: WorkerGraph, *fragments: str) -> str:
    assert not graph.is_valid()
    with pytest.raises(InvalidGraph) as info:
        graph.validate()
    message = str(info.value)
    for fragment in fragments:
        assert fragment in message, message
    return message


# ------------------------------------------------------------------- validity


def test_minimal_graph_is_valid() -> None:
    graph = minimal_graph()
    graph.validate()
    assert graph.is_valid()
    assert list(graph.nodes) == [0]
    assert graph.nodes[0].type is T.EDITOR
    assert graph.edges == [WorkerEdge(0, END, E)]
    assert graph.end_source() == 0
    assert graph.topological_order() == [0]
    assert graph.reaches_end(0)


def test_minimal_graph_uses_given_properties() -> None:
    graph = minimal_graph({"role": 2})
    assert graph.nodes[0].properties == {"role": 2}


def test_diamond_is_valid() -> None:
    diamond().validate()


# ---------------------------------------------------------- invariant failures


def test_cycle_rejected() -> None:
    # Two EDITORs feeding each other VIEW; one also feeds END.
    graph = make({1: T.EDITOR, 2: T.EDITOR}, [(1, 2, V), (2, 1, V), (1, END, E)])
    message = assert_invalid(graph, "cycle", "node 1 (editor)", "node 2 (editor)")
    assert "->" in message
    with pytest.raises(InvalidGraph, match="cycle"):
        graph.topological_order()


def test_unreachable_worker_rejected() -> None:
    graph = make({0: T.EDITOR, 5: T.VIEWER}, [(0, END, E)])
    assert_invalid(graph, "node 5 (viewer)", "no path to END")
    assert not graph.reaches_end(5)
    assert graph.reaches_end(0)


def test_vieweditor_missing_required_view() -> None:
    graph = make({0: T.EDITOR, 1: T.VIEWEDITOR}, [(0, 1, R), (1, END, E)])
    assert_invalid(graph, "node 1 (vieweditor)", "0 view")


def test_critic_edit_max_exceeded() -> None:
    graph = make(
        {1: T.EDITOR, 2: T.EDITOR, 3: T.CRITIC},
        [(1, 3, V), (1, 3, E), (2, 3, E), (3, END, E)],
    )
    assert_invalid(graph, "node 3 (critic)", "2 edit", "1..1")


def test_end_with_zero_inputs() -> None:
    graph = make({0: T.EDITOR}, [])
    assert_invalid(graph, "END", "found 0")
    with pytest.raises(InvalidGraph):
        graph.end_source()


def test_end_with_two_inputs() -> None:
    graph = make({0: T.EDITOR, 1: T.EDITOR}, [(0, END, E), (1, END, E)])
    assert_invalid(graph, "END", "found 2", "edge 0->END [edit]", "edge 1->END [edit]")
    with pytest.raises(InvalidGraph):
        graph.end_source()


def test_end_receiving_view() -> None:
    graph = make({0: T.EDITOR}, [(0, END, V)])
    assert_invalid(graph, "edge 0->END [view]", "END accepts only edit")


def test_duplicate_edge() -> None:
    graph = make({0: T.VIEWER, 1: T.EDITOR}, [(0, 1, V), (0, 1, V), (1, END, E)])
    assert_invalid(graph, "edge 0->1 [view]", "duplicate")


def test_kind_not_produced_by_source() -> None:
    graph = make({0: T.VIEWEDITOR, 1: T.EDITOR, 2: T.VIEWER}, [(2, 0, V), (0, 1, R), (1, END, E)])
    assert_invalid(graph, "edge 0->1 [report]", "node 0 (vieweditor)", "does not output report")


def test_kind_not_accepted_by_destination() -> None:
    # EDITOR has no EDIT slot.
    graph = make({0: T.EDITOR, 1: T.EDITOR}, [(0, 1, E), (1, END, E)])
    assert_invalid(graph, "edge 0->1 [edit]", "node 1 (editor)", "no edit input slot")


def test_missing_endpoint_and_self_loop() -> None:
    graph = make({0: T.EDITOR}, [(0, 9, V), (0, END, E)])
    assert_invalid(graph, "edge 0->9 [view]", "does not exist")
    graph = make({0: T.EDITOR}, [(7, 0, V), (0, END, E)])
    assert_invalid(graph, "edge 7->0 [view]", "source")
    graph = make({0: T.EDITOR}, [(0, 0, V), (0, END, E)])
    assert_invalid(graph, "edge 0->0 [view]", "self-loop")


def test_node_key_mismatch() -> None:
    graph = WorkerGraph(nodes={1: WorkerNode(2, T.EDITOR, {})}, edges=[WorkerEdge(1, END, E)])
    assert_invalid(graph, "node 2 (editor)", "key 1")


# ------------------------------------------------------------------------ JSON


def test_json_round_trip() -> None:
    graph = diamond()
    data = graph.to_json()
    restored = WorkerGraph.from_json(json.loads(json.dumps(data)))
    assert restored == graph
    assert restored.to_json() == data
    assert restored.structure_hash() == graph.structure_hash()
    end_edges = [e for e in restored.edges if e.dst == END]
    assert end_edges == [WorkerEdge(4, END, E)]
    restored.validate()


def test_minimal_graph_json_round_trip() -> None:
    graph = minimal_graph({"role": 1, "strategy": 3})
    assert WorkerGraph.from_json(json.loads(json.dumps(graph.to_json()))) == graph


# ---------------------------------------------------------------------- hashing


def test_structure_hash_invariant_under_relabelling_and_properties() -> None:
    a = diamond((1, 2, 3, 4))
    b = diamond((40, 7, 12, 0))
    for node in b.nodes.values():
        node.properties = {"role": 99}
    b.edges.reverse()
    assert a.structure_hash() == b.structure_hash()


def test_structure_hash_changes_with_structure() -> None:
    base = diamond()
    h = base.structure_hash()

    extra_report = diamond()
    extra_report.edges.append(WorkerEdge(2, 4, R))
    assert extra_report.structure_hash() != h

    retyped = diamond()
    retyped.nodes[4] = WorkerNode(4, T.PICKMERGER, {})
    assert retyped.structure_hash() != h

    rekinded = diamond()
    rekinded.edges = [e for e in rekinded.edges if e != WorkerEdge(1, 2, R)]
    assert rekinded.structure_hash() != h

    assert minimal_graph().structure_hash() != h


def test_structure_hash_distinguishes_symmetric_wiring() -> None:
    # Same node types and kinds, different VIEW wiring from the two viewers.
    a = make(
        {1: T.VIEWER, 2: T.VIEWER, 3: T.VIEWEDITOR, 4: T.CRITIC},
        [(1, 3, V), (2, 4, V), (3, 4, E), (4, END, E)],
    )
    b = make(
        {1: T.VIEWER, 2: T.VIEWER, 3: T.VIEWEDITOR, 4: T.CRITIC},
        [(1, 3, V), (1, 4, V), (2, 3, V), (3, 4, E), (4, END, E)],
    )
    a.validate()
    b.validate()
    assert a.structure_hash() != b.structure_hash()


# ----------------------------------------------------------------- ordering etc.


def test_topological_order_of_diamond() -> None:
    assert diamond((1, 2, 3, 4)).topological_order() == [1, 2, 3, 4]
    # Tie between EDITOR and VIEWEDITOR is broken by id.
    assert diamond((1, 3, 2, 4)).topological_order() == [1, 2, 3, 4]
    assert diamond((10, 30, 20, 0)).topological_order() == [10, 20, 30, 0]


def test_topological_order_tiebreak_among_sources() -> None:
    graph = make(
        {5: T.VIEWER, 2: T.VIEWER, 9: T.VIEWEDITOR},
        [(5, 9, V), (2, 9, V), (9, END, E)],
    )
    graph.validate()
    assert graph.topological_order() == [2, 5, 9]


def test_incoming_outgoing_and_end_source() -> None:
    graph = diamond()
    assert graph.end_source() == 4
    assert graph.incoming(END) == [WorkerEdge(4, END, E)]
    assert set(graph.outgoing(1)) == {
        WorkerEdge(1, 2, V),
        WorkerEdge(1, 2, R),
        WorkerEdge(1, 3, V),
        WorkerEdge(1, 4, V),
    }
    assert set(graph.incoming(4)) == {WorkerEdge(1, 4, V), WorkerEdge(2, 4, E), WorkerEdge(3, 4, E)}
    assert all(graph.reaches_end(n) for n in graph.nodes)


def test_inputs_by_source_groups_kinds() -> None:
    graph = diamond()
    assert graph.inputs_by_source(2) == {1: {V, R}}
    assert graph.inputs_by_source(4) == {1: {V}, 2: {E}, 3: {E}}
    assert list(graph.inputs_by_source(4)) == [1, 2, 3]
    assert graph.inputs_by_source(END) == {4: {E}}
    assert graph.inputs_by_source(1) == {}
