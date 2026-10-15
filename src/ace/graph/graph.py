"""Worker graph: nodes, typed edges, validation, ordering, hashing, JSON.

See GRAPH_NEAT_DESIGN.md sections 1.3 and 7.2.

Choices made where the contract is open:

* **Validation order.** ``validate`` checks, in order: node table consistency (dict key ==
  ``node.id``, ids are ints, END is not a node id), edge legality (src/dst exist, no
  self-loops, no duplicate ``(src, dst, kind)``, kind in the source's outputs and in the
  destination's input table, END accepts only EDIT), acyclicity, the END invariant, slot
  arity per worker, and reachability of END. The first violation found raises
  ``InvalidGraph`` whose message names the offending node (``node 3 (critic)``) or edge
  (``edge 1->END [view]``). Properties are not validated here; that is
  ``properties.validate_properties``'s job.
* **Reachability** follows edges of any kind.
* **topological_order** is Kahn's algorithm with a min-heap on node id, so ties are always
  broken by the smallest id. It raises ``InvalidGraph`` on a cycle.
* **structure_hash** uses Weisfeiler-Lehman style colour refinement instead of id-based
  relabelling. Every node (END included) starts with its type as its colour; each round a
  node's colour becomes the sha1 of its previous colour plus the sorted multisets of
  ``(kind, colour)`` of its in-neighbours and of its out-neighbours. After
  ``len(nodes) + 1`` rounds (enough for colours to stabilise on a graph of that size) the
  final digest is sha1 over the sorted node colours and the sorted edge list
  ``(colour(src), colour(dst), kind)``. No id ever enters the computation, so the hash is
  invariant under relabelling by construction; properties are ignored. Two different
  structures collide only if they are indistinguishable by colour refinement, which for
  small typed DAGs is practically never and is acceptable for deduplication.
* **minimal_graph** with ``editor_properties=None`` uses
  ``properties.default_properties(EDITOR)`` when that module is importable, else ``{}``.
"""
from __future__ import annotations

import hashlib
import heapq
import json
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from ace.graph.types import END, WORKER_SPECS, InvalidGraph, Kind, WorkerType

NodeId = int | str
"""A worker id (int) or the END sentinel."""


@dataclass
class WorkerNode:
    """One worker in the graph. ``id`` is the innovation number of its node gene."""

    id: int
    type: WorkerType
    properties: dict[str, int] = field(default_factory=dict)

    def label(self) -> str:
        """Human-readable name used in error messages, e.g. ``node 3 (critic)``."""
        return f"node {self.id} ({self.type.value})"


@dataclass(frozen=True)
class WorkerEdge:
    """A typed edge. ``dst`` is a worker id or ``END``."""

    src: int
    dst: int | str
    kind: Kind

    def label(self) -> str:
        """Human-readable name used in error messages, e.g. ``edge 1->END [edit]``."""
        kind = self.kind.value if isinstance(self.kind, Kind) else str(self.kind)
        return f"edge {self.src}->{self.dst} [{kind}]"


@dataclass
class WorkerGraph:
    """A DAG of workers with an explicit END node (§1)."""

    nodes: dict[int, WorkerNode]
    edges: list[WorkerEdge]

    # ------------------------------------------------------------------ queries

    def incoming(self, node_id: NodeId) -> list[WorkerEdge]:
        """Edges whose destination is ``node_id`` (a worker id or END), in edge order."""
        return [e for e in self.edges if e.dst == node_id]

    def outgoing(self, node_id: int) -> list[WorkerEdge]:
        """Edges whose source is ``node_id``, in edge order."""
        return [e for e in self.edges if e.src == node_id]

    def inputs_by_source(self, node_id: NodeId) -> dict[int, set[Kind]]:
        """Incoming edge kinds grouped by source worker id (keys in ascending order)."""
        grouped: dict[int, set[Kind]] = defaultdict(set)
        for edge in self.incoming(node_id):
            grouped[edge.src].add(edge.kind)
        return {src: grouped[src] for src in sorted(grouped)}

    def reaches_end(self, node_id: int) -> bool:
        """Whether a directed path (edges of any kind) leads from ``node_id`` to END."""
        adjacency = self._adjacency()
        seen: set[NodeId] = {node_id}
        stack: list[NodeId] = [node_id]
        while stack:
            current = stack.pop()
            for nxt in adjacency.get(current, ()):
                if nxt == END:
                    return True
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        return False

    def topological_order(self) -> list[int]:
        """Workers in dependency order (END excluded); ties broken by smallest id.

        Raises ``InvalidGraph`` naming a cycle if the graph is not acyclic.
        """
        indegree: dict[int, int] = {nid: 0 for nid in self.nodes}
        children: dict[int, list[int]] = defaultdict(list)
        for edge in self.edges:
            if edge.dst == END or edge.dst not in indegree or edge.src not in indegree:
                continue
            indegree[edge.dst] += 1  # type: ignore[index]
            children[edge.src].append(edge.dst)  # type: ignore[arg-type]
        heap = [nid for nid, deg in indegree.items() if deg == 0]
        heapq.heapify(heap)
        order: list[int] = []
        while heap:
            nid = heapq.heappop(heap)
            order.append(nid)
            for child in children[nid]:
                indegree[child] -= 1
                if indegree[child] == 0:
                    heapq.heappush(heap, child)
        if len(order) != len(self.nodes):
            cycle = self._find_cycle()
            path = " -> ".join(self.nodes[n].label() for n in cycle)
            raise InvalidGraph(f"graph contains a cycle: {path}")
        return order

    def end_source(self) -> int:
        """The single worker feeding END. Raises ``InvalidGraph`` if not exactly one edge."""
        into_end = self.incoming(END)
        if len(into_end) != 1:
            raise InvalidGraph(
                f"END must have exactly one incoming edge, found {len(into_end)}"
                + (": " + ", ".join(e.label() for e in into_end) if into_end else "")
            )
        return into_end[0].src

    # --------------------------------------------------------------- validation

    def validate(self) -> None:
        """Check every invariant of §1.3 plus edge legality; raise ``InvalidGraph``."""
        self._validate_nodes()
        self._validate_edges()
        self.topological_order()  # acyclicity
        self._validate_end()
        self._validate_arity()
        self._validate_reachability()

    def is_valid(self) -> bool:
        """``validate()`` as a boolean."""
        try:
            self.validate()
        except InvalidGraph:
            return False
        return True

    def _validate_nodes(self) -> None:
        for key, node in self.nodes.items():
            if not isinstance(key, int) or isinstance(key, bool):
                raise InvalidGraph(f"node key {key!r} is not an int")
            if node.id != key:
                raise InvalidGraph(f"{node.label()} is stored under key {key}")
            if not isinstance(node.type, WorkerType):
                raise InvalidGraph(f"node {key} has unknown worker type {node.type!r}")

    def _validate_edges(self) -> None:
        seen: set[tuple[int, NodeId, Kind]] = set()
        for edge in self.edges:
            name = edge.label()
            if not isinstance(edge.kind, Kind):
                raise InvalidGraph(f"{name}: unknown kind {edge.kind!r}")
            if edge.src not in self.nodes:
                raise InvalidGraph(f"{name}: source {edge.src!r} is not a worker in the graph")
            if edge.dst != END and edge.dst not in self.nodes:
                raise InvalidGraph(f"{name}: destination {edge.dst!r} does not exist")
            if edge.src == edge.dst:
                raise InvalidGraph(f"{name}: self-loop on {self.nodes[edge.src].label()}")
            triple = (edge.src, edge.dst, edge.kind)
            if triple in seen:
                raise InvalidGraph(f"{name}: duplicate edge")
            seen.add(triple)
            src = self.nodes[edge.src]
            if edge.kind not in WORKER_SPECS[src.type].outputs:
                raise InvalidGraph(
                    f"{name}: {src.label()} does not output {edge.kind.value}"
                )
            if edge.dst == END:
                if edge.kind is not Kind.EDIT:
                    raise InvalidGraph(f"{name}: END accepts only edit edges")
            else:
                dst = self.nodes[edge.dst]  # type: ignore[index]
                if WORKER_SPECS[dst.type].slot(edge.kind) is None:
                    raise InvalidGraph(
                        f"{name}: {dst.label()} has no {edge.kind.value} input slot"
                    )

    def _validate_end(self) -> None:
        into_end = self.incoming(END)
        if len(into_end) != 1:
            raise InvalidGraph(
                f"END must have exactly one incoming edge, found {len(into_end)}"
                + (": " + ", ".join(e.label() for e in into_end) if into_end else "")
            )
        if into_end[0].kind is not Kind.EDIT:
            raise InvalidGraph(f"{into_end[0].label()}: END accepts only edit edges")

    def _validate_arity(self) -> None:
        counts: dict[int, dict[Kind, int]] = {nid: defaultdict(int) for nid in self.nodes}
        for edge in self.edges:
            if edge.dst != END:
                counts[edge.dst][edge.kind] += 1  # type: ignore[index]
        for nid in sorted(self.nodes):
            node = self.nodes[nid]
            spec = WORKER_SPECS[node.type]
            for kind in Kind:
                count = counts[nid][kind]
                slot = spec.slot(kind)
                if slot is None:
                    if count:
                        raise InvalidGraph(
                            f"{node.label()} has {count} {kind.value} input(s) but no "
                            f"{kind.value} slot"
                        )
                    continue
                if not slot.accepts(count):
                    upper = "inf" if slot.max is None else str(slot.max)
                    raise InvalidGraph(
                        f"{node.label()} has {count} {kind.value} input(s), "
                        f"requires {slot.min}..{upper}"
                    )

    def _validate_reachability(self) -> None:
        # Reverse BFS from END: every worker must be found.
        parents: dict[NodeId, list[int]] = defaultdict(list)
        for edge in self.edges:
            parents[edge.dst].append(edge.src)
        seen: set[NodeId] = {END}
        stack: list[NodeId] = [END]
        while stack:
            current = stack.pop()
            for p in parents.get(current, ()):
                if p not in seen:
                    seen.add(p)
                    stack.append(p)
        for nid in sorted(self.nodes):
            if nid not in seen:
                raise InvalidGraph(f"{self.nodes[nid].label()} has no path to END")

    # ------------------------------------------------------------------ helpers

    def _adjacency(self) -> dict[NodeId, list[NodeId]]:
        adjacency: dict[NodeId, list[NodeId]] = defaultdict(list)
        for edge in self.edges:
            adjacency[edge.src].append(edge.dst)
        return adjacency

    def _find_cycle(self) -> list[int]:
        """Return one cycle as a closed node path ``[a, b, ..., a]`` (empty if none)."""
        adjacency = self._adjacency()
        white, grey, black = 0, 1, 2
        colour: dict[int, int] = {nid: white for nid in self.nodes}
        for start in sorted(self.nodes):
            if colour[start] != white:
                continue
            path: list[int] = [start]
            iters = [iter(sorted(n for n in adjacency.get(start, ()) if n in self.nodes))]
            colour[start] = grey
            while iters:
                nxt = next(iters[-1], None)
                if nxt is None:
                    colour[path.pop()] = black
                    iters.pop()
                    continue
                assert isinstance(nxt, int)
                if colour[nxt] == grey:
                    return path[path.index(nxt):] + [nxt]
                if colour[nxt] == white:
                    colour[nxt] = grey
                    path.append(nxt)
                    iters.append(
                        iter(sorted(n for n in adjacency.get(nxt, ()) if n in self.nodes))
                    )
        return []

    # ------------------------------------------------------------------ hashing

    def structure_hash(self) -> str:
        """Id- and property-independent sha1 of the graph structure (see module doc)."""

        def h(payload: Any) -> str:
            return hashlib.sha1(
                json.dumps(payload, separators=(",", ":")).encode()
            ).hexdigest()

        colours: dict[NodeId, str] = {nid: n.type.value for nid, n in self.nodes.items()}
        colours[END] = END
        ins: dict[NodeId, list[WorkerEdge]] = defaultdict(list)
        outs: dict[NodeId, list[WorkerEdge]] = defaultdict(list)
        for edge in self.edges:
            ins[edge.dst].append(edge)
            outs[edge.src].append(edge)
        for _ in range(len(colours) + 1):
            new = {
                nid: h(
                    [
                        colour,
                        sorted([e.kind.value, colours.get(e.src, "?")] for e in ins[nid]),
                        sorted([e.kind.value, colours.get(e.dst, "?")] for e in outs[nid]),
                    ]
                )
                for nid, colour in colours.items()
            }
            colours = new
        edges = sorted(
            [colours.get(e.src, "?"), colours.get(e.dst, "?"), e.kind.value]
            for e in self.edges
        )
        return h({"nodes": sorted(colours.values()), "edges": edges})

    # --------------------------------------------------------------------- JSON

    def to_json(self) -> dict:
        """JSON-serialisable dict: nodes sorted by id, edges in stored order."""
        return {
            "nodes": [
                {
                    "id": node.id,
                    "type": node.type.value,
                    "properties": dict(node.properties),
                }
                for _, node in sorted(self.nodes.items())
            ],
            "edges": [
                {"src": e.src, "dst": e.dst, "kind": e.kind.value} for e in self.edges
            ],
        }

    @staticmethod
    def from_json(data: dict) -> WorkerGraph:
        """Inverse of ``to_json``. Does not validate."""
        nodes: dict[int, WorkerNode] = {}
        for raw in data["nodes"]:
            node = WorkerNode(
                id=int(raw["id"]),
                type=WorkerType(raw["type"]),
                properties={str(k): int(v) for k, v in raw.get("properties", {}).items()},
            )
            nodes[node.id] = node
        edges = [
            WorkerEdge(
                src=int(raw["src"]),
                dst=END if raw["dst"] == END else int(raw["dst"]),
                kind=Kind(raw["kind"]),
            )
            for raw in data["edges"]
        ]
        return WorkerGraph(nodes=nodes, edges=edges)


def minimal_graph(editor_properties: dict[str, int] | None = None) -> WorkerGraph:
    """One EDITOR (id 0) feeding END with an EDIT edge."""
    if editor_properties is None:
        try:
            from ace.graph.properties import default_properties
        except ImportError:
            editor_properties = {}
        else:
            editor_properties = default_properties(WorkerType.EDITOR)
    node = WorkerNode(id=0, type=WorkerType.EDITOR, properties=dict(editor_properties))
    return WorkerGraph(nodes={0: node}, edges=[WorkerEdge(0, END, Kind.EDIT)])
