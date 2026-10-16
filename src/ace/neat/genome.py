"""Genomes: node and connection genes that encode a worker graph (GRAPH_NEAT_DESIGN.md 7.3).

A :class:`Genome` holds node genes (workers, keyed by their global node innovation) and
connection genes (typed edges, keyed by their connection innovation). Node and connection
innovations come from one shared :class:`~ace.neat.innovation.InnovationDB` counter, so
they never collide. :meth:`Genome.to_graph` turns the enabled genes into a
:class:`~ace.graph.graph.WorkerGraph` whose node ids are the node innovations.

Contract readings / additions (documented per the contract's ambiguity rule):

* **Equality.** ``Genome`` is declared with ``eq=False``: two genomes compare equal only if
  they are the same object, and genomes are hashable by identity (usable in sets / as
  dict keys). Compare structure with :meth:`Genome.structure_hash` or ``to_json()``.
  ``NodeGene`` and ``ConnectionGene`` keep dataclass value equality.
* **validate()** first checks gene-level consistency, then runs
  ``to_graph().validate()``. Gene-level checks, all raising ``InvalidGraph``: every dict
  key equals its gene's ``innovation``; every node gene type is a ``WorkerType``; every
  connection kind is a ``Kind``; every connection (enabled *or disabled*) has ``src`` an
  existing node gene and ``dst`` an existing node gene or ``END``; connection innovations
  are unique and disjoint from node innovations; no two connection genes share the same
  ``(src, dst, kind)``. "Matched genes share a type" is a property of a *pair* of
  genomes and is checked by :meth:`Genome.assert_types_match`.
* **copy(new_id=True)** deep-copies every gene (including the property dicts) and keeps
  ``generation``, ``parents``, ``operators`` (a new list), ``fitness``,
  ``adjusted_fitness`` and ``species_id`` unchanged. Only ``id`` changes (a fresh uuid4
  hex) unless ``new_id=False``. Callers producing offspring are expected to set
  ``generation``/``parents``/``operators`` and reset fitness themselves.
* **Gene queries** (:meth:`Genome.connections_into`, :meth:`Genome.connections_from`,
  :meth:`Genome.has_connection`, :meth:`Genome.find_connection`) consider *all*
  connection genes, enabled or not, unless ``enabled_only=True`` is passed. Results are
  sorted by innovation.
* :meth:`Genome.num_workers` is the number of node genes (every node gene is a worker and
  ``to_graph`` keeps all of them).
* **JSON**: enums are written as their ``.value``; a connection whose ``dst`` is END is
  written as the string ``"END"``; nodes and connections are lists sorted by innovation;
  ``parents`` is written as a list and read back as a tuple. ``from_json`` does not
  validate.
* **minimal_genome(properties=None)** uses ``properties.default_properties(EDITOR)`` when
  that module is importable, else ``{}`` (same rule as ``minimal_graph``).
* **Recording fields** ``created_at``, ``evaluated_at`` (ISO-8601 UTC strings),
  ``eval_duration_s`` and ``task_scores`` are optional bookkeeping for the run recording
  (they default to ``None`` / ``[]``, are kept by ``copy`` and round-trip through JSON;
  ``from_json`` tolerates their absence).
* Convenience builders :meth:`Genome.add_node` and :meth:`Genome.add_connection` allocate
  innovations through an ``InnovationDB`` and insert the gene; they do not validate.
"""
from __future__ import annotations

import copy as _copy
import uuid
from dataclasses import dataclass, field
from typing import Any

from ace.graph.graph import WorkerEdge, WorkerGraph, WorkerNode
from ace.graph.types import END, InvalidGraph, Kind, WorkerType
from ace.neat.innovation import InnovationDB


def _default_properties(t: WorkerType) -> dict[str, int]:
    """``properties.default_properties(t)`` if that module is importable, else ``{}``."""
    try:
        from ace.graph.properties import default_properties
    except ImportError:  # pragma: no cover - properties.py is optional for this module
        return {}
    return default_properties(t)


def _new_id() -> str:
    return uuid.uuid4().hex


@dataclass
class NodeGene:
    """One worker. ``innovation`` is its global node id."""

    innovation: int
    type: WorkerType
    properties: dict[str, int] = field(default_factory=dict)


@dataclass
class ConnectionGene:
    """One typed edge between node innovations (``dst`` may be ``END``)."""

    innovation: int
    src: int
    dst: int | str
    kind: Kind
    enabled: bool = True


@dataclass(eq=False)
class Genome:
    """A NEAT genome encoding a worker graph. Identity equality (see module docstring)."""

    id: str
    nodes: dict[int, NodeGene]
    connections: dict[int, ConnectionGene]
    generation: int = 0
    parents: tuple[str, ...] = ()
    operators: list[str] = field(default_factory=list)
    fitness: float | None = None
    adjusted_fitness: float | None = None
    species_id: int | None = None
    # Run-recording bookkeeping (recording spec 11.4); set by Population, never by operators.
    created_at: str | None = None
    evaluated_at: str | None = None
    eval_duration_s: float | None = None
    task_scores: list[float] = field(default_factory=list)
    # Raw fitness of every minibatch this genome was scored on (elites are re-scored each
    # generation by minibatch evaluators); ``fitness`` is then their mean.
    fitness_history: list[float] = field(default_factory=list)

    # ------------------------------------------------------------------ graph

    def to_graph(self) -> WorkerGraph:
        """The worker graph of all node genes and the *enabled* connection genes.

        Node ids are node innovations; properties dicts are copied. Edges are in
        connection-innovation order.
        """
        nodes = {
            inn: WorkerNode(id=inn, type=gene.type, properties=dict(gene.properties))
            for inn, gene in sorted(self.nodes.items())
        }
        edges = [
            WorkerEdge(src=c.src, dst=c.dst, kind=c.kind)
            for _, c in sorted(self.connections.items())
            if c.enabled
        ]
        return WorkerGraph(nodes=nodes, edges=edges)

    def structure_hash(self) -> str:
        """``to_graph().structure_hash()``: id- and property-independent structure digest."""
        return self.to_graph().structure_hash()

    def num_workers(self) -> int:
        """Number of worker node genes."""
        return len(self.nodes)

    # ------------------------------------------------------------- validation

    def validate(self) -> None:
        """Gene-level consistency checks, then ``to_graph().validate()``.

        Raises ``InvalidGraph`` naming the offending gene (see module docstring).
        """
        for key, node in self.nodes.items():
            if node.innovation != key:
                raise InvalidGraph(f"node gene {node.innovation} is stored under key {key}")
            if not isinstance(node.type, WorkerType):
                raise InvalidGraph(f"node gene {key} has unknown worker type {node.type!r}")
        seen_triples: dict[tuple[int, int | str, Kind], int] = {}
        for key, conn in self.connections.items():
            name = f"connection gene {key} ({conn.src}->{conn.dst} [{getattr(conn.kind, 'value', conn.kind)}])"
            if conn.innovation != key:
                raise InvalidGraph(f"{name} is stored under key {key} but has innovation {conn.innovation}")
            if key in self.nodes:
                raise InvalidGraph(f"{name}: innovation collides with a node gene innovation")
            if not isinstance(conn.kind, Kind):
                raise InvalidGraph(f"{name}: unknown kind {conn.kind!r}")
            if conn.src not in self.nodes:
                raise InvalidGraph(f"{name}: source {conn.src!r} is not a node gene")
            if conn.dst != END and conn.dst not in self.nodes:
                raise InvalidGraph(f"{name}: destination {conn.dst!r} is not a node gene or END")
            triple = (conn.src, conn.dst, conn.kind)
            if triple in seen_triples:
                raise InvalidGraph(
                    f"{name}: duplicates connection gene {seen_triples[triple]}"
                )
            seen_triples[triple] = key
        self.to_graph().validate()

    def is_valid(self) -> bool:
        """``validate()`` as a boolean."""
        try:
            self.validate()
        except InvalidGraph:
            return False
        return True

    def assert_types_match(self, other: Genome) -> None:
        """Raise ``InvalidGraph`` if a shared innovation denotes different things.

        Checks that every node innovation present in both genomes has the same worker type,
        and that no innovation is a node gene in one genome and a connection gene in the
        other.
        """
        for inn in sorted(self.nodes.keys() & other.nodes.keys()):
            mine, theirs = self.nodes[inn].type, other.nodes[inn].type
            if mine != theirs:
                raise InvalidGraph(
                    f"node innovation {inn} has type {getattr(mine, 'value', mine)} in genome "
                    f"{self.id} but {getattr(theirs, 'value', theirs)} in genome {other.id}"
                )
        for inn in sorted(
            (self.nodes.keys() & other.connections.keys())
            | (self.connections.keys() & other.nodes.keys())
        ):
            raise InvalidGraph(
                f"innovation {inn} is a node gene in one genome and a connection gene in the "
                f"other ({self.id}, {other.id})"
            )

    # ---------------------------------------------------------------- queries

    def enabled_connections(self) -> list[ConnectionGene]:
        """Enabled connection genes, sorted by innovation."""
        return [c for _, c in sorted(self.connections.items()) if c.enabled]

    def _select(self, *, enabled_only: bool, **match: Any) -> list[ConnectionGene]:
        return [
            c
            for _, c in sorted(self.connections.items())
            if (c.enabled or not enabled_only)
            and all(getattr(c, attr) == value for attr, value in match.items())
        ]

    def connections_into(self, node: int | str, *, enabled_only: bool = False) -> list[ConnectionGene]:
        """Connection genes whose ``dst`` is ``node`` (a node innovation or END)."""
        return self._select(enabled_only=enabled_only, dst=node)

    def connections_from(self, node: int, *, enabled_only: bool = False) -> list[ConnectionGene]:
        """Connection genes whose ``src`` is ``node``."""
        return self._select(enabled_only=enabled_only, src=node)

    def find_connection(
        self, src: int, dst: int | str, kind: Kind, *, enabled_only: bool = False
    ) -> ConnectionGene | None:
        """The connection gene ``(src, dst, kind)``, or None."""
        found = self._select(enabled_only=enabled_only, src=src, dst=dst, kind=Kind(kind))
        return found[0] if found else None

    def has_connection(
        self, src: int, dst: int | str, kind: Kind, *, enabled_only: bool = False
    ) -> bool:
        """Whether a connection gene ``(src, dst, kind)`` exists (enabled or not by default)."""
        return self.find_connection(src, dst, kind, enabled_only=enabled_only) is not None

    # --------------------------------------------------------------- builders

    def add_node(
        self,
        innovations: InnovationDB,
        key: tuple,
        properties: dict[str, int] | None = None,
    ) -> NodeGene:
        """Allocate ``innovations.node(key)`` and insert a node gene of type ``key[-1]``.

        ``properties`` defaults to the type's default properties. If the innovation is
        already present the existing gene is returned unchanged. Does not validate.
        """
        node_type = WorkerType(key[-1])
        inn = innovations.node(key)
        if inn in self.nodes:
            return self.nodes[inn]
        gene = NodeGene(
            innovation=inn,
            type=node_type,
            properties=dict(properties) if properties is not None else _default_properties(node_type),
        )
        self.nodes[inn] = gene
        return gene

    def add_connection(
        self,
        innovations: InnovationDB,
        src: int,
        dst: int | str,
        kind: Kind,
        *,
        enabled: bool = True,
    ) -> ConnectionGene:
        """Allocate ``innovations.connection(src, dst, kind)`` and insert the gene.

        If the gene already exists its ``enabled`` flag is set to ``enabled`` and it is
        returned. Does not validate.
        """
        kind = Kind(kind)
        inn = innovations.connection(src, dst, kind)
        existing = self.connections.get(inn)
        if existing is not None:
            existing.enabled = enabled
            return existing
        gene = ConnectionGene(innovation=inn, src=src, dst=dst, kind=kind, enabled=enabled)
        self.connections[inn] = gene
        return gene

    # ------------------------------------------------------------------- copy

    def copy(self, *, new_id: bool = True) -> Genome:
        """Deep copy of the genes; a fresh uuid hex id unless ``new_id=False``.

        ``generation``, ``parents``, ``operators`` (new list), ``fitness``,
        ``adjusted_fitness`` and ``species_id`` are kept as they are.
        """
        return Genome(
            id=_new_id() if new_id else self.id,
            nodes={
                inn: NodeGene(g.innovation, g.type, dict(g.properties))
                for inn, g in self.nodes.items()
            },
            connections={
                inn: _copy.copy(c) for inn, c in self.connections.items()
            },
            generation=self.generation,
            parents=tuple(self.parents),
            operators=list(self.operators),
            fitness=self.fitness,
            adjusted_fitness=self.adjusted_fitness,
            species_id=self.species_id,
            created_at=self.created_at,
            evaluated_at=self.evaluated_at,
            eval_duration_s=self.eval_duration_s,
            task_scores=list(self.task_scores),
        )

    # ------------------------------------------------------------------- JSON

    def to_json(self) -> dict:
        """JSON-serialisable dict (enums as values, END as ``"END"``)."""
        return {
            "id": self.id,
            "generation": self.generation,
            "parents": list(self.parents),
            "operators": list(self.operators),
            "fitness": self.fitness,
            "adjusted_fitness": self.adjusted_fitness,
            "species_id": self.species_id,
            "created_at": self.created_at,
            "evaluated_at": self.evaluated_at,
            "eval_duration_s": self.eval_duration_s,
            "task_scores": list(self.task_scores),
            "fitness_history": list(self.fitness_history),
            "nodes": [
                {
                    "innovation": g.innovation,
                    "type": g.type.value,
                    "properties": dict(g.properties),
                }
                for _, g in sorted(self.nodes.items())
            ],
            "connections": [
                {
                    "innovation": c.innovation,
                    "src": c.src,
                    "dst": END if c.dst == END else c.dst,
                    "kind": c.kind.value,
                    "enabled": c.enabled,
                }
                for _, c in sorted(self.connections.items())
            ],
        }

    @staticmethod
    def from_json(data: dict) -> Genome:
        """Inverse of :meth:`to_json`. Does not validate."""
        nodes: dict[int, NodeGene] = {}
        for raw in data.get("nodes", []):
            gene = NodeGene(
                innovation=int(raw["innovation"]),
                type=WorkerType(raw["type"]),
                properties={str(k): int(v) for k, v in (raw.get("properties") or {}).items()},
            )
            nodes[gene.innovation] = gene
        connections: dict[int, ConnectionGene] = {}
        for raw in data.get("connections", []):
            dst = raw["dst"]
            gene = ConnectionGene(
                innovation=int(raw["innovation"]),
                src=int(raw["src"]),
                dst=END if dst == END else int(dst),
                kind=Kind(raw["kind"]),
                enabled=bool(raw.get("enabled", True)),
            )
            connections[gene.innovation] = gene
        fitness = data.get("fitness")
        adjusted = data.get("adjusted_fitness")
        species_id = data.get("species_id")
        return Genome(
            id=str(data["id"]),
            nodes=nodes,
            connections=connections,
            generation=int(data.get("generation", 0)),
            parents=tuple(str(p) for p in data.get("parents", ()) or ()),
            operators=[str(o) for o in data.get("operators", []) or []],
            fitness=None if fitness is None else float(fitness),
            adjusted_fitness=None if adjusted is None else float(adjusted),
            species_id=None if species_id is None else int(species_id),
            created_at=data.get("created_at"),
            evaluated_at=data.get("evaluated_at"),
            eval_duration_s=(
                None if data.get("eval_duration_s") is None else float(data["eval_duration_s"])
            ),
            task_scores=[float(x) for x in data.get("task_scores", []) or []],
            fitness_history=[float(x) for x in data.get("fitness_history", []) or []],
        )


def minimal_genome(
    innovations: InnovationDB,
    properties: dict[str, int] | None = None,
    *,
    generation: int = 0,
) -> Genome:
    """One EDITOR (node key ``("init", 0, EDITOR)``) feeding END with an EDIT connection.

    On a fresh ``InnovationDB`` the editor gets innovation 0 (matching ``minimal_graph``)
    and the END connection innovation 1.
    """
    genome = Genome(id=_new_id(), nodes={}, connections={}, generation=generation)
    editor = genome.add_node(innovations, ("init", 0, WorkerType.EDITOR), properties)
    genome.add_connection(innovations, editor.innovation, END, Kind.EDIT)
    return genome


__all__ = ["ConnectionGene", "Genome", "NodeGene", "minimal_genome"]
