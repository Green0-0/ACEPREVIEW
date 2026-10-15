"""Global innovation database (GRAPH_NEAT_DESIGN.md section 7.4).

:class:`InnovationDB` hands out innovation numbers for structural events:

* ``node(key)`` for node genes, where ``key`` is an operator-specific tuple such as
  ``("init", index, WorkerType.EDITOR)`` or ``("split", conn_innovation, type)``;
* ``connection(src, dst, kind)`` for connection genes, keyed on node innovations.

Node and connection innovations share **one** monotonically increasing counter (classic
NEAT style), so a node id never equals a connection id. The counter starts at 0, so on a
fresh database the first allocated key (normally ``("init", 0, WorkerType.EDITOR)`` for the
minimal graph) receives id 0, matching ``minimal_graph()``'s editor id. Keys are never reset;
the same key always yields the same id. Not thread/async safe (not required).

Key equality is plain tuple equality. Because ``Kind`` and ``WorkerType`` are ``str``
enums, ``Kind.VIEW == "view"``; such keys therefore collide by design (they denote the
same event). ``connection`` normalises ``kind`` to :class:`Kind`.

JSON key serialisation (lossless, used by :meth:`InnovationDB.save`): every key element is
encoded as a two-element tagged list ``[tag, value]``:

=============== ======================================== ==========================
tag             Python value                             JSON value
=============== ======================================== ==========================
``"WorkerType"``  ``WorkerType`` member                   its ``.value`` string
``"Kind"``        ``Kind`` member                         its ``.value`` string
``"bool"``        ``bool``                                JSON bool
``"int"``         ``int``                                 JSON int
``"float"``       ``float``                               JSON number
``"str"``         ``str`` (including ``END``)             JSON string
``"none"``        ``None``                                ``null``
``"tuple"``       nested ``tuple`` (or list, decoded as a tuple)  list of tagged elements
=============== ======================================== ==========================

A whole key is a list of tagged elements. Enum tags are checked before ``str``/``int`` and
``bool`` before ``int``, so the exact Python types come back on load and
``InnovationDB.load(path)`` compares equal to the saved database. Connection entries are
stored as ``{"src": int, "dst": int | "END", "kind": "<Kind value>", "id": int}``.

File format::

    {"version": 1, "next_id": N,
     "nodes": [{"key": [[tag, value], ...], "id": i}, ...],
     "connections": [{"src": s, "dst": d, "kind": k, "id": i}, ...]}
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from ace.graph.types import Kind, WorkerType

FORMAT_VERSION = 1

ConnectionKey = tuple[int, int | str, Kind]


def encode_key_element(value: Any) -> list[Any]:
    """Encode one key element as a ``[tag, value]`` list (see module docstring)."""
    if isinstance(value, WorkerType):
        return ["WorkerType", value.value]
    if isinstance(value, Kind):
        return ["Kind", value.value]
    if isinstance(value, bool):
        return ["bool", value]
    if isinstance(value, int):
        return ["int", value]
    if isinstance(value, float):
        return ["float", value]
    if isinstance(value, str):
        return ["str", value]
    if value is None:
        return ["none", None]
    if isinstance(value, (tuple, list)):
        return ["tuple", [encode_key_element(v) for v in value]]
    raise TypeError(f"cannot encode innovation key element of type {type(value).__name__}: {value!r}")


def decode_key_element(item: Any) -> Any:
    """Inverse of :func:`encode_key_element`."""
    if not isinstance(item, list) or len(item) != 2:
        raise ValueError(f"malformed innovation key element: {item!r}")
    tag, value = item
    match tag:
        case "WorkerType":
            return WorkerType(value)
        case "Kind":
            return Kind(value)
        case "bool":
            return bool(value)
        case "int":
            return int(value)
        case "float":
            return float(value)
        case "str":
            return str(value)
        case "none":
            return None
        case "tuple":
            return tuple(decode_key_element(v) for v in value)
    raise ValueError(f"unknown innovation key tag: {tag!r}")


def encode_key(key: tuple) -> list[Any]:
    """Encode a whole node key tuple."""
    return [encode_key_element(v) for v in key]


def decode_key(data: list[Any]) -> tuple:
    """Decode a whole node key."""
    return tuple(decode_key_element(v) for v in data)


class InnovationDB:
    """Persistent, global mapping from structural events to innovation numbers."""

    def __init__(self) -> None:
        self.next_id: int = 0
        self._nodes: dict[tuple, int] = {}
        self._connections: dict[ConnectionKey, int] = {}

    # -- allocation -------------------------------------------------------------------

    def _allocate(self) -> int:
        new_id = self.next_id
        self.next_id += 1
        return new_id

    def node(self, key: tuple) -> int:
        """Return the innovation of node key ``key``, allocating one if new."""
        if not isinstance(key, tuple):
            raise TypeError(f"node key must be a tuple, got {type(key).__name__}")
        existing = self._nodes.get(key)
        if existing is not None:
            return existing
        encode_key(key)  # fail fast on keys that could not be saved
        new_id = self._allocate()
        self._nodes[key] = new_id
        return new_id

    def connection(self, src: int, dst: int | str, kind: Kind) -> int:
        """Return the innovation of connection ``(src, dst, kind)``, allocating if new."""
        key = self._connection_key(src, dst, kind)
        existing = self._connections.get(key)
        if existing is not None:
            return existing
        new_id = self._allocate()
        self._connections[key] = new_id
        return new_id

    # -- lookup without allocation ----------------------------------------------------

    def peek_node(self, key: tuple) -> int | None:
        """Return the innovation of ``key`` if already allocated, else None."""
        return self._nodes.get(key)

    def peek_connection(self, src: int, dst: int | str, kind: Kind) -> int | None:
        """Return the innovation of ``(src, dst, kind)`` if already allocated, else None."""
        return self._connections.get(self._connection_key(src, dst, kind))

    def node_entries(self) -> dict[tuple, int]:
        """A copy of all node keys and their innovations."""
        return dict(self._nodes)

    def connection_entries(self) -> dict[ConnectionKey, int]:
        """A copy of all connection keys ``(src, dst, kind)`` and their innovations."""
        return dict(self._connections)

    @staticmethod
    def _connection_key(src: int, dst: int | str, kind: Kind) -> ConnectionKey:
        if isinstance(src, bool) or not isinstance(src, int):
            raise TypeError(f"connection src must be an int node innovation, got {src!r}")
        if isinstance(dst, bool) or not isinstance(dst, (int, str)):
            raise TypeError(f"connection dst must be an int or END, got {dst!r}")
        return (src, dst, Kind(kind))

    # -- dunder -----------------------------------------------------------------------

    def __len__(self) -> int:
        """Number of allocated innovations (nodes + connections)."""
        return len(self._nodes) + len(self._connections)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, InnovationDB):
            return NotImplemented
        return (
            self.next_id == other.next_id
            and self._nodes == other._nodes
            and self._connections == other._connections
        )

    __hash__ = None  # type: ignore[assignment]  # mutable

    def __repr__(self) -> str:
        return (
            f"InnovationDB(next_id={self.next_id}, nodes={len(self._nodes)}, "
            f"connections={len(self._connections)})"
        )

    # -- persistence ------------------------------------------------------------------

    def to_json(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dict (format in the module docstring)."""
        return {
            "version": FORMAT_VERSION,
            "next_id": self.next_id,
            "nodes": [
                {"key": encode_key(key), "id": i}
                for key, i in sorted(self._nodes.items(), key=lambda kv: kv[1])
            ],
            "connections": [
                {"src": src, "dst": dst, "kind": kind.value, "id": i}
                for (src, dst, kind), i in sorted(self._connections.items(), key=lambda kv: kv[1])
            ],
        }

    @staticmethod
    def from_json(data: dict[str, Any]) -> InnovationDB:
        """Inverse of :meth:`to_json`; validates id uniqueness and the counter."""
        version = data.get("version", FORMAT_VERSION)
        if version != FORMAT_VERSION:
            raise ValueError(f"unsupported innovation DB format version {version!r}")
        db = InnovationDB()
        seen: set[int] = set()
        for entry in data.get("nodes", []):
            key = decode_key(entry["key"])
            node_id = int(entry["id"])
            if node_id in seen or key in db._nodes:
                raise ValueError(f"duplicate innovation entry for node key {key!r} / id {node_id}")
            seen.add(node_id)
            db._nodes[key] = node_id
        for entry in data.get("connections", []):
            dst = entry["dst"]
            ckey = InnovationDB._connection_key(int(entry["src"]), dst, Kind(entry["kind"]))
            conn_id = int(entry["id"])
            if conn_id in seen or ckey in db._connections:
                raise ValueError(f"duplicate innovation entry for connection {ckey!r} / id {conn_id}")
            seen.add(conn_id)
            db._connections[ckey] = conn_id
        floor = max(seen) + 1 if seen else 0
        next_id = int(data.get("next_id", floor))
        if next_id < floor:
            raise ValueError(f"next_id {next_id} is not above the largest stored id {floor - 1}")
        db.next_id = next_id
        return db

    def save(self, path: Path) -> None:
        """Write the database to ``path`` as JSON (atomically; parents created)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(self.to_json(), indent=1), encoding="utf-8")
        os.replace(tmp, path)

    @staticmethod
    def load(path: Path) -> InnovationDB:
        """Read a database previously written by :meth:`save`."""
        return InnovationDB.from_json(json.loads(Path(path).read_text(encoding="utf-8")))
