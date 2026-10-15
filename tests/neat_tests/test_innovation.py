"""Tests for ace.neat.innovation."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from ace.graph.types import END, Kind, WorkerType
from ace.neat.innovation import InnovationDB, decode_key, encode_key


def test_first_init_editor_is_zero() -> None:
    db = InnovationDB()
    assert db.next_id == 0
    assert db.node(("init", 0, WorkerType.EDITOR)) == 0
    assert db.next_id == 1


def test_same_key_same_id_and_shared_counter() -> None:
    db = InnovationDB()
    n0 = db.node(("init", 0, WorkerType.EDITOR))
    c0 = db.connection(n0, END, Kind.EDIT)
    n1 = db.node(("split", c0, WorkerType.CRITIC))
    c1 = db.connection(n0, n1, Kind.EDIT)
    ids = [n0, c0, n1, c1]
    assert ids == sorted(ids) == list(range(4))
    assert len(set(ids)) == 4
    assert db.node(("init", 0, WorkerType.EDITOR)) == n0
    assert db.connection(n0, END, Kind.EDIT) == c0
    assert db.connection(n0, END, "edit") == c0  # kind normalised
    assert len(db) == 4
    assert db.next_id == 4


def test_distinct_keys_distinct_ids() -> None:
    db = InnovationDB()
    a = db.node(("source", 3, Kind.VIEW, WorkerType.VIEWER))
    b = db.node(("source", 3, Kind.REPORT, WorkerType.VIEWER))
    c = db.connection(1, 2, Kind.VIEW)
    d = db.connection(1, 2, Kind.REPORT)
    e = db.connection(2, 1, Kind.VIEW)
    assert len({a, b, c, d, e}) == 5


def test_peek_does_not_allocate() -> None:
    db = InnovationDB()
    assert db.peek_node(("init", 0, WorkerType.EDITOR)) is None
    assert db.peek_connection(0, END, Kind.EDIT) is None
    assert db.next_id == 0 and len(db) == 0
    n = db.node(("init", 0, WorkerType.EDITOR))
    c = db.connection(n, END, Kind.EDIT)
    assert db.peek_node(("init", 0, WorkerType.EDITOR)) == n
    assert db.peek_connection(n, END, Kind.EDIT) == c
    assert db.next_id == 2


def test_invalid_keys() -> None:
    db = InnovationDB()
    with pytest.raises(TypeError):
        db.node(["init", 0])  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        db.node(("bad", object()))
    assert db.next_id == 0
    with pytest.raises(ValueError):
        db.connection(0, 1, "nope")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        db.connection("0", 1, Kind.VIEW)  # type: ignore[arg-type]


def test_key_encoding_lossless() -> None:
    key = ("merge", 3, 7, WorkerType.JUDGEMERGER, Kind.EDIT, END, True, 1.5, None, (1, "x"))
    encoded = encode_key(key)
    assert json.loads(json.dumps(encoded)) == encoded
    decoded = decode_key(json.loads(json.dumps(encoded)))
    assert decoded == key
    assert [type(v) for v in decoded] == [type(v) for v in key]
    assert encoded[3] == ["WorkerType", "judgemerger"]
    assert encoded[4] == ["Kind", "edit"]
    assert encoded[6] == ["bool", True]


def test_save_load_roundtrip(tmp_path: Path) -> None:
    db = InnovationDB()
    n0 = db.node(("init", 0, WorkerType.EDITOR))
    c0 = db.connection(n0, END, Kind.EDIT)
    n1 = db.node(("critic", c0, WorkerType.CRITIC))
    db.connection(n0, n1, Kind.EDIT)
    db.connection(n1, END, Kind.EDIT)
    db.node(("source", n1, Kind.VIEW, WorkerType.VIEWER))
    path = tmp_path / "sub" / "innovations.json"
    db.save(path)
    loaded = InnovationDB.load(path)
    assert loaded == db
    assert loaded.next_id == db.next_id
    assert len(loaded) == len(db)
    # Types survive the round trip.
    keys = loaded.node_entries()
    assert ("critic", c0, WorkerType.CRITIC) in keys
    for key in keys:
        assert isinstance(key[-1], WorkerType)
    conn_keys = loaded.connection_entries()
    assert (n1, END, Kind.EDIT) in conn_keys
    # Continuing allocation never collides with stored ids.
    new = loaded.node(("init", 1, WorkerType.VIEWER))
    assert new == db.next_id
    assert loaded.node(("init", 0, WorkerType.EDITOR)) == n0


def test_load_rejects_bad_counter(tmp_path: Path) -> None:
    db = InnovationDB()
    db.node(("init", 0, WorkerType.EDITOR))
    data = db.to_json()
    data["next_id"] = 0
    with pytest.raises(ValueError):
        InnovationDB.from_json(data)
    data = db.to_json()
    data["connections"].append({"src": 0, "dst": "END", "kind": "edit", "id": 0})
    with pytest.raises(ValueError):
        InnovationDB.from_json(data)


def test_empty_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "i.json"
    InnovationDB().save(path)
    assert InnovationDB.load(path) == InnovationDB()
