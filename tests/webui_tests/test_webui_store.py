"""RunStore / JsonlFile caching, tailing and invalidation; pure layout helpers."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from ace.graph.graph import minimal_graph
from ace.webui import analysis
from ace.webui.store import InvalidName, JsonlFile, NotFound, RunStore, check_name


def _write(path: Path, records: list[dict]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")


def test_jsonl_tail_reads_only_new_bytes(tmp_path):
    path = tmp_path / "x.jsonl"
    _write(path, [{"a": 1}, {"a": 2}])
    f = JsonlFile(path)
    assert f.refresh() is True
    offset = f.offset
    assert [r["a"] for r in f.records] == [1, 2]
    assert f.refresh() is False
    with path.open("a") as fh:
        fh.write(json.dumps({"a": 3}) + "\n")
    assert f.refresh() is True
    assert f.offset > offset
    assert [r["a"] for r in f.records] == [1, 2, 3]


def test_jsonl_truncation_and_rewrite_reload(tmp_path):
    path = tmp_path / "x.jsonl"
    _write(path, [{"a": 1}, {"a": 2}])
    f = JsonlFile(path)
    f.refresh()
    _write(path, [{"a": 9}])  # shorter: reload from start
    f.refresh()
    assert [r["a"] for r in f.records] == [9]
    _write(path, [{"a": 8}])  # same size, new mtime: reload
    st = path.stat()
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 10_000_000))
    f.refresh()
    assert [r["a"] for r in f.records] == [8]
    path.unlink()
    f.refresh()
    assert f.records == []


def test_jsonl_skips_malformed_lines(tmp_path):
    path = tmp_path / "x.jsonl"
    path.write_text('{"a": 1}\n{"a": \n{"a": 2}\n', encoding="utf-8")
    f = JsonlFile(path)
    f.refresh()
    assert [r["a"] for r in f.records] == [1, 2]


def test_check_name():
    assert check_name("run-1.a_b") == "run-1.a_b"
    for bad in ("..", ".", "a/b", "../x", ".hidden", "", "a\\b"):
        with pytest.raises(InvalidName):
            check_name(bad)


def test_store_discovery(tmp_path):
    (tmp_path / "r1").mkdir()
    _write(tmp_path / "r1" / "generations.jsonl", [{"generation": 0}])
    (tmp_path / "r2").mkdir()
    (tmp_path / "r2" / "run.json").write_text("{}", encoding="utf-8")
    (tmp_path / "nothing").mkdir()
    (tmp_path / "file.txt").write_text("x", encoding="utf-8")
    store = RunStore(tmp_path)
    assert store.run_ids() == ["r1", "r2"]
    with pytest.raises(NotFound):
        store.get("nothing")
    with pytest.raises(InvalidName):
        store.get("..")
    assert RunStore(tmp_path / "missing").run_ids() == []


def test_index_rebuilds_on_change(tmp_path):
    run_dir = tmp_path / "r"
    run_dir.mkdir()
    _write(run_dir / "genomes.jsonl", [{"id": "a", "generation": 0, "parents": []}])
    _write(run_dir / "generations.jsonl", [{"generation": 0}])
    run = RunStore(tmp_path).get("r")
    assert list(run.index().genomes) == ["a"]
    with (run_dir / "genomes.jsonl").open("a") as fh:
        fh.write(json.dumps({"id": "b", "generation": 1, "parents": ["a"]}) + "\n")
        fh.write(json.dumps({"id": "a", "generation": 5, "parents": []}) + "\n")  # duplicate: first wins
    index = run.index()
    assert list(index.genomes) == ["a", "b"]
    assert index.genomes["a"]["generation"] == 0
    assert index.children == {"a": ["b"]}


def test_layout_minimal_graph():
    layout = analysis.layout(minimal_graph())
    nodes = {n["id"]: n for n in layout["nodes"]}
    assert nodes[0]["level"] == 0 and nodes["END"]["level"] == 1
    assert nodes[0]["y"] == 0.0 and nodes["END"]["y"] == 1.0
    assert nodes[0]["x"] == 0.5
    assert nodes["END"]["type"] == "end" and nodes["END"]["properties_resolved"] == {}
    assert layout["edges"] == [{"src": 0, "dst": "END", "kind": "edit"}]
