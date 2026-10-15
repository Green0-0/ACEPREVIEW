"""Run discovery, cached JSON/JSONL loading with offset-based tailing, and per-run indexes.

A **run** is any directory directly under ``runs_root`` that contains ``run.json`` or
``generations.jsonl``; its ``run_id`` is the directory name. Everything here only *reads*.

Caching rules:

* A JSONL file is parsed incrementally. The store remembers the byte offset just past the
  last complete line it consumed and, on the next access, ``stat``s the file: if the size
  grew it reads only the new bytes; if the size shrank, or the size is unchanged but the
  mtime moved (the file was rewritten in place), it re-reads from the start; otherwise the
  cached records are returned untouched. A trailing line without ``\\n`` (a write in
  progress) is left for the next poll; malformed complete lines are skipped.
* JSON snapshots (``run.json``, ``innovations.json``) are re-parsed when their
  ``(mtime_ns, size)`` changes.
* Derived indexes (genomes by id, children, generations by number, species snapshots,
  evaluations by ``(genome, task)``) are rebuilt only when one of their source files changed.
"""
from __future__ import annotations

import json
import os
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

RUN_FILE = "run.json"
EVENTS_FILE = "events.jsonl"
GENERATIONS_FILE = "generations.jsonl"
GENOMES_FILE = "genomes.jsonl"
SPECIES_FILE = "species.jsonl"
EVALUATIONS_FILE = "evaluations.jsonl"
INNOVATIONS_FILE = "innovations.json"
ROLLOUTS_DIR = "rollouts"

SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


class NotFound(Exception):
    """An unknown run, genome, generation, evaluation, worker or rollout (HTTP 404)."""


class InvalidName(Exception):
    """A path component that is not a plain file/directory name (HTTP 400)."""


def check_name(name: str, what: str = "id") -> str:
    """Return ``name`` if it is a plain directory/file name, else raise :class:`InvalidName`."""
    if (
        not isinstance(name, str)
        or not SAFE_NAME_RE.match(name)
        or name in (".", "..")
        or name.startswith(".")
    ):
        raise InvalidName(f"invalid {what} {name!r}: must match ^[A-Za-z0-9_.-]+$ and not start with '.'")
    return name


# --------------------------------------------------------------------------- file caches


class JsonlFile:
    """Incrementally parsed JSONL file (see module docstring for invalidation)."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.records: list[dict[str, Any]] = []
        self.offset = 0
        self.mtime_ns: int | None = None
        self.version = 0  # bumped whenever ``records`` changes

    def _reset(self) -> None:
        if self.records or self.offset:
            self.version += 1
        self.records = []
        self.offset = 0
        self.mtime_ns = None

    def refresh(self) -> bool:
        """Bring ``records`` up to date with the file; return whether anything changed."""
        try:
            st = os.stat(self.path)
        except OSError:
            had = bool(self.records)
            self._reset()
            return had
        size, mtime = st.st_size, st.st_mtime_ns
        if size < self.offset or (size == self.offset and mtime != self.mtime_ns and self.mtime_ns is not None):
            # Truncated or rewritten in place: start over.
            self.records = []
            self.offset = 0
            self.version += 1
        if size == self.offset:
            self.mtime_ns = mtime
            return False
        with open(self.path, "rb") as fh:
            fh.seek(self.offset)
            chunk = fh.read(size - self.offset)
        end = chunk.rfind(b"\n")
        if end < 0:
            return False  # only a partial line so far
        complete = chunk[: end + 1]
        self.offset += len(complete)
        self.mtime_ns = mtime
        added = False
        for raw in complete.split(b"\n"):
            if not raw.strip():
                continue
            try:
                record = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue  # truncated write from a crash
            if isinstance(record, dict):
                self.records.append(record)
                added = True
        if added:
            self.version += 1
        return added


class JsonFile:
    """A JSON snapshot re-parsed when its ``(mtime_ns, size)`` changes."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.data: Any = None
        self.key: tuple[int, int] | None = None
        self.version = 0

    def refresh(self) -> bool:
        try:
            st = os.stat(self.path)
        except OSError:
            changed = self.data is not None
            self.data, self.key = None, None
            if changed:
                self.version += 1
            return changed
        key = (st.st_mtime_ns, st.st_size)
        if key == self.key:
            return False
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return False  # mid-write; keep the previous snapshot and retry next time
        self.data, self.key = data, key
        self.version += 1
        return True


# --------------------------------------------------------------------------- one run


@dataclass
class RunIndex:
    """Derived tables of one run, rebuilt when a source file changes."""

    genomes: dict[str, dict[str, Any]] = field(default_factory=dict)
    genome_order: list[str] = field(default_factory=list)
    children: dict[str, list[str]] = field(default_factory=dict)
    by_generation: dict[int, list[str]] = field(default_factory=dict)
    by_species: dict[int, list[str]] = field(default_factory=dict)
    generations: dict[int, dict[str, Any]] = field(default_factory=dict)
    species: list[dict[str, Any]] = field(default_factory=list)
    evaluations: dict[tuple[str, int], dict[str, Any]] = field(default_factory=dict)
    evaluations_by_genome: dict[str, list[dict[str, Any]]] = field(default_factory=dict)


class Run:
    """One run directory: cached files plus lazily rebuilt indexes. Thread-safe."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.run_id = directory.name
        self._lock = threading.RLock()
        self.run_file = JsonFile(directory / RUN_FILE)
        self.innovations_file = JsonFile(directory / INNOVATIONS_FILE)
        self.events_file = JsonlFile(directory / EVENTS_FILE)
        self.generations_file = JsonlFile(directory / GENERATIONS_FILE)
        self.genomes_file = JsonlFile(directory / GENOMES_FILE)
        self.species_file = JsonlFile(directory / SPECIES_FILE)
        self.evaluations_file = JsonlFile(directory / EVALUATIONS_FILE)
        self._index = RunIndex()
        self._index_key: tuple[int, ...] | None = None

    # -- raw access -------------------------------------------------------------------

    def run_json(self) -> dict[str, Any] | None:
        with self._lock:
            self.run_file.refresh()
            data = self.run_file.data
            return data if isinstance(data, dict) else None

    def innovations(self) -> dict[str, Any] | None:
        with self._lock:
            self.innovations_file.refresh()
            data = self.innovations_file.data
            return data if isinstance(data, dict) else None

    def events(self) -> list[dict[str, Any]]:
        with self._lock:
            self.events_file.refresh()
            return list(self.events_file.records)

    def generation_records(self) -> list[dict[str, Any]]:
        """Generation records sorted by generation (a re-logged generation: last wins)."""
        index = self.index()
        return [index.generations[g] for g in sorted(index.generations)]

    def genome_records(self) -> list[dict[str, Any]]:
        index = self.index()
        return [index.genomes[gid] for gid in index.genome_order]

    def species_snapshots(self) -> list[dict[str, Any]]:
        return list(self.index().species)

    def evaluation_records(self) -> list[dict[str, Any]]:
        with self._lock:
            self.evaluations_file.refresh()
            return list(self.evaluations_file.records)

    # -- indexes ----------------------------------------------------------------------

    def index(self) -> RunIndex:
        with self._lock:
            for f in (self.generations_file, self.genomes_file, self.species_file, self.evaluations_file):
                f.refresh()
            key = (
                self.generations_file.version,
                self.genomes_file.version,
                self.species_file.version,
                self.evaluations_file.version,
            )
            if key != self._index_key:
                self._index = self._build_index()
                self._index_key = key
            return self._index

    def _build_index(self) -> RunIndex:
        idx = RunIndex()
        for record in self.genomes_file.records:
            gid = record.get("id")
            if not isinstance(gid, str) or gid in idx.genomes:
                continue  # first record wins
            idx.genomes[gid] = record
            idx.genome_order.append(gid)
            gen = record.get("generation")
            if isinstance(gen, int):
                idx.by_generation.setdefault(gen, []).append(gid)
            sid = record.get("species_id")
            if isinstance(sid, int):
                idx.by_species.setdefault(sid, []).append(gid)
        for gid in idx.genome_order:
            for parent in idx.genomes[gid].get("parents") or []:
                kids = idx.children.setdefault(str(parent), [])
                if gid not in kids:
                    kids.append(gid)
        for record in self.generations_file.records:
            gen = record.get("generation")
            if isinstance(gen, int):
                idx.generations[gen] = record
        idx.species = sorted(
            (r for r in self.species_file.records if isinstance(r.get("generation"), int)),
            key=lambda r: r["generation"],
        )
        for record in self.evaluations_file.records:
            gid = record.get("genome_id")
            task = record.get("task_index", 0)
            if not isinstance(gid, str) or not isinstance(task, int):
                continue
            if (gid, task) in idx.evaluations:
                previous = idx.evaluations[(gid, task)]
                idx.evaluations_by_genome[gid] = [
                    r for r in idx.evaluations_by_genome[gid] if r is not previous
                ]
            idx.evaluations[(gid, task)] = record  # last record wins
            idx.evaluations_by_genome.setdefault(gid, []).append(record)
        for records in idx.evaluations_by_genome.values():
            records.sort(key=lambda r: r.get("task_index", 0))
        return idx

    # -- lookups ----------------------------------------------------------------------

    def genome(self, genome_id: str) -> dict[str, Any]:
        record = self.index().genomes.get(genome_id)
        if record is None:
            raise NotFound(f"genome {genome_id!r} not found in run {self.run_id!r}")
        return record

    def generation(self, generation: int) -> dict[str, Any]:
        record = self.index().generations.get(generation)
        if record is None:
            raise NotFound(f"generation {generation} not found in run {self.run_id!r}")
        return record

    def evaluation(self, genome_id: str, task_index: int) -> dict[str, Any]:
        record = self.index().evaluations.get((genome_id, task_index))
        if record is None:
            raise NotFound(f"evaluation ({genome_id!r}, task {task_index}) not found in run {self.run_id!r}")
        return record

    def rollout_path(self, genome_id: str, task_index: int, rollout_id: str) -> Path:
        """Path of a rollout file; every component must be a plain name."""
        check_name(genome_id, "genome id")
        check_name(rollout_id, "rollout id")
        name = rollout_id if rollout_id.endswith(".json") else f"{rollout_id}.json"
        root = (self.directory / ROLLOUTS_DIR).resolve()
        path = (root / genome_id / str(int(task_index)) / name).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise NotFound(f"rollout {rollout_id!r} not found for genome {genome_id!r} task {task_index}")
        return path

    # -- status -----------------------------------------------------------------------

    def status(self) -> str:
        """``run.json`` status, else the last ``run_end`` event, else running/unknown."""
        run = self.run_json()
        if run and isinstance(run.get("status"), str):
            return run["status"]
        events = self.events()
        for event in reversed(events):
            if event.get("event") == "run_end" and isinstance(event.get("status"), str):
                return event["status"]
        return "running" if events else "unknown"


# --------------------------------------------------------------------------- the store


class RunStore:
    """Discovers runs under ``runs_root`` and caches one :class:`Run` per directory."""

    def __init__(self, runs_root: Path) -> None:
        self.runs_root = Path(runs_root)
        self._runs: dict[str, Run] = {}
        self._lock = threading.Lock()

    @staticmethod
    def is_run_dir(path: Path) -> bool:
        return path.is_dir() and ((path / RUN_FILE).is_file() or (path / GENERATIONS_FILE).is_file())

    def run_ids(self) -> list[str]:
        if not self.runs_root.is_dir():
            return []
        ids = []
        for child in sorted(self.runs_root.iterdir()):
            if SAFE_NAME_RE.match(child.name) and not child.name.startswith(".") and self.is_run_dir(child):
                ids.append(child.name)
        return ids

    def runs(self) -> list[Run]:
        return [self.get(run_id) for run_id in self.run_ids()]

    def get(self, run_id: str) -> Run:
        check_name(run_id, "run id")
        directory = self.runs_root / run_id
        if not self.is_run_dir(directory):
            with self._lock:
                self._runs.pop(run_id, None)
            raise NotFound(f"run {run_id!r} not found")
        with self._lock:
            run = self._runs.get(run_id)
            if run is None:
                run = self._runs[run_id] = Run(directory)
            return run


__all__ = [
    "InvalidName",
    "JsonFile",
    "JsonlFile",
    "NotFound",
    "Run",
    "RunIndex",
    "RunStore",
    "check_name",
]
