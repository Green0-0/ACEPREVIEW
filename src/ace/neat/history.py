"""Evolution history: lineage, species and fitness logs (GRAPH_NEAT_DESIGN.md section 10.6).

Files under ``directory`` (the directory is created on the first write):

* ``genomes.jsonl``      one record per genome, logged at most once per genome id::

      {"id", "generation", "parents": [...], "operators": [...], "species_id",
       "fitness", "adjusted_fitness", "structure_hash", "num_workers", "genome": <to_json()>}

* ``species.jsonl``      one record per :meth:`History.log_species` call (one generation)::

      {"generation", "threshold", "species": [{"id", "size", "member_ids",
       "representative_id", "best_fitness", "created", "last_improved"}, ...]}

* ``generations.jsonl``  ``{"generation": g, **stats}`` per :meth:`History.log_generation`.
* ``innovations.json``   the :class:`InnovationDB` (see ``innovation.py``).
* ``run.json``           run metadata (recording spec 11.1), rewritten atomically
                         (temp file + ``os.replace``) by :meth:`History.log_run`.
* ``events.jsonl``       ``{"ts", "event", ...}`` progress events (:meth:`History.log_event`).
* ``evaluations.jsonl``  one record per (genome, task) evaluation (:meth:`History.log_evaluation`).

Genome records also carry (recording spec 11.4) ``created_at``, ``evaluated_at``,
``eval_duration_s`` (read from the genome, default ``None``), ``worker_types``
(``{type value: count}`` over ``to_graph().nodes``, ``{}`` without a graph) and
``task_scores`` (default ``[]``).

Contract readings:

* ``log_genome`` is duck-typed: it only needs ``id``; every other attribute
  (``to_json``, ``parents``, ``operators``, ``generation``, ``species_id``, ``fitness``,
  ``adjusted_fitness``, ``to_graph``) is read with ``getattr`` and missing values are logged
  as ``None`` (``[]`` for parents/operators). ``num_workers = len(to_graph().nodes)``.
* "Logged exactly once": a genome id already in ``genomes.jsonl`` (including records written
  by an earlier process, re-read on construction) is silently skipped, so later fitness
  updates are not recorded. The population should therefore log a genome after it has been
  evaluated.
* ``lineage`` walks parents breadth-first, de-duplicates, skips parents that were never
  logged, and returns the genome's own record first and the oldest ancestors last (ancestors
  stably sorted by descending generation when all have one).
* JSON values: enums are written as their ``.value``, paths as strings, sets/tuples as lists.
  Malformed lines (e.g. a truncated write from a crash) are skipped with a warning when
  reading, and the next append starts on a fresh line.
"""
from __future__ import annotations

import json
import os
import warnings
from collections import deque
from collections.abc import Iterator
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ace.graph.clock import utc_now
from ace.neat.innovation import InnovationDB

if TYPE_CHECKING:
    from ace.neat.genome import Genome
    from ace.neat.speciation import Species

GENOMES_FILE = "genomes.jsonl"
SPECIES_FILE = "species.jsonl"
GENERATIONS_FILE = "generations.jsonl"
INNOVATIONS_FILE = "innovations.json"
RUN_FILE = "run.json"
EVENTS_FILE = "events.jsonl"
EVALUATIONS_FILE = "evaluations.jsonl"


def _json_default(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (set, frozenset)):
        return sorted(value, key=repr)
    if isinstance(value, tuple):
        return list(value)
    to_json = getattr(value, "to_json", None)
    if callable(to_json):
        return to_json()
    raise TypeError(f"object of type {type(value).__name__} is not JSON serialisable")


def _dumps(record: dict[str, Any]) -> str:
    return json.dumps(record, default=_json_default, separators=(",", ":"))


def worker_type_counts(nodes: Any) -> dict[str, int]:
    """``{worker type value: count}`` over a graph's ``nodes`` mapping (sorted keys)."""
    counts: dict[str, int] = {}
    values = nodes.values() if hasattr(nodes, "values") else nodes
    for node in values:
        node_type = getattr(node, "type", None)
        if node_type is None:
            continue
        key = node_type.value if isinstance(node_type, Enum) else str(node_type)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


class History:
    """Append-only JSONL logs of every genome, species snapshot and generation summary."""

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)
        self._logged: set[str] = {record["id"] for record in self.iter_genomes() if "id" in record}

    # -- paths ------------------------------------------------------------------------

    @property
    def genomes_path(self) -> Path:
        return self.directory / GENOMES_FILE

    @property
    def species_path(self) -> Path:
        return self.directory / SPECIES_FILE

    @property
    def generations_path(self) -> Path:
        return self.directory / GENERATIONS_FILE

    @property
    def innovations_path(self) -> Path:
        return self.directory / INNOVATIONS_FILE

    @property
    def run_path(self) -> Path:
        return self.directory / RUN_FILE

    @property
    def events_path(self) -> Path:
        return self.directory / EVENTS_FILE

    @property
    def evaluations_path(self) -> Path:
        return self.directory / EVALUATIONS_FILE

    @staticmethod
    def now() -> str:
        """The current time as an ISO-8601 UTC string."""
        return utc_now()

    def _append(self, path: Path, record: dict[str, Any]) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        line = _dumps(record)
        prefix = ""
        if path.exists() and path.stat().st_size > 0:
            with path.open("rb") as fh:
                fh.seek(-1, 2)
                if fh.read(1) != b"\n":
                    prefix = "\n"  # isolate a truncated trailing write on its own line
        with path.open("a", encoding="utf-8") as fh:
            fh.write(prefix + line + "\n")

    # -- writing ----------------------------------------------------------------------

    def is_logged(self, genome_id: str) -> bool:
        """Whether ``genome_id`` already has a record in ``genomes.jsonl``."""
        return genome_id in self._logged

    def log_genome(self, genome: Genome) -> None:
        """Append ``genome`` to ``genomes.jsonl`` unless its id was already logged."""
        genome_id = genome.id
        if genome_id in self._logged:
            return
        structure_hash: str | None = None
        num_workers: int | None = None
        worker_types: dict[str, int] = {}
        to_graph = getattr(genome, "to_graph", None)
        if callable(to_graph):
            graph = to_graph()
            hash_fn = getattr(graph, "structure_hash", None)
            if callable(hash_fn):
                structure_hash = hash_fn()
            nodes = getattr(graph, "nodes", None)
            if nodes is not None:
                num_workers = len(nodes)
                worker_types = worker_type_counts(nodes)
        to_json = getattr(genome, "to_json", None)
        record = {
            "id": genome_id,
            "generation": getattr(genome, "generation", None),
            "parents": list(getattr(genome, "parents", ()) or ()),
            "operators": list(getattr(genome, "operators", ()) or ()),
            "species_id": getattr(genome, "species_id", None),
            "fitness": getattr(genome, "fitness", None),
            "adjusted_fitness": getattr(genome, "adjusted_fitness", None),
            "structure_hash": structure_hash,
            "num_workers": num_workers,
            "genome": to_json() if callable(to_json) else None,
            "created_at": getattr(genome, "created_at", None),
            "evaluated_at": getattr(genome, "evaluated_at", None),
            "eval_duration_s": getattr(genome, "eval_duration_s", None),
            "worker_types": worker_types,
            "task_scores": list(getattr(genome, "task_scores", None) or []),
        }
        self._append(self.genomes_path, record)
        self._logged.add(genome_id)

    def log_species(self, generation: int, species: list[Species], threshold: float) -> None:
        """Append one snapshot of all species for ``generation`` to ``species.jsonl``."""
        entries = []
        for sp in species:
            members = list(getattr(sp, "members", []) or [])
            representative = getattr(sp, "representative", None)
            entries.append(
                {
                    "id": getattr(sp, "id", None),
                    "size": len(members),
                    "member_ids": [getattr(m, "id", None) for m in members],
                    "representative_id": getattr(representative, "id", None),
                    "best_fitness": getattr(sp, "best_fitness", None),
                    "created": getattr(sp, "created", None),
                    "last_improved": getattr(sp, "last_improved", None),
                }
            )
        self._append(
            self.species_path,
            {"generation": generation, "threshold": threshold, "species": entries},
        )

    def log_generation(self, generation: int, stats: dict) -> None:
        """Append ``{"generation": generation, **stats}`` to ``generations.jsonl``.

        The ``generation`` argument wins over a ``"generation"`` key inside ``stats``.
        """
        record: dict[str, Any] = {"generation": generation}
        record.update({k: v for k, v in stats.items() if k != "generation"})
        self._append(self.generations_path, record)

    def log_run(self, record: dict) -> None:
        """Rewrite ``run.json`` with ``record`` atomically (temp file + ``os.replace``)."""
        self.directory.mkdir(parents=True, exist_ok=True)
        tmp = self.run_path.with_name(self.run_path.name + ".tmp")
        tmp.write_text(
            json.dumps(record, default=_json_default, indent=2), encoding="utf-8"
        )
        os.replace(tmp, self.run_path)

    def log_event(self, event: str, **fields: Any) -> None:
        """Append ``{"ts": now, "event": event, **fields}`` to ``events.jsonl``."""
        record: dict[str, Any] = {"ts": self.now(), "event": event}
        record.update({k: v for k, v in fields.items() if k not in ("ts", "event")})
        self._append(self.events_path, record)

    def log_evaluation(self, record: dict) -> None:
        """Append one (genome, task) evaluation record to ``evaluations.jsonl``."""
        self._append(self.evaluations_path, record)

    def save_innovations(self, db: InnovationDB) -> None:
        """Write ``db`` to ``innovations.json``."""
        db.save(self.innovations_path)

    def load_innovations(self) -> InnovationDB | None:
        """Read ``innovations.json``, or None if it does not exist."""
        if not self.innovations_path.exists():
            return None
        return InnovationDB.load(self.innovations_path)

    # -- reading ----------------------------------------------------------------------

    @staticmethod
    def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
        if not path.exists():
            return
        with path.open("r", encoding="utf-8") as fh:
            lines = fh.readlines()
        for index, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                # A truncated write from an interrupted run; skip it rather than lose the log.
                warnings.warn(f"{path}:{index}: skipping malformed JSONL line", stacklevel=2)

    def iter_genomes(self) -> Iterator[dict[str, Any]]:
        """Yield genome records in logging order."""
        yield from self._iter_jsonl(self.genomes_path)

    def iter_species(self) -> Iterator[dict[str, Any]]:
        """Yield species snapshots in logging order."""
        yield from self._iter_jsonl(self.species_path)

    def iter_generations(self) -> Iterator[dict[str, Any]]:
        """Yield generation summaries in logging order."""
        yield from self._iter_jsonl(self.generations_path)

    def iter_events(self) -> Iterator[dict[str, Any]]:
        """Yield progress events in logging order."""
        yield from self._iter_jsonl(self.events_path)

    def iter_evaluations(self) -> Iterator[dict[str, Any]]:
        """Yield evaluation records in logging order."""
        yield from self._iter_jsonl(self.evaluations_path)

    def load_run(self) -> dict | None:
        """``run.json`` as a dict, or None when it does not exist (or is unreadable)."""
        if not self.run_path.exists():
            return None
        try:
            return json.loads(self.run_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            warnings.warn(f"{self.run_path}: unreadable run.json", stacklevel=2)
            return None

    def evaluations_for(self, genome_id: str) -> list[dict]:
        """Evaluation records of ``genome_id``, sorted by task index."""
        records = [r for r in self.iter_evaluations() if r.get("genome_id") == genome_id]
        return sorted(records, key=lambda r: r.get("task_index", 0))

    def load_genome_records(self) -> dict[str, dict[str, Any]]:
        """Map genome id -> its logged record (first record wins)."""
        records: dict[str, dict[str, Any]] = {}
        for record in self.iter_genomes():
            records.setdefault(record["id"], record)
        return records

    def lineage(self, genome_id: str) -> list[dict]:
        """Records of ``genome_id`` and all its logged ancestors, breadth-first.

        The genome's own record is first, the oldest ancestors last; each ancestor appears
        once even if reachable through several parents, and parents that were never logged
        are skipped. Ancestors are discovered breadth-first; when every ancestor record has
        an integer ``generation`` they are then stably sorted by generation, newest first, so
        "oldest last" also holds when parents come from different generations. Returns
        ``[]`` if ``genome_id`` was never logged.
        """
        records = self.load_genome_records()
        if genome_id not in records:
            return []
        chain: list[dict] = []
        seen = {genome_id}
        queue: deque[str] = deque([genome_id])
        while queue:
            current = queue.popleft()
            record = records[current]
            chain.append(record)
            for parent in record.get("parents") or []:
                if parent in seen or parent not in records:
                    continue
                seen.add(parent)
                queue.append(parent)
        generations = [r.get("generation") for r in chain[1:]]
        if all(isinstance(g, int) and not isinstance(g, bool) for g in generations):
            # Stable: keeps breadth-first order among equal generations.
            chain[1:] = sorted(chain[1:], key=lambda r: -r["generation"])
        return chain
