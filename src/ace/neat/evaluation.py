"""Fitness evaluation of genomes (GRAPH_NEAT_DESIGN.md section 10.7).

Two evaluators implement the :class:`Evaluator` protocol
(``async evaluate(genome) -> float``, raw fitness, higher is better):

* :class:`MockEvaluator` scores the *structure* of a genome without running anything. It is
  deterministic and network-free, for tests and dry runs of the evolution loop.
* :class:`TaskEvaluator` runs the genome's worker graph on a list of :class:`Task`\\ s and
  returns the mean task score.

MockEvaluator formula (``g = genome.to_graph()``, ``n = len(g.nodes)``)::

    score = 1.00 * |distinct worker types in g|
          + 0.50 * [END source is a JUDGEMERGER or PICKMERGER]
          + 0.25 * |CRITIC workers|
          - 0.10 * max(0, n - 6)
          + noise * (2u - 1)

where ``u`` in ``[0, 1)`` is drawn from ``random.Random(int(structure_hash, 16))``, so the
noise term is a reproducible function of the structure (``noise`` defaults to 0). If the
genome's ``structure_hash`` is a key of the ``scores`` override mapping, that value is
returned instead of the formula. Properties never influence the mock score.

TaskEvaluator contract readings:

* Every task gets a fresh ``Session`` and a fresh :class:`RunContext` (``settings``,
  ``client_factory``, ``rollout_dir``); the client factory defaults to
  :func:`~ace.graph.llm.make_client_factory` of ``settings``.
* **Rollout isolation.** The Session of task ``i`` of genome ``g`` is created with the
  *absolute* rollout dir ``<root>/<g.id>/<i>/`` (``Session`` joins ``project_dir /
  rollout_dir``, so an absolute path escapes the project), and the same path is
  ``RunContext.rollout_dir``. ``root`` is the constructor's ``rollout_dir``
  (``scripts/run_neat.py`` passes ``<history_dir>/rollouts``); when it is ``None`` a
  private ``tempfile.mkdtemp(prefix="agent2-rollouts-")`` directory is created on first
  use (``rollout_root``) and never deleted, so rollouts stay inspectable. Each Session
  therefore starts from an empty rollout dir instead of re-parsing every rollout of the
  run (which grew quadratically), and nothing is ever written into a task's project
  folder: the graph runtime writes only through the Session's rollout dir. A root that
  lies inside a task's project folder on a non-hidden path is rejected with
  ``ValueError`` (``Session.refresh_files`` would read the rollout JSON as project files
  and change the pristine base).
* One ``asyncio.Semaphore(concurrency)`` per evaluator (per event loop) bounds the number of
  concurrently running task graphs *across all* concurrent ``evaluate`` calls (the population
  evaluates genomes concurrently). Session construction, the graph run and scoring all
  happen while holding the semaphore.
* A :class:`GraphFailure` (the runner wraps every worker exception in one) scores 0.0 for
  that task and is logged with ``logging``. Exceptions raised by the scorer itself, or by
  ``Session`` construction, are *not* caught: they indicate a broken task definition.
* ``last_results`` holds the per-task records of the most recent ``evaluate`` call to
  finish; ``results_by_genome`` keeps them for every genome id. Record keys: ``task``
  (index), ``project_dir``, ``score``, ``failure`` (``None`` or ``"Type: message"``),
  ``node_id`` (failing node, if known), ``diff_chars``, ``rollout_dir``.

Run recording (recording spec 11.5): both evaluators accept ``history: History | None``.
When it is set they append one record per (genome, task) to ``evaluations.jsonl`` and the
``evaluation_start`` / ``evaluation_end`` events to ``events.jsonl``.
:class:`TaskEvaluator` records are built by :func:`build_evaluation_record` and contain
every node of the graph (status ``ok``, ``failed`` or ``skipped``; a sibling cancelled
because another worker of its level failed is ``skipped`` but keeps its timings).
``rollout_ids`` are the worker's reported ids plus any rollout JSON in the task's rollout
dir whose ``config_path`` is ``graph/<type>/<node id>`` (so a failed worker's rollout is
found too); ``llm_calls`` / ``tool_calls`` count ``assistant`` / ``tool`` messages in
those rollouts' ``raw_chat``. ``failure_type`` is the type name of the exception that
caused the ``GraphFailure`` (its ``__cause__``) when there is one, else of the
``GraphFailure`` itself (e.g. ``AceError``). ``order`` is ``GraphResult.order`` on success
and the started workers by start time on failure. ``describe()`` returns the
``run.json`` evaluator dict (``{"type": "mock" | "task", "tasks": [...]}``).

Helpers for task files (used by ``scripts/run_neat.py``): :func:`nonempty_diff` (builtin
scorer: 1.0 iff the edit is non-empty), :func:`resolve_scorer` (``"nonempty_diff"`` or
``"module:function"``; SWE-bench style scorers that run a test harness on the Edit plug in
this way) and :func:`load_tasks` (a JSON list of ``{"project_dir", "prompt", "scorer"}``;
relative project dirs resolve against the JSON file's directory).
"""
from __future__ import annotations

import asyncio
import importlib
import inspect
import json
import logging
import random
import statistics
import tempfile
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from ace.graph.clock import seconds_between, utc_now
from ace.graph.llm import ModelSettings, make_client_factory
from ace.graph.types import MERGER_TYPES, GraphFailure, Kind, WorkerType

if TYPE_CHECKING:
    from agent2.core.session import Session
    from ace.graph.edits import Edit
    from ace.graph.graph import WorkerGraph
    from ace.graph.runner import WorkerOutputs
    from ace.neat.genome import Genome
    from ace.neat.history import History

logger = logging.getLogger(__name__)

Scorer = Callable[["Edit", "Session"], "Awaitable[float] | float"]

TYPE_WEIGHT = 1.0
MERGER_END_BONUS = 0.5
CRITIC_BONUS = 0.25
SIZE_PENALTY = 0.1
SIZE_FREE_WORKERS = 6


@runtime_checkable
class Evaluator(Protocol):
    """Anything that can assign a raw fitness to a genome."""

    async def evaluate(self, genome: Genome) -> float:
        """Raw fitness of ``genome``; higher is better."""
        ...


# --------------------------------------------------------------------------- mock


class MockEvaluator:
    """Deterministic structural score for tests: rewards specific structures.

    See the module docstring for the formula. ``calls`` counts ``evaluate`` invocations and
    ``evaluated_ids`` records the genome ids in call order (tests use them to check that
    elites are not re-evaluated). ``delay`` (seconds) makes each call yield to the loop.
    """

    def __init__(
        self,
        *,
        noise: float = 0.0,
        scores: dict[str, float] | None = None,
        delay: float = 0.0,
        history: History | None = None,
    ) -> None:
        self.noise = noise
        self.scores = dict(scores or {})
        self.delay = delay
        self.history = history
        self.calls = 0
        self.evaluated_ids: list[str] = []

    def describe(self) -> dict[str, Any]:
        """The ``evaluator`` entry of ``run.json``."""
        return {"type": "mock", "tasks": []}

    def score(self, genome: Genome) -> float:
        """The synchronous mock score of ``genome``."""
        graph = genome.to_graph()
        structure_hash = graph.structure_hash()
        if structure_hash in self.scores:
            return float(self.scores[structure_hash])
        types = [node.type for node in graph.nodes.values()]
        value = TYPE_WEIGHT * len(set(types))
        if graph.nodes[graph.end_source()].type in MERGER_TYPES:
            value += MERGER_END_BONUS
        value += CRITIC_BONUS * sum(1 for t in types if t is WorkerType.CRITIC)
        value -= SIZE_PENALTY * max(0, len(types) - SIZE_FREE_WORKERS)
        if self.noise:
            u = random.Random(int(structure_hash, 16)).random()
            value += self.noise * (2.0 * u - 1.0)
        return value

    async def evaluate(self, genome: Genome) -> float:
        """Async wrapper around :meth:`score`."""
        self.calls += 1
        self.evaluated_ids.append(genome.id)
        started = utc_now()
        if self.history is not None:
            self.history.log_event("evaluation_start", genome_id=genome.id, task_index=0)
        if self.delay:
            await asyncio.sleep(self.delay)
        value = self.score(genome)
        if self.history is not None:
            finished = utc_now()
            duration = seconds_between(started, finished)
            self.history.log_evaluation(
                {
                    "genome_id": genome.id,
                    "generation": getattr(genome, "generation", None),
                    "task_index": 0,
                    "score": value,
                    "mock": True,
                    "started_at": started,
                    "finished_at": finished,
                    "duration_s": duration,
                }
            )
            self.history.log_event(
                "evaluation_end", genome_id=genome.id, task_index=0, score=value,
                failure=None, duration_s=duration,
            )
        return value


# -------------------------------------------------------------------------- tasks


@dataclass
class Task:
    """One evaluation task: a project, the prompt given to the graph, and a scorer."""

    project_dir: Path
    prompt: str
    scorer: Scorer


def nonempty_diff(edit: Edit, session: Session) -> float:
    """Builtin scorer: 1.0 if the edit changes anything, else 0.0."""
    return 0.0 if edit.is_empty() else 1.0


BUILTIN_SCORERS: dict[str, Scorer] = {"nonempty_diff": nonempty_diff}


def scorer_name(scorer: Scorer) -> str:
    """Inverse of :func:`resolve_scorer` where possible: builtin name or ``module:qualname``."""
    for name, fn in BUILTIN_SCORERS.items():
        if fn is scorer:
            return name
    module = getattr(scorer, "__module__", None)
    qualname = getattr(scorer, "__qualname__", None) or getattr(scorer, "__name__", None)
    if module and qualname:
        return f"{module}:{qualname}"
    return repr(scorer)


def resolve_scorer(spec: str) -> Scorer:
    """``"nonempty_diff"`` or ``"package.module:function"`` -> scorer callable."""
    if spec in BUILTIN_SCORERS:
        return BUILTIN_SCORERS[spec]
    module_name, sep, attr = spec.partition(":")
    if not sep or not module_name or not attr:
        raise ValueError(
            f"unknown scorer {spec!r}: expected one of {sorted(BUILTIN_SCORERS)} or 'module:function'"
        )
    target: Any = importlib.import_module(module_name)
    for part in attr.split("."):
        target = getattr(target, part)
    if not callable(target):
        raise TypeError(f"scorer {spec!r} is not callable")
    return target


def load_tasks(path: Path | str) -> list[Task]:
    """Load a JSON list of ``{"project_dir", "prompt", "scorer"}`` task objects.

    ``scorer`` defaults to ``"nonempty_diff"``; relative ``project_dir`` values resolve
    against the directory containing the JSON file.
    """
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a JSON list of tasks")
    tasks: list[Task] = []
    for index, raw in enumerate(data):
        try:
            project_dir = Path(raw["project_dir"]).expanduser()
            prompt = str(raw["prompt"])
        except (KeyError, TypeError) as exc:
            raise ValueError(f"{path}: task {index} needs 'project_dir' and 'prompt'") from exc
        if not project_dir.is_absolute():
            project_dir = (path.parent / project_dir).resolve()
        tasks.append(Task(project_dir, prompt, resolve_scorer(raw.get("scorer", "nonempty_diff"))))
    return tasks


# ----------------------------------------------------------------- task evaluator


class TaskEvaluator:
    """Runs a genome's worker graph on every task; fitness is the mean task score."""

    def __init__(
        self,
        tasks: list[Task],
        settings: ModelSettings,
        client_factory: Callable[[], Any] | None = None,
        concurrency: int = 4,
        rollout_dir: Path | None = None,
        history: History | None = None,
    ) -> None:
        if concurrency < 1:
            raise ValueError(f"concurrency must be >= 1, got {concurrency}")
        self.tasks = list(tasks)
        self.settings = settings
        self.client_factory = client_factory if client_factory is not None else make_client_factory(settings)
        self.concurrency = concurrency
        self.rollout_dir = Path(rollout_dir).resolve() if rollout_dir is not None else None
        self.history = history
        self.last_results: list[dict[str, Any]] = []
        self.results_by_genome: dict[str, list[dict[str, Any]]] = {}
        self._semaphore: asyncio.Semaphore | None = None
        self._semaphore_loop: asyncio.AbstractEventLoop | None = None

    def describe(self) -> dict[str, Any]:
        """The ``evaluator`` entry of ``run.json``."""
        return {
            "type": "task",
            "tasks": [
                {
                    "index": i,
                    "project_dir": str(t.project_dir),
                    "prompt": t.prompt,
                    "scorer": scorer_name(t.scorer),
                }
                for i, t in enumerate(self.tasks)
            ],
        }

    def _get_semaphore(self) -> asyncio.Semaphore:
        """One semaphore per running event loop (tests may call ``asyncio.run`` repeatedly)."""
        loop = asyncio.get_running_loop()
        if self._semaphore is None or self._semaphore_loop is not loop:
            self._semaphore = asyncio.Semaphore(self.concurrency)
            self._semaphore_loop = loop
        return self._semaphore

    @property
    def rollout_root(self) -> Path:
        """Absolute root of all rollout dirs (a fresh temp dir when none was given)."""
        if self.rollout_dir is None:
            self.rollout_dir = Path(tempfile.mkdtemp(prefix="agent2-rollouts-")).resolve()
        return self.rollout_dir

    def task_rollout_dir(self, genome: Genome, index: int, task: Task) -> Path:
        """``<rollout_root>/<genome.id>/<index>``; rejects a root scanned as project files."""
        path = self.rollout_root / genome.id / str(index)
        project = Path(task.project_dir).resolve()
        if path.is_relative_to(project):
            inside = path.relative_to(project).parts
            if not any(part.startswith(".") for part in inside):
                raise ValueError(
                    f"rollout dir {path} lies inside project {project} on a non-hidden path; "
                    "its rollout files would be read as project files"
                )
        return path

    async def _run_task(self, genome: Genome, index: int, task: Task) -> dict[str, Any]:
        from agent2.core.session import Session
        from ace.graph.runner import GraphRunner, RunContext

        record: dict[str, Any] = {
            "task": index,
            "project_dir": str(task.project_dir),
            "score": 0.0,
            "failure": None,
            "node_id": None,
            "diff_chars": 0,
        }
        rollout_dir = self.task_rollout_dir(genome, index, task)
        record["rollout_dir"] = str(rollout_dir)
        async with self._get_semaphore():
            started = utc_now()
            if self.history is not None:
                self.history.log_event("evaluation_start", genome_id=genome.id, task_index=index)
            session = Session(str(task.project_dir), rollout_dir=str(rollout_dir))
            ctx = RunContext(
                session,
                task.prompt,
                self.settings,
                self.client_factory,
                rollout_dir=rollout_dir,
            )
            graph = genome.to_graph()
            result = None
            failure: GraphFailure | None = None
            try:
                result = await GraphRunner(graph).run(ctx)
            except GraphFailure as exc:
                failure = exc
                record["failure"] = f"{type(exc).__name__}: {exc}"
                record["node_id"] = getattr(exc, "node_id", None)
                logger.warning("genome %s failed on task %d (%s): %s",
                               genome.id, index, task.project_dir, record["failure"])
            if result is not None:
                score = task.scorer(result.edit, session)
                if inspect.isawaitable(score):
                    score = await score
                record["score"] = float(score)
                record["diff_chars"] = len(result.diff)
            if self.history is not None:
                finished = utc_now()
                full = build_evaluation_record(
                    genome=genome, graph=graph, task_index=index, task=task, session=session,
                    rollout_dir=rollout_dir, started_at=started, finished_at=finished,
                    score=record["score"], result=result, failure=failure,
                )
                self.history.log_evaluation(full)
                self.history.log_event(
                    "evaluation_end", genome_id=genome.id, task_index=index,
                    score=record["score"], failure=record["failure"],
                    duration_s=full["duration_s"],
                )
        return record

    async def evaluate(self, genome: Genome) -> float:
        """Mean task score of ``genome`` (0.0 with no tasks; a failed task scores 0.0)."""
        records = list(
            await asyncio.gather(*(self._run_task(genome, i, t) for i, t in enumerate(self.tasks)))
        )
        self.last_results = records
        self.results_by_genome[genome.id] = records
        if not records:
            return 0.0
        return statistics.fmean(r["score"] for r in records)


# --------------------------------------------------------------- recording helpers


def _load_rollouts(rollout_dir: Path) -> dict[str, dict[str, Any]]:
    """Rollout id -> rollout JSON for every readable ``*.json`` in ``rollout_dir``."""
    rollouts: dict[str, dict[str, Any]] = {}
    if not rollout_dir.is_dir():
        return rollouts
    for path in sorted(rollout_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data, dict):
            rollouts[str(data.get("id") or path.stem)] = data
    return rollouts


def _count_roles(rollout: dict[str, Any]) -> tuple[int, int]:
    llm = tools = 0
    for message in rollout.get("raw_chat") or []:
        role = message.get("role") if isinstance(message, dict) else None
        if role == "assistant":
            llm += 1
        elif role == "tool":
            tools += 1
    return llm, tools


def _view_summary(view: Any, session: Session) -> dict[str, int]:
    if view is None:
        return {"files": 0, "entries": 0, "rendered_chars": 0}
    entries = sum(len(fv.levels) + len(fv.line_ranges) for fv in view.files.values())
    try:
        from ace.graph.rendering import render_view

        rendered = len(render_view(session, view))
    except Exception:  # noqa: BLE001 - a summary must never fail the evaluation
        rendered = 0
    return {"files": len(view.files), "entries": entries, "rendered_chars": rendered}


def _failure_type(exc: BaseException) -> str:
    cause = exc.__cause__
    return type(cause).__name__ if cause is not None else type(exc).__name__


def build_evaluation_record(
    *,
    genome: Genome,
    graph: WorkerGraph,
    task_index: int,
    task: Task,
    session: Session,
    rollout_dir: Path,
    started_at: str,
    finished_at: str,
    score: float,
    result: Any = None,
    failure: GraphFailure | None = None,
) -> dict[str, Any]:
    """One ``evaluations.jsonl`` record (recording spec 11.5)."""
    from ace.graph.properties import resolve

    if result is not None:
        outputs: dict[int, WorkerOutputs] = dict(result.outputs)
        timings: dict[int, tuple[str, str]] = dict(getattr(result, "timings", {}) or {})
        order = list(result.order)
        failed_node = None
    else:
        outputs = dict(getattr(failure, "partial_outputs", {}) or {})
        timings = dict(getattr(failure, "partial_timings", {}) or {})
        failed_node = getattr(failure, "failed_node", getattr(failure, "node_id", None))
        order = [nid for nid, _ in sorted(timings.items(), key=lambda kv: (kv[1][0], kv[0]))]

    rollouts = _load_rollouts(rollout_dir)
    by_config: dict[str, list[str]] = {}
    for rid, data in sorted(rollouts.items(), key=lambda kv: (str(kv[1].get("date_deployed")), kv[0])):
        by_config.setdefault(str(data.get("config_path")), []).append(rid)

    workers: list[dict[str, Any]] = []
    for nid in sorted(graph.nodes):
        node = graph.nodes[nid]
        try:
            properties = resolve(node.type, node.properties)
        except Exception:  # noqa: BLE001 - record raw indices for an unresolvable gene
            properties = dict(node.properties)
        produced = outputs.get(nid)
        if failed_node is not None and nid == failed_node:
            status, error = "failed", f"{type(failure).__name__}: {failure}"
        elif produced is not None:
            status, error = "ok", None
        else:
            status, error = "skipped", None
        rollout_ids = list(produced.rollout_ids) if produced is not None else []
        for rid in by_config.get(f"graph/{node.type.value}/{nid}", []):
            if rid not in rollout_ids:
                rollout_ids.append(rid)
        llm_calls = tool_calls = 0
        for rid in rollout_ids:
            if rid in rollouts:
                a, t = _count_roles(rollouts[rid])
                llm_calls += a
                tool_calls += t
        start, end = timings.get(nid, (None, None))
        view = produced.view if produced is not None else None
        edit = produced.edit if produced is not None else None
        edit_chars = 0
        if edit is not None:
            try:
                edit_chars = len(edit.unified_diff(session))
            except Exception:  # noqa: BLE001
                edit_chars = 0
        workers.append(
            {
                "node_id": nid,
                "type": node.type.value,
                "properties": properties,
                "started_at": start,
                "finished_at": end,
                "duration_s": seconds_between(start, end) if start and end else None,
                "status": status,
                "error": error,
                "rollout_ids": rollout_ids,
                "rollout_dir": str(rollout_dir),
                "llm_calls": llm_calls,
                "tool_calls": tool_calls,
                "report": produced.report if produced is not None else None,
                "view": view.to_json() if view is not None else None,
                "view_summary": _view_summary(view, session),
                "edit_touched_paths": sorted(edit.touched_paths()) if edit is not None else [],
                "edit_chars": edit_chars,
            }
        )

    inputs: dict[str, dict[str, list[int]]] = {}
    for nid in sorted(graph.nodes):
        entry: dict[str, list[int]] = {"views_from": [], "edits_from": [], "reports_from": []}
        for edge in graph.incoming(nid):
            key = {Kind.VIEW: "views_from", Kind.EDIT: "edits_from", Kind.REPORT: "reports_from"}[edge.kind]
            entry[key].append(edge.src)
        inputs[str(nid)] = {k: sorted(v) for k, v in entry.items()}

    diff = result.diff if result is not None else None
    return {
        "genome_id": genome.id,
        "generation": getattr(genome, "generation", None),
        "task_index": task_index,
        "task": {
            "project_dir": str(task.project_dir),
            "prompt": task.prompt,
            "scorer": scorer_name(task.scorer),
        },
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_s": seconds_between(started_at, finished_at),
        "score": float(score),
        "failure": None if failure is None else f"{type(failure).__name__}: {failure}",
        "failure_node_id": failed_node,
        "failure_type": None if failure is None else _failure_type(failure),
        "diff": diff,
        "touched_paths": sorted(result.edit.touched_paths()) if result is not None else [],
        "diff_chars": len(diff) if diff is not None else 0,
        "order": order,
        "workers": workers,
        "inputs": inputs,
    }


__all__ = [
    "BUILTIN_SCORERS",
    "Evaluator",
    "MockEvaluator",
    "Scorer",
    "Task",
    "TaskEvaluator",
    "build_evaluation_record",
    "load_tasks",
    "nonempty_diff",
    "resolve_scorer",
    "scorer_name",
]
