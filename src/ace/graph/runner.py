"""Worker-graph runtime: run context, payload containers and the graph runner.

See GRAPH_NEAT_DESIGN.md section 9.

**Disk writes.** The graph runtime never writes into the project folder except through the
Session's rollout dir (``Session.rollout_path``, where every worker agent saves its
rollout JSON). Workers edit private ``Workspace`` overlays that are never committed; the
END edit is returned as an :class:`~ace.graph.edits.Edit`. Callers choose where
rollouts go by constructing the Session with ``rollout_dir`` (an absolute path escapes
the project folder) and record that same path in :attr:`RunContext.rollout_dir`, which is
informational: the runner itself does not read it.

Choices made where the contract is open:

* :class:`ModelSettings` lives in :mod:`ace.graph.llm` and is re-exported here.
* :class:`RunContext.codegraph <RunContext>` is lazy: pass ``codegraph=None`` (the
  default) and the :class:`~ace.graph.codegraph.CodebaseGraph` is built from the
  pristine session on first attribute access, then cached. It may also be assigned.
  Because ``codegraph`` now has a default, ``run_id`` defaults to a fresh uuid hex.
* :class:`WorkerInputs` gains ``source_types`` (source node id -> worker type) so workers
  can label reports and edits by the type of the worker that produced them. Every field
  defaults to an empty dict.
* Worker classes come from ``ace.graph.workers.WORKERS`` (imported lazily inside
  :meth:`GraphRunner.run`) unless ``GraphRunner(graph, workers=...)`` overrides them.
* Node properties are resolved with ``properties.resolve`` right before the worker runs,
  so a bad property index fails that node with a :class:`GraphFailure`.
* Scheduling: the runner repeatedly takes every not-yet-run worker whose incoming-edge
  sources have all finished (one topological level, ascending id), runs them concurrently
  as tasks, waits until all finish or one fails, cancels the rest on failure, and collects
  results with ``asyncio.gather(return_exceptions=True)``. If several workers of a level
  fail, the failure of the lowest node id is raised.
* Missing payloads are detected as soon as the producing worker finishes: for each of its
  outgoing edges (END included, in edge order) the carried payload must not be ``None``,
  otherwise :class:`GraphFailure` names the edge.
* Errors: a non-``GraphFailure`` exception from a worker becomes
  ``GraphFailure("node N (type) failed: ...")`` chained to the original. A
  ``GraphFailure`` (e.g. ``AceError``) is re-raised unchanged except that, if it does not
  already carry one, it gets a ``node_id`` attribute and a note naming the node.
* ``GraphResult.order`` lists worker ids in the order they were started (levels in
  sequence, ascending id within a level).
* **Timings (run recording, spec 11.5).** ``GraphResult.timings`` maps every worker id to
  ``(started_at, finished_at)`` ISO-8601 UTC strings (:func:`ace.graph.clock.utc_now`),
  taken right before property resolution and right after the worker returned or raised.
  A ``GraphFailure`` raised by :meth:`GraphRunner.run` additionally carries
  ``partial_outputs`` (outputs of every worker that finished successfully, including
  successful siblings of the failing level), ``partial_timings`` (every worker that
  started, cancelled siblings included) and ``failed_node`` (the failing node id or
  ``None``). These attributes are additive; messages and ``node_id`` are unchanged.
"""
from __future__ import annotations

import asyncio
from dataclasses import InitVar, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable
from uuid import uuid4

from ace.graph.clock import utc_now
from ace.graph.edits import Edit
from ace.graph.graph import WorkerEdge, WorkerGraph, WorkerNode
from ace.graph.llm import ModelSettings
from ace.graph.types import END, GraphFailure, Kind, WorkerType
from ace.graph.views import View

if TYPE_CHECKING:
    from agent2.core.session import Session
    from ace.graph.codegraph import CodebaseGraph
    from ace.graph.workers.base import Worker

__all__ = [
    "GraphFailure",
    "GraphResult",
    "GraphRunner",
    "ModelSettings",
    "RunContext",
    "WorkerInputs",
    "WorkerOutputs",
]


@dataclass
class RunContext:
    """Everything a worker needs that is shared across one graph run.

    ``codegraph`` is lazily built from ``session`` on first access when not supplied.
    ``rollout_dir`` records where the session saves worker rollouts (normally
    ``session.rollout_path``); it is informational only.
    """

    session: Session
    prompt: str
    settings: ModelSettings
    client_factory: Callable[[], Any]
    codegraph: InitVar[CodebaseGraph | None] = None
    run_id: str = field(default_factory=lambda: uuid4().hex)
    rollout_dir: Path | None = None

    def __post_init__(self, codegraph: CodebaseGraph | None) -> None:
        self._codegraph = codegraph


def _get_codegraph(self: RunContext) -> CodebaseGraph:
    """The run's codebase graph, built from the pristine session on first access."""
    if self._codegraph is None:
        from ace.graph.codegraph import CodebaseGraph

        self._codegraph = CodebaseGraph(self.session)
    return self._codegraph


def _set_codegraph(self: RunContext, value: CodebaseGraph | None) -> None:
    self._codegraph = value


# Installed after class creation so the dataclass ``__init__`` keeps ``codegraph`` as a
# keyword/positional argument while attribute access goes through the lazy property.
RunContext.codegraph = property(_get_codegraph, _set_codegraph)  # type: ignore[assignment]


@dataclass
class WorkerInputs:
    """Payloads delivered to one worker, grouped by source node id (never flattened)."""

    views: dict[int, View] = field(default_factory=dict)
    edits: dict[int, Edit] = field(default_factory=dict)
    reports: dict[int, str] = field(default_factory=dict)
    source_types: dict[int, WorkerType] = field(default_factory=dict)


@dataclass
class WorkerOutputs:
    """What one worker produced. ``None`` means the worker does not output that kind."""

    view: View | None = None
    edit: Edit | None = None
    report: str | None = None
    rollout_ids: list[str] = field(default_factory=list)


@dataclass
class GraphResult:
    """Result of a successful graph run: the END edit, its diff and every worker's output."""

    edit: Edit
    diff: str
    outputs: dict[int, WorkerOutputs]
    order: list[int]
    timings: dict[int, tuple[str, str]] = field(default_factory=dict)


_PAYLOAD_ATTR: dict[Kind, str] = {Kind.VIEW: "view", Kind.EDIT: "edit", Kind.REPORT: "report"}
_INPUT_ATTR: dict[Kind, str] = {Kind.VIEW: "views", Kind.EDIT: "edits", Kind.REPORT: "reports"}


def _payload(edge: WorkerEdge, outputs: dict[int, WorkerOutputs], graph: WorkerGraph) -> Any:
    """The payload ``edge`` carries, or ``GraphFailure`` naming the edge if it is missing."""
    produced = outputs.get(edge.src)
    value = getattr(produced, _PAYLOAD_ATTR[edge.kind]) if produced is not None else None
    if value is None:
        failure = GraphFailure(
            f"{edge.label()}: {graph.nodes[edge.src].label()} produced no "
            f"{edge.kind.value} payload"
        )
        failure.node_id = edge.src  # type: ignore[attr-defined]  # the worker at fault
        raise failure
    return value


class GraphRunner:
    """Executes a validated :class:`WorkerGraph` on one task."""

    def __init__(
        self,
        graph: WorkerGraph,
        workers: dict[WorkerType, type[Worker]] | None = None,
    ) -> None:
        """Validate ``graph`` (raises ``InvalidGraph``).

        ``workers`` overrides the worker registry (tests); by default
        ``ace.graph.workers.WORKERS`` is consulted when the graph runs.
        """
        graph.validate()
        self.graph = graph
        self.workers = workers

    def _registry(self) -> dict[WorkerType, type[Worker]]:
        if self.workers is not None:
            return self.workers
        from ace.graph.workers import IMPORT_ERRORS, WORKERS

        missing = sorted({n.type for n in self.graph.nodes.values()} - set(WORKERS))
        if missing:
            details = "; ".join(
                f"{t.value}" + (f" ({IMPORT_ERRORS[t.value]})" if t.value in IMPORT_ERRORS else "")
                for t in missing
            )
            raise GraphFailure(f"no worker implementation registered for: {details}")
        return WORKERS

    def build_inputs(self, node_id: int, outputs: dict[int, WorkerOutputs]) -> WorkerInputs:
        """Collect exactly the payloads carried by ``node_id``'s incoming edges."""
        inputs = WorkerInputs()
        for edge in self.graph.incoming(node_id):
            getattr(inputs, _INPUT_ATTR[edge.kind])[edge.src] = _payload(
                edge, outputs, self.graph
            )
            inputs.source_types[edge.src] = self.graph.nodes[edge.src].type
        for name in ("views", "edits", "reports", "source_types"):
            mapping = getattr(inputs, name)
            setattr(inputs, name, {key: mapping[key] for key in sorted(mapping)})
        return inputs

    async def _run_node(
        self,
        ctx: RunContext,
        node: WorkerNode,
        inputs: WorkerInputs,
        registry: dict[WorkerType, type[Worker]],
        timings: dict[int, tuple[str, str]] | None = None,
    ) -> WorkerOutputs:
        started = utc_now()
        if timings is not None:
            timings[node.id] = (started, started)
        try:
            return await self._run_node_inner(ctx, node, inputs, registry)
        finally:
            if timings is not None:
                timings[node.id] = (started, utc_now())

    async def _run_node_inner(
        self,
        ctx: RunContext,
        node: WorkerNode,
        inputs: WorkerInputs,
        registry: dict[WorkerType, type[Worker]],
    ) -> WorkerOutputs:
        try:
            from ace.graph.properties import resolve

            try:
                worker_cls = registry[node.type]
            except KeyError:
                raise GraphFailure(
                    f"{node.label()}: no worker implementation for {node.type.value}"
                ) from None
            props = resolve(node.type, node.properties)
            result = await worker_cls().run(ctx, node, inputs, props)
            if not isinstance(result, WorkerOutputs):
                raise GraphFailure(
                    f"{node.label()} returned {type(result).__name__}, expected WorkerOutputs"
                )
            return result
        except GraphFailure as exc:
            if getattr(exc, "node_id", None) is None:
                exc.node_id = node.id  # type: ignore[attr-defined]
                exc.add_note(f"raised by {node.label()}")
            raise
        except Exception as exc:
            failure = GraphFailure(f"{node.label()} failed: {type(exc).__name__}: {exc}")
            failure.node_id = node.id  # type: ignore[attr-defined]
            raise failure from exc

    def _check_outgoing(self, node_id: int, outputs: dict[int, WorkerOutputs]) -> None:
        for edge in self.graph.outgoing(node_id):
            _payload(edge, outputs, self.graph)

    async def _run_level(
        self,
        ctx: RunContext,
        ready: list[int],
        outputs: dict[int, WorkerOutputs],
        registry: dict[WorkerType, type[Worker]],
        timings: dict[int, tuple[str, str]] | None = None,
        completed: dict[int, WorkerOutputs] | None = None,
    ) -> dict[int, WorkerOutputs]:
        inputs = {nid: self.build_inputs(nid, outputs) for nid in ready}
        tasks = {
            nid: asyncio.create_task(
                self._run_node(ctx, self.graph.nodes[nid], inputs[nid], registry, timings),
                name=f"graph-{ctx.run_id}-node-{nid}",
            )
            for nid in ready
        }
        try:
            _, pending = await asyncio.wait(tasks.values(), return_when=asyncio.FIRST_EXCEPTION)
        finally:
            for task in tasks.values():
                if not task.done():
                    task.cancel()
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        by_id = dict(zip(tasks, results))
        failures = [
            (nid, r) for nid, r in by_id.items()
            if isinstance(r, BaseException) and not isinstance(r, asyncio.CancelledError)
        ]
        if failures:
            if completed is not None:
                completed.update(
                    (nid, r) for nid, r in by_id.items() if isinstance(r, WorkerOutputs)
                )
            raise min(failures, key=lambda item: item[0])[1]
        cancelled = [nid for nid, r in by_id.items() if isinstance(r, asyncio.CancelledError)]
        if cancelled:  # cancelled from outside the runner
            raise asyncio.CancelledError()
        return by_id  # type: ignore[return-value]

    async def run(self, ctx: RunContext) -> GraphResult:
        """Run every worker level by level; raise ``GraphFailure`` on any worker failure."""
        outputs: dict[int, WorkerOutputs] = {}
        timings: dict[int, tuple[str, str]] = {}
        try:
            return await self._run(ctx, outputs, timings)
        except GraphFailure as exc:
            exc.partial_outputs = dict(outputs)  # type: ignore[attr-defined]
            exc.partial_timings = dict(timings)  # type: ignore[attr-defined]
            exc.failed_node = getattr(exc, "node_id", None)  # type: ignore[attr-defined]
            raise

    async def _run(
        self,
        ctx: RunContext,
        outputs: dict[int, WorkerOutputs],
        timings: dict[int, tuple[str, str]],
    ) -> GraphResult:
        registry = self._registry()
        order: list[int] = []
        remaining = set(self.graph.nodes)
        while remaining:
            ready = sorted(
                nid for nid in remaining
                if all(e.src in outputs for e in self.graph.incoming(nid))
            )
            if not ready:  # unreachable for a validated (acyclic) graph
                raise GraphFailure(f"no runnable workers among {sorted(remaining)}")
            order.extend(ready)
            level = await self._run_level(
                ctx, ready, outputs, registry, timings=timings, completed=outputs
            )
            outputs.update(level)
            remaining.difference_update(ready)
            for nid in ready:
                self._check_outgoing(nid, outputs)
        end_edge = self.graph.incoming(END)[0]
        edit = _payload(end_edge, outputs, self.graph)
        return GraphResult(
            edit=edit,
            diff=edit.unified_diff(ctx.session),
            outputs=outputs,
            order=order,
            timings=dict(timings),
        )
