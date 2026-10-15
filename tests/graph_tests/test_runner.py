"""Tests for graph/runner.py using fake (LLM-free) workers."""
from __future__ import annotations

import asyncio
import functools
from typing import Any, Awaitable, Callable

import pytest

from ace.graph.edits import Edit
from ace.graph.graph import WorkerEdge, WorkerGraph, WorkerNode
from ace.graph.properties import default_properties
from ace.graph.runner import (
    GraphResult,
    GraphRunner,
    ModelSettings,
    RunContext,
    WorkerInputs,
    WorkerOutputs,
)
from ace.graph.types import END, GraphFailure, InvalidGraph, Kind, WorkerType
from ace.graph.views import Level, View
from ace.graph.workers.base import Worker

Handler = Callable[[WorkerNode, WorkerInputs], Awaitable[WorkerOutputs]]

V, E, R = Kind.VIEW, Kind.EDIT, Kind.REPORT
T = WorkerType


def sync(fn: Callable[..., Awaitable[Any]]) -> Callable[..., Any]:
    """Run an async test body with ``asyncio.run`` (no async pytest plugin is installed)."""

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return asyncio.run(fn(*args, **kwargs))

    return wrapper


def node(nid: int, t: WorkerType) -> WorkerNode:
    return WorkerNode(nid, t, default_properties(t))


def view_of(nid: int) -> View:
    view = View()
    view.add_file(f"pkg/file{nid}.py", Level.COLLAPSED)
    return view


def edit_of(nid: int) -> Edit:
    return Edit({"notes.txt": f"edited by {nid}\n".encode()})


async def default_handler(n: WorkerNode, inputs: WorkerInputs) -> WorkerOutputs:
    """Produce every kind the type outputs, tagged with the node id."""
    from ace.graph.types import WORKER_SPECS

    outs = WORKER_SPECS[n.type].outputs
    return WorkerOutputs(
        view=view_of(n.id) if V in outs else None,
        edit=edit_of(n.id) if E in outs else None,
        report=f"report {n.id}" if R in outs else None,
    )


class Log:
    def __init__(self) -> None:
        self.events: list[tuple[str, int]] = []
        self.inputs: dict[int, WorkerInputs] = {}
        self.props: dict[int, dict[str, Any]] = {}


def fake_workers(
    log: Log, handlers: dict[int, Handler] | None = None
) -> dict[WorkerType, type[Worker]]:
    """One fake Worker subclass per type; per-node handlers override the default."""
    handlers = handlers or {}
    classes: dict[WorkerType, type[Worker]] = {}
    for t in WorkerType:

        async def run(self, ctx, n, inputs, props):  # noqa: ANN001
            log.events.append(("start", n.id))
            log.inputs[n.id] = inputs
            log.props[n.id] = props
            result = await handlers.get(n.id, default_handler)(n, inputs)
            log.events.append(("end", n.id))
            return result

        classes[t] = type(f"Fake{t.name.title()}", (Worker,), {"type": t, "run": run})
    return classes


def diamond() -> WorkerGraph:
    """VIEWER 1 -> {EDITOR 2, VIEWEDITOR 3} -> JUDGEMERGER 4 -> END."""
    nodes = {
        1: node(1, T.VIEWER),
        2: node(2, T.EDITOR),
        3: node(3, T.VIEWEDITOR),
        4: node(4, T.JUDGEMERGER),
    }
    edges = [
        WorkerEdge(1, 2, V),
        WorkerEdge(1, 2, R),
        WorkerEdge(1, 3, V),
        WorkerEdge(2, 4, E),
        WorkerEdge(2, 4, V),
        WorkerEdge(2, 4, R),
        WorkerEdge(3, 4, E),
        WorkerEdge(4, END, E),
    ]
    graph = WorkerGraph(nodes, edges)
    graph.validate()
    return graph


@pytest.fixture
def ctx(session) -> RunContext:
    return RunContext(
        session=session,
        prompt="Fix the bug.",
        settings=ModelSettings(model="fake-model"),
        client_factory=lambda: None,
        run_id="test-run",
    )


@sync
async def test_diamond_order_and_grouped_inputs(ctx):
    log = Log()
    result = await GraphRunner(diamond(), workers=fake_workers(log)).run(ctx)

    assert isinstance(result, GraphResult)
    assert result.order == [1, 2, 3, 4]
    starts = [nid for kind, nid in log.events if kind == "start"]
    ends = {nid: i for i, (kind, nid) in enumerate(log.events) if kind == "end"}
    first_start = {nid: i for i, (kind, nid) in enumerate(log.events) if kind == "start"}
    assert starts[0] == 1 and starts[-1] == 4
    assert ends[1] < first_start[2] and ends[1] < first_start[3]
    assert ends[2] < first_start[4] and ends[3] < first_start[4]

    viewer_in = log.inputs[1]
    assert viewer_in == WorkerInputs()

    editor_in = log.inputs[2]
    assert editor_in.views == {1: view_of(1)}
    assert editor_in.reports == {1: "report 1"}
    assert editor_in.edits == {}
    assert editor_in.source_types == {1: T.VIEWER}

    vieweditor_in = log.inputs[3]
    assert vieweditor_in.views == {1: view_of(1)}
    assert vieweditor_in.reports == {}  # no REPORT edge 1->3
    assert vieweditor_in.source_types == {1: T.VIEWER}

    merger_in = log.inputs[4]
    assert merger_in.edits == {2: edit_of(2), 3: edit_of(3)}
    assert merger_in.views == {2: view_of(2)}  # only the VIEW edge from 2
    assert merger_in.reports == {2: "report 2"}
    assert merger_in.source_types == {2: T.EDITOR, 3: T.VIEWEDITOR}
    assert list(merger_in.edits) == [2, 3]

    # Properties are resolved to concrete values.
    assert isinstance(log.props[2]["max_steps"], int)
    assert isinstance(log.props[4]["context_budget"], int)

    assert set(result.outputs) == {1, 2, 3, 4}
    assert result.edit == edit_of(4)


@sync
async def test_result_diff_matches_end_edit(ctx, session):
    result = await GraphRunner(diamond(), workers=fake_workers(Log())).run(ctx)
    assert result.diff == edit_of(4).unified_diff(session)
    assert "+edited by 4" in result.diff
    assert "-line one" in result.diff


@sync
async def test_siblings_run_concurrently(ctx):
    editor_started = asyncio.Event()
    vieweditor_started = asyncio.Event()

    async def editor(n: WorkerNode, inputs: WorkerInputs) -> WorkerOutputs:
        editor_started.set()
        await asyncio.wait_for(vieweditor_started.wait(), timeout=2)
        return await default_handler(n, inputs)

    async def vieweditor(n: WorkerNode, inputs: WorkerInputs) -> WorkerOutputs:
        vieweditor_started.set()
        await asyncio.wait_for(editor_started.wait(), timeout=2)
        return await default_handler(n, inputs)

    log = Log()
    result = await GraphRunner(
        diamond(), workers=fake_workers(log, {2: editor, 3: vieweditor})
    ).run(ctx)
    assert result.edit == edit_of(4)
    # Both started before either finished.
    events = log.events
    assert events.index(("start", 3)) < events.index(("end", 2))
    assert events.index(("start", 2)) < events.index(("end", 3))


@sync
async def test_missing_payload_names_edge(ctx):
    async def editor_without_edit(n: WorkerNode, inputs: WorkerInputs) -> WorkerOutputs:
        return WorkerOutputs(view=view_of(n.id), edit=None, report="r")

    log = Log()
    runner = GraphRunner(diamond(), workers=fake_workers(log, {2: editor_without_edit}))
    with pytest.raises(GraphFailure) as info:
        await runner.run(ctx)
    assert "edge 2->4 [edit]" in str(info.value)
    assert 4 not in log.inputs  # downstream work abandoned


@sync
async def test_missing_end_payload_names_edge(ctx):
    async def merger_without_edit(n: WorkerNode, inputs: WorkerInputs) -> WorkerOutputs:
        return WorkerOutputs(view=view_of(n.id))

    runner = GraphRunner(diamond(), workers=fake_workers(Log(), {4: merger_without_edit}))
    with pytest.raises(GraphFailure, match=r"edge 4->END \[edit\]"):
        await runner.run(ctx)


@sync
async def test_worker_exception_becomes_graph_failure_naming_node(ctx):
    sibling_cancelled = asyncio.Event()

    async def broken(n: WorkerNode, inputs: WorkerInputs) -> WorkerOutputs:
        await asyncio.sleep(0)
        raise ValueError("boom")

    async def slow(n: WorkerNode, inputs: WorkerInputs) -> WorkerOutputs:
        try:
            await asyncio.sleep(5)
        except asyncio.CancelledError:
            sibling_cancelled.set()
            raise
        return await default_handler(n, inputs)

    log = Log()
    runner = GraphRunner(diamond(), workers=fake_workers(log, {3: broken, 2: slow}))
    with pytest.raises(GraphFailure) as info:
        await runner.run(ctx)
    assert "node 3 (vieweditor)" in str(info.value)
    assert "boom" in str(info.value)
    assert isinstance(info.value.__cause__, ValueError)
    assert info.value.node_id == 3
    assert sibling_cancelled.is_set()
    assert 4 not in log.inputs


@sync
async def test_graph_failure_from_worker_passes_through(ctx):
    class Custom(GraphFailure):
        pass

    async def failing(n: WorkerNode, inputs: WorkerInputs) -> WorkerOutputs:
        raise Custom("bad ace")

    runner = GraphRunner(diamond(), workers=fake_workers(Log(), {3: failing}))
    with pytest.raises(Custom) as info:
        await runner.run(ctx)
    assert info.value.node_id == 3


@sync
async def test_non_outputs_return_is_failure(ctx):
    async def wrong(n: WorkerNode, inputs: WorkerInputs) -> Any:
        return {"edit": None}

    runner = GraphRunner(diamond(), workers=fake_workers(Log(), {1: wrong}))
    with pytest.raises(GraphFailure, match=r"node 1 \(viewer\)"):
        await runner.run(ctx)


@sync
async def test_missing_worker_class_is_failure(ctx):
    workers = fake_workers(Log())
    del workers[T.JUDGEMERGER]
    with pytest.raises(GraphFailure, match="node 4"):
        await GraphRunner(diamond(), workers=workers).run(ctx)


def test_invalid_graph_rejected_in_init():
    graph = WorkerGraph({1: node(1, T.VIEWER)}, [WorkerEdge(1, END, V)])
    with pytest.raises(InvalidGraph):
        GraphRunner(graph, workers={})


def test_codegraph_is_lazy_and_assignable(session):
    sentinel = object()
    ctx = RunContext(session, "p", ModelSettings("m"), lambda: None, sentinel)  # type: ignore[arg-type]
    assert ctx.codegraph is sentinel
    ctx.codegraph = None
    assert ctx._codegraph is None
    built = ctx.codegraph
    assert built is ctx.codegraph  # cached
    assert ("pkg/models.py", "") in built.nodes
    assert len(RunContext(session, "p", ModelSettings("m"), lambda: None).run_id) == 32


def test_registry_loads_lazily():
    import ace.graph.workers as workers_pkg

    registry = workers_pkg.WORKERS
    assert isinstance(registry, dict)
    for t, cls in registry.items():
        assert cls.type is t


# ----------------------------------------------------------- run recording (spec 11.5)


@sync
async def test_result_timings_cover_every_worker(ctx):
    from ace.graph.clock import parse_ts

    result = await GraphRunner(diamond(), workers=fake_workers(Log())).run(ctx)
    assert set(result.timings) == {1, 2, 3, 4}
    for start, end in result.timings.values():
        assert parse_ts(start) <= parse_ts(end)
        assert parse_ts(start).utcoffset().total_seconds() == 0
    # level order: the viewer finished before the merger started
    assert parse_ts(result.timings[1][1]) <= parse_ts(result.timings[4][0])


@sync
async def test_graph_failure_carries_partial_results(ctx):
    async def broken(n: WorkerNode, inputs: WorkerInputs) -> WorkerOutputs:
        raise ValueError("boom")

    runner = GraphRunner(diamond(), workers=fake_workers(Log(), {3: broken}))
    with pytest.raises(GraphFailure) as info:
        await runner.run(ctx)
    exc = info.value
    assert "node 3 (vieweditor)" in str(exc)  # message unchanged
    assert exc.node_id == 3 and exc.failed_node == 3
    # the viewer and the successful sibling editor finished; the merger never ran
    assert set(exc.partial_outputs) == {1, 2}
    assert all(isinstance(o, WorkerOutputs) for o in exc.partial_outputs.values())
    assert set(exc.partial_timings) == {1, 2, 3}
    assert 4 not in exc.partial_timings


@sync
async def test_missing_payload_failure_has_partial_attributes(ctx):
    async def editor_without_edit(n: WorkerNode, inputs: WorkerInputs) -> WorkerOutputs:
        return WorkerOutputs(view=view_of(n.id), edit=None, report="r")

    runner = GraphRunner(diamond(), workers=fake_workers(Log(), {2: editor_without_edit}))
    with pytest.raises(GraphFailure) as info:
        await runner.run(ctx)
    assert info.value.failed_node == 2
    assert {1, 2, 3} <= set(info.value.partial_outputs)
